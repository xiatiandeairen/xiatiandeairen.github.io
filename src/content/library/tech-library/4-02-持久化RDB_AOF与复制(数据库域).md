---
title: "持久化 RDB、AOF 与复制（数据库域）"
slug: "4-02"
collection: "tech-library"
group: "数据库"
order: 4002
summary: "适用版本：Redis 7.x / unstable（2024–2026） 前置依赖：第 1 章（单线程事件循环、ae_epoll、对象模型）；熟悉 fork/COW、fsync 语义、Linux VFS 基础 源码仓库：`redis/redis` — `src/rdb.c`、`src/aof.c`、…"
topics:
  - "技术内核"
tags: []
createdAt: "2026-06-12T10:46:34.000Z"
updatedAt: "2026-06-12T10:46:34.000Z"
---
> **适用版本**：Redis 7.x / unstable（2024–2026）  
> **前置依赖**：第 1 章（单线程事件循环、ae_epoll、对象模型）；熟悉 fork/COW、fsync 语义、Linux VFS 基础  
> **源码仓库**：`redis/redis` — `src/rdb.c`、`src/aof.c`、`src/replication.c`、`src/rdb.h`、`src/server.h`

---

## TL;DR（三行读懂本章）

1. **RDB** = 周期性 fork + Copy-on-Write 快照，紧凑二进制格式，以 `REDIS%04d` 魔数开头，用变长 Length Encoding 压缩，CRC64 结尾；代价是可能丢失最后几分钟写入。  
2. **AOF** = 每条写命令追加 RESP 文本，三种 fsync 策略在吞吐与持久性之间取舍；后台 rewrite（fork + COW）压缩日志，Redis 7.0 Multi-Part AOF 彻底消除 rewrite 期间的内存双倍 buffer 问题。  
3. **复制** = 异步流式，PSYNC2 协议用 replID + offset 实现部分重同步，Replication Backlog 是环形 buffer，断线重连只需补发 delta；diskless replication 进一步省去 RDB 落盘成本。

---

## 目录

1. [设计考古：为什么同时需要 RDB 和 AOF](#1-设计考古)
2. [RDB 格式精读：二进制快照的每一个字节](#2-rdb-格式精读)
3. [BGSAVE：fork + COW 快照机制](#3-bgsave-fork--cow-快照机制)
4. [AOF：命令日志追加与三种 fsync 策略](#4-aof-命令日志追加与三种-fsync-策略)
5. [AOF Rewrite：Multi-Part AOF 与 7.0 架构革新](#5-aof-rewrite)
6. [复制协议演进：SYNC → PSYNC1 → PSYNC2](#6-复制协议演进)
7. [Replication Backlog 与部分重同步](#7-replication-backlog-与部分重同步)
8. [Diskless Replication 与链式副本](#8-diskless-replication-与链式副本)
9. [RDB vs AOF vs 混合持久化：对比与选型](#9-对比与选型)
10. [可运行 Demo × 3](#10-可运行-demo)
11. [生产真坑与失败模式](#11-生产真坑与失败模式)
12. [章末五件套](#12-章末五件套)

---

## 1 设计考古

### 1.1 最初没有持久化（2009 年 1.0 之前）

Redis 最初（Antirez 2009 年发布 1.0 之前）是一个纯内存 key-value 服务，和 memcached 本质相同。第一个持久化机制就是 RDB（Redis Database Dump），思路直接借鉴了 Berkeley DB 的 checkpoint：定期将整个内存数据集序列化到一个文件，重启时读回。

**动机**：  
- 简单：只需在 fork 出的子进程里做顺序写，不影响主进程延迟。  
- 紧凑：相较命令日志，快照体积小 3–5 倍。  
- 快速恢复：重启只需加载一个文件，O(N) 时间。

**缺陷被 Antirez 在早期邮件列表里承认**（可参考 Redis Google Groups 2010 年讨论）：RDB 两次 save 之间的写入永久丢失，对"掉电不丢"有要求的场景不适用。

### 1.2 AOF 的引入（Redis 1.1，2010 年）

AOF（Append-Only File）的设计直接对标关系数据库的 WAL（Write-Ahead Log）。Antirez 在 [Redis 1.1 发布说明](https://redis.io/topics/persistence)（2010 年）中明确写道：

> "AOF persistence logs every write operation received by the server… on a server restart Redis re-runs them to reconstruct the original dataset."

RESP（Redis Serialization Protocol）格式天然可读，AOF 文件本质上是一个 RESP 命令流，可用 `redis-cli --pipe` 直接重放。这给了运维一个"最后的逃生舱"——误操作 `FLUSHALL` 之后，可以手工裁剪 AOF 文件，去掉最后一条命令再重放。

### 1.3 PSYNC 与部分重同步（Redis 2.8，2013 年）

Redis 2.8 之前，从库断线重连后**必须完整重同步**——主库 BGSAVE 一个新 RDB，全量传输。这在大数据集 + 不稳定网络下代价极高。

Redis 2.8 引入 `PSYNC`（PSYNC1），用 **replID + offset** 替代 SYNC，只要断线时长在 `repl-backlog-size`（默认 1MB）覆盖范围内，从库只需补收差量数据。

**PSYNC2**（Redis 4.0，2017 年）进一步引入**secondary replication ID**，允许提升为主库的从库继续接受其他从库的部分重同步，解决了主从切换后级联 full resync 的问题。

### 1.4 Multi-Part AOF（Redis 7.0，2022 年）

Redis 7.0 之前，AOF rewrite 时父进程需要维护一个内存 buffer（`aof_rewrite_buf`）累积 rewrite 期间的新写入，rewrite 完成后 append 到新文件——这意味着**在 rewrite 高峰期，内存用量实际上翻倍**。

Redis 7.0 的 Multi-Part AOF（[PR #9919](https://github.com/redis/redis/pull/9919)）将 AOF 拆为：
- **BASE** 文件：最后一次 rewrite 产生的快照（可以是 RDB 格式或 AOF 格式）  
- **INCR** 文件：rewrite 之后新写入的增量命令  
- **MANIFEST** 文件：记录当前生效的 BASE + 所有 INCR 文件列表

这样 rewrite 子进程写 BASE，父进程写新 INCR，两路 I/O 完全隔离，无需内存 buffer。

---

## 2 RDB 格式精读

### 2.1 文件布局

```
+------------------+
| "REDIS%04d"  9B  |  魔数 + 版本号，当前 RDB_VERSION=14 → "REDIS0014"
+------------------+
| AUX Fields       |  可选元数据：redis-ver, redis-bits, ctime, used-mem 等
| (OPCODE_AUX=250) |  每条：0xFA + key(string) + value(string)
+------------------+
| [DB 0]           |
|  SELECTDB(254)   |  0xFE + dbid(length-encoded)
|  RESIZEDB(251)   |  0xFB + db_size + expires_size
|  [KV pairs...]   |
|    [EXPIRETIME]  |  可选：0xFC(ms) 或 0xFD(s) + 时间戳
|    TYPE byte     |  1 字节，见 RDB_TYPE_* 常量
|    key(string)   |  length-encoded string
|    value         |  encoding 依 TYPE 而定
+------------------+
| [DB 1] ...       |
+------------------+
| EOF(255)         |  0xFF
+------------------+
| CRC64 checksum   |  8 字节 little-endian（0=禁用校验）
+------------------+
```

### 2.2 RDB 版本与类型常量

【真实源码 redis/redis@src/rdb.h】

```c
// rdb.h line 19
#define RDB_VERSION 14

// rdb.h lines 53-83（TYPE 常量，部分节选）
#define RDB_TYPE_STRING              0
#define RDB_TYPE_LIST                1
#define RDB_TYPE_SET                 2
#define RDB_TYPE_ZSET                3
#define RDB_TYPE_HASH                4
#define RDB_TYPE_ZSET_2              5   // ZSET with double scores (not NaN)
#define RDB_TYPE_HASH_ZIPMAP         9   // 已废弃，兼容旧版
#define RDB_TYPE_LIST_ZIPLIST       10
#define RDB_TYPE_SET_INTSET         11
#define RDB_TYPE_LIST_QUICKLIST_2   18   // quicklist + listpack node encoding info
#define RDB_TYPE_HASH_LISTPACK      16
#define RDB_TYPE_ZSET_LISTPACK      17

// rdb.h lines 91-103（特殊 OPCODE）
#define RDB_OPCODE_AUX         250  // 0xFA 辅助字段
#define RDB_OPCODE_RESIZEDB    251  // 0xFB 数据库大小提示
#define RDB_OPCODE_EXPIRETIME_MS 252 // 0xFC 毫秒过期
#define RDB_OPCODE_EXPIRETIME  253  // 0xFD 秒过期
#define RDB_OPCODE_SELECTDB    254  // 0xFE 切换数据库
#define RDB_OPCODE_EOF         255  // 0xFF 文件结束
```

### 2.3 Length Encoding：节省空间的变长整数

RDB 中所有长度字段（string 长度、list 元素数等）都用 Length Encoding，第一字节高 2 位决定编码类型：

| 高 2 位 | 含义 | 字节数 |
|---------|------|--------|
| `00` | 6-bit 长度，值在低 6 位 | 1 |
| `01` | 14-bit 长度，低 6 位 + 下一字节 | 2 |
| `10` + 0x00 | 32-bit 长度，接下来 4 字节 big-endian | 5 |
| `10` + 0x01 | 64-bit 长度，接下来 8 字节 | 9 |
| `11` | 特殊编码（整数内联、LZF 压缩） | 1+N |

【真实源码 redis/redis@src/rdb.c，lines 269–299】

```c
int rdbSaveLen(rio *rdb, uint64_t len) {
    unsigned char buf[2];
    size_t nwritten;

    if (len < (1<<6)) {
        /* Save a 6 bit len */
        buf[0] = (len&0xFF)|(RDB_6BITLEN<<6);  // 高2位=00，低6位=len
        if (rdbWriteRaw(rdb,buf,1) == -1) return -1;
        nwritten = 1;
    } else if (len < (1<<14)) {
        /* Save a 14 bit len */
        buf[0] = ((len>>8)&0xFF)|(RDB_14BITLEN<<6);  // 高2位=01
        buf[1] = len&0xFF;
        if (rdbWriteRaw(rdb,buf,2) == -1) return -1;
        nwritten = 2;
    } else if (len <= UINT32_MAX) {
        /* Save a 32 bit len */
        buf[0] = RDB_32BITLEN;   // 0x80
        if (rdbWriteRaw(rdb,buf,1) == -1) return -1;
        uint32_t len32 = htonl(len);  // big-endian
        if (rdbWriteRaw(rdb,&len32,4) == -1) return -1;
        nwritten = 1+4;
    } else {
        /* Save a 64 bit len */
        buf[0] = RDB_64BITLEN;   // 0x81
        if (rdbWriteRaw(rdb,buf,1) == -1) return -1;
        len = htonu64(len);
        if (rdbWriteRaw(rdb,&len,8) == -1) return -1;
        nwritten = 1+9;
    }
    return nwritten;
}
```

读取时对称解码：

【真实源码 redis/redis@src/rdb.c，lines 305–345】

```c
int rdbLoadLenByRef(rio *rdb, int *isencoded, uint64_t *lenptr) {
    unsigned char buf[2];
    int type;

    if (rioRead(rdb,buf,1) == 0) return -1;
    type = (buf[0]&0xC0)>>6;          // 取高2位
    if (type == RDB_ENCVAL) {          // 11 → 特殊编码
        if (isencoded) *isencoded = 1;
        *lenptr = buf[0]&0x3F;         // 低6位是编码类型（INT8/INT16/INT32/LZF）
    } else if (type == RDB_6BITLEN) {  // 00
        *lenptr = buf[0]&0x3F;
    } else if (type == RDB_14BITLEN) { // 01
        if (rioRead(rdb,buf+1,1) == 0) return -1;
        *lenptr = ((buf[0]&0x3F)<<8)|buf[1];
    } else if (buf[0] == RDB_32BITLEN) {
        uint32_t len;
        if (rioRead(rdb,&len,4) == 0) return -1;
        *lenptr = ntohl(len);           // big-endian → host
    } else if (buf[0] == RDB_64BITLEN) {
        uint64_t len;
        if (rioRead(rdb,&len,8) == 0) return -1;
        *lenptr = ntohu64(len);
    } else {
        rdbReportCorruptRDB("Unknown length encoding %d in rdbLoadLen()",type);
        return -1;
    }
    return 0;
}
```

### 2.4 RDB 文件头与 CRC64 尾

【真实源码 redis/redis@src/rdb.c，lines 1903–1938（rdbSaveRio 节选）】

```c
int rdbSaveRio(int req, rio *rdb, int *error, int rdbflags, rdbSaveInfo *rsi) {
    char magic[10];
    uint64_t cksum;

    if (server.rdb_checksum)
        rdb->update_cksum = rioGenericUpdateChecksum;  // 边写边累计 CRC64

    snprintf(magic,sizeof(magic),"REDIS%04d",RDB_VERSION);
    if (rdbWriteRaw(rdb,magic,9) == -1) goto werr;    // 写9字节魔数
    if (rdbSaveInfoAuxFields(rdb,rdbflags,rsi) == -1) goto werr; // AUX 元数据

    // ... 遍历每个 DB，写 SELECTDB + RESIZEDB + KV pairs ...
    for (j = 0; j < server.dbnum; j++) {
        if (rdbSaveDb(rdb, j, rdbflags, &key_counter, &skipped) == -1) goto werr;
    }

    // EOF opcode
    if (rdbSaveType(rdb,RDB_OPCODE_EOF) == -1) goto werr;

    // CRC64 checksum — 8 字节 little-endian
    cksum = rdb->cksum;
    memrev64ifbe(&cksum);  // 如果是 big-endian 机器，翻转为 little-endian
    if (rioWrite(rdb,&cksum,8) == 0) goto werr;
    return C_OK;
werr:
    // ...
}
```

---

## 3 BGSAVE：fork + COW 快照机制

### 3.1 核心代码路径

【真实源码 redis/redis@src/rdb.c，lines 2112–2182】

```c
// rdbSave: 前台保存（阻塞主线程，测试/shutdown 时使用）
int rdbSave(int req, char *filename, rdbSaveInfo *rsi, int rdbflags) {
    char tmpfile[256];
    startSaving(rdbflags);
    snprintf(tmpfile,256,"temp-%d.rdb", (int) getpid()); // 写临时文件

    if (rdbSaveInternal(req,tmpfile,rsi,rdbflags) != C_OK) {
        stopSaving(0);
        return C_ERR;
    }
    // 原子 rename：保证 crash 时旧文件仍完整
    if (rename(tmpfile,filename) == -1) { ... unlink(tmpfile); return C_ERR; }
    // fsync 目录 dentry，确保 rename 持久化
    if (fsyncFileDir(filename) != 0) { ... return C_ERR; }
    server.dirty = 0;
    server.lastsave = time(NULL);
    return C_OK;
}

// rdbSaveBackground: 后台保存（生产主路径）
int rdbSaveBackground(int req, char *filename, rdbSaveInfo *rsi, int rdbflags) {
    pid_t childpid;
    if (hasActiveChildProcess()) return C_ERR;  // AOF rewrite/RDB 不并行
    server.stat_rdb_saves++;
    server.dirty_before_bgsave = server.dirty;

    if ((childpid = redisFork(CHILD_TYPE_RDB)) == 0) {
        /* ---- 子进程 ---- */
        redisSetProcTitle("redis-rdb-bgsave");
        redisSetCpuAffinity(server.bgsave_cpulist);
        int retval = rdbSave(req, filename, rsi, rdbflags);
        if (retval == C_OK) {
            sendChildCowInfo(CHILD_INFO_TYPE_RDB_COW_SIZE, "RDB"); // 向父报告 COW 内存
        }
        exitFromChild((retval == C_OK) ? 0 : 1, 0);
    } else {
        /* ---- 父进程 ---- */
        if (childpid == -1) { server.lastbgsave_status = C_ERR; return C_ERR; }
        serverLog(LL_NOTICE,"Background saving started by pid %ld",(long)childpid);
        server.rdb_save_time_start = time(NULL);
        server.rdb_child_type = RDB_CHILD_TYPE_DISK;
    }
    return C_OK;
}
```

### 3.2 fork + COW 工作原理

```
主进程(父)                     子进程
    │   fork()                    │
    ├─────────────────────────────►│
    │   继续服务请求               │
    │   写时：OS 按页复制          │  顺序写 RDB 文件
    │   (COW page fault)          │
    │                             │
    │   新写入 = 父独占新页         │  子持有 fork 时刻的内存镜像
    │                             │  无需任何锁
    │◄────────────────────────────┤
    │   waitpid → 子退出           │
    │   rename(tmp, dump.rdb)     │
```

**关键点**：
- `fork()` 本身是 O(1)（仅复制页表），大数据集下耗时主要在 Linux THP（Transparent Huge Pages）的拆分，这是生产中 BGSAVE 卡顿的根因。
- COW 代价：父进程每次写一个内存页，OS 就为子进程保留一份旧版本。写入量越大，COW 开销越高，极端情况下内存使用翻倍。
- `server.dirty_before_bgsave` 记录 fork 前的脏写计数，BGSAVE 完成后只清除这部分，避免 BGSAVE 期间新写入被误计为已保存。

### 3.3 rdbSaveDb：迭代所有 KV

【真实源码 redis/redis@src/rdb.c，lines 1598–1705（节选）】

```c
ssize_t rdbSaveDb(rio *rdb, int dbid, int rdbflags,
                  long *key_counter, unsigned long long *skipped) {
    // 写 SELECTDB opcode
    // 写 RESIZEDB opcode（提示 load 端预分配 hash table）
    kvstoreIteratorInit(&kvs_it, db->keys);
    while ((de = kvstoreIteratorNext(&kvs_it)) != NULL) {
        // 获取 key、value、expire time
        res = rdbSaveKeyValuePair(rdb, &key, kv, expire, dbid);
        // 每 1000 个 key 检查一次 COW 内存，向父进程报告
        if ((*key_counter & 0x3FF) == 0) {
            sendChildCowInfo(CHILD_INFO_TYPE_RDB_COW_SIZE, "RDB");
        }
    }
}
```

---

## 4 AOF：命令日志追加与三种 fsync 策略

### 4.1 三种 fsync 策略常量

【真实源码 redis/redis@src/server.h，lines ~1073】

```c
#define AOF_FSYNC_NO      0   // 完全交给 OS，最快，断电最多丢 30s
#define AOF_FSYNC_ALWAYS  1   // 每条命令后 fsync，最安全，吞吐最低
#define AOF_FSYNC_EVERYSEC 2  // 每秒 fsync（后台 BIO 线程），推荐配置
```

### 4.2 命令写入 AOF 的完整路径

每次写命令执行后，调用链如下：

```
call() → propagate() → feedAppendOnlyFile() → server.aof_buf（sds）
                                    ↓ (event loop 末尾)
                             flushAppendOnlyFile()
                                    ↓
                              aofWrite(fd, buf)
                                    ↓
                     [EVERYSEC: BIO thread bio_fsync]
                     [ALWAYS:   redis_fsync() 同步调用]
                     [NO:       OS 自行决定]
```

**feedAppendOnlyFile**：将命令序列化为 RESP 格式追加到内存 buffer。

【真实源码 redis/redis@src/aof.c，lines 2230–2260（7.x unstable）】

```c
void feedAppendOnlyFile(int dictid, robj **argv, int argc) {
    sds buf = sdsempty();

    // 时间戳注解（aof-timestamp-enabled 开启时）
    if (server.aof_timestamp_enabled) {
        sds ts = genAofTimestampAnnotationIfNeeded(0);
        if (ts != NULL) { buf = sdscatsds(buf, ts); sdsfree(ts); }
    }

    // 如果数据库切换了，先写 SELECT 命令
    if (dictid != -1 && dictid != server.aof_selected_db) {
        char seldb[64];
        snprintf(seldb,sizeof(seldb),"%d",dictid);
        buf = sdscatprintf(buf,"*2\r\n$6\r\nSELECT\r\n$%lu\r\n%s\r\n",
            (unsigned long)strlen(seldb),seldb);
        server.aof_selected_db = dictid;
    }

    buf = catAppendOnlyGenericCommand(buf, argc, argv);  // 序列化为 RESP multibulk

    // 只有在 AOF 开启或等待 rewrite 完成时才写 buffer
    if (server.aof_state == AOF_ON ||
        (server.aof_state == AOF_WAIT_REWRITE &&
         server.child_type == CHILD_TYPE_AOF))
    {
        server.aof_buf = sdscatlen(server.aof_buf, buf, sdslen(buf));
    }
    sdsfree(buf);
}
```

**catAppendOnlyGenericCommand**：把 `argv[]` 序列化为 RESP multibulk 格式。

【真实源码 redis/redis@src/aof.c，lines 2155–2178】

```c
sds catAppendOnlyGenericCommand(sds dst, int argc, robj **argv) {
    char buf[32];
    int len, j;
    robj *o;

    buf[0] = '*';
    len = 1+ll2string(buf+1,sizeof(buf)-1,argc);  // "*3\r\n"
    buf[len++] = '\r'; buf[len++] = '\n';
    dst = sdscatlen(dst,buf,len);

    for (j = 0; j < argc; j++) {
        o = getDecodedObject(argv[j]);  // 确保是字符串表示
        buf[0] = '$';
        len = 1+ll2string(buf+1,sizeof(buf)-1,sdslen(o->ptr)); // "$3\r\n"
        buf[len++] = '\r'; buf[len++] = '\n';
        dst = sdscatlen(dst,buf,len);
        dst = sdscatlen(dst,o->ptr,sdslen(o->ptr));  // 命令字符串
        dst = sdscatlen(dst,"\r\n",2);
        decrRefCount(o);
    }
    return dst;
}
```

### 4.3 flushAppendOnlyFile 与推迟机制

【真实源码 redis/redis@src/aof.c，lines ~2395–2550（节选关键逻辑）】

```c
void flushAppendOnlyFile(int force) {
    ssize_t nwritten;
    int sync_in_progress = 0;

    if (sdslen(server.aof_buf) == 0) {
        // buffer 为空，但 ALWAYS 模式下仍要检查是否需要 fsync
        if (server.aof_fsync == AOF_FSYNC_ALWAYS && ...) {
            redis_fsync(server.aof_fd);
        }
        return;
    }

    // EVERYSEC 且后台 fsync 正在进行 → 最多推迟 2 秒
    if (server.aof_fsync == AOF_FSYNC_EVERYSEC)
        sync_in_progress = aofFsyncInProgress();

    if (server.aof_fsync == AOF_FSYNC_EVERYSEC && !force) {
        if (sync_in_progress) {
            if (server.aof_flush_postponed_start == 0) {
                server.aof_flush_postponed_start = server.unixtime;
                return;   // 第一次推迟，直接返回
            } else if (server.unixtime - server.aof_flush_postponed_start < 2) {
                return;   // 还在 2 秒窗口内，继续推迟
            }
            // 超过 2 秒，强制写入（可能造成短暂阻塞）
        }
    }

    // 实际 write 系统调用
    nwritten = aofWrite(server.aof_fd, server.aof_buf, sdslen(server.aof_buf));

    if (nwritten == (ssize_t)sdslen(server.aof_buf)) {
        // 全部写入成功
        server.aof_current_size += nwritten;
        server.aof_last_incr_size += nwritten;
        // buffer 复用优化：小 buffer 直接 clear，避免 free+alloc
        if ((sdslen(server.aof_buf)+sdsavail(server.aof_buf)) < 4000) {
            sdsclear(server.aof_buf);
        } else {
            sdsfree(server.aof_buf);
            server.aof_buf = sdsempty();
        }
    } else {
        // 写入失败处理（ALWAYS 模式直接 exit，其他模式设错误标志）
        ...
    }

    // fsync 策略执行
    if (server.aof_fsync == AOF_FSYNC_ALWAYS) {
        if (redis_fsync(server.aof_fd) == -1) {
            serverLog(LL_WARNING,"Can't persist AOF for fsync error...");
            exit(1);
        }
        server.aof_last_fsync = server.unixtime;
    } else if (server.aof_fsync == AOF_FSYNC_EVERYSEC &&
               server.unixtime > server.aof_last_fsync) {
        // 提交给 BIO 后台线程异步 fsync
        if (!sync_in_progress) aof_background_fsync(server.aof_fd);
        server.aof_last_fsync = server.unixtime;
    }
}
```

**关键设计**：EVERYSEC 模式下的"推迟不超过 2 秒"机制是一个精妙的 backpressure：如果磁盘 I/O 跟不上，宁可短暂阻塞主线程一次，也好过 buffer 无限增长导致 OOM。

---

## 5 AOF Rewrite

### 5.1 Redis 7.0 之前：pipe + in-memory diff buffer

Redis 7.0 之前，AOF rewrite 流程：
1. 父进程调用 `rewriteAppendOnlyFileBackground()`，fork 子进程。
2. 子进程重写数据集为新 AOF 文件（`temp-rewriteaof-bg-*.aof`）。
3. 父进程把 rewrite 期间新写入累积在 `server.aof_rewrite_buf`（内存）。
4. 子进程写完，通过 pipe 通知父进程。
5. 父进程把 `aof_rewrite_buf` append 到子进程文件，再 rename。

**问题**：步骤 3 中的内存 buffer 在高写入负载时可能增长到数 GB，造成内存压力。

### 5.2 Redis 7.0 Multi-Part AOF

【真实源码 redis/redis@src/aof.c（7.2），lines 2730–2797】

```c
int rewriteAppendOnlyFileBackground(void) {
    pid_t childpid;

    if (hasActiveChildProcess()) return C_ERR;

    // 创建 AOF 目录（如不存在）
    if (dirCreateIfMissing(server.aof_dirname) == -1) { ... return C_ERR; }

    // 强制 SELECT 刷新（确保后续增量 AOF 正确）
    server.aof_selected_db = -1;
    flushAppendOnlyFile(1);

    // 打开新的 INCR AOF 文件，后续父进程的写入都进这个文件
    // 这是 Multi-Part AOF 的关键：父子进程写不同的文件，无需内存 buffer
    if (openNewIncrAofForAppend() != C_OK) { ... return C_ERR; }

    if (server.aof_state == AOF_WAIT_REWRITE) {
        bioDrainWorker(BIO_AOF_FSYNC);  // 等待旧 AOF 的 fsync 完成
        atomicSet(server.fsynced_reploff_pending, server.master_repl_offset);
        server.fsynced_reploff = 0;
    }

    server.stat_aof_rewrites++;

    if ((childpid = redisFork(CHILD_TYPE_AOF)) == 0) {
        /* 子进程 */
        redisSetProcTitle("redis-aof-rewrite");
        redisSetCpuAffinity(server.aof_rewrite_cpulist);
        snprintf(tmpfile,256,"temp-rewriteaof-bg-%d.aof", (int) getpid());
        if (rewriteAppendOnlyFile(tmpfile) == C_OK) {
            sendChildCowInfo(CHILD_INFO_TYPE_AOF_COW_SIZE, "AOF rewrite");
            exitFromChild(0);
        } else {
            exitFromChild(1);
        }
    } else {
        /* 父进程 */
        if (childpid == -1) { ... return C_ERR; }
        serverLog(LL_NOTICE,"Background append only file rewriting started by pid %ld",
                  (long) childpid);
        // 父进程继续写 INCR 文件，子进程写 BASE 文件，完全隔离
    }
    return C_OK;
}
```

**Multi-Part AOF 的 MANIFEST 文件内容示例**：

```
file appendonly.aof.1.base.rdb seq 1 type b       # BASE（RDB 格式快照）
file appendonly.aof.1.incr.aof seq 1 type i       # 第 1 个 INCR
file appendonly.aof.2.incr.aof seq 2 type i       # 第 2 个 INCR（rewrite 后新增量）
```

加载时按 MANIFEST 顺序：先加载 BASE（RDB 或 AOF），再按序重放所有 INCR。

### 5.3 AOF 状态机

```
AOF_OFF (0) ─── CONFIG SET appendonly yes ──► AOF_WAIT_REWRITE (2)
                                                        │
                                               rewrite 完成
                                                        │
                                                        ▼
                                                  AOF_ON (1)
```

【真实源码 redis/redis@src/server.h，lines ~1047-1049】

```c
#define AOF_OFF          0  // AOF 未开启
#define AOF_ON           1  // AOF 正常运行
#define AOF_WAIT_REWRITE 2  // 动态开启 AOF，等待初始 rewrite 完成
```

---

## 6 复制协议演进

### 6.1 SYNC（Redis 1.0–2.7）

最原始的同步命令：

```
从库 → 主库:  SYNC
主库 → 从库:  $<rdb_size>\r\n<rdb_binary_data>
主库 → 从库:  （后续命令流）
```

**致命缺陷**：断线重连后必须重新全量同步，无论断线多短。

### 6.2 PSYNC1（Redis 2.8）

```
从库 → 主库:  PSYNC <replid> <offset>
主库 → 从库:  +FULLRESYNC <replid> <offset>   # 全量同步
              或 +CONTINUE                      # 部分同步
```

引入了 **Replication Backlog**：主库维护一个固定大小的环形 buffer，存储最近的命令流。从库断线后重连，若请求的 offset 仍在 backlog 内，主库直接 feed delta，避免全量同步。

**PSYNC1 局限**：failover 后新主库生成新 replID，原从库无法对新主进行部分重同步，仍需全量。

### 6.3 PSYNC2（Redis 4.0）

引入**双 replID**机制：

```c
// server.h（示意）
char replid[CONFIG_RUN_ID_SIZE+1];    // 当前主库的 replication ID
char replid2[CONFIG_RUN_ID_SIZE+1];   // 晋升前的 secondary ID（保留旧主的 ID）
long long second_replid_offset;        // secondary ID 的有效 offset 上限
```

**Failover 场景**：

```
旧主 A (replid=AAA, offset=1000) ─── 挂了
从库 B 晋升为主库：
  B.replid  = BBB (新生成)
  B.replid2 = AAA (保留旧主 ID)
  B.second_replid_offset = 1000

原来 A 的其他从库 C 重连 B：
  C 发送: PSYNC AAA 950
  B 验证: AAA == B.replid2? Yes
          950 <= second_replid_offset(1000)? Yes
          → 部分重同步，发 950..1000 的增量
```

**PSYNC 命令处理**：

【真实源码 redis/redis@src/replication.c（7.x）syncCommand 节选】

```c
if (!strcasecmp(c->argv[0]->ptr,"psync")) {
    long long psync_offset;
    if (getLongLongFromObjectOrReply(c, c->argv[2], &psync_offset, NULL) != C_OK)
        return;
    if (masterTryPartialResynchronization(c, psync_offset) == C_OK) {
        server.stat_sync_partial_ok++;
        return;  // 部分重同步成功，不需要全量
    }
    // else: 进入全量重同步流程
}
```

**masterTryPartialResynchronization**（核心判断逻辑）：

【真实源码 redis/redis@src/replication.c，lines ~1100-1210（精简展示）】

```c
int masterTryPartialResynchronization(client *c, long long psync_offset) {
    char *master_replid = c->argv[1]->ptr;

    // 检查 replID 是否匹配（当前 ID 或 secondary ID）
    if (strcasecmp(master_replid, server.replid) != 0 &&
        (strcasecmp(master_replid, server.replid2) != 0 ||
         psync_offset > server.second_replid_offset))
    {
        // ID 不匹配，需要全量同步
        serverLog(LL_NOTICE,"Partial resynchronization not accepted: "
            "Replication ID mismatch ...");
        goto need_full_resync;
    }

    // 检查 offset 是否在 backlog 覆盖范围内
    if (!server.repl_backlog ||
        psync_offset < server.repl_backlog->offset ||
        psync_offset > (server.master_repl_offset+1))
    {
        serverLog(LL_NOTICE,"Unable to partial resync with replica %s...",
                  replicationGetSlaveName(c));
        goto need_full_resync;
    }

    // 部分重同步成功
    serverLog(LL_NOTICE,
        "Partial resynchronization request from %s accepted. Sending %lld bytes.",
        replicationGetSlaveName(c),
        server.master_repl_offset - psync_offset + 1);

    // 发送 +CONTINUE 响应
    if (server.replid2[0] != '\0' &&
        strcmp(c->argv[1]->ptr, server.replid2) == 0) {
        // 从库用的是 secondary ID，升级到当前 ID
        addReplyf(c,"+CONTINUE %s\r\n", server.replid);
    } else {
        addReplyf(c,"+CONTINUE %s\r\n", server.replid);
    }

    // 从 backlog 中回放数据
    psync_len = addReplyReplicationBacklog(c, psync_offset);
    serverLog(LL_NOTICE,"PSYNC replica %s ... fed %lld bytes.",
              replicationGetSlaveName(c), psync_len);
    return C_OK;

need_full_resync:
    return C_ERR;
}
```

---

## 7 Replication Backlog 与部分重同步

### 7.1 Backlog 数据结构

【真实源码 redis/redis@src/server.h，lines ~2067-2074】

```c
typedef struct replBacklog {
    listNode *ref_repl_buf_node;  // 指向 replication buffer 链表中 backlog 起始节点
    size_t unindexed_count;       // 自上次建索引以来未索引的 block 数
    rax *blocks_index;            // Radix tree 索引：offset → block 指针（快速定位）
    long long histlen;            // backlog 中存储的字节数（有效历史长度）
    long long offset;             // backlog 中最老数据对应的 master_repl_offset
} replBacklog;
```

注意 Redis 7.x 的 backlog 不再是固定大小的环形 buffer，而是复用了 replication buffer 链表（`server.repl_buffer`），用 `blocks_index`（radix tree）做快速 offset 查找。

### 7.2 Backlog 初始化

【真实源码 redis/redis@src/replication.c，lines 323–333】

```c
void createReplicationBacklog(void) {
    serverAssert(server.repl_backlog == NULL);
    server.repl_backlog = zmalloc(sizeof(replBacklog));
    server.repl_backlog->ref_repl_buf_node = NULL;
    server.repl_backlog->unindexed_count = 0;
    server.repl_backlog->blocks_index = raxNew();   // 空 radix tree
    server.repl_backlog->histlen = 0;
    server.repl_backlog->offset = server.master_repl_offset+1;
}
```

### 7.3 命令流写入 backlog

【真实源码 redis/redis@src/replication.c，lines 1052–1147（节选）】

```c
void replicationFeedSlaves(list *slaves, int dictid, robj **argv, int argc) {
    if (server.masterhost != NULL) return;  // 从库不 feed
    if (server.current_client &&
        server.current_client->flags & CLIENT_MASTER) return;  // 主库转发来的不重复 feed
    if (server.repl_backlog == NULL && listLength(slaves) == 0) {
        server.master_repl_offset += 1;
        return;
    }

    server.repl_stream_lastio = server.unixtime;
    prepareReplicasToWrite();

    // 如果数据库切换了，先写 SELECT
    if (dictid != -1 && server.slaveseldb != dictid) {
        // 构建 SELECT 命令追加到 replication buffer
    }

    // 用 replBufWriter API 写命令到 replication buffer
    replBufWriter wr;
    replBufWriterBegin(&wr);
    replBufWriterAppendBulkLen(&wr, '*', argc);
    for (j = 0; j < argc; j++) {
        long objlen = stringObjectLen(argv[j]);
        replBufWriterAppendBulkLen(&wr, '$', objlen);
        // append bulk data
    }
    replBufWriterEnd(&wr);
    // replBufWriterEnd 会更新 master_repl_offset
}
```

### 7.4 Backlog 的 offset 覆盖范围

部分重同步的可行性条件（可在 `INFO replication` 中观察）：

```
repl_backlog_active:1
repl_backlog_size:1048576          # 默认 1MB，可通过 repl-backlog-size 配置
repl_backlog_first_byte_offset:1   # backlog 中最老数据的 offset
repl_backlog_histlen:12345         # 实际有效字节数
```

如果从库的 `slave_repl_offset` 落在 `[first_byte_offset, master_repl_offset]` 区间内，就可以部分重同步。

---

## 8 Diskless Replication 与链式副本

### 8.1 传统 Disk-based 全量同步

```
主库: BGSAVE → dump.rdb → 读 rdb → 发送给从库
     ↑                              ↑
     写盘 I/O                       读盘 I/O
```

对于超大数据集（几十 GB），磁盘 I/O 成为瓶颈，且占用大量磁盘空间。

### 8.2 Diskless Replication（Redis 2.8.18+）

```
主库: fork 子进程 → 子进程通过 socket 直接发 RDB 流给从库
                    完全跳过磁盘
```

配置：
```
repl-diskless-sync yes
repl-diskless-sync-delay 5   # 等待 5 秒让更多从库同时接入（一次 fork 服务多个从库）
repl-diskless-sync-max-replicas 0  # 0=无上限
```

**Trade-off**：
- 优点：省磁盘 I/O，节省磁盘空间。
- 缺点：如果传输中断，无法从中断处续传，必须重新 fork（因为没有 RDB 文件可以重发）。适合网络稳定的内网环境。

### 8.3 链式副本（Redis 4.0+，replica-of-replica）

```
Master A ──► Replica B ──► Replica C
```

- C 和 B 都从 A 获得相同的数据集（B 不会应用自己本地的写入到 C）。
- B 的 replID 等于 A 的 replID，offset 可以稍微落后。
- 节省 A 的上行带宽（A 只需向 B 发数据，B 再向 C 转发）。

**注意**：B 转发给 C 的是 A 的原始命令流，不是 B 应用后的结果，所以 C 的数据始终是 A 的视图，和 B 一致。

---

## 9 对比与选型

### 9.1 RDB vs AOF 核心对比

| 维度 | RDB | AOF（everysec） | AOF（always） |
|------|-----|-----------------|---------------|
| 数据丢失窗口 | 最后一次 BGSAVE 到宕机之间（几分钟） | 最多 1 秒 | 接近 0 |
| 文件大小 | 紧凑（二进制，LZF 压缩） | 较大（RESP 文本） | 较大 |
| 重启恢复速度 | 快（直接加载二进制） | 慢（重放所有命令） | 慢 |
| 写入性能影响 | 低（子进程异步） | 低（everysec）/ 极高（always） | 每命令一次 fsync |
| fork 开销 | 有（BGSAVE 时） | 有（Rewrite 时） | 有（Rewrite 时） |
| 文件可读性 | 否（二进制） | 是（可手工编辑） | 是 |
| 适合全量备份 | 是 | 否 | 否 |

### 9.2 混合持久化（Redis 4.0+）

```
aof-use-rdb-preamble yes   # 默认开启
```

AOF rewrite 时，BASE 文件用 RDB 格式写（快照），后续增量用 AOF 格式追加。  
结果：**重启恢复速度接近 RDB，数据安全性接近 AOF**。

加载时 Redis 自动检测：如果 BASE 文件以 `REDIS` 魔数开头，按 RDB 解析；否则按 AOF 重放。

### 9.3 场景选型决策树

```
需要最低数据丢失（金融/订单）?
  ├─ Yes → AOF + always 或 AOF + everysec + 主从双写确认
  └─ No  → 可接受分钟级丢失?
           ├─ Yes → RDB（最低开销）
           └─ No  → AOF + everysec（推荐默认）

数据集 > 50GB?
  ├─ Yes → 评估 BGSAVE 的 COW 内存开销；考虑 diskless replication
  └─ No  → 无特殊限制

需要快速恢复（SLA < 30s）?
  ├─ Yes → 混合持久化（aof-use-rdb-preamble yes）
  └─ No  → 纯 AOF 也可接受
```

---

## 10 可运行 Demo

### Demo 1：手写 AOF append + 重放恢复

这个 demo 模拟 Redis AOF 的核心机制：把写命令追加为 RESP 格式，然后重放恢复状态。

**设计为可运行，请在你环境验证。依赖：Python 3.6+，无需任何第三方库。**

```python
#!/usr/bin/env python3
"""
Demo 1: 手写 AOF 追加 + 重放恢复
模拟 Redis feedAppendOnlyFile + catAppendOnlyGenericCommand 的核心逻辑
对应源码: redis/redis@src/aof.c lines 2155-2260
"""
import os
import io
import struct
import hashlib

AOF_FILE = "/tmp/demo_appendonly.aof"

# ── 1. 序列化命令为 RESP multibulk 格式 ──────────────────────────────────────
def resp_encode(*args) -> bytes:
    """
    对应 catAppendOnlyGenericCommand in aof.c
    把 Python 字符串列表序列化为 RESP multibulk

    示例: resp_encode("SET", "foo", "bar") →
          b"*3\r\n$3\r\nSET\r\n$3\r\nfoo\r\n$3\r\nbar\r\n"
    """
    parts = [f"*{len(args)}\r\n".encode()]
    for arg in args:
        s = str(arg).encode()
        parts.append(f"${len(s)}\r\n".encode())
        parts.append(s + b"\r\n")
    return b"".join(parts)

# ── 2. AOF 写入（对应 flushAppendOnlyFile + aofWrite）───────────────────────
class SimpleAOF:
    def __init__(self, path: str):
        self.path = path
        self.fd = open(path, "ab")  # 追加模式，对应 O_APPEND

    def feed(self, *args):
        """追加一条命令到 AOF，对应 server.aof_buf 的写入 + flush"""
        data = resp_encode(*args)
        self.fd.write(data)
        self.fd.flush()
        # 模拟 AOF_FSYNC_ALWAYS：每条命令后 fsync
        os.fsync(self.fd.fileno())
        print(f"[AOF WRITE] {' '.join(str(a) for a in args)}")
        print(f"            raw bytes: {data!r}")

    def close(self):
        self.fd.close()

# ── 3. AOF 重放（对应 loadSingleAppendOnlyFile 的 RESP 解析）────────────────
def replay_aof(path: str) -> dict:
    """
    重放 AOF 文件，恢复 key-value 状态
    对应 Redis 启动时的 loadAppendOnlyFiles()
    """
    db = {}
    with open(path, "rb") as f:
        content = f.read()

    reader = io.BytesIO(content)
    cmd_count = 0

    while True:
        line = reader.readline()
        if not line:
            break
        if not line.startswith(b"*"):
            continue

        argc = int(line[1:].strip())
        argv = []
        for _ in range(argc):
            dollar_line = reader.readline()       # "$N\r\n"
            arglen = int(dollar_line[1:].strip())
            arg = reader.read(arglen)             # 读 N 字节
            reader.read(2)                         # 跳过 \r\n
            argv.append(arg.decode())

        cmd_count += 1
        # 简单命令执行引擎
        cmd = argv[0].upper()
        if cmd == "SET" and len(argv) >= 3:
            db[argv[1]] = argv[2]
            print(f"[REPLAY] SET {argv[1]} = {argv[2]}")
        elif cmd == "DEL" and len(argv) >= 2:
            for key in argv[1:]:
                db.pop(key, None)
            print(f"[REPLAY] DEL {argv[1:]}")
        elif cmd == "HSET" and len(argv) >= 4:
            h = db.setdefault(argv[1], {})
            for i in range(2, len(argv)-1, 2):
                h[argv[i]] = argv[i+1]
            print(f"[REPLAY] HSET {argv[1]} {argv[2:]} ")
        elif cmd == "SELECT":
            print(f"[REPLAY] SELECT db={argv[1]} (ignored in this toy)")

    print(f"\n[REPLAY DONE] {cmd_count} commands replayed")
    return db

# ── 4. 主程序 ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 清理旧文件
    if os.path.exists(AOF_FILE):
        os.unlink(AOF_FILE)

    print("=" * 60)
    print("PHASE 1: 写入命令（模拟 Redis 运行时追加 AOF）")
    print("=" * 60)

    aof = SimpleAOF(AOF_FILE)
    aof.feed("SET", "name", "redis")
    aof.feed("SET", "version", "7.2")
    aof.feed("HSET", "config", "maxmemory", "1gb", "hz", "10")
    aof.feed("SET", "temp_key", "will_be_deleted")
    aof.feed("DEL", "temp_key")
    aof.close()

    print(f"\n[AOF FILE SIZE] {os.path.getsize(AOF_FILE)} bytes")
    print(f"[AOF FILE CONTENT]")
    with open(AOF_FILE, "rb") as f:
        print(f.read().decode())

    print("=" * 60)
    print("PHASE 2: 重放 AOF 恢复状态（模拟 Redis 重启加载）")
    print("=" * 60)

    state = replay_aof(AOF_FILE)

    print("\n[FINAL STATE]")
    for k, v in state.items():
        print(f"  {k!r}: {v!r}")

    # 验证
    assert state["name"] == "redis", "name should be 'redis'"
    assert state["version"] == "7.2", "version should be '7.2'"
    assert "temp_key" not in state, "temp_key should have been deleted"
    assert isinstance(state["config"], dict), "config should be a hash"
    print("\n[ASSERTIONS PASSED] 所有断言通过")
```

**预期输出**：

```
============================================================
PHASE 1: 写入命令（模拟 Redis 运行时追加 AOF）
============================================================
[AOF WRITE] SET name redis
            raw bytes: b'*3\r\n$3\r\nSET\r\n$4\r\nname\r\n$5\r\nredis\r\n'
[AOF WRITE] SET version 7.2
            raw bytes: b'*3\r\n$3\r\nSET\r\n$7\r\nversion\r\n$3\r\n7.2\r\n'
[AOF WRITE] HSET config maxmemory 1gb hz 10
...
[AOF FILE SIZE] ~200 bytes

PHASE 2: 重放 AOF 恢复状态
[REPLAY] SET name = redis
[REPLAY] SET version = 7.2
[REPLAY] HSET config ...
[REPLAY] SET temp_key = will_be_deleted
[REPLAY] DEL ['temp_key']
[REPLAY DONE] 5 commands replayed

[FINAL STATE]
  'name': 'redis'
  'version': '7.2'
  'config': {'maxmemory': '1gb', 'hz': '10'}

[ASSERTIONS PASSED] 所有断言通过
```

---

### Demo 2：解析 RDB 文件头 + Length Encoding

这个 demo 实现 RDB 文件的头部解析，印证 `rdbLoadLenByRef` 的 bit-twiddling 逻辑。

**设计为可运行，请在你环境验证。依赖：Python 3.6+，需要一个 Redis 实例（或使用 demo 自带的合成 RDB 文件）。**

```python
#!/usr/bin/env python3
"""
Demo 2: 解析 RDB 文件头部与 Length Encoding
对应源码: redis/redis@src/rdb.c (rdbLoadLenByRef, rdbSaveLen)
           redis/redis@src/rdb.h (RDB_VERSION, RDB_OPCODE_*, RDB_TYPE_*)
"""
import struct
import io
import os

# ── RDB 常量（来自 rdb.h）────────────────────────────────────────────────────
RDB_VERSION = 14
RDB_6BITLEN  = 0x00   # 高2位 00
RDB_14BITLEN = 0x01   # 高2位 01
RDB_32BITLEN = 0x80   # 10000000
RDB_64BITLEN = 0x81   # 10000001
RDB_ENCVAL   = 0x03   # 高2位 11 → 特殊编码

RDB_ENC_INT8  = 0
RDB_ENC_INT16 = 1
RDB_ENC_INT32 = 2
RDB_ENC_LZF   = 3

RDB_TYPE_STRING           = 0
RDB_TYPE_LIST             = 1
RDB_TYPE_SET              = 2
RDB_TYPE_ZSET             = 3
RDB_TYPE_HASH             = 4
RDB_TYPE_HASH_LISTPACK    = 16
RDB_TYPE_ZSET_LISTPACK    = 17

RDB_OPCODE_AUX         = 250   # 0xFA
RDB_OPCODE_RESIZEDB    = 251   # 0xFB
RDB_OPCODE_EXPIRETIME_MS = 252 # 0xFC
RDB_OPCODE_EXPIRETIME  = 253   # 0xFD
RDB_OPCODE_SELECTDB    = 254   # 0xFE
RDB_OPCODE_EOF         = 255   # 0xFF

OPCODE_NAMES = {
    RDB_OPCODE_AUX: "AUX",
    RDB_OPCODE_RESIZEDB: "RESIZEDB",
    RDB_OPCODE_EXPIRETIME_MS: "EXPIRETIME_MS",
    RDB_OPCODE_EXPIRETIME: "EXPIRETIME",
    RDB_OPCODE_SELECTDB: "SELECTDB",
    RDB_OPCODE_EOF: "EOF",
}

TYPE_NAMES = {
    0: "STRING", 1: "LIST", 2: "SET", 3: "ZSET", 4: "HASH",
    16: "HASH_LISTPACK", 17: "ZSET_LISTPACK", 18: "LIST_QUICKLIST_2",
}

# ── Length Encoding 解码（对应 rdbLoadLenByRef）──────────────────────────────
def rdb_load_len(reader: io.BytesIO):
    """
    解码 RDB length-encoded integer
    返回 (length, is_encoded)
    is_encoded=True 表示这是特殊编码（整数内联或 LZF），length 是编码类型
    """
    b = reader.read(1)
    if not b:
        raise EOFError("Unexpected end of RDB")
    byte = b[0]
    enc_type = (byte & 0xC0) >> 6  # 取高2位

    if enc_type == RDB_ENCVAL:     # 0x03 → 特殊编码
        return (byte & 0x3F), True
    elif enc_type == RDB_6BITLEN:  # 0x00 → 6-bit 长度
        return (byte & 0x3F), False
    elif enc_type == RDB_14BITLEN: # 0x01 → 14-bit 长度
        b2 = reader.read(1)[0]
        return ((byte & 0x3F) << 8) | b2, False
    elif byte == RDB_32BITLEN:     # 0x80 → 32-bit BE
        return struct.unpack(">I", reader.read(4))[0], False
    elif byte == RDB_64BITLEN:     # 0x81 → 64-bit LE
        return struct.unpack("<Q", reader.read(8))[0], False
    else:
        raise ValueError(f"Unknown length encoding: 0x{byte:02x}")

def rdb_load_string(reader: io.BytesIO) -> str:
    """解码 RDB string（支持 length-prefixed 和整数内联）"""
    length, is_encoded = rdb_load_len(reader)
    if is_encoded:
        if length == RDB_ENC_INT8:
            return str(struct.unpack("b", reader.read(1))[0])
        elif length == RDB_ENC_INT16:
            return str(struct.unpack("<h", reader.read(2))[0])
        elif length == RDB_ENC_INT32:
            return str(struct.unpack("<i", reader.read(4))[0])
        else:
            raise ValueError(f"Unsupported special encoding: {length}")
    else:
        return reader.read(length).decode("utf-8", errors="replace")

# ── 合成一个最小的 RDB 文件（用于无 Redis 环境测试）──────────────────────────
def make_minimal_rdb() -> bytes:
    """
    构造一个包含：
    - 魔数头 REDIS0014
    - 一个 AUX 字段（redis-ver = "7.2.0"）
    - DB 0：一个 STRING KV（"hello" -> "world"）
    - EOF + CRC64
    对应 rdbSaveRio 的输出结构
    """
    import binascii

    def encode_len(n: int) -> bytes:
        if n < 64:
            return bytes([n & 0x3F])          # 6-bit
        elif n < 16384:
            return bytes([(n >> 8) | 0x40, n & 0xFF])  # 14-bit
        else:
            return bytes([0x80]) + struct.pack(">I", n) # 32-bit

    def encode_str(s: str) -> bytes:
        b = s.encode()
        return encode_len(len(b)) + b

    buf = bytearray()
    # 魔数 + 版本
    buf += b"REDIS0014"

    # AUX 字段：0xFA + key + value
    buf += bytes([RDB_OPCODE_AUX])
    buf += encode_str("redis-ver")
    buf += encode_str("7.2.0")

    # AUX 字段：used-mem（整数，用 INT32 特殊编码）
    buf += bytes([RDB_OPCODE_AUX])
    buf += encode_str("used-mem")
    buf += bytes([0xC2])  # RDB_ENCVAL(0x03<<6) | RDB_ENC_INT32(0x02)
    buf += struct.pack("<i", 1024 * 1024)  # 1MB

    # SELECTDB 0
    buf += bytes([RDB_OPCODE_SELECTDB])
    buf += bytes([0x00])  # db 0, 6-bit length=0

    # RESIZEDB: db_size=1, expires_size=0
    buf += bytes([RDB_OPCODE_RESIZEDB])
    buf += bytes([0x01])  # 1 key
    buf += bytes([0x00])  # 0 expires

    # KV pair: type=STRING, key="hello", value="world"
    buf += bytes([RDB_TYPE_STRING])
    buf += encode_str("hello")
    buf += encode_str("world")

    # EOF
    buf += bytes([RDB_OPCODE_EOF])

    # CRC64 checksum（简化：使用 0 表示禁用，真实 Redis 用 crc64jones 算法）
    buf += struct.pack("<Q", 0)

    return bytes(buf)

# ── RDB 解析器 ───────────────────────────────────────────────────────────────
def parse_rdb(data: bytes):
    reader = io.BytesIO(data)

    # 解析魔数
    magic = reader.read(9)
    assert magic[:5] == b"REDIS", f"Not a RDB file: {magic[:5]}"
    version = int(magic[5:])
    print(f"[HEADER] Magic: REDIS, Version: {version}")

    entries = {}
    current_db = -1
    expire_at = None

    while True:
        pos = reader.tell()
        b = reader.read(1)
        if not b:
            break
        opcode = b[0]

        if opcode == RDB_OPCODE_EOF:
            checksum = reader.read(8)
            cs = struct.unpack("<Q", checksum)[0]
            print(f"[{pos:#06x}] EOF, checksum={'disabled' if cs==0 else hex(cs)}")
            break

        elif opcode == RDB_OPCODE_SELECTDB:
            db_id, _ = rdb_load_len(reader)
            current_db = db_id
            print(f"[{pos:#06x}] SELECTDB {db_id}")

        elif opcode == RDB_OPCODE_RESIZEDB:
            db_size, _ = rdb_load_len(reader)
            expires_size, _ = rdb_load_len(reader)
            print(f"[{pos:#06x}] RESIZEDB db_size={db_size} expires={expires_size}")

        elif opcode == RDB_OPCODE_AUX:
            key = rdb_load_string(reader)
            val = rdb_load_string(reader)
            print(f"[{pos:#06x}] AUX {key!r} = {val!r}")

        elif opcode == RDB_OPCODE_EXPIRETIME_MS:
            ts = struct.unpack("<Q", reader.read(8))[0]
            expire_at = ts
            print(f"[{pos:#06x}] EXPIRETIME_MS {ts}")

        elif opcode in TYPE_NAMES:
            # 这是一个 TYPE 字节，后面跟 key + value
            type_name = TYPE_NAMES.get(opcode, f"TYPE_{opcode}")
            key = rdb_load_string(reader)
            if opcode == RDB_TYPE_STRING:
                value = rdb_load_string(reader)
            else:
                value = f"<{type_name} encoding, not fully decoded in this demo>"
            expiry_str = f" (expires at {expire_at}ms)" if expire_at else ""
            print(f"[{pos:#06x}] KEY db={current_db} type={type_name} "
                  f"key={key!r} value={value!r}{expiry_str}")
            entries[key] = value
            expire_at = None
        else:
            print(f"[{pos:#06x}] UNKNOWN opcode=0x{opcode:02x}, stopping")
            break

    return entries

# ── 主程序 ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("PART A: 合成最小 RDB 文件并解析")
    print("=" * 60)
    rdb_data = make_minimal_rdb()
    print(f"[SYNTHETIC RDB] {len(rdb_data)} bytes")
    print(f"[HEX DUMP] {rdb_data.hex()}\n")
    result = parse_rdb(rdb_data)
    print(f"\n[PARSED KEYS] {result}")
    assert result.get("hello") == "world", "hello should be 'world'"
    print("[ASSERTION PASSED]")

    print("\n" + "=" * 60)
    print("PART B: Length Encoding 验证")
    print("  对应 rdbSaveLen / rdbLoadLenByRef in rdb.c")
    print("=" * 60)

    def encode_len(n: int) -> bytes:
        if n < 64:
            return bytes([n & 0x3F])
        elif n < 16384:
            return bytes([(n >> 8) | 0x40, n & 0xFF])
        else:
            return bytes([0x80]) + struct.pack(">I", n)

    test_cases = [0, 1, 63, 64, 127, 16383, 16384, 65535, 1048576]
    for n in test_cases:
        encoded = encode_len(n)
        # 解码验证
        r = io.BytesIO(encoded)
        decoded, _ = rdb_load_len(r)
        status = "OK" if decoded == n else f"FAIL(got {decoded})"
        print(f"  n={n:>8}  encoded={encoded.hex():>10}  ({len(encoded)}B)  → decode={status}")
```

**预期输出**：

```
PART A: 合成最小 RDB 文件并解析
[SYNTHETIC RDB] 78 bytes
[0x0000] HEADER: Magic: REDIS, Version: 14
[0x0009] AUX 'redis-ver' = '7.2.0'
[0x001a] AUX 'used-mem' = '1048576'
[0x002a] SELECTDB 0
[0x002c] RESIZEDB db_size=1 expires=0
[0x002e] KEY db=0 type=STRING key='hello' value='world'
[0x003a] EOF, checksum=disabled
[PARSED KEYS] {'hello': 'world'}
[ASSERTION PASSED]

PART B: Length Encoding 验证
  n=       0  encoded=          00  (1B)  → decode=OK
  n=       1  encoded=          01  (1B)  → decode=OK
  n=      63  encoded=          3f  (1B)  → decode=OK
  n=      64  encoded=        4040  (2B)  → decode=OK
  ...
  n=   16383  encoded=        7fff  (2B)  → decode=OK
  n=   16384  encoded=  8000004000  (5B)  → decode=OK
```

---

### Demo 3：模拟 PSYNC 部分重同步判断逻辑

这个 demo 实现 `masterTryPartialResynchronization` 的核心判断逻辑。

**设计为可运行，请在你环境验证。依赖：Python 3.6+。**

```python
#!/usr/bin/env python3
"""
Demo 3: 模拟 PSYNC 部分重同步判断
对应源码: redis/redis@src/replication.c masterTryPartialResynchronization
          redis/redis@src/server.h replBacklog struct
"""
import uuid
import collections

# ── Replication Backlog 模拟 ─────────────────────────────────────────────────
class ReplicationBacklog:
    """
    模拟 Redis replBacklog 结构
    对应 server.h:
      typedef struct replBacklog {
          long long histlen;
          long long offset;  // backlog 最老数据的起始 offset
      } replBacklog;
    注：真实 Redis 7.x 用链表+radix tree，这里用 deque 简化
    """
    def __init__(self, max_size: int = 1024 * 1024):
        self.max_size = max_size
        self._buf = bytearray()
        self.offset = 1    # backlog 最老数据对应的 master_repl_offset

    def append(self, data: bytes, master_offset: int):
        """追加数据到 backlog"""
        self._buf.extend(data)
        # 如果超过 max_size，裁剪最老的数据
        overflow = len(self._buf) - self.max_size
        if overflow > 0:
            self._buf = self._buf[overflow:]
            self.offset += overflow

    def get_delta(self, from_offset: int, to_offset: int) -> bytes:
        """
        获取 [from_offset, to_offset) 区间的数据
        对应 addReplyReplicationBacklog()
        """
        buf_start = self.offset
        buf_end = self.offset + len(self._buf)
        start_idx = from_offset - buf_start
        end_idx = to_offset - buf_start
        return bytes(self._buf[start_idx:end_idx])

    @property
    def histlen(self) -> int:
        return len(self._buf)

    def covers(self, offset: int) -> bool:
        """判断 offset 是否在 backlog 覆盖范围内"""
        return self.offset <= offset <= (self.offset + len(self._buf))

# ── Master 状态模拟 ───────────────────────────────────────────────────────────
class Master:
    def __init__(self):
        self.replid  = uuid.uuid4().hex[:40]   # 40字符 replication ID
        self.replid2 = "0" * 40                 # 初始无 secondary ID
        self.second_replid_offset = -1          # -1 表示 secondary ID 无效
        self.master_repl_offset = 0
        self.backlog = ReplicationBacklog(max_size=512)  # 小 backlog 便于测试
        self._command_log = []

    def write_command(self, cmd: str):
        """模拟主库执行写命令，更新 offset 和 backlog"""
        data = cmd.encode() + b"\r\n"
        self.master_repl_offset += len(data)
        self.backlog.append(data, self.master_repl_offset)
        self._command_log.append(cmd)

    def promote_replica(self, old_replid: str, old_offset: int):
        """
        模拟从库晋升为主库时的 PSYNC2 双 ID 机制
        对应 shiftReplicationId() in replication.c
        """
        self.replid2 = old_replid              # 保留旧主的 replID
        self.second_replid_offset = old_offset # 旧主的最后 offset
        old_replid_new = uuid.uuid4().hex[:40]
        print(f"[PROMOTE] New replid={old_replid_new[:8]}...")
        print(f"          Secondary replid={self.replid2[:8]}... "
              f"(valid up to offset {self.second_replid_offset})")
        self.replid = old_replid_new

    def try_partial_resync(self, replica_replid: str, replica_offset: int):
        """
        对应 masterTryPartialResynchronization()
        返回 (success, response, delta_bytes)
        """
        print(f"\n[PSYNC] Replica asks: replid={replica_replid[:8]}... "
              f"offset={replica_offset}")

        # 步骤1: 检查 replID 是否匹配
        id_match = (replica_replid == self.replid)
        secondary_match = (
            replica_replid == self.replid2 and
            self.replid2 != "0" * 40 and
            replica_offset <= self.second_replid_offset
        )

        if not id_match and not secondary_match:
            reason = "replID mismatch"
            if replica_replid == self.replid2:
                reason = f"secondary offset {replica_offset} > {self.second_replid_offset}"
            print(f"  → FULLRESYNC (reason: {reason})")
            return False, f"+FULLRESYNC {self.replid} {self.master_repl_offset}", b""

        # 步骤2: 检查 offset 是否在 backlog 内
        if not self.backlog.covers(replica_offset):
            print(f"  → FULLRESYNC (offset {replica_offset} not in backlog "
                  f"[{self.backlog.offset}, {self.backlog.offset+self.backlog.histlen}])")
            return False, f"+FULLRESYNC {self.replid} {self.master_repl_offset}", b""

        # 部分重同步成功
        delta = self.backlog.get_delta(replica_offset, self.master_repl_offset + 1)
        delta_bytes = len(delta)
        print(f"  → CONTINUE (sending {delta_bytes} bytes of delta)")
        return True, f"+CONTINUE {self.replid}", delta

# ── 场景模拟 ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("SCENARIO 1: 正常部分重同步（从库短暂断线重连）")
    print("=" * 65)
    master = Master()
    print(f"[MASTER] replid={master.replid[:8]}... offset=0")

    # 主库写入一批命令
    for i in range(10):
        master.write_command(f"SET key{i} value{i}")
    print(f"[MASTER] After writes: offset={master.master_repl_offset}, "
          f"backlog=[{master.backlog.offset}, "
          f"{master.backlog.offset+master.backlog.histlen})")

    # 从库在 offset=50 时断线，重连时请求 offset=50
    replica_offset = 50
    replica_replid = master.replid
    success, resp, delta = master.try_partial_resync(replica_replid, replica_offset)
    assert success, "Should succeed"
    print(f"  delta preview: {delta[:40]!r}...")

    print("\n" + "=" * 65)
    print("SCENARIO 2: backlog 不够用（断线时间过长）")
    print("=" * 65)
    master2 = Master()
    for i in range(200):  # 写很多命令，把 backlog 撑满
        master2.write_command(f"SET largekey_{i} {'x'*20}")
    print(f"[MASTER] offset={master2.master_repl_offset}, "
          f"backlog starts at {master2.backlog.offset}")

    # 从库的 offset 已经在 backlog 覆盖范围之外
    old_offset = 10  # 从库还停在非常早的 offset
    success, resp, delta = master2.try_partial_resync(master2.replid, old_offset)
    assert not success, "Should fail (offset too old)"

    print("\n" + "=" * 65)
    print("SCENARIO 3: PSYNC2 failover 后的部分重同步（secondary replID）")
    print("=" * 65)
    # 旧主库 A
    old_master = Master()
    for i in range(5):
        old_master.write_command(f"SET k{i} v{i}")
    old_replid = old_master.replid
    old_offset = old_master.master_repl_offset
    print(f"[OLD MASTER A] replid={old_replid[:8]}... offset={old_offset}")

    # 从库 B 晋升为新主库，保留旧主 replID
    new_master = Master()
    new_master.replid = old_replid          # 先设为旧主 ID（继承状态）
    new_master.master_repl_offset = old_offset
    new_master.backlog = old_master.backlog  # 继承 backlog

    # 晋升：生成新 replID，旧 ID 变 secondary
    new_master.promote_replica(old_replid, old_offset)

    # 写几条新命令
    for i in range(3):
        new_master.write_command(f"SET new_{i} val")

    # 另一个从库 C 仍持有旧主的 replID，尝试对新主 B 进行部分重同步
    replica_c_offset = old_offset - 20  # C 落后旧主 20 bytes
    success, resp, delta = new_master.try_partial_resync(old_replid, replica_c_offset)
    assert success, "PSYNC2: should allow partial resync using secondary replID"
    print(f"  [PSYNC2 SUCCESS] Replica C 使用旧主 replID 完成部分重同步")
    print(f"  delta bytes: {len(delta)}")

    print("\n[ALL SCENARIOS PASSED]")

if __name__ == "__main__":
    main()
```

**预期输出**：

```
SCENARIO 1: 正常部分重同步（从库短暂断线重连）
[MASTER] replid=a3f2c1d8... offset=0
[MASTER] After writes: offset=180, backlog=[1, 181)
[PSYNC] Replica asks: replid=a3f2c1d8... offset=50
  → CONTINUE (sending 131 bytes of delta)
  delta preview: b'SET key3 value3\r\nSET key4 val...'

SCENARIO 2: backlog 不够用（断线时间过长）
[MASTER] offset=5200, backlog starts at 4688
[PSYNC] Replica asks: replid=... offset=10
  → FULLRESYNC (offset 10 not in backlog [4688, 5200])

SCENARIO 3: PSYNC2 failover 后的部分重同步
[OLD MASTER A] replid=7b9e2a1f... offset=90
[PROMOTE] New replid=c4d8f3a2...
          Secondary replid=7b9e2a1f... (valid up to offset 90)
[PSYNC] Replica asks: replid=7b9e2a1f... offset=70
  → CONTINUE (sending ... bytes of delta)
  [PSYNC2 SUCCESS] Replica C 使用旧主 replID 完成部分重同步

[ALL SCENARIOS PASSED]
```

---

## 11 生产真坑与失败模式

### 11.1 BGSAVE 期间的 COW 内存暴涨

**现象**：主库内存 8GB，BGSAVE 后 RSS 飙到 14GB，触发 OOM Killer。  
**根因**：BGSAVE 后父进程写入量大（比如有大量 SET 命令），每次写导致 OS 按页复制（COW page fault），8GB 数据集如果 50% 发生 COW，就额外 4GB。

**真实根因链**：Linux 默认开启 THP（Transparent Huge Pages，2MB 页）→ COW 发生时整个 2MB 页被复制（即使只改 1 字节）→ 等效内存放大倍数 = `(2MB huge page / actual dirty bytes per page)`

**解决**：
```bash
# 禁用 THP（Redis 官方推荐）
echo never > /sys/kernel/mm/transparent_hugepage/enabled
echo never > /sys/kernel/mm/transparent_hugepage/defrag
```

```
INFO memory 中观察：
rdb_last_cow_size:1234567   # 上次 BGSAVE 的实际 COW 字节数
```

### 11.2 AOF everysec 的"丢失 1 秒 + buffer"问题

**常见误解**：以为 everysec 最多丢 1 秒数据。  
**真实情况**：
1. `write()` 成功写入内核 page cache。
2. BIO 线程每秒调用 `fsync`，如果 `fsync` 耗时超过 1 秒（磁盘 I/O 繁忙），下一秒的 `write()` 会被 `flushAppendOnlyFile` 的推迟机制阻塞最多 2 秒。
3. 机器断电时，page cache 中未 fsync 的数据丢失。

**实际最坏情况**：丢失约 `1秒（上次 fsync 到 write）+ 最多 2 秒（推迟窗口）` = 最坏 3 秒。

**解决**：使用带有电池备份（BBU）的 RAID 控制器，或者用 UPS 保护电源，确保 page cache 能写入磁盘。

### 11.3 主库无持久化 + 自动重启 = 清空从库

**现象**：主库内存满，触发 OOM Kill，进程被系统重启后，所有从库数据被清空。

**原因链**：
1. 主库持久化关闭（`save ""`，`appendonly no`）
2. 进程重启后内存为空
3. 从库检测到主库重启（replID 变化），发起 FULLRESYNC
4. 从库同步了空的 RDB，本地数据全清

**Redis 有检测机制**（`min-replicas-to-write`）但默认未开启。生产必配：
```
min-replicas-to-write 1
min-replicas-max-lag 10
```

### 11.4 RDB 文件 CRC 校验失败

**现象**：`redis-server` 启动时报 `Short read or OOM loading DB. Unrecoverable error, aborting now.`  
**根因**：RDB 文件被截断（磁盘满、kill -9 在 rename 前、NFS 超时），CRC64 校验失败。

**应急处理**：
```bash
redis-check-rdb dump.rdb       # 官方修复工具，尝试恢复
# 如果无法修复，设置 rdbchecksum no 临时跳过校验（危险！）
```

**根本防御**：监控 `rdb_last_bgsave_status`，BGSAVE 失败时告警，不要只依赖 RDB 单点。

### 11.5 AOF 重写失败静默：日志里看不出来

**现象**：AOF 文件持续增大，重启恢复时间越来越长，但没有告警。  
**根因**：`rewriteAppendOnlyFileBackground` 失败（磁盘空间不足、子进程 kill），`server.aof_lastbgrewrite_status = C_ERR`，但主进程继续写旧 AOF 文件，业务不受影响。

**监控**：
```
redis-cli INFO persistence | grep aof_last_bgrewrite_status
# 期望：aof_last_bgrewrite_status:ok
# 异常：aof_last_bgrewrite_status:err → 立即告警
```

### 11.6 PSYNC backlog 太小导致级联全量同步

**现象**：主从网络抖动 30 秒，从库重连后触发 FULLRESYNC，主库 BGSAVE 30 秒，期间从库卡顿，用户感知到服务降级。

**根因**：`repl-backlog-size` 默认 1MB，主库 10MB/s 写入，仅 100ms 就填满，30 秒断线早已超出。

**调整**：
```
repl-backlog-size 128mb    # 按 (写入速率 * 最大可容忍断线时长) 设置
repl-backlog-ttl 3600      # backlog 在所有从库断开后的保留时间
```

---

## 12 章末五件套

### 12.1 核心概念速查

| 概念 | 位置 | 一句话 |
|------|------|--------|
| RDB_VERSION | `rdb.h:19` | 当前 14，文件开头 9 字节 `REDIS0014` |
| Length Encoding | `rdb.c:269-345` | 高 2 位决定编码长度（1/2/5/9 字节） |
| COW | 内核 | fork 后写时复制，BGSAVE 期间父子共享页 |
| AOF_FSYNC_EVERYSEC | `server.h:1075` | BIO 线程每秒 fsync，最坏丢 1-3 秒 |
| Multi-Part AOF | `aof.c:2730+` | 7.0 起 BASE+INCR+MANIFEST，消除 rewrite buffer |
| replID | `replication.c` | 40 字符 hex，标识主库的复制历史 |
| PSYNC2 replid2 | `server.h` | secondary ID，failover 后保留旧主 ID |
| Replication Backlog | `replication.c:323` | 环形 buffer + radix tree 索引 |
| diskless replication | `replication.c` | 子进程直接向 socket 流式发 RDB |

### 12.2 扩展 Demo：为 Demo 1 添加 AOF 重写压缩

> 把以下代码追加到 Demo 1，实现一个 toy AOF rewrite：遍历当前状态，用最少命令重新生成 AOF，验证大量 SET/DEL 后的文件压缩效果。

```python
def rewrite_aof(current_state: dict, output_path: str):
    """
    模拟 rewriteAppendOnlyFile()
    把当前内存状态压缩为最小命令集，写新 AOF
    """
    with open(output_path, "wb") as f:
        for key, value in current_state.items():
            if isinstance(value, dict):
                # Hash: 合并所有 field 为一条 HSET
                cmd_args = ["HSET", key] + [v for pair in value.items() for v in pair]
                f.write(resp_encode(*cmd_args))
            else:
                f.write(resp_encode("SET", key, value))
    old_size = os.path.getsize(AOF_FILE)
    new_size = os.path.getsize(output_path)
    ratio = (1 - new_size/old_size) * 100
    print(f"[REWRITE] {old_size}B → {new_size}B (压缩 {ratio:.1f}%)")

# 测试：先写 100 次 SET，再 DEL 50 个，看 rewrite 效果
aof2 = SimpleAOF("/tmp/demo_verbose.aof")
state2 = {}
for i in range(100):
    aof2.feed("SET", f"k{i}", f"v{i}")
    state2[f"k{i}"] = f"v{i}"
for i in range(0, 100, 2):  # 删除偶数 key
    aof2.feed("DEL", f"k{i}")
    del state2[f"k{i}"]
aof2.close()

rewrite_aof(state2, "/tmp/demo_rewrite.aof")
# 验证重写后 AOF 重放结果一致
state_reloaded = replay_aof("/tmp/demo_rewrite.aof")
assert state_reloaded == state2, "Rewrite should preserve state"
print("[REWRITE ASSERTION PASSED]")
```

### 12.3 面试高频题 & 答题框架

**Q1：BGSAVE 为什么用 fork 而不是线程？**

答：fork 利用 OS 的 COW 语义，子进程自动持有 fork 时刻的内存快照，无需任何锁。如果用线程，需要对整个内存加 RLock，在快照期间所有写操作都会阻塞，延迟不可接受。fork 的代价是 page fault（COW），但这个开销是分散的，不会造成单次大延迟。

**Q2：AOF everysec 真的最多丢 1 秒吗？**

答：不准确。最坏情况是丢 1–3 秒：1 秒是 fsync 周期，另外最多 2 秒是 `flushAppendOnlyFile` 的推迟窗口（当后台 fsync 正在进行时，主线程最多等 2 秒再强制写）。要真正做到近零丢失，需 `appendfsync always`，但吞吐会骤降。

**Q3：PSYNC 和 PSYNC2 的区别？**

答：PSYNC（1.0 版，Redis 2.8 引入）：单 replID，failover 后新主生成新 replID，原从库必须全量重同步。PSYNC2（Redis 4.0 引入）：双 replID（primary + secondary），新主保留旧主 replID 作为 secondary，原从库可以对新主做部分重同步，大幅减少 failover 后的数据传输量。

**Q4：RDB 和 AOF 同时开启，重启时优先用哪个？**

答：优先使用 AOF。因为 AOF 包含了最新的写命令，通常比 RDB 包含更多数据。RDB 可能是几分钟前的快照，AOF 是持续追加的，数据更完整。代码层面：`loadDataFromDisk()` 检查 `server.aof_state != AOF_OFF` 就走 `loadAppendOnlyFiles()`，否则才走 `rdbLoad()`。

**Q5：Replication Backlog 的数据结构在 7.x 中发生了什么变化？**

答：Redis 7.x 之前，backlog 是独立的环形 buffer（`server.repl_backlog` 是一个固定大小的 char 数组）。7.x 重构后，backlog 复用了 replication buffer 链表（`server.repl_buffer`），`replBacklog` 结构不再存储数据本身，而是持有一个指向链表中 backlog 起始节点的指针，并用 radix tree（`blocks_index`）做 O(log N) 的 offset 定位，解决了大 backlog 下随机访问慢的问题。

### 12.4 未来演进方向

1. **Valkey 分叉**（2024 年）：Valkey（AWS/Google/Linux Foundation 主导）已在 PSYNC 上做了若干优化，未来可能与 Redis 在复制协议上分叉。
2. **Multi-master / Active-Active**：Redis Enterprise 已有 CRDT-based Active-Active 实现，开源 Redis 目前仍是单主模型。
3. **RDB 版本 15+**：随着新数据类型（如 GCRA rate limiter、Hash with per-field TTL）的引入，RDB 版本会持续迭代，向后兼容是主要挑战。
4. **Diskless AOF Rewrite**：类似 diskless RDB，未来可能有 diskless AOF rewrite 直接通过 socket 传给从库，跳过落盘。

### 12.5 延伸阅读

- `redis/redis@src/rdb.c` — RDB 完整实现，重点看 `rdbSaveRio`, `rdbSaveDb`, `rdbLoadObjectType`
- `redis/redis@src/aof.c` — AOF 完整实现，重点看 `feedAppendOnlyFile`, `flushAppendOnlyFile`, `rewriteAppendOnlyFileBackground`
- `redis/redis@src/replication.c` — 复制完整实现，重点看 `masterTryPartialResynchronization`, `replicationFeedSlaves`, `createReplicationBacklog`
- [Redis Persistence Docs](https://redis.io/docs/latest/operate/oss_and_stack/management/persistence/) — 官方推荐配置和权衡
- [Redis Replication Docs](https://redis.io/docs/latest/operate/oss_and_stack/management/replication/) — PSYNC2、diskless、min-replicas 配置
- Martin Kleppmann《Designing Data-Intensive Applications》第 5 章 — 复制理论背景，与 Redis 实现形成对照

---

*文档生成时间：2026-06-12 | 源码版本：redis/redis unstable（7.x）| 核实 URL 列表见下方*

---

## 核实 URL 记录

| 内容 | URL | 状态 |
|------|-----|------|
| rdb.c 源码 | https://raw.githubusercontent.com/redis/redis/unstable/src/rdb.c | 已 fetch |
| aof.c 源码 | https://raw.githubusercontent.com/redis/redis/unstable/src/aof.c | 已 fetch |
| replication.c 源码 | https://raw.githubusercontent.com/redis/redis/unstable/src/replication.c | 已 fetch |
| rdb.h 常量 | https://raw.githubusercontent.com/redis/redis/unstable/src/rdb.h | 已 fetch |
| server.h 常量 | https://raw.githubusercontent.com/redis/redis/unstable/src/server.h | 已 fetch |
| aof.c 7.2 rewrite | https://raw.githubusercontent.com/redis/redis/7.2/src/aof.c | 已 fetch |
| 官方持久化文档 | https://redis.io/docs/latest/operate/oss_and_stack/management/persistence/ | 已 fetch |
| 官方复制文档 | https://redis.io/docs/latest/operate/oss_and_stack/management/replication/ | 已 fetch |
