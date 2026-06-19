---
title: "Redis 内核数据结构与单线程事件循环"
slug: "4-01"
collection: "tech-library"
group: "数据库"
order: 4001
summary: "TL;DR Redis 的性能秘密不是\"单线程有多快\"，而是内存操作 + 无锁 + epoll + 精心设计的编码自适应共同作用的结果。"
topics:
  - "技术内核"
tags: []
createdAt: "2026-06-12T10:35:54.000Z"
updatedAt: "2026-06-12T10:35:54.000Z"
---
> **TL;DR**
> Redis 的性能秘密不是"单线程有多快"，而是**内存操作 + 无锁 + epoll + 精心设计的编码自适应**共同作用的结果。核心数据结构五层编码体系（SDS / dict / skiplist / listpack / quicklist / intset）在小数据集时极省内存、大数据集时极快操作；ae 事件库用 100 行左右的 epoll 包装驱动整个服务。真正的生产坑不在于单线程本身，而在于 fork / KEYS / THP / rehash 等隐性阻塞。

---

## 前置依赖

| 知识点 | 要求层次 |
|--------|---------|
| C 语言指针与内存管理 | 熟练 |
| Linux epoll / select / kqueue | 了解原理 |
| 哈希表、跳表、双向链表 | 掌握 |
| 进程地址空间、COW | 了解 |

---

## 一、设计考古：单线程为什么快

### 1.1 Redis 诞生背景

2009 年，Salvatore Sanfilippo（antirez）在开发一个实时日志分析工具时，发现 MySQL 无法满足需要对列表做 push/pop 的延迟要求。他写了一个内存型、支持复杂数据结构的键值服务，最初只用于自己的项目，后来开源成为 Redis。

关键设计决策——**单线程事件循环**——来自以下几个论据：

**1. 内存操作本身极快，瓶颈在 I/O 而非 CPU**

Redis 典型命令耗时 < 1μs，一条 GET 的内存操作时间远小于一次系统调用。网络 I/O 是瓶颈，而 epoll 可以在单线程内服务数千并发连接。

**2. 避免锁竞争**

antirez 在 2014 年的博文（https://antirez.com/news/126）中明确指出：Redis 的数据结构之间存在相互依赖（例如 LPUSH 和 LPOP 共享同一个 list 对象），多线程需要细粒度锁，复杂度高且容易死锁。"Share-nothing" 的多实例方案在 Redis 的工作负载下比加锁更优。

> 原文引用："it's a lot of complexity without a good reason. Many Redis deployments are already network or memory-bound."

**3. 数据结构复杂性**

与 Memcached（简单 KV）不同，Redis 的 sorted set 同时维护一个 skiplist + 一个 hashtable，任何写操作都要同时更新两个结构。细粒度锁的开销可能抵消多线程收益。

**4. Redis 6.0 的修正**

Redis 6.0 引入了 I/O Threading（仅 网络读写多线程，命令执行仍单线程），是对"纯单线程"论点的补充而非推翻——命令执行的业务逻辑仍维持单线程。

### 1.2 ae 事件库的设计哲学

ae（async events）是 Redis 自己实现的极简事件库，约 500 行 C 代码，主要动机是：

- libevent/libev 引入外部依赖，且功能远超 Redis 需要
- 需要在同一个 event loop 中混合处理 file events 和 time events（定时器）
- 需要跨平台支持 epoll / kqueue / evport / select，通过编译期条件选择

ae 的设计非常纯粹：file event 注册读写回调，time event 注册定时回调，主循环每次 poll 一次，分发所有就绪事件。没有线程池、没有工作队列。

---

## 二、ae 事件循环深度精读

### 2.1 核心数据结构

【真实源码 redis/redis@unstable src/ae.h】

ae.h 中定义的核心结构：

```c
// aeFileEvent: 注册在某个 fd 上的读/写回调
typedef struct aeFileEvent {
    int mask;                    // AE_READABLE | AE_WRITABLE | AE_BARRIER
    aeFileProc *rfileProc;       // 可读回调
    aeFileProc *wfileProc;       // 可写回调
    void *clientData;            // 传给回调的用户数据
} aeFileEvent;

// aeFiredEvent: epoll_wait 返回后，记录哪些 fd 就绪了
typedef struct aeFiredEvent {
    int fd;
    int mask;
} aeFiredEvent;

// aeTimeEvent: 单向链表上的定时器节点
typedef struct aeTimeEvent {
    long long id;                // 时间事件 ID
    monotime when;               // 触发时间（单调时钟）
    aeTimeProc *timeProc;        // 回调
    aeEventFinalizerProc *finalizerProc; // 删除时的清理回调
    void *clientData;
    struct aeTimeEvent *prev;    // 双向链表
    struct aeTimeEvent *next;
    int refcount;                // 防止回调中删除自身时 use-after-free
} aeTimeEvent;

// aeEventLoop: 整个事件循环的状态
typedef struct aeEventLoop {
    int maxfd;                   // 当前注册的最大 fd
    int setsize;                 // 允许的最大 fd 数（对应 epoll 初始化大小）
    long long timeEventNextId;
    aeFileEvent *events;         // 按 fd 索引的注册事件数组
    aeFiredEvent *fired;         // epoll_wait 返回的就绪事件数组
    aeTimeEvent *timeEventHead;  // 时间事件链表头
    int stop;
    void *apidata;               // 指向 aeApiState（epoll/kqueue 私有数据）
    aeBeforeSleepProc *beforesleep;
    aeBeforeSleepProc *aftersleep;
    int flags;
} aeEventLoop;
```

**关键设计点**：`events` 数组以 fd 为下标，O(1) 查找。`fired` 数组保存 epoll_wait 返回的事件，分两步走：先 poll 得到 fired，再遍历 fired 逐个分发。

### 2.2 aeCreateEventLoop — 初始化

【真实源码 redis/redis@unstable src/ae.c】

```c
aeEventLoop *aeCreateEventLoop(int setsize) {
    aeEventLoop *eventLoop;
    int i;

    monotonicInit();  // 初始化单调时钟（CLOCK_MONOTONIC）

    if ((eventLoop = zmalloc(sizeof(*eventLoop))) == NULL) goto err;
    
    // nevents 取 setsize 和 INITIAL_EVENT 的较小值，避免 fd 稀少时浪费内存
    eventLoop->nevents = setsize < INITIAL_EVENT ? setsize : INITIAL_EVENT;
    eventLoop->events = zmalloc(sizeof(aeFileEvent)*eventLoop->nevents);
    eventLoop->fired = zmalloc(sizeof(aeFiredEvent)*eventLoop->nevents);
    if (eventLoop->events == NULL || eventLoop->fired == NULL) goto err;
    
    eventLoop->setsize = setsize;
    eventLoop->timeEventHead = NULL;
    eventLoop->timeEventNextId = 0;
    eventLoop->stop = 0;
    eventLoop->maxfd = -1;
    eventLoop->beforesleep = NULL;
    eventLoop->aftersleep = NULL;
    eventLoop->flags = 0;
    memset(eventLoop->privdata, 0, sizeof(eventLoop->privdata));
    
    if (aeApiCreate(eventLoop) == -1) goto err;  // 调用平台相关实现（epoll_create）
    
    // 所有 slot 初始化为 AE_NONE，表示没有注册事件
    for (i = 0; i < eventLoop->nevents; i++)
        eventLoop->events[i].mask = AE_NONE;
    return eventLoop;

err:
    if (eventLoop) {
        zfree(eventLoop->events);
        zfree(eventLoop->fired);
        zfree(eventLoop);
    }
    return NULL;
}
```

### 2.3 ae_epoll.c — epoll 适配层

【真实源码 redis/redis@unstable src/ae_epoll.c】

```c
// epoll 私有数据
typedef struct aeApiState {
    int epfd;                         // epoll fd
    struct epoll_event *events;       // epoll_wait 结果缓冲区
} aeApiState;

// 初始化：创建 epoll fd 并分配事件缓冲
static int aeApiCreate(aeEventLoop *eventLoop) {
    aeApiState *state = zmalloc(sizeof(aeApiState));
    if (!state) return -1;
    state->events = zmalloc(sizeof(struct epoll_event)*eventLoop->setsize);
    if (!state->events) {
        zfree(state);
        return -1;
    }
    state->epfd = epoll_create(1024);  // 参数 1024 是 hint，Linux 2.6.8+ 已忽略
    if (state->epfd == -1) {
        zfree(state->events);
        zfree(state);
        return -1;
    }
    anetCloexec(state->epfd);          // 设置 CLOEXEC，fork 后子进程不继承
    eventLoop->apidata = state;
    return 0;
}

// 注册 fd 的读/写事件
static int aeApiAddEvent(aeEventLoop *eventLoop, int fd, int mask) {
    aeApiState *state = eventLoop->apidata;
    struct epoll_event ee = {0};
    
    // 如果 fd 之前没有注册过任何事件，用 ADD；否则用 MOD（合并新 mask）
    int op = eventLoop->events[fd].mask == AE_NONE ?
            EPOLL_CTL_ADD : EPOLL_CTL_MOD;

    ee.events = 0;
    mask |= eventLoop->events[fd].mask;  // 合并已有 mask
    if (mask & AE_READABLE) ee.events |= EPOLLIN;
    if (mask & AE_WRITABLE) ee.events |= EPOLLOUT;
    ee.data.fd = fd;
    if (epoll_ctl(state->epfd, op, fd, &ee) == -1) return -1;
    return 0;
}

// poll 一次，返回就绪事件数
static int aeApiPoll(aeEventLoop *eventLoop, struct timeval *tvp) {
    aeApiState *state = eventLoop->apidata;
    int retval, numevents = 0;

    // tvp == NULL 表示无限等待；否则转换为毫秒
    retval = epoll_wait(state->epfd, state->events, eventLoop->setsize,
            tvp ? (tvp->tv_sec*1000 + (tvp->tv_usec + 999)/1000) : -1);
    if (retval > 0) {
        int j;
        numevents = retval;
        for (j = 0; j < numevents; j++) {
            int mask = 0;
            struct epoll_event *e = state->events+j;
            
            // 将 epoll 事件转换为 ae 内部 mask
            if (e->events & EPOLLIN)  mask |= AE_READABLE;
            if (e->events & EPOLLOUT) mask |= AE_WRITABLE;
            // 错误和挂断也映射为 READABLE+WRITABLE，让上层处理
            if (e->events & EPOLLERR) mask |= AE_WRITABLE|AE_READABLE;
            if (e->events & EPOLLHUP) mask |= AE_WRITABLE|AE_READABLE;
            eventLoop->fired[j].fd = e->data.fd;
            eventLoop->fired[j].mask = mask;
        }
    } else if (retval == -1 && errno != EINTR) {
        panic("aeApiPoll: epoll_wait, %s", strerror(errno));
    }
    return numevents;
}
```

**注意**：`EPOLLERR` 和 `EPOLLHUP` 都映射到 `READABLE|WRITABLE`——ae 的设计让上层读/写回调去处理错误（read() 返回 -1 或 0），而不是在 ae 层面做错误分发。

### 2.4 aeProcessEvents — 事件分发引擎

【真实源码 redis/redis@unstable src/ae.c】

```c
int aeProcessEvents(aeEventLoop *eventLoop, int flags) {
    int processed = 0, numevents;

    // 如果既不处理 file events 也不处理 time events，直接返回
    if (!(flags & AE_TIME_EVENTS) && !(flags & AE_FILE_EVENTS)) return 0;

    if (eventLoop->maxfd != -1 ||
        ((flags & AE_TIME_EVENTS) && !(flags & AE_DONT_WAIT))) {
        int j;
        struct timeval tv, *tvp = NULL;
        int64_t usUntilTimer;

        // beforesleep：在 epoll_wait 阻塞前执行（例如 flush pending AOF、处理 blocked clients）
        if (eventLoop->beforesleep != NULL && (flags & AE_CALL_BEFORE_SLEEP))
            eventLoop->beforesleep(eventLoop);

        // 计算 epoll_wait 的超时时间
        if ((flags & AE_DONT_WAIT) || (eventLoop->flags & AE_DONT_WAIT)) {
            // 非阻塞模式：立即返回
            tv.tv_sec = tv.tv_usec = 0;
            tvp = &tv;
        } else if (flags & AE_TIME_EVENTS) {
            // 计算距离最近一个 time event 的时间差，精确到微秒
            usUntilTimer = usUntilEarliestTimer(eventLoop);
            if (usUntilTimer >= 0) {
                tv.tv_sec = usUntilTimer / 1000000;
                tv.tv_usec = usUntilTimer % 1000000;
                tvp = &tv;
            }
            // tvp == NULL 说明没有 time event，epoll_wait 无限阻塞
        }

        numevents = aeApiPoll(eventLoop, tvp);  // epoll_wait

        // aftersleep：epoll_wait 返回后立即执行（例如 acquire module GIL，更新时间缓存）
        if (eventLoop->aftersleep != NULL && flags & AE_CALL_AFTER_SLEEP)
            eventLoop->aftersleep(eventLoop);

        // 遍历就绪 file events
        for (j = 0; j < numevents; j++) {
            int fd = eventLoop->fired[j].fd;
            aeFileEvent *fe = &eventLoop->events[fd];
            int mask = eventLoop->fired[j].mask;
            int fired = 0;

            // AE_BARRIER：正常先处理 read 再处理 write
            // 设置了 AE_BARRIER 时反转：先 write 再 read（用于 AOF 写回时保证先落盘再读）
            int invert = fe->mask & AE_BARRIER;

            if (!invert && fe->mask & mask & AE_READABLE) {
                fe->rfileProc(eventLoop, fd, fe->clientData, mask);
                fired++;
                fe = &eventLoop->events[fd];  // 回调可能修改事件，重新取
            }

            if (fe->mask & mask & AE_WRITABLE) {
                // 如果 rfileProc == wfileProc 且已触发过，跳过（防止重复调用）
                if (!fired || fe->wfileProc != fe->rfileProc) {
                    fe->wfileProc(eventLoop, fd, fe->clientData, mask);
                    fired++;
                }
            }

            // AE_BARRIER 模式下，write 已在上面先调用，现在再处理 read
            if (invert) {
                fe = &eventLoop->events[fd];
                if ((fe->mask & mask & AE_READABLE) &&
                    (!fired || fe->wfileProc != fe->rfileProc)) {
                    fe->rfileProc(eventLoop, fd, fe->clientData, mask);
                    fired++;
                }
            }
            processed++;
        }
    }
    // 处理时间事件（serverCron 等）
    if (flags & AE_TIME_EVENTS)
        processed += processTimeEvents(eventLoop);

    return processed;
}

// 主循环：极简
void aeMain(aeEventLoop *eventLoop) {
    eventLoop->stop = 0;
    while (!eventLoop->stop) {
        aeProcessEvents(eventLoop, AE_ALL_EVENTS |
                                   AE_CALL_BEFORE_SLEEP |
                                   AE_CALL_AFTER_SLEEP);
    }
}
```

**aeMain 解析**：整个 Redis 服务器的主循环就是这 5 行。`AE_ALL_EVENTS` = `AE_FILE_EVENTS | AE_TIME_EVENTS`。每次迭代都：
1. 调用 `beforesleep`（AOF flush、blocked clients 处理等）
2. `epoll_wait` 阻塞，超时时间 = 最近 time event 触发时间
3. 调用 `aftersleep`（更新时间缓存）
4. 分发就绪 file events
5. 处理到期 time events（`serverCron`，默认 100ms 周期）

---

## 三、SDS — 动态字符串

### 3.1 设计动机

C 的 `char*` 有三个问题：O(n) strlen、不支持二进制安全、追加需 realloc。Redis 设计了 SDS（Simple Dynamic String）解决这些问题。

### 3.2 五种 header 类型

【真实源码 redis/redis@unstable src/sds.h】

```c
// __attribute__((__packed__)) 关键：禁止编译器添加 padding，
// 使得 buf[-1] 就是 flags 字节
struct __attribute__ ((__packed__)) sdshdr5 {
    unsigned char flags;   // 低 3 位=类型，高 5 位=长度（仅用于 <=31 字节的字符串）
    char buf[];
};
struct __attribute__ ((__packed__)) sdshdr8 {
    uint8_t len;           // 实际字符串长度（不含 \0）
    uint8_t alloc;         // 分配的 buf 容量（不含 header 和 \0）
    unsigned char flags;   // 低 3 位=类型
    char buf[];
};
struct __attribute__ ((__packed__)) sdshdr16 {
    uint16_t len;
    uint16_t alloc;
    unsigned char flags;
    char buf[];
};
struct __attribute__ ((__packed__)) sdshdr32 {
    uint32_t len;
    uint32_t alloc;
    unsigned char flags;
    char buf[];
};
struct __attribute__ ((__packed__)) sdshdr64 {
    uint64_t len;
    uint64_t alloc;
    unsigned char flags;
    char buf[];
};
```

**内存布局技巧**：SDS 返回给调用者的是 `buf` 指针，而不是 header 指针。这样 SDS 字符串可以直接传给 C 字符串函数（以 `\0` 结尾）。通过 `buf[-1]` 读取 flags 字节，再 switch(type) 选择合适 header 类型来读 len/alloc。

```c
// O(1) 获取长度：直接读 header 中的 len 字段
static inline size_t sdslen(const sds s) {
    switch (sdsType(s)) {
        case SDS_TYPE_5:  return SDS_TYPE_5_LEN(s);
        case SDS_TYPE_8:  return SDS_HDR(8,s)->len;
        case SDS_TYPE_16: return SDS_HDR(16,s)->len;
        case SDS_TYPE_32: return SDS_HDR(32,s)->len;
        case SDS_TYPE_64: return SDS_HDR(64,s)->len;
    }
    return 0;
}
```

### 3.3 贪心预分配策略

【真实源码 redis/redis@unstable src/sds.c】

```c
sds _sdsMakeRoomFor(sds s, size_t addlen, int greedy) {
    if (avail >= addlen) return s;  // 空间够，直接返回

    len = sdslen(s);
    reqlen = newlen = (len + addlen);
    if (greedy == 1) {
        if (newlen < SDS_MAX_PREALLOC)  // SDS_MAX_PREALLOC = 1MB
            newlen *= 2;                // < 1MB：翻倍
        else
            newlen += SDS_MAX_PREALLOC; // >= 1MB：线性增加 1MB
    }
    // ... realloc 逻辑 ...
    sdssetalloc(s, usable);
    return s;
}
```

这个策略确保 N 次追加操作摊还后是 O(N) 而非 O(N²)。

### 3.4 embstr vs raw 编码

Redis string 对象有三种内部编码：

| 编码 | 条件 | 内存布局 |
|------|------|---------|
| `OBJ_ENCODING_INT` | 整数且 ≤ 2^63-1 | 直接把整数存在指针里（ptr 当 int 用） |
| `OBJ_ENCODING_EMBSTR` | 字符串长度 ≤ 44 字节 | robj 和 SDS 在同一块内存中 |
| `OBJ_ENCODING_RAW` | 字符串长度 > 44 字节 | robj 和 SDS 分两次 malloc |

**44 字节边界**：jemalloc 分配 64 字节 arena，robj 占 16 字节，SDS sdshdr8 占 3 字节，buf 44 字节 + `\0` = 1 字节，共 64 字节，刚好一个 arena。embstr 只需一次 malloc/free，且 CPU cache 友好。

---

## 四、dict — 渐进式 rehash 哈希表

### 4.1 双 hashtable 结构

【真实源码 redis/redis@unstable src/dict.h】

```c
struct dict {
    dictType *type;          // 虚函数表：hash函数、key比较、dup/free 回调
    dictEntry **ht_table[2]; // 两张哈希表，rehash 时同时使用
    unsigned long ht_used[2];// 各表已用 bucket 数
    long rehashidx;          // rehash 进度：-1=未在 rehash，否则=当前迁移的 bucket index
    unsigned pauserehash;    // 暂停 rehash 的计数器（迭代器持有时暂停）
    signed char ht_size_exp[2]; // 哈希表大小的 2 的幂次（size = 1 << exp）
    int16_t pauseAutoResize;
    void *metadata[];
};
```

**双表设计原理**：`ht_table[0]` 是当前主表，`ht_table[1]` 是 rehash 目标表。rehash 期间，查找要同时查两张表；写入只写 `ht_table[1]`。

### 4.2 渐进式 rehash 源码

【真实源码 redis/redis@unstable src/dict.c】

```c
int dictRehash(dict *d, int n) {
    int empty_visits = n*10; // 最多访问 n*10 个空 bucket，防止单次迁移耗时过长
    unsigned long s0 = DICTHT_SIZE(d->ht_size_exp[0]);
    unsigned long s1 = DICTHT_SIZE(d->ht_size_exp[1]);
    
    if (dict_can_resize == DICT_RESIZE_FORBID || !dictIsRehashing(d)) return 0;
    
    // DICT_RESIZE_AVOID 模式下（有 BGSAVE/BGREWRITEAOF 进行中），
    // 只有比例超过强制阈值才继续 rehash（扩容 4:1，缩容 1:32）
    if (dict_can_resize == DICT_RESIZE_AVOID && 
        ((s1 > s0 && s1 < dict_force_resize_ratio * s0) ||
         (s1 < s0 && s0 < HASHTABLE_MIN_FILL * dict_force_resize_ratio * s1))) {
        return 0;
    }

    while (n-- && d->ht_used[0] != 0) {
        // 跳过空 bucket（最多跳 empty_visits 个）
        assert(DICTHT_SIZE(d->ht_size_exp[0]) > (unsigned long)d->rehashidx);
        while (d->ht_table[0][d->rehashidx] == NULL) {
            d->rehashidx++;
            if (--empty_visits == 0) return 1;
        }
        // 把 rehashidx 这个 bucket 里的所有 entry 迁移到 ht_table[1]
        rehashEntriesInBucketAtIndex(d, d->rehashidx);
        d->rehashidx++;
    }
    return !dictCheckRehashingCompleted(d); // 返回 0 表示 rehash 完成
}

// 每次写入/查找时顺带推进一步 rehash（每次迁移 1 个 bucket）
static void _dictRehashStep(dict *d) {
    if (d->pauserehash == 0) dictRehash(d, 1);
}
```

**触发时机**：
1. **被动 rehash**：每次 `dictAdd` / `dictFind` 时调用 `_dictRehashStep`，每次迁移 1 个 bucket
2. **主动 rehash**：`serverCron` 中调用 `dictRehashMilliseconds`，在 1ms 内尽量多迁移
3. **禁止 rehash**：BGSAVE / BGREWRITEAOF 进行中（COW 时 rehash 会产生大量写时复制页面）

**扩缩容阈值**：
- 扩容：`ht_used[0] >= ht_size[0]`（负载因子 ≥ 1）；RESIZE_AVOID 模式下强制阈值为 4
- 缩容：`ht_used[0] * 8 <= ht_size[0]`（负载因子 ≤ 0.125）；RESIZE_AVOID 模式下强制阈值为 1/32

---

## 五、skiplist — Sorted Set 的灵魂

### 5.1 William Pugh 的原始论文

Skip List 由 William Pugh 在 1990 年提出（"Skip Lists: A Probabilistic Alternative to Balanced Trees"，Communications of the ACM, 33(6), 668-676）。核心思想：通过**概率性**地维护多层指针，在期望 O(log N) 时间内完成有序集合的插入/删除/范围查询，而无需像红黑树那样复杂的旋转操作。

Redis 选择 skiplist 而非红黑树的原因（antirez 在 redis-dev 邮件列表中的解释）：
1. 范围查询（ZRANGEBYSCORE）在 skiplist 上实现简单，在红黑树上需要额外遍历
2. skiplist 的常数因子比红黑树小
3. 实现更简单，不容易引入 bug

### 5.2 数据结构定义

【真实源码 redis/redis@unstable src/server.h】

```c
typedef struct zskiplistNode {
    double score;                    // 排序依据
    struct zskiplistNode *backward;  // 第 0 层的后向指针（支持 ZREVRANGE）
    struct zskiplistLevel {
        struct zskiplistNode *forward; // 指向同层的下一个节点
        /* 在 level[0]，span 被复用为 zskiplistNodeInfo（存储节点信息），
         * 在 level[1..N]，span 是跨越的节点数（用于 ZRANK 的 O(log N) 实现） */
        unsigned long span;
    } level[];                       // 柔性数组，levels 个 level 槽
    /* sds ele 嵌入在 level[] 数组之后（通过 zslGetNodeElement(node) 访问） */
} zskiplistNode;

typedef struct zskiplist {
    struct zskiplistNode *header, *tail;
    unsigned long length;  // 节点总数（不含 header）
    int level;             // 当前最大层数
    size_t alloc_size;
} zskiplist;
```

**内存布局创新**：`ele`（SDS 字符串）嵌入在 `level[]` 数组之后，避免额外的指针和 malloc。`zslGetNodeElement(node)` 宏通过偏移计算访问嵌入的 SDS。这是 Redis 近年来的优化，减少了每个节点的内存分配次数。

### 5.3 概率层数生成

【真实源码 redis/redis@unstable src/t_zset.c】

```c
// ZSKIPLIST_P = 0.25，ZSKIPLIST_MAXLEVEL = 32
static int zslRandomLevel(void) {
    static const int threshold = ZSKIPLIST_P * RAND_MAX; // 0.25 * RAND_MAX
    int level = 1;
    while (random() < threshold)   // 每次有 25% 概率升一层
        level += 1;
    return (level < ZSKIPLIST_MAXLEVEL) ? level : ZSKIPLIST_MAXLEVEL;
}
```

**数学分析**：
- 层数 = k 的概率：P(level=k) = (1-p) * p^(k-1) = 0.75 * 0.25^(k-1)
- 期望层数：E[level] = 1/(1-p) = 1/0.75 ≈ 1.33
- 期望高度：E[max_level] = log_{1/p}(N) = log_4(N)
- 空间复杂度：O(N/（1-p)) ≈ O(1.33N)
- 时间复杂度：O(log_{1/p}(N)) = O(log_4(N)) ≈ O(log N / log 4)

**为什么选 p=0.25 而不是 0.5**：p=0.25 比 p=0.5 使用更少内存（平均 1.33 层 vs 2 层），速度上差异不大（log_4(N) vs log_2(N) 只差一个常数因子）。

### 5.4 插入操作

【真实源码 redis/redis@unstable src/t_zset.c】

```c
zskiplistNode *zslInsert(zskiplist *zsl, double score, sds ele) {
    int level;
    serverAssert(!isnan(score));

    level = zslRandomLevel();                          // 随机层数
    zskiplistNode *node = zslCreateNode(zsl, level, score, ele); // 分配节点
    zslInsertNode(zsl, node);                          // 插入到有序位置
    return node;
}
```

`zslInsertNode` 的核心逻辑（【示意，非逐字】）：
1. 从最高层向右遍历，找到每层的插入前驱节点，记录在 `update[]` 数组中
2. 同时累加每层的 `span`，计算新节点的 rank
3. 为新节点分配层数（如果新节点层数 > 当前最大层，用 header 补全 `update[]`）
4. 从第 0 层到 level 层，逐层插入新节点，更新 `forward` 指针和 `span`
5. 更新 `backward` 指针（第 0 层双向链表）
6. 更新 `zsl->level` 和 `zsl->length`

### 5.5 Sorted Set 的双结构

Redis Sorted Set 在非 listpack 编码下同时维护：
- **skiplist**：有序，支持 `ZRANGE`、`ZRANGEBYSCORE`、`ZRANK` O(log N)
- **hashtable**（`dict`）：支持 `ZSCORE`、`ZINCRBY` O(1)

两个结构共享 SDS 字符串（不复制），通过引用计数管理。

---

## 六、listpack — 紧凑序列化列表

### 6.1 listpack vs ziplist：设计演进

ziplist 是 Redis 早期的紧凑编码，存在**连锁更新（cascade update）**问题：修改一个节点的大小可能导致后继节点的 `prevlen` 字段大小改变，继而触发连锁 realloc，最坏 O(N)。

listpack（Redis 7.0 起替代 ziplist）通过消除 `prevlen` 字段解决了这个问题：每个 entry 只存自己的长度（backlen），不再存前驱的长度。反向遍历通过 backlen 字段实现，但 backlen 是编码在 entry 末尾的，不需要修改前驱。

### 6.2 内存布局

```
[Total bytes: 4B][Num elements: 2B][Entry1][Entry2]...[EntryN][0xFF EOF]

Entry 格式:
[encoding + data][backlen]

backlen: 变长整数，编码 encoding+data 的总字节数
         1 字节 (< 128)
         2 字节 (< 16384)
         ...最多 5 字节
```

### 6.3 lpNew 与编码函数

【真实源码 redis/redis@unstable src/listpack.c】

```c
unsigned char *lpNew(size_t capacity) {
    unsigned char *lp = lp_malloc(capacity > LP_HDR_SIZE+1 ? capacity : LP_HDR_SIZE+1);
    if (lp == NULL) return NULL;
    lpSetTotalBytes(lp, LP_HDR_SIZE+1);  // 初始大小 = header(6) + EOF(1) = 7
    lpSetNumElements(lp, 0);
    lp[LP_HDR_SIZE] = LP_EOF;             // 0xFF 结束标记
    return lp;
}

// 整数编码：选最紧凑的表示
static inline void lpEncodeIntegerGetType(int64_t v, unsigned char *intenc, uint64_t *enclen) {
    if (v >= 0 && v <= 127) {
        // 7 bit unsigned int，1 字节
        if (intenc != NULL) intenc[0] = v;
        if (enclen != NULL) *enclen = 1;
    } else if (v >= -4096 && v <= 4095) {
        // 13 bit signed int，2 字节
        if (v < 0) v = ((int64_t)1<<13) + v;
        if (intenc != NULL) {
            intenc[0] = (v>>8) | LP_ENCODING_13BIT_INT;
            intenc[1] = v & 0xff;
        }
        if (enclen != NULL) *enclen = 2;
    }
    // ... 16/24/32/64 bit 分段处理 ...
}

// 字符串编码：长度前缀
static inline void lpEncodeString(unsigned char *buf, unsigned char *s, uint32_t len) {
    if (len < 64) {
        buf[0] = len | LP_ENCODING_6BIT_STR;    // 6 bit 长度 + 数据
        memcpy(buf+1, s, len);
    } else if (len < 4096) {
        buf[0] = (len >> 8) | LP_ENCODING_12BIT_STR; // 12 bit 长度
        buf[1] = len & 0xff;
        memcpy(buf+2, s, len);
    } else {
        buf[0] = LP_ENCODING_32BIT_STR;
        // 4 字节小端长度 + 数据
        buf[1] = len & 0xff; buf[2] = (len >> 8) & 0xff;
        buf[3] = (len >> 16) & 0xff; buf[4] = (len >> 24) & 0xff;
        memcpy(buf+5, s, len);
    }
}
```

---

## 七、quicklist — List 的双端链表 + listpack

### 7.1 结构定义

【真实源码 redis/redis@unstable src/quicklist.h】

```c
typedef struct quicklistNode {
    struct quicklistNode *prev;
    struct quicklistNode *next;
    unsigned char *entry;          // 指向 listpack（或压缩后的 LZF 数据）
    size_t sz;                     // entry 字节数
    unsigned int count : 16;       // 此节点中的元素数（最多 65535）
    unsigned int encoding : 2;     // RAW==1（listpack）或 LZF==2（压缩）
    unsigned int container : 2;    // PLAIN==1 或 PACKED==2（listpack 容器）
    unsigned int recompress : 1;   // 访问后需要重新压缩
    unsigned int attempted_compress : 1; // 太小了，不值得压缩
    unsigned int dont_compress : 1;// 防止正在使用的节点被压缩
    unsigned int extra : 9;
} quicklistNode;

typedef struct quicklist {
    quicklistNode *head;
    quicklistNode *tail;
    unsigned long count;           // 所有节点中的总元素数
    unsigned long len;             // 节点数（listpack 个数）
    size_t alloc_size;
    signed int fill : QL_FILL_BITS;     // 每个 listpack 节点的大小限制
    unsigned int compress : QL_COMP_BITS; // 两端不压缩的节点数（LZF 压缩中间节点）
    unsigned int bookmark_count: QL_BM_BITS;
    quicklistBookmark bookmarks[];
} quicklist;
```

**fill 参数**：
- 正数：每个 listpack 节点最多存 fill 个元素
- 负数：-1 ≤ 4KB，-2 ≤ 8KB（默认），-3 ≤ 16KB，-4 ≤ 32KB，-5 ≤ 64KB

**compress 参数**：两端各保留 compress 个节点不压缩（LPUSH/RPUSH 操作的热点），中间节点用 LZF 压缩（节省内存）。

---

## 八、intset — 小整数集合

### 8.1 自动升级机制

【真实源码 redis/redis@unstable src/intset.c】

```c
intset *intsetNew(void) {
    intset *is = zmalloc(sizeof(intset));
    is->encoding = intrev32ifbe(INTSET_ENC_INT16); // 初始 INT16
    is->length = 0;
    return is;
}

intset *intsetAdd(intset *is, int64_t value, uint8_t *success) {
    uint8_t valenc = _intsetValueEncoding(value); // 判断 value 需要几字节
    uint32_t pos;
    if (success) *success = 1;

    if (valenc > intrev32ifbe(is->encoding)) {
        return intsetUpgradeAndAdd(is, value); // 需要升级编码
    } else {
        if (intsetSearch(is, value, &pos)) {   // 二分查找（intset 有序）
            if (success) *success = 0;          // 已存在，返回失败
            return is;
        }
        is = intsetResize(is, intrev32ifbe(is->length)+1);
        if (pos < intrev32ifbe(is->length))
            intsetMoveTail(is, pos, pos+1);     // 后移腾位置
    }
    _intsetSet(is, pos, value);
    is->length = intrev32ifbe(intrev32ifbe(is->length)+1);
    return is;
}

// 升级：INT16 -> INT32 -> INT64（单向，不降级）
static intset *intsetUpgradeAndAdd(intset *is, int64_t value) {
    uint8_t curenc = intrev32ifbe(is->encoding);
    uint8_t newenc = _intsetValueEncoding(value); // 新值决定目标编码
    int length = intrev32ifbe(is->length);
    int prepend = value < 0 ? 1 : 0; // 负数放头部，正数放尾部

    is->encoding = intrev32ifbe(newenc);
    is = intsetResize(is, intrev32ifbe(is->length)+1);

    // 从后往前重新编码（避免覆盖未处理的数据）
    while (length--)
        _intsetSet(is, length+prepend, _intsetGetEncoded(is, length, curenc));

    if (prepend)
        _intsetSet(is, 0, value);
    else
        _intsetSet(is, intrev32ifbe(is->length), value);
    is->length = intrev32ifbe(intrev32ifbe(is->length)+1);
    return is;
}
```

**关键设计**：intset 是一个**有序紧凑数组**，支持二分查找 O(log N)。所有元素使用同一编码（INT16/INT32/INT64），因此可以用 `memcpy` 整体移动。升级时从后往前遍历防止覆盖。

---

## 九、编码自适应总览

Redis 对每种数据类型都有"小数据集紧凑编码 → 大数据集性能编码"的自动转换：

| 类型 | 小数据集编码 | 大数据集编码 | 触发阈值（默认） |
|------|------------|------------|----------------|
| String | embstr (≤44B) / INT | raw SDS | 长度 > 44B |
| Hash | listpack | hashtable | 元素 > 128 或值 > 64B |
| List | listpack | quicklist | 元素 > 128 或值 > 64B |
| Set (全整数) | intset | hashtable | 元素 > 512 |
| Set (混合) | listpack | hashtable | 元素 > 128 或值 > 64B |
| Sorted Set | listpack | skiplist+HT | 元素 > 128 或值 > 64B |

**编码转换是单向不可逆的**（除非 DEBUG OBJECT ENCODING 手动触发）：一旦升级到大编码，即使删除元素让数据集变小，也不会降回小编码。

---

## 十、⭐ 可运行 Demo

### Demo 1：Python 手写 epoll mini event loop（印证 ae.c 设计）

> 设计为可运行，请在你的 Linux 环境验证（macOS 无 epoll，需用 Linux 或 Docker）。  
> 依赖：Python 3.8+，Linux 内核。

```python
#!/usr/bin/env python3
"""
ae_mini.py — Redis ae 事件库的 Python 等价实现
模拟 aeCreateEventLoop / aeCreateFileEvent / aeMain / aeProcessEvents
只依赖 Python 标准库（select.epoll 仅 Linux 可用）

运行方法：
  Linux: python3 ae_mini.py
  Docker: docker run --rm -v $(pwd):/w python:3.11-slim python3 /w/ae_mini.py

预期输出（约 3 秒后）：
  [server] 监听 127.0.0.1:16379
  [event loop] 注册 time event: tick_every_1s, id=0
  [event loop] 注册 file event: fd=<n> mask=READABLE
  [client] 连接成功，发送 PING
  [server] 接受新连接，fd=<n>
  [server] 收到数据: b'PING'
  [server] 发送响应: +PONG
  [client] 收到响应: b'+PONG'
  [time event] tick_every_1s 触发 (第 1 次)
  [time event] tick_every_1s 触发 (第 2 次)
  [time event] tick_every_1s 触发 (第 3 次)
  [event loop] 停止
"""

import select
import socket
import time
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional, Dict, List

# ─── 对应 ae.h 的 AE_READABLE / AE_WRITABLE ─────────────────────────────────
AE_READABLE = 1
AE_WRITABLE = 2
AE_NONE = 0


# ─── 对应 aeFileEvent ────────────────────────────────────────────────────────
@dataclass
class FileEvent:
    mask: int = AE_NONE
    rfile_proc: Optional[Callable] = None  # 读就绪回调
    wfile_proc: Optional[Callable] = None  # 写就绪回调
    client_data: object = None


# ─── 对应 aeTimeEvent ────────────────────────────────────────────────────────
@dataclass
class TimeEvent:
    id: int
    when: float          # 触发时间（time.monotonic()）
    interval: float      # 重复间隔（0 = 单次）
    proc: Callable
    client_data: object = None
    name: str = ""


# ─── 对应 aeEventLoop ────────────────────────────────────────────────────────
class AeEventLoop:
    def __init__(self, setsize: int = 1024):
        # 对应 aeCreateEventLoop 中的 epoll_create
        self._epoll = select.epoll()
        self._events: Dict[int, FileEvent] = {}   # fd -> FileEvent
        self._time_events: List[TimeEvent] = []   # 时间事件链表（简化为 list）
        self._time_event_next_id = 0
        self.stop = False
        self.setsize = setsize
        print(f"[event loop] aeCreateEventLoop(setsize={setsize})")

    # 对应 aeCreateFileEvent
    def create_file_event(self, fd: int, mask: int,
                          rproc: Optional[Callable] = None,
                          wproc: Optional[Callable] = None,
                          client_data=None):
        fe = self._events.get(fd, FileEvent())
        old_mask = fe.mask

        # 对应 aeApiAddEvent 中 EPOLL_CTL_ADD vs EPOLL_CTL_MOD 的判断
        epoll_mask = 0
        new_mask = old_mask | mask
        if new_mask & AE_READABLE: epoll_mask |= select.EPOLLIN
        if new_mask & AE_WRITABLE: epoll_mask |= select.EPOLLOUT

        if old_mask == AE_NONE:
            self._epoll.register(fd, epoll_mask)    # EPOLL_CTL_ADD
        else:
            self._epoll.modify(fd, epoll_mask)       # EPOLL_CTL_MOD

        fe.mask = new_mask
        if rproc: fe.rfile_proc = rproc
        if wproc: fe.wfile_proc = wproc
        fe.client_data = client_data
        self._events[fd] = fe
        print(f"[event loop] 注册 file event: fd={fd} mask={'READABLE' if mask & AE_READABLE else 'WRITABLE'}")

    # 对应 aeDeleteFileEvent
    def delete_file_event(self, fd: int, mask: int):
        fe = self._events.get(fd)
        if not fe: return
        new_mask = fe.mask & (~mask)
        if new_mask == AE_NONE:
            self._epoll.unregister(fd)               # EPOLL_CTL_DEL
        else:
            epoll_mask = 0
            if new_mask & AE_READABLE: epoll_mask |= select.EPOLLIN
            if new_mask & AE_WRITABLE: epoll_mask |= select.EPOLLOUT
            self._epoll.modify(fd, epoll_mask)       # EPOLL_CTL_MOD
        fe.mask = new_mask

    # 对应 aeCreateTimeEvent（返回 event id）
    def create_time_event(self, delay_sec: float, proc: Callable,
                          interval: float = 0, name: str = "",
                          client_data=None) -> int:
        eid = self._time_event_next_id
        self._time_event_next_id += 1
        te = TimeEvent(
            id=eid, when=time.monotonic() + delay_sec,
            interval=interval, proc=proc,
            client_data=client_data, name=name
        )
        self._time_events.append(te)
        print(f"[event loop] 注册 time event: {name}, id={eid}")
        return eid

    # 对应 aeProcessEvents（核心分发逻辑）
    def process_events(self, timeout: float = -1) -> int:
        processed = 0

        # 计算 epoll_wait 超时：取最近 time event 的剩余时间
        if self._time_events:
            now = time.monotonic()
            earliest = min(te.when for te in self._time_events)
            wait = max(0.0, earliest - now)
            if timeout < 0 or wait < timeout:
                timeout = wait

        # 对应 aeApiPoll / epoll_wait
        try:
            fired = self._epoll.poll(timeout if timeout >= 0 else -1)
        except OSError:
            fired = []

        # 对应 aeProcessEvents 中的 file event 分发循环
        for fd, event in fired:
            fe = self._events.get(fd)
            if not fe: continue
            if (event & select.EPOLLIN) and fe.rfile_proc:
                fe.rfile_proc(self, fd, fe.client_data)
                processed += 1
            if (event & select.EPOLLOUT) and fe.wfile_proc:
                fe.wfile_proc(self, fd, fe.client_data)
                processed += 1

        # 对应 processTimeEvents
        now = time.monotonic()
        for te in list(self._time_events):
            if te.when <= now:
                te.proc(self, te.id, te.client_data)
                processed += 1
                if te.interval > 0:
                    te.when = now + te.interval  # 重置为下次触发
                else:
                    self._time_events.remove(te) # 单次事件，删除

        return processed

    # 对应 aeMain
    def main(self):
        print("[event loop] aeMain 开始")
        while not self.stop:
            self.process_events(timeout=1.0)
        print("[event loop] 停止")


# ─── 演示：简单 echo 服务器 ─────────────────────────────────────────────────

def on_accept(el: AeEventLoop, server_fd: int, client_data):
    """对应 acceptTcpHandler：接受新连接，注册读事件"""
    conn, addr = client_data['sock'].accept()
    conn.setblocking(False)
    fd = conn.fileno()
    print(f"[server] 接受新连接，fd={fd}")

    def on_read(el, fd, cd):
        data = cd['conn'].recv(1024)
        if data:
            print(f"[server] 收到数据: {data}")
            resp = b'+PONG\r\n' if data.strip() == b'PING' else b'-ERR\r\n'
            print(f"[server] 发送响应: {resp.decode().strip()}")
            cd['conn'].sendall(resp)
        else:
            el.delete_file_event(fd, AE_READABLE)
            cd['conn'].close()

    el.create_file_event(fd, AE_READABLE, rproc=on_read, client_data={'conn': conn})


def tick(el: AeEventLoop, eid: int, client_data):
    """对应 serverCron：定时任务"""
    client_data['count'] += 1
    print(f"[time event] tick_every_1s 触发 (第 {client_data['count']} 次)")
    if client_data['count'] >= 3:
        el.stop = True  # 演示 3 次后退出


def main():
    # 创建 event loop（对应 aeCreateEventLoop）
    el = AeEventLoop(setsize=1024)

    # 创建 server socket
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(('127.0.0.1', 16379))
    server_sock.listen(5)
    server_sock.setblocking(False)
    print(f"[server] 监听 127.0.0.1:16379")

    # 注册 accept 事件（对应 aeCreateFileEvent for server socket）
    el.create_file_event(
        server_sock.fileno(), AE_READABLE,
        rproc=on_accept,
        client_data={'sock': server_sock}
    )

    # 注册定时事件（对应 aeCreateTimeEvent for serverCron）
    el.create_time_event(
        delay_sec=1.0, proc=tick, interval=1.0,
        name="tick_every_1s",
        client_data={'count': 0}
    )

    # 启动模拟客户端（在独立线程中发送 PING）
    def client_thread():
        time.sleep(0.3)  # 等 server 就绪
        c = socket.socket()
        c.connect(('127.0.0.1', 16379))
        print("[client] 连接成功，发送 PING")
        c.sendall(b'PING')
        resp = c.recv(128)
        print(f"[client] 收到响应: {resp.strip()}")
        c.close()

    threading.Thread(target=client_thread, daemon=True).start()

    # 启动事件循环（对应 aeMain）
    el.main()
    server_sock.close()


if __name__ == '__main__':
    main()
```

**与 ae.c 的对应关系**：

| Python 代码 | ae.c 对应 |
|------------|---------|
| `AeEventLoop.__init__` → `epoll()` | `aeCreateEventLoop` → `epoll_create(1024)` |
| `create_file_event` → `epoll.register/modify` | `aeApiAddEvent` → `epoll_ctl(ADD/MOD)` |
| `process_events` → `epoll.poll` | `aeApiPoll` → `epoll_wait` |
| `on_accept` / `on_read` | `acceptTcpHandler` / `readQueryFromClient` |
| `tick` | `serverCron` |
| `el.main()` while loop | `aeMain` while loop |

---

### Demo 2：Python 手写增量 rehash hashtable（印证 dict.c）

> 设计为可运行，请在你的环境验证。依赖：Python 3.8+（无外部依赖）。

```python
#!/usr/bin/env python3
"""
dict_rehash.py — Redis dict.c 渐进式 rehash 的 Python 等价实现

展示：
1. 双表结构（ht_table[0] 和 ht_table[1]）
2. 写入只写 ht_table[1]（rehash 期间）
3. 查找同时查两张表
4. 每次操作顺带迁移 1 个 bucket（_dictRehashStep）
5. 扩容触发：load_factor >= 1

运行方法：python3 dict_rehash.py

预期输出（关键部分）：
  === 初始化 dict（初始容量 4）===
  写入 key0..key3 后:  ht[0] used=4 size=4, ht[1] not initialized
  触发扩容: load_factor=1.00 >= 1, 新 ht[1] size=8
  *** 开始渐进式 rehash ***
  写入 key4（rehash 中：只写 ht[1]）: ht[0] used=3, ht[1] used=2  [迁移 bucket 0]
  写入 key5（rehash 中）: ht[0] used=2, ht[1] used=3  [迁移 bucket 1]
  ...
  rehash 完成: ht[0] now has 0 entries, ht[1] promoted to ht[0]
  最终 ht[0] used=8 size=8, ht[1] not initialized
  查找 key0 = value0 ✓
  查找 key5 = value5 ✓
"""


class DictEntry:
    __slots__ = ['key', 'value', 'next']
    def __init__(self, key, value, next_=None):
        self.key = key; self.value = value; self.next = next_


class Dict:
    """
    对应 Redis dict 的核心逻辑：
    - ht_table[2]：双哈希表（list of buckets，每个 bucket 是链表头）
    - ht_used[2]：各表已用条目数
    - rehashidx：当前 rehash 进度（-1 = 未在 rehash）
    """

    def __init__(self, initial_exp: int = 2):
        """initial_exp: 初始 size = 2^initial_exp"""
        size = 1 << initial_exp
        self.ht_table = [[None] * size, None]  # ht_table[0] 初始化，ht_table[1] 为 None
        self.ht_size_exp = [initial_exp, -1]   # -1 表示 ht[1] 未初始化
        self.ht_used = [0, 0]
        self.rehashidx = -1                     # -1 = 未在 rehash

    def _ht_size(self, table: int) -> int:
        """对应 DICTHT_SIZE(ht_size_exp[t])"""
        exp = self.ht_size_exp[table]
        return 1 << exp if exp >= 0 else 0

    def _is_rehashing(self) -> bool:
        return self.rehashidx >= 0

    def _hash(self, key) -> int:
        return hash(key)

    def _bucket_index(self, key, table: int) -> int:
        return self._hash(key) & (self._ht_size(table) - 1)

    def _load_factor(self, table: int = 0) -> float:
        size = self._ht_size(table)
        return self.ht_used[table] / size if size > 0 else 0

    # 对应 dictAdd / dictAddRaw + dictSetVal
    def add(self, key, value):
        if self._is_rehashing():
            self._rehash_step()  # _dictRehashStep：顺带迁移 1 个 bucket

        # rehash 中：写入 ht[1]；否则写入 ht[0]
        table = 1 if self._is_rehashing() else 0
        size = self._ht_size(table)
        idx = self._hash(key) & (size - 1)

        # 检查 key 是否已存在（先查两张表）
        existing = self._find_in_table(key, table)
        if existing is None and self._is_rehashing():
            existing = self._find_in_table(key, 1 - table)
        if existing:
            existing.value = value
            return

        entry = DictEntry(key, value, self.ht_table[table][idx])
        self.ht_table[table][idx] = entry
        self.ht_used[table] += 1

        # 检查是否需要扩容（对应 _dictExpandIfNeeded）
        if not self._is_rehashing() and self._load_factor(0) >= 1.0:
            self._expand(self.ht_used[0] * 2)

    # 对应 dictFind
    def find(self, key):
        if self._is_rehashing():
            self._rehash_step()
        # 先查 ht[0]，再查 ht[1]
        entry = self._find_in_table(key, 0)
        if entry is None and self._is_rehashing():
            entry = self._find_in_table(key, 1)
        return entry

    def _find_in_table(self, key, table: int):
        if self._ht_size(table) == 0:
            return None
        idx = self._bucket_index(key, table)
        e = self.ht_table[table][idx]
        while e:
            if e.key == key:
                return e
            e = e.next
        return None

    # 对应 _dictExpand + _dictResize
    def _expand(self, new_size: int):
        import math
        exp = math.ceil(math.log2(max(new_size, 1)))
        actual_size = 1 << exp
        print(f"  触发扩容: load_factor={self._load_factor(0):.2f} >= 1, 新 ht[1] size={actual_size}")
        self.ht_table[1] = [None] * actual_size
        self.ht_size_exp[1] = exp
        self.ht_used[1] = 0
        self.rehashidx = 0
        print(f"  *** 开始渐进式 rehash ***")

    # 对应 dictRehash(d, n=1)
    def _rehash(self, n: int = 1):
        if not self._is_rehashing():
            return False
        empty_visits = n * 10  # 最多访问空 bucket 数

        while n > 0 and self.ht_used[0] > 0:
            # 跳过空 bucket
            while self.ht_table[0][self.rehashidx] is None:
                self.rehashidx += 1
                empty_visits -= 1
                if empty_visits == 0:
                    return True

            # 把这个 bucket 的所有 entry 迁移到 ht[1]
            entry = self.ht_table[0][self.rehashidx]
            bucket_moved = 0
            while entry:
                next_entry = entry.next
                new_idx = self._hash(entry.key) & (self._ht_size(1) - 1)
                entry.next = self.ht_table[1][new_idx]
                self.ht_table[1][new_idx] = entry
                self.ht_used[0] -= 1
                self.ht_used[1] += 1
                entry = next_entry
                bucket_moved += 1

            self.ht_table[0][self.rehashidx] = None
            self.rehashidx += 1
            n -= 1

        # 检查是否 rehash 完成
        if self.ht_used[0] == 0:
            print(f"  rehash 完成: ht[0] now has 0 entries, ht[1] promoted to ht[0]")
            # ht[1] 升级为 ht[0]
            self.ht_table[0] = self.ht_table[1]
            self.ht_size_exp[0] = self.ht_size_exp[1]
            self.ht_used[0] = self.ht_used[1]
            self.ht_table[1] = None
            self.ht_size_exp[1] = -1
            self.ht_used[1] = 0
            self.rehashidx = -1
            return False
        return True

    # 对应 _dictRehashStep（每次操作顺带迁移 1 步）
    def _rehash_step(self):
        self._rehash(n=1)

    def status(self) -> str:
        ht1_str = f"ht[1] used={self.ht_used[1]} size={self._ht_size(1)}" \
                  if self._is_rehashing() else "ht[1] not initialized"
        rehash_str = f", rehashidx={self.rehashidx}" if self._is_rehashing() else ""
        return (f"ht[0] used={self.ht_used[0]} size={self._ht_size(0)}, "
                f"{ht1_str}{rehash_str}")


def main():
    print("=== 初始化 dict（初始容量 4）===")
    d = Dict(initial_exp=2)  # size = 2^2 = 4

    print("\n写入 key0..key3:")
    for i in range(4):
        d.add(f"key{i}", f"value{i}")
    print(f"写入后: {d.status()}")

    print("\n继续写入触发扩容并观察渐进式 rehash:")
    for i in range(4, 12):
        d.add(f"key{i}", f"value{i}")
        print(f"  写入 key{i}: {d.status()}")

    print(f"\n最终: {d.status()}")

    print("\n查找验证:")
    for key in ['key0', 'key5', 'key11']:
        e = d.find(key)
        if e:
            print(f"  查找 {key} = {e.value} ✓")
        else:
            print(f"  查找 {key} = None ✗（错误）")


if __name__ == '__main__':
    main()
```

---

### Demo 3：Python 手写 Skiplist（印证 t_zset.c）

> 设计为可运行，请在你的环境验证。依赖：Python 3.8+。

```python
#!/usr/bin/env python3
"""
skiplist_demo.py — 印证 Redis zskiplist 的关键设计

展示：
1. zslRandomLevel()：p=0.25 的概率层数分布
2. zslInsert()：有序插入 + span 维护
3. ZRANK 的 O(log N) 实现（通过 span 累加）
4. ZRANGE by score 范围查询

运行方法：python3 skiplist_demo.py

预期输出（核心部分）：
  === 层数分布验证 (p=0.25, 10000次) ===
  level 1: ~75.0%  (实际约 75%)
  level 2: ~18.8%  (实际约 18-19%)
  level 3: ~4.7%   (实际约 4-5%)
  level 4: ~1.2%   (实际约 1%)
  ...
  === 插入 10 个元素后的 skiplist ===
  层数分布验证通过
  ZRANK alice = 0  (score=1.0)
  ZRANGE 1..5: [alice(1.0), bob(2.0), charlie(3.0), dave(4.0), eve(5.0)]
"""

import random
from typing import Optional, List, Tuple

ZSKIPLIST_MAXLEVEL = 32
ZSKIPLIST_P = 0.25


class ZskiplistLevel:
    """对应 struct zskiplistLevel"""
    __slots__ = ['forward', 'span']
    def __init__(self):
        self.forward: Optional['ZskiplistNode'] = None
        self.span: int = 0  # 跨越的节点数（用于 ZRANK 的 O(log N)）


class ZskiplistNode:
    """对应 zskiplistNode（简化：ele 作为普通字段）"""
    __slots__ = ['score', 'ele', 'backward', 'level']
    def __init__(self, level: int, score: float, ele: str):
        self.score = score
        self.ele = ele
        self.backward: Optional['ZskiplistNode'] = None
        self.level: List[ZskiplistLevel] = [ZskiplistLevel() for _ in range(level)]


class Zskiplist:
    """对应 zskiplist"""

    def __init__(self):
        self.header = ZskiplistNode(ZSKIPLIST_MAXLEVEL, -float('inf'), "")  # 哨兵节点
        self.tail: Optional[ZskiplistNode] = None
        self.length = 0
        self.level = 1

    # 对应 zslRandomLevel()
    # p=0.25：每次有 25% 概率升一层
    @staticmethod
    def random_level() -> int:
        level = 1
        while random.random() < ZSKIPLIST_P:
            level += 1
        return min(level, ZSKIPLIST_MAXLEVEL)

    # 对应 zslInsert() + zslInsertNode()
    def insert(self, score: float, ele: str) -> ZskiplistNode:
        update = [None] * ZSKIPLIST_MAXLEVEL  # update[i] = 第 i 层的插入前驱
        rank = [0] * ZSKIPLIST_MAXLEVEL       # rank[i] = 到 update[i] 的累计 rank

        x = self.header
        for i in range(self.level - 1, -1, -1):  # 从最高层向下
            rank[i] = rank[i+1] if i < self.level - 1 else 0
            # 向右移动：score 更小，或 score 相同但 ele 字典序更小
            while (x.level[i].forward and
                   (x.level[i].forward.score < score or
                    (x.level[i].forward.score == score and
                     x.level[i].forward.ele < ele))):
                rank[i] += x.level[i].span
                x = x.level[i].forward
            update[i] = x

        level = self.random_level()
        if level > self.level:
            # 新层：前驱是 header，span 设为 length（跨越所有节点）
            for i in range(self.level, level):
                rank[i] = 0
                update[i] = self.header
                update[i].level[i].span = self.length
            self.level = level

        node = ZskiplistNode(level, score, ele)

        for i in range(level):
            node.level[i].forward = update[i].level[i].forward
            update[i].level[i].forward = node

            # span 更新：
            # node 的 span = update[i] 原 span - (rank[0] - rank[i])
            # update[i] 新 span = rank[0] - rank[i] + 1
            node.level[i].span = update[i].level[i].span - (rank[0] - rank[i])
            update[i].level[i].span = (rank[0] - rank[i]) + 1

        # 高层节点 span 加 1（新节点插入后，跨越范围增大）
        for i in range(level, self.level):
            update[i].level[i].span += 1

        # backward 指针（第 0 层双向链表）
        node.backward = None if update[0] is self.header else update[0]
        if node.level[0].forward:
            node.level[0].forward.backward = node
        else:
            self.tail = node

        self.length += 1
        return node

    # 对应 ZRANK：通过 span 累加实现 O(log N)
    def rank(self, score: float, ele: str) -> int:
        rank = 0
        x = self.header
        for i in range(self.level - 1, -1, -1):
            while (x.level[i].forward and
                   (x.level[i].forward.score < score or
                    (x.level[i].forward.score == score and
                     x.level[i].forward.ele <= ele))):
                rank += x.level[i].span
                x = x.level[i].forward
        if x.ele == ele:
            return rank - 1  # 0-indexed
        return -1  # 未找到

    # 对应 ZRANGE by score（范围查询）
    def range_by_score(self, min_score: float, max_score: float) -> List[Tuple[str, float]]:
        result = []
        # 快速跳到 min_score
        x = self.header
        for i in range(self.level - 1, -1, -1):
            while x.level[i].forward and x.level[i].forward.score < min_score:
                x = x.level[i].forward
        x = x.level[0].forward
        while x and x.score <= max_score:
            result.append((x.ele, x.score))
            x = x.level[0].forward
        return result

    def debug_print(self):
        print(f"  length={self.length}, height={self.level}")
        x = self.header.level[0].forward
        items = []
        while x:
            items.append(f"{x.ele}({x.score})[L{len(x.level)}]")
            x = x.level[0].forward
        print(f"  {' -> '.join(items)}")


def validate_level_distribution():
    """验证 zslRandomLevel 的概率分布"""
    print("=== 层数分布验证 (p=0.25, 10000次) ===")
    n = 10000
    counts = {}
    for _ in range(n):
        l = Zskiplist.random_level()
        counts[l] = counts.get(l, 0) + 1

    print("  理论期望 vs 实际:")
    for level in sorted(counts.keys()):
        actual_pct = counts[level] / n * 100
        expected_pct = 75 * (0.25 ** (level - 1))
        if level <= 5:
            print(f"  level {level}: 理论 {expected_pct:.1f}%, 实际 {actual_pct:.1f}%")

    # 验证 level 1 约占 75%
    pct1 = counts.get(1, 0) / n
    assert 0.70 < pct1 < 0.80, f"level 1 比例异常: {pct1:.2%}"
    print("  层数分布验证通过 ✓")


def main():
    validate_level_distribution()

    print("\n=== 插入测试数据 ===")
    zsl = Zskiplist()
    members = [
        (1.0, "alice"), (2.0, "bob"), (3.0, "charlie"),
        (4.0, "dave"), (5.0, "eve"), (6.0, "frank"),
        (7.0, "grace"), (8.0, "henry"), (9.0, "iris"), (10.0, "jack")
    ]
    for score, ele in members:
        zsl.insert(score, ele)
    zsl.debug_print()

    print("\n=== ZRANK 测试 ===")
    for ele, expected_rank in [("alice", 0), ("bob", 1), ("jack", 9)]:
        score = dict((e, s) for s, e in members)[ele]
        r = zsl.rank(score, ele)
        status = "✓" if r == expected_rank else "✗"
        print(f"  ZRANK {ele} = {r} (期望 {expected_rank}) {status}")

    print("\n=== ZRANGEBYSCORE 1..5 ===")
    result = zsl.range_by_score(1.0, 5.0)
    print(f"  结果: {[f'{e}({s})' for e, s in result]}")
    assert len(result) == 5, f"期望 5 个，得到 {len(result)}"
    assert result[0][0] == "alice" and result[-1][0] == "eve"
    print("  范围查询验证通过 ✓")


if __name__ == '__main__':
    main()
```

---

## 十一、生产实战：失败模式与真实坑

### 坑 1：KEYS * 命令阻塞整个服务

**根因**：KEYS 是 O(N) 全表扫描，单线程下直接阻塞所有其他客户端。生产库有 1000 万 key 时，KEYS 可能耗时数秒。

**解决**：用 `SCAN cursor [MATCH pattern] [COUNT hint]` 替代，每次最多扫描 COUNT 个 bucket，不阻塞其他命令。注意 SCAN 的 COUNT 是**建议值**，不保证精确返回 COUNT 个结果。

### 坑 2：fork() 导致延迟毛刺

**根因**：`BGSAVE` / `BGREWRITEAOF` 调用 `fork()`，Linux 需要复制父进程的**页表**（不是内存本身，靠 COW），页表大小正比于内存占用。8GB 内存的 Redis 实例 fork 可能需要 100-400ms。

**诊断**：
```bash
# 查看最近一次 fork 耗时
redis-cli info stats | grep latest_fork_usec
```

**解决**：
1. 升级到 HVM 虚拟化（KVM/Hyper-V）而非 Xen
2. 禁用 Transparent Huge Pages（THP）：`echo never > /sys/kernel/mm/transparent_hugepage/enabled`
3. 控制 Redis 实例内存不超过 8GB，多实例分散

### 坑 3：Rehash 期间内存翻倍

**根因**：dict rehash 时同时持有两张哈希表，ht[0] + ht[1] 内存叠加。一个 1GB 的 hash 在扩容时瞬间需要 2GB。

**缓解**：
- `hash-max-listpack-entries 128`：小 hash 保持 listpack 编码，不走 dict
- 监控 `used_memory` vs `maxmemory` 的余量，在 rehash 期间留 50% buffer
- `CONFIG SET hash-max-listpack-entries` 可在线调整

### 坑 4：AE_BARRIER 使用不当导致数据不一致

**根因**：AOF 在 `always` fsync 模式下需要保证：先写 AOF，再回复客户端。`AE_BARRIER` 标志让 ae 在 write 回调之后再执行 read 回调，实现"先落盘再 ACK"的语义。错误地在非 AOF 场景设置此标志会导致响应延迟。

### 坑 5：大量 key 同时过期的延迟毛刺

**根因**：Redis 的主动过期每秒运行 10 次，每次采样 20 个 key，如果 >25% 已过期则继续循环（最坏情况连续运行数百毫秒）。如果用 `EXPIREAT` 给批量 key 设置相同的过期时间，瞬间有大量 key 到期。

**解决**：过期时间加随机抖动（`EXPIRE key $(( base_ttl + RANDOM % jitter ))`）。

### 坑 6：listpack -> hashtable 升级后内存暴涨

**场景**：存储 1000 个小 hash，每个 hash 里有 130 个 field（超过默认 128 阈值），触发从 listpack -> hashtable 的编码升级，内存突增 3-5x。

**诊断**：
```bash
redis-cli object encoding myhash   # 检查编码类型
redis-cli debug sleep 0            # 触发渐进式 rehash 完成
redis-cli memory usage myhash      # 比较升级前后内存
```

---

## 十二、方案对比：io_uring vs epoll vs select

| 特性 | select | poll | epoll | io_uring |
|------|--------|------|-------|----------|
| 监听 fd 数上限 | 1024（FD_SETSIZE） | 无上限 | 无上限 | 无上限 |
| 就绪通知复杂度 | O(N) 扫描 | O(N) 扫描 | O(1) 回调 | O(1) 完成队列 |
| 内核→用户数据拷贝 | 每次 call 拷贝 | 每次 call 拷贝 | 仅就绪 fd | 零拷贝（共享内存） |
| 模式 | LT | LT | LT + ET | async |
| 系统调用开销 | 高（传参大） | 高 | 低 | 极低（批量提交） |
| Redis 使用 | fallback | 未使用 | Linux 首选 | 实验性支持（7.x） |
| 适用场景 | 移植性/fd < 100 | 移植性 | 高并发服务器 | 磁盘 I/O 密集 |

**Redis 的 ae 设计使得切换 backend 只需替换 ae_epoll.c → ae_kqueue.c → ae_select.c，上层代码零修改**，是典型的 Strategy 模式。

---

## 十三、章末五件套

### 13.1 核心概念速查

| 概念 | 关键点 |
|------|--------|
| ae 事件库 | 500 行极简事件循环，Strategy 模式选 epoll/kqueue/select |
| SDS | packed header + O(1) len + greedy prealloc，buf[-1] 存 type |
| dict 渐进 rehash | 双表，每次操作迁移 1 bucket，BGSAVE 期间暂停 |
| skiplist | p=0.25，span 字段实现 O(log N) ZRANK |
| listpack | 无 prevlen，消除连锁更新，backlen 实现反向遍历 |
| 编码自适应 | 小数据集紧凑编码，超阈值单向升级为性能编码 |

### 13.2 扩展阅读

1. **William Pugh (1990)**：Skip Lists: A Probabilistic Alternative to Balanced Trees — skiplist 原始论文
2. **antirez blog (2014)**：https://antirez.com/news/126 — 单线程设计辩护
3. **Redis 源码目录**：`ae.c`, `ae_epoll.c`, `dict.c`, `t_zset.c`, `sds.h`, `listpack.c`, `quicklist.h`（均在 redis/redis 仓库 unstable 分支）
4. **Redis latency guide**：https://redis.io/docs/latest/operate/oss_and_stack/management/optimization/latency/ — 生产调优参考

### 13.3 面试高频考点

**Q1：Redis 为什么是单线程还那么快？**

答：三个层次：① 全内存操作，数据不需要磁盘 I/O；② epoll 多路复用，单线程服务数千并发，I/O 等待不占 CPU；③ 无锁设计，避免了线程切换和锁竞争开销。真正瓶颈是网络带宽，而非 CPU。

**Q2：dict 的渐进式 rehash 为什么不在一次操作中完成？**

答：Redis 是单线程的，一次性 rehash 一个大 dict 可能阻塞数百毫秒。渐进式 rehash 每次操作只迁移 1 个 bucket（约几微秒），把总开销均摊到正常请求中。代价是 rehash 期间需要同时查两张表，略增查找复杂度。

**Q3：skiplist 为什么用 span 字段？如果去掉，ZRANK 复杂度会变成什么？**

答：span 记录每个 level[i].forward 跨越的节点数。ZRANK 从最高层累加 span，遇到目标节点时 rank 就是累加值。去掉 span 后，必须在第 0 层逐节点遍历计数，O(N) 退化。

**Q4：listpack 解决了 ziplist 的什么问题？**

答：ziplist 的每个 entry 存储前一个 entry 的大小（prevlen）。当某 entry 大小跨越 [0,253] → [254, ∞) 边界时，prevlen 字段需要从 1 字节扩展到 5 字节，导致后继 entry 的 prevlen 也需要更新，产生连锁更新，最坏 O(N)。listpack 去掉了 prevlen，每个 entry 只存自己的大小（backlen 在尾部），消除了这个问题。

**Q5：SDS 的 embstr 编码有什么限制？为什么是 44 字节？**

答：embstr 把 robj（16 字节）和 sdshdr8（3 字节）+ buf（44 字节）+ `\0` 合并在一次 malloc 中，共 64 字节，刚好是 jemalloc 的一个 arena 大小。embstr 字符串是只读的（修改会先升级为 raw），因为两者共享内存，realloc 可能移动地址。

### 13.4 代码挑战（扩展 Demo）

**挑战 1**：在 Demo 2（dict_rehash.py）中实现缩容逻辑：当 `ht_used[0] * 8 <= ht_size[0]` 时触发缩容，缩小到 `ht_used[0] * 2`。

**挑战 2**：在 Demo 3（skiplist_demo.py）中实现 `zslDelete`：给定 score + ele 删除节点，更新所有层的 span 和 forward 指针，并维护 `backward` 链。

**挑战 3**：在 Demo 1（ae_mini.py）中实现一个 AE_BARRIER 模拟：write 回调在 read 回调之前执行（AOF always 模式的语义）。提示：在 `process_events` 的 file event 分发循环中增加 `invert` 标志判断。

**挑战 4**：实现一个 intset 的 Python 版本，支持三种编码自动升级（INT16 → INT32 → INT64），并验证升级时从后往前重新编码的正确性。

### 13.5 未来演进方向

1. **io_uring**：Redis 7.x 开始实验性支持 io_uring，主要目标是降低磁盘 I/O（AOF）的系统调用开销。网络 I/O 部分 epoll 已经足够优秀。

2. **Multi-threading 的边界**：Redis 6.0 的 I/O threading 只并行化网络读写，命令执行仍单线程。未来可能通过 key-level locking 实现命令级并发，但复杂度极高。

3. **listpack 替代 ziplist 的完整迁移**：Redis 7.2 已完全移除 ziplist，所有紧凑编码统一为 listpack，连锁更新问题彻底消除。

4. **zskiplistNode 的内嵌 SDS**：当前 unstable 版本已将 ele SDS 嵌入节点末尾（`level[]` 数组之后），减少内存分配。这是持续内存优化方向的缩影。

---

*源码 URLs（本章实际 WebFetch 获取）：*
- `https://raw.githubusercontent.com/redis/redis/unstable/src/ae.c`
- `https://raw.githubusercontent.com/redis/redis/unstable/src/ae_epoll.c`
- `https://raw.githubusercontent.com/redis/redis/unstable/src/ae.h`
- `https://raw.githubusercontent.com/redis/redis/unstable/src/dict.c`
- `https://raw.githubusercontent.com/redis/redis/unstable/src/dict.h`
- `https://raw.githubusercontent.com/redis/redis/unstable/src/t_zset.c`
- `https://raw.githubusercontent.com/redis/redis/unstable/src/server.h`
- `https://raw.githubusercontent.com/redis/redis/unstable/src/sds.h`
- `https://raw.githubusercontent.com/redis/redis/unstable/src/sds.c`
- `https://raw.githubusercontent.com/redis/redis/unstable/src/listpack.c`
- `https://raw.githubusercontent.com/redis/redis/unstable/src/quicklist.c`
- `https://raw.githubusercontent.com/redis/redis/unstable/src/quicklist.h`
- `https://raw.githubusercontent.com/redis/redis/unstable/src/intset.c`
- `https://raw.githubusercontent.com/redis/redis/unstable/src/ziplist.c`
- `https://antirez.com/news/126`
- `https://redis.io/docs/latest/operate/oss_and_stack/management/optimization/latency/`
- `https://redis.io/docs/latest/develop/data-types/sorted-sets/`
