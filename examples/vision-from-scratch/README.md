# vision-from-scratch

《计算机视觉从零：手写卷积神经网络》配套代码。

从零用 TypeScript 拆解卷积神经网络：从 `conv2d` 一路搭到一个能在 CPU 上几秒内跑完的 tiny-CNN。

- **纯算法，零 ML 框架**：所有算子（卷积/池化/BatchNorm/残差）都手写，反向传播由本仓自带的 autograd 引擎驱动。
- **离线、确定性**：不下载任何数据集、不联网、不需要 API key。所有随机性都走种子 PRNG（`core/rng.ts`），同一 seed 跑出 bit-for-bit 一致的结果。
- **诚实数字**：训练曲线、grad-check 误差、加速比都是代码真算/真测出来的。toy 合成图像上的绝对准确率偏乐观——可迁移的是机制和趋势（loss 下降、残差比非残差收敛快、BatchNorm 稳定训练、数据增强缩小 train/test gap），不是绝对值。

## 运行

每个 stage 是一个独立可跑的脚本（加载即跑 `main()`）。需要先安装依赖：

```bash
npm install
```

然后按章节运行：

```bash
npm run stage01   # 卷积：im2col 把卷积摊成 matmul，可视化卷积核与 feature map
npm run stage02   # 池化：maxpool / avgpool 前向 + 梯度路由
npm run stage03   # 卷积反向：用 gradCheck 验证 conv/pool 的解析梯度
npm run stage04   # 感受野：随深度/步长增长，ASCII 可视化
npm run stage05   # BatchNorm：训练 batch 统计 vs 推理 running 统计
npm run stage06   # 残差：residual vs plain 的收敛差距
npm run stage07   # tiny-CNN：在合成图像上端到端训练 + 混淆矩阵
npm run stage08   # 数据增强：随机平移/翻转对 train/test gap 的影响
```

类型检查：

```bash
npm run typecheck
```

## 结构

```
src/
  core/            # 全书共享底座（一次写好，stage 只调用不改）
    autograd.ts    # Tensor + 反向传播；conv2d(im2col)/maxpool2d/avgpool2d/flatten
    nn.ts          # Module 基类 + Conv2d/Pool/BatchNorm2d/Linear/ReLU/Sequential/ResidualBlock
    optim.ts       # SGD(momentum)/Adam/AdamW + 梯度裁剪 + cosine warmup
    rng.ts         # 种子 PRNG (mulberry32) + 正态/均匀采样 + kaiming/xavier 初始化
    data.ts        # 合成图像生成器（几何形状/笔画）+ batch 迭代器 + 增强变换
    metrics.ts     # 损失/准确率/混淆矩阵/gradCheck/计时 + ASCII 图（loss/直方图/热图/感受野）
  stage01-..08-*.ts
```

`core/` 是冻结契约：每个 stage 只从 `core` 组装，不重造算子。这样全书的"诚实数字"建立在同一套经过 `gradCheck` 验证的底座上。
