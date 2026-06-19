---
title: "Capstone:从零搭一个视觉模型并训练"
slug: "5-16"
collection: "ai-research-compass"
group: "计算机视觉专家课程"
order: 5016
summary: "这一章把你从\"逐章学过卷积、ResNet、BN、ViT、自监督、扩散\"带到\"能独立把这些组件拼成一个真正跑得起来、收敛得对、调得动、看得懂的视觉项目\"。"
topics:
  - "AI 研究"
tags: []
createdAt: "2026-06-19T05:58:31.000Z"
updatedAt: "2026-06-19T05:58:31.000Z"
---
> 这一章把你从"逐章学过卷积、ResNet、BN、ViT、自监督、扩散"带到"能独立把这些组件拼成一个真正跑得起来、收敛得对、调得动、看得懂的视觉项目"。前面每章都在拆解单个零件;这一章是装配线——你将亲手走完"先 overfit 一个 batch 验证实现 → 上数据增强 → 选迁移学习还是从零训 → 诊断过拟合/欠拟合 → 用 Grad-CAM/注意力看模型到底在看哪"的完整闭环,并拿到一份可复用的训练报告模板。读完你应当能对任意一个新视觉任务,自己搭出一条不踩坑的训练管线。

## 一句话引言

> 一个能跑的视觉项目 = **正确的实现** + **对的优化配方** + **诚实的诊断**。三者缺一,模型要么训不动、要么过拟合、要么你根本不知道它学到了什么。本章用一条分类主线(从零小 ResNet / 微调 timm ViT)走通这三件事,再给一条最小扩散副线,把生成也串起来。

## 1. 动机:为什么"会每个零件"不等于"会做项目"

你已经能手推 BN 的反向、能算 ViT 的注意力复杂度、能从 ELBO 推出扩散的 ε-预测损失。但如果现在丢给你一个 CIFAR-10 和一张 GPU,让你"训出一个 90%+ 的分类器",大概率第一次会卡在这些地方:

- loss 不降,或者降到一半 NaN 了——**你分不清是实现 bug、学习率太大、还是数据没归一化**;
- 训练 acc 99%、测试 acc 70%——**你知道这叫过拟合,但不知道下一步该加增强、加正则、还是减模型**;
- 换了个增强策略,acc 掉了 3 个点——**你说不清是增强太强、还是 epoch 不够让它"吃下"增强**;
- 模型说这是猫,但你不知道它是看了猫脸还是看了背景的草地——**一旦分布偏移就会翻车,而你毫无预警**。

**做项目的核心技能不是写模型(那是前几章的事),而是建立一套"出了问题能定位、定位了能修"的工作流。** 工程界把这套工作流总结成几条铁律,本章逐条落地:

1. **先 overfit 一个 batch**——在堆任何技巧之前,用最小代价证明"前向、反向、loss、优化器"这条链路是通的;
2. **一次只改一个变量(消融)**——增强、正则、学习率、架构,分开验证,否则收益和损失互相抵消你永远看不清;
3. **能迁移就别从零训**——除非数据量足够大(回扣第 07 章那条"随数据量翻转的曲线"),迁移学习几乎总是更快更好;
4. **训练完必须可视化**——Grad-CAM / 注意力图是模型的"X 光片",不看就上线等于闭眼开车。

预备:你需要会用 PyTorch 写 `Dataset`/`DataLoader`、`nn.Module`、`optimizer.step()` 这一套基本循环。本章不重复教这些样板,只写**能体现机制、决定成败**的核心片段。环境假设:PyTorch ≥ 2.0、一张能放下 batch=128 小模型的 GPU(显存 ≥ 6GB 足够 CIFAR 规模)。

---

## 2. 主线选择:分类(推荐)vs 扩散

两条主线都能"把组件串起来",但对一个**第一次完整做项目**的读者,推荐先走**图像分类**,理由是工程化的:

| 维度 | ① 图像分类 | ② 最小扩散生成 |
|---|---|---|
| 评估信号 | top-1 准确率,**单一标量、立刻可读** | 生成质量(FID/肉眼),**慢、主观、易自欺** |
| 调试难度 | overfit 一个 batch 几秒见效 | 要采样几十步才看到一张图,反馈慢 |
| 回扣章节 | 01 卷积 / 03 ResNet+BN / 07 ViT / 08 自监督预训练 | 11 扩散 / 03 BN(U-Net 里) |
| 失败模式 | 直观(acc 曲线) | 隐蔽(loss 降了但样本糊) |
| 迁移学习 | 成熟生态(timm 几百个预训练权重) | 小数据从零训即可见效 |

**结论:本章正文以分类为主线讲透完整工作流(§3-§8),扩散作为副线给一份能跑的最小骨架(§9)。** 工作流的方法论(overfit 一个 batch、消融、诊断)对两条线通用——你在分类上练熟的诊断习惯,直接迁移到扩散和任何后续项目。

---

## 3. 第一步永远是 overfit 一个 batch:用最小代价证明实现是对的

### 问题:怎么在不浪费几小时训练的前提下,确认"我这套代码根本没写错"

新手最常见的浪费:写完模型直接开训,跑了两小时发现 loss 不降,然后开始盲目调学习率、换优化器、改架构——**在实现可能有 bug 的前提下调超参,纯属浪费**。

**正确的第一步:拿一个 batch(比如 16 张图),反复在它上面训练,直到 loss → 0、训练 acc → 100%。** 这件事如果做不到,说明实现链路有 bug,任何超参都救不了。

### 机制:为什么 overfit 一个 batch 能定位绝大多数 bug

一个能正确执行"前向 → 算 loss → 反向 → 更新"的网络,**一定有能力记住 16 个样本**(参数量远大于 16 个样本的信息量,这是纯记忆,不需要泛化)。如果它连记都记不住,问题必然出在这条链路的某个环节,且可以二分定位:

```
overfit 一个 batch 失败 → 按这个顺序排查:
1. loss 一开始就是 nan/inf      → 数据没归一化 / 学习率爆炸 / log(0)
2. loss 纹丝不动(完全水平)     → 梯度没回传:忘了 loss.backward() /
                                   optimizer.zero_grad() 漏了 /
                                   requires_grad=False / 输入和标签接反
3. loss 缓慢降但降不到 0         → 学习率太小 / 模型容量真的不够(罕见)
4. loss 降到 0 但 acc 不是 100%  → acc 计算写错(argmax 维度错)
5. 初始 loss 远偏离 ln(num_classes) → 输出层/softmax 接错
```

**关键数值检查——初始 loss 应该约等于 ln(类别数)。** 这是个能心算的健全性检查(sanity check):训练刚开始时网络输出接近均匀分布,交叉熵损失 = E[−ln pᵧ] ≈ −ln(1/C) = ln C。CIFAR-10 是 ln 10 ≈ **2.30**;ImageNet-1k 是 ln 1000 ≈ **6.91**。如果你的初始 loss 是 2.30 附近,说明输出层、softmax、标签对齐都没问题;如果是 0.5 或者 50,立刻知道哪里接错了,**不用等训练**。

### 代码:overfit 一个 batch 的标准模板

```python
import torch, torch.nn as nn, math

def overfit_one_batch(model, x, y, device, steps=200, lr=1e-3):
    """x:[B,3,H,W]  y:[B]。目标:loss→0, acc→100%。跑不到就是实现有 bug。"""
    model.to(device).train()
    x, y = x.to(device), y.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    # 健全性检查 1:初始 loss ≈ ln(num_classes)
    with torch.no_grad():
        init_loss = crit(model(x), y).item()
    C = model.num_classes
    print(f"init loss={init_loss:.3f}  expect≈ln({C})={math.log(C):.3f}")

    for it in range(steps):
        opt.zero_grad()                 # 漏这句 → loss 纹丝不动(梯度累积错)
        logits = model(x)
        loss = crit(logits, y)
        loss.backward()                 # 漏这句 → 梯度全 0
        opt.step()
        if it % 20 == 0 or it == steps - 1:
            acc = (logits.argmax(1) == y).float().mean().item()
            print(f"it={it:3d}  loss={loss.item():.4f}  acc={acc:.3f}")
    # 健全性检查 2:结束时 loss 应接近 0,acc 应=1.0
```

**做项目时第一件事就是跑这个函数。** 它花你 10 秒,能省你 2 小时的瞎调。**注意:此时绝对不要开数据增强**(增强会让"完美记忆 16 张图"变难甚至不可能,干扰这个纯净的链路检查)。

### 分析:这一步还能顺带验证什么

- **梯度量级**:在循环里打印 `sum(p.grad.norm()**2 for p in model.parameters())**0.5`,正常应在 0.1~10 量级。若 >100 持续不降,学习率太大或缺归一化;若 <1e-4,梯度消失(回扣第 03 章:这正是 ResNet 残差连接要解决的问题——深网络从零训若不用残差,这一步就会看到梯度消失)。
- **数据管线**:把 `x` 反归一化后 `plt.imshow` 出来,确认图没被你的 transform 搞坏(通道顺序错、归一化参数错都会在这一眼看出来)。

---

## 4. 数据增强:它到底在做什么,以及为什么"先 overfit 再加增强"

### 问题:增强是"免费午餐"吗?加了一定涨点吗?

不是。增强是**用训练难度换泛化能力**:它人为扩大训练分布,逼模型学到对扰动不变的特征,但代价是**训练更难、需要更多 epoch 才能收敛**。理解这个 trade-off,才能解释"为什么加了强增强反而掉点"——往往是 epoch 不够,模型还没"吃下"增强。

### 机制:三类增强的数学本质

**(1) 几何/光度增强(RandAugment 一类)——保持标签不变的图像变换。**
数学上,设原样本 (x, y),增强是一族变换 {Tₖ},满足语义不变:label(Tₖ(x)) = y。模型在 {Tₖ(x)} 上都要预测 y,等价于在损失里加了一个**不变性约束**:让 f(Tₖ(x)) 对 k 尽量不敏感。RandAugment 的设计精髓是把"选哪些变换、各自多强"压成两个超参 (N, M):每张图随机选 N 个操作、每个强度 M,**省掉了 AutoAugment 那种昂贵的策略搜索**。

**(2) Mixup——在样本对之间做凸组合。**
取两个样本 (xᵢ, yᵢ)、(xⱼ, yⱼ),λ ~ Beta(α, α):

```
x̃ = λ·xᵢ + (1−λ)·xⱼ          # 像素线性插值
ỹ = λ·yᵢ + (1−λ)·yⱼ          # one-hot 标签也线性插值(软标签)
loss = λ·CE(f(x̃), yᵢ) + (1−λ)·CE(f(x̃), yⱼ)   # 等价于对软标签算 CE
```

为什么有效?它强制模型在**样本之间的线性路径上**输出也线性变化,即鼓励决策边界平滑、惩罚过度自信。从正则角度,它约束了模型的局部线性行为,等价于一种数据依赖的、比 label smoothing 更强的正则。α 越大,λ 越接近 0.5,混合越激进。

**(3) CutMix——把一块区域整体粘贴过去。**
从 xⱼ 抠一个矩形框 B 贴到 xᵢ 上,标签按面积比例混合:

```
x̃ = (1−M_B)⊙xᵢ + M_B⊙xⱼ      # M_B 是框 B 的 0/1 掩码
λ = 1 − area(B)/area(image)   # 保留 xᵢ 的面积占比
ỹ = λ·yᵢ + (1−λ)·yⱼ
```

相比 Mixup 的全图半透明叠加(产生不自然的"鬼影"),CutMix 保留了**局部图像的真实统计**,同时逼模型不能只盯一块区域(那块可能被替换了),从而关注整个目标——这对定位也有帮助。

### 代码:Mixup / CutMix 的核心实现

```python
import numpy as np, torch

def mixup_cutmix(x, y, num_classes, alpha=0.2, cutmix_p=0.5):
    """返回混合后的 x 和软标签 y_soft:[B,C]。在 collate 后、喂模型前调用。"""
    B = x.size(0)
    perm = torch.randperm(B, device=x.device)        # 把 batch 自身打乱配对
    lam = np.random.beta(alpha, alpha)
    y1 = torch.nn.functional.one_hot(y, num_classes).float()
    y2 = y1[perm]

    if np.random.rand() < cutmix_p:                  # CutMix
        H, W = x.shape[-2:]
        r = np.sqrt(1.0 - lam)
        cut_h, cut_w = int(H * r), int(W * r)
        cy, cx = np.random.randint(H), np.random.randint(W)
        y0, y0e = np.clip([cy - cut_h//2, cy + cut_h//2], 0, H)
        x0, x0e = np.clip([cx - cut_w//2, cx + cut_w//2], 0, W)
        x[:, :, y0:y0e, x0:x0e] = x[perm, :, y0:y0e, x0:x0e]
        lam = 1 - (y0e - y0) * (x0e - x0) / (H * W)   # 用真实面积修正 lam
    else:                                            # Mixup
        x = lam * x + (1 - lam) * x[perm]
    y_soft = lam * y1 + (1 - lam) * y2
    return x, y_soft
# 训练时:loss = -(y_soft * log_softmax(logits, dim=1)).sum(1).mean()
```

**坑:用了软标签后,损失必须用"软标签交叉熵"**(`-(y_soft * log_softmax).sum(1).mean()`),不能再用 `CrossEntropyLoss(logits, y)`(后者只接受整数硬标签)。这是新手最常踩的——混合了图却忘了改 loss,标签还是硬的,增强等于白做。

### 分析:增强对训练曲线的影响(怎么读出"增强生效了")

- **训练 acc 会降低、收敛变慢**——这是**正常且预期**的。强增强下训练 acc 可能只有 80% 多(因为每张图都被改过,记不住),但**测试 acc 反而更高**。如果你看到"加增强后训练 acc 掉、测试 acc 也掉",大概率是 epoch 不够,延长训练通常恢复。
- **train-test gap 收窄**是增强生效的最直接信号。无增强时 gap 可能 30 个点(过拟合),加 RandAug+Mixup 后 gap 收到 5 个点以内。
- **何时别用强增强**:迁移学习只微调几个 epoch 时,Mixup/CutMix 收益小甚至有害(模型还没来得及适应混合分布)。小数据微调优先用轻量几何增强 + 适度 RandAug。

---

## 5. 主线 A:从零搭一个小 ResNet(回扣 01/03 章)

### 问题:CIFAR-10 这种 32×32 小图,该怎么搭网络?

直接套 ImageNet 版 ResNet-50 是**错的**:它第一层是 7×7 stride-2 卷积 + maxpool,把 224×224 一上来降到 56×56。对 32×32 的图这么干,几层下来空间维度就没了(32→16→8→4→2→1)。**CIFAR 版 ResNet 要改头**:第一层用 3×3 stride-1、去掉开头的 maxpool,保住空间分辨率。这是个真实的、新手必栽的坑——架构要匹配输入尺寸。

### 机制:残差块 + BN 的最小实现(回扣 03 章)

回扣第 03 章:残差块把"连乘"改成"连加"给梯度修高速路,BN 把每个通道的尺度钉死。CIFAR-ResNet 用 BasicBlock(两个 3×3 卷积 + 残差),不用 bottleneck(小图没必要省那点算力)。

```python
import torch.nn as nn, torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)  # bias=False:后面有BN
        self.bn1   = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(c_out)
        # shortcut:维度不匹配时用 1×1 卷积对齐(回扣 03 章 shortcut 对齐)
        self.short = nn.Sequential()
        if stride != 1 or c_in != c_out:
            self.short = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride, bias=False),
                nn.BatchNorm2d(c_out))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))           # 注意:残差相加前不加 relu
        out = out + self.short(x)                 # 连加,不是连乘
        return F.relu(out)                        # 相加后才 relu

class CifarResNet(nn.Module):
    """ResNet-20 风格:3 个 stage,每 stage n 个 block,通道 16→32→64。"""
    def __init__(self, n=3, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        self.stem = nn.Sequential(                # CIFAR 专用 stem:3×3 s1,无 maxpool
            nn.Conv2d(3, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True))
        self.layer1 = self._make(16, 16, n, 1)    # 32×32
        self.layer2 = self._make(16, 32, n, 2)    # 16×16
        self.layer3 = self._make(32, 64, n, 2)    # 8×8
        self.pool = nn.AdaptiveAvgPool2d(1)        # 全局平均池化
        self.fc = nn.Linear(64, num_classes)

    def _make(self, c_in, c_out, n, stride):
        layers = [BasicBlock(c_in, c_out, stride)]
        layers += [BasicBlock(c_out, c_out, 1) for _ in range(n - 1)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer3(self.layer2(self.layer1(x)))
        x = self.pool(x).flatten(1)
        return self.fc(x)
```

### 分析:参数量、FLOPs、感受野(亲手算一遍)

**参数量(以 n=3,即 ResNet-20 为例)。** 主要在卷积。一个 3×3 卷积参数 = c_out · (c_in · 3 · 3)(无 bias)。逐 stage 估算 conv 参数:

```
stem:        16·(3·9)          = 432
layer1(×3):  每块 2 个 16→16    → 3·2·(16·16·9)        = 3·2·2304 = 13,824
layer2:      首块 16→32 + 32→32 + 短路1×1(16→32)
             首块: 16·32·9 + 32·32·9 + 16·32·1 = 4608+9216+512 = 14,336
             另2块: 2·2·(32·32·9) = 2·2·9216 = 36,864
layer3:      首块 32→64 + 64→64 + 短路: 32·64·9+64·64·9+32·64 = 18432+36864+2048=57,344
             另2块: 2·2·(64·64·9) = 147,456
fc:          64·10 + 10(bias)  = 650
BN:          每个 conv 后 2·C,总计约 ~3,000(可忽略量级)
─────────────────────────────────────────
总计 ≈ 0.27 M 参数(ResNet-20 公认约 0.27M,可对账)
```

**FLOPs。** 一个卷积层的乘加 FLOPs = H_out · W_out · c_out · (c_in · k · k)。空间维度大的浅层最贵:layer1 在 32×32 上跑,单个 16→16 的 3×3 卷积 = 32·32·16·(16·9) = 32·32·16·144 ≈ 2.36 MFLOPs。整网约 **40 MFLOPs**(待核,数量级正确;ResNet-20 在 CIFAR 上的常见报告约 41 MFLOPs)。**关键洞察:FLOPs 集中在浅层大特征图,参数集中在深层多通道层**——压缩模型时这两处分别下手。

**感受野(回扣 01 章逐层递推)。** stem 3×3 起始 RF=3,之后每个 3×3 stride-1 卷积 RF += 2·(累积 stride),每次 stride-2 把后续步长翻倍。ResNet-20 末层理论感受野已远超 32(覆盖整图),所以全局平均池化前每个位置都"看过全图"——这是它能在 32×32 上工作的几何前提。

---

## 6. 主线 B:微调 timm 的预训练 ViT(回扣 07/08 章)

### 问题:从零训 vs 迁移学习,什么时候选哪个

回扣第 07 章那条**随数据量翻转的曲线**:ViT 弱归纳偏置,从零训需要海量数据(JFT-300M 级)才能超过 CNN。CIFAR-10 只有 5 万张训练图,**从零训 ViT 必然过拟合、远不如小 ResNet**——这正是第 07 章练习 2 的"曲线最左端陷阱"。

**但迁移学习改变了游戏。** 一个在 ImageNet(或更大数据,经第 08 章自监督预训练如 MAE)上预训练好的 ViT,已经学到了通用视觉特征。微调它到 CIFAR,等于"站在巨人肩膀上"——**迁移学习把 ViT 的弱先验问题转嫁给了预训练阶段的大数据,下游小数据只需适配**。这是实践中绝大多数情况的正确选择。

### 机制:两种迁移策略 + 学习率分层

**(1) 线性探测(linear probing):冻结 backbone,只训分类头。** 把预训练模型当固定特征提取器。回扣第 08 章——这正是评估自监督表征质量的标准协议:特征好不好,看冻结后线性头能达到多高 acc。

**(2) 全量微调(full fine-tuning):所有参数都训,但 backbone 用更小的学习率。** 这是性能上限更高的做法。关键机制是**分层学习率(layer-wise lr / discriminative lr)**:预训练特征已经很好,大学习率会"震碎"它(catastrophic forgetting);新初始化的分类头则需要大学习率快速学。所以 backbone 用小 lr(如 1e-5),head 用大 lr(如 1e-3)。

```python
import timm, torch

def build_and_param_groups(model_name="vit_small_patch16_224",
                           num_classes=10, mode="finetune"):
    # timm 一行加载预训练权重 + 自动替换分类头为 num_classes
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    model.num_classes = num_classes

    if mode == "linear_probe":
        for n, p in model.named_parameters():
            p.requires_grad = "head" in n          # 只训分类头(timm ViT 头叫 head)
        groups = [{"params": [p for p in model.parameters() if p.requires_grad],
                   "lr": 1e-3}]
    else:  # full finetune,backbone 与 head 分层 lr
        head_params, bb_params = [], []
        for n, p in model.named_parameters():
            (head_params if "head" in n else bb_params).append(p)
        groups = [{"params": bb_params,   "lr": 1e-5},   # backbone:小心别震碎预训练
                  {"params": head_params, "lr": 1e-3}]   # head:新初始化,要大 lr
    return model, groups
```

### 坑:输入分辨率与归一化必须匹配预训练

- **分辨率**:`vit_*_patch16_224` 期望 **224×224** 输入。CIFAR 是 32×32,直接喂会因 patch 数不匹配而维度错(或位置编码对不上)。要么把图 resize 到 224(简单但浪费算力),要么用 timm 的位置编码插值(`resample_abs_pos_embed`,回扣第 07 章练习 4 的提示)。
- **归一化均值/方差**:必须用**预训练时的**统计量,不是 CIFAR 自己的。用 `timm.data.resolve_data_config` + `create_transform` 自动取对应模型的 mean/std——**用错归一化是迁移翻车的头号原因**,模型看到的输入分布和预训练时不一致,特征全废。

```python
from timm.data import resolve_data_config, create_transform
cfg = resolve_data_config({}, model=model)   # 自动拿到该模型的 input_size/mean/std
train_tf = create_transform(**cfg, is_training=True,
                            auto_augment="rand-m9-mstd0.5")  # RandAugment 一行接入
```

### 分析:参数量与"训哪些参数"的代价

`vit_small_patch16_224` 约 **22M 参数**(待核,timm 标称约 22.1M)。线性探测只训分类头(384→10 ≈ 3850 个参数),显存和算力都极省、但性能上限低;全量微调训全部 22M,显存翻倍(要存所有梯度和优化器状态),但 acc 更高。**决策规则**:数据 < 1 万张 → 线性探测或冻结大部分;数据几万张以上、算力够 → 全量微调 + 分层 lr。

---

## 7. 训练配方:AdamW + cosine + warmup,以及每个旋钮的作用

### 问题:为什么是 AdamW + cosine,不是 SGD + step decay

这是现代视觉训练(尤其 Transformer 系)的默认配方,每个选择都有理由:

**AdamW = Adam + 解耦权重衰减(decoupled weight decay)。** 回顾普通 Adam 把 L2 正则混进梯度里,经过自适应缩放后,权重衰减的实际强度会被每个参数的梯度二阶矩扭曲——大梯度参数被少衰减,这不是我们想要的。AdamW 把权重衰减**从梯度里拆出来、直接作用在参数上**:

```
普通 Adam(L2):  g ← g + λ·θ,  然后 θ ← θ − lr·Adam_update(g)   # λ 被 1/√v 扭曲
AdamW:           θ ← θ − lr·Adam_update(g) − lr·λ·θ            # 衰减项独立、不被扭曲
```

这个看似微小的改动,在 Transformer 上对泛化影响显著(原论文 Loshchilov & Hutter 2019)。**对 ViT/ConvNext,几乎一律用 AdamW;纯 CNN(ResNet)上 SGD+momentum 仍很有竞争力。**

**Cosine 学习率 + warmup。** 学习率 η 随训练步 t 的调度:

```
warmup(前 t_w 步):   η(t) = η_max · t / t_w           # 线性升温
cosine(之后):        η(t) = η_min + ½(η_max−η_min)(1 + cos(π·(t−t_w)/(T−t_w)))
```

- **warmup 为什么必要**:训练初期参数随机,BN 的 running stats 还没稳、Adam 的二阶矩估计 v 还不准,此时直接上大 lr 容易把模型推向坏区域甚至 NaN。warmup 用几百步线性升温,给统计量"预热"的时间。**Transformer 对 warmup 尤其敏感**(没有 warmup 经常训崩)。
- **cosine 为什么比 step decay 好**:平滑衰减,末期 lr 趋近 0,让模型在损失面上"精修"到一个更平坦的极小值(泛化更好),且没有 step decay 那种"突然掉一档"的不稳定。

### 代码:完整最小训练循环 + 评估

```python
import torch, math
from torch.optim.lr_scheduler import LambdaLR

def make_cosine_warmup(opt, warmup_steps, total_steps, min_ratio=0.0):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)                  # 线性 warmup
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * prog))
    return LambdaLR(opt, lr_lambda)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()                                                # 切 BN/dropout 到推理行为!
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item(); total += y.numel()
    return correct / total                                      # top-1

def train(model, param_groups, train_loader, val_loader, device,
          epochs=100, weight_decay=0.05, mixup_fn=None, num_classes=10):
    model.to(device)
    opt = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    total = epochs * len(train_loader)
    sched = make_cosine_warmup(opt, warmup_steps=5*len(train_loader), total_steps=total)
    scaler = torch.cuda.amp.GradScaler()                        # 混合精度,省显存+加速
    best = 0.0
    for ep in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if mixup_fn is not None:                            # §4 的 Mixup/CutMix
                x, y_soft = mixup_fn(x, y, num_classes)
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                logits = model(x)
                if mixup_fn is not None:                        # 软标签交叉熵
                    loss = -(y_soft * logits.log_softmax(1)).sum(1).mean()
                else:
                    loss = torch.nn.functional.cross_entropy(
                        logits, y, label_smoothing=0.1)         # label smoothing 防过度自信
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪,防爆炸
            scaler.step(opt); scaler.update()
            sched.step()                                        # 每个 step 更新 lr(不是每 epoch)
        acc = evaluate(model, val_loader, device)
        best = max(best, acc)
        print(f"epoch {ep:3d}  val_top1={acc:.4f}  lr={sched.get_last_lr()[0]:.2e}")
    return best
```

### 分析:几个数值与显存的账

- **label smoothing ε=0.1**:把硬标签 [0,…,1,…,0] 改成 [ε/C,…,1−ε+ε/C,…,ε/C],惩罚过度自信、提升泛化和校准。**注意:已经用 Mixup 软标签时通常不叠加 label smoothing**(两者都在软化标签,叠加过度反而欠拟合)。
- **混合精度(AMP)**:用 fp16/bf16 做前向反向,显存约减半、吞吐近翻倍。`GradScaler` 防 fp16 下小梯度下溢为 0(把 loss 放大再反向,更新前缩回)。**bf16(若硬件支持)比 fp16 更稳,可不用 scaler**。
- **梯度裁剪 max_norm=1.0**:把梯度范数截断在 1,防止偶发大梯度把训练带崩,Transformer 训练几乎标配。
- **显存估算**:显存 ≈ 模型参数 + 梯度 + 优化器状态 + 激活值。AdamW 每参数额外存 2 份状态(m、v),所以 fp32 下"参数+梯度+Adam 状态" = 4×参数显存。激活值随 batch 和分辨率线性增长,常是大头——**OOM 时先降 batch 或开梯度检查点(gradient checkpointing,用算力换显存)**。

---

## 8. 诊断与可视化:过拟合/欠拟合,以及 Grad-CAM 看模型在看哪

### 问题:训练曲线长什么样代表什么,以及"模型到底看了哪"

光看一个最终 acc 数字不够。**诊断靠的是 train/val 两条曲线的相对关系**:

```
欠拟合(underfit):  train acc 和 val acc 都低,且都还在涨
   → 模型容量不够 / 训练不充分 / lr 太小 / 正则太强
   → 对策:加大模型、训更久、调大 lr、减弱增强与 weight decay

过拟合(overfit):   train acc 高(≈100%),val acc 明显更低且开始下降
   → 模型记住了训练集噪声,gap 大
   → 对策:加增强(§4)、加 weight decay/dropout、减小模型、early stopping、要更多数据

恰好(good fit):    train 略高于 val,gap 小且 val 仍在缓慢爬升
   → 这是目标状态
```

**关键诊断习惯:同时盯 train_loss、val_loss、val_acc 三条线,而不是只看 val_acc。** val_loss 先于 val_acc 反映过拟合——往往 val_acc 还没掉、val_loss 已经开始回升(模型对错的样本更自信了)。

### 机制:Grad-CAM——用梯度加权特征图,看分类决策的空间依据

回扣第 01 章:深层卷积特征图的每个通道是一种高层模式的"激活地图",且因感受野覆盖全图,空间位置对应原图区域。**Grad-CAM 的思想**:对某个类别 c 的得分 yᶜ,求它对最后一个卷积层特征图 Aᵏ(第 k 个通道)的梯度,**沿空间做全局平均得到该通道的重要性权重 αᵏ**,再加权求和、过 ReLU,得到一张"哪里对预测类 c 贡献大"的热力图:

```
αᵏ_c = (1/Z) · Σᵢ Σⱼ  ∂yᶜ / ∂Aᵏᵢⱼ        # 对通道 k 的梯度做全局平均池化 = 该通道重要性
L_GradCAM = ReLU( Σₖ αᵏ_c · Aᵏ )          # 加权求和后 ReLU(只保留正贡献区域)
```

为什么对梯度做全局平均就是重要性?因为 αᵏ 近似"通道 k 的激活整体增大一点,类得分 yᶜ 涨多少"的一阶灵敏度;ReLU 是因为我们只关心**支持该类**的区域(正梯度),抑制负贡献。Grad-CAM 不需要改网络结构、对任意 CNN 通用(CAM 原版要求特定全局池化结构,Grad-CAM 用梯度解除了这个限制)。

### 代码:Grad-CAM 最小实现(hook 拿激活与梯度)

```python
import torch, torch.nn.functional as F

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.acts = self.grads = None
        target_layer.register_forward_hook(self._save_act)       # 抓前向激活
        target_layer.register_full_backward_hook(self._save_grad) # 抓反向梯度

    def _save_act(self, m, i, o):  self.acts = o.detach()
    def _save_grad(self, m, gi, go): self.grads = go[0].detach()

    def __call__(self, x, class_idx=None):
        logits = self.model(x)                                   # [1,C]
        if class_idx is None: class_idx = logits.argmax(1).item()
        self.model.zero_grad()
        logits[0, class_idx].backward()                          # 只对该类得分回传
        # acts/grads: [1, K, h, w]
        alpha = self.grads.mean(dim=(2, 3), keepdim=True)        # 通道重要性 αᵏ:空间平均
        cam = F.relu((alpha * self.acts).sum(dim=1, keepdim=True))# 加权求和 + ReLU
        cam = F.interpolate(cam, size=x.shape[-2:],              # 上采样回原图尺寸
                            mode="bilinear", align_corners=False)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8) # 归一化到 [0,1]
        return cam[0, 0].cpu().numpy(), class_idx
# CNN 用最后一个卷积层做 target_layer(如 model.layer3[-1].conv2)。
# ViT 没有空间卷积图,Grad-CAM 要在最后一个 block 的 token 上重排成 2D,
# 或直接用注意力 rollout——见 §源码导读 pytorch-grad-cam 的 ViT 适配。
```

### 分析:可视化能抓出什么真实问题

- **学到了背景而非目标**(shortcut learning):热力图亮在草地/水面而不是动物身上 → 数据集有偏(比如船总在水里),模型学了捷径,**换背景就翻车**。这是只看 acc 永远发现不了的隐患。
- **ViT 的注意力图**:回扣第 07 章,可视化 [CLS] token 对各 patch 的注意力权重,自监督预训练(如 DINO)的 ViT 注意力会自发聚焦到物体上、甚至呈现分割效果——这是第 08 章自监督学到好表征的直观证据。
- **诚实原则**:可视化是诊断工具不是炫图工具。看到热力图合理别急着高兴,要专门找**模型答错的样本**看热力图,那里才暴露真问题。

---

## 9. 副线:最小扩散模型(回扣 11 章,在 MNIST/CIFAR 上生成)

如果你想走生成主线,这里给一份能跑的最小 DDPM 骨架,直接复用第 11 章推出的结论(前向闭式加噪 `xₜ=√(ᾱₜ)x₀+√(1−ᾱₜ)ε`、ε-预测损失 `L=E‖ε−εθ(xₜ,t)‖²`、DDPM 采样)。**方法论完全一样:先 overfit 一小撮样本验证实现,再上全量。**

```python
import torch, torch.nn as nn, math

# ---- 噪声调度:预计算 βₜ、αₜ、ᾱₜ(回扣 11 章 §2)----
def make_schedule(T=1000, device="cuda"):
    beta = torch.linspace(1e-4, 0.02, T, device=device)     # 线性调度
    alpha = 1 - beta
    abar = torch.cumprod(alpha, dim=0)                       # ᾱₜ = ∏ αₛ
    return beta, alpha, abar

def q_sample(x0, t, abar, eps):                             # 前向一步到位(闭式解)
    a = abar[t].view(-1, 1, 1, 1)
    return a.sqrt() * x0 + (1 - a).sqrt() * eps             # xₜ=√ᾱₜ·x0+√(1−ᾱₜ)·ε

def train_step(model, x0, abar, T, opt):
    B = x0.size(0)
    t = torch.randint(0, T, (B,), device=x0.device)        # 每样本独立采时间步
    eps = torch.randn_like(x0)
    xt = q_sample(x0, t, abar, eps)
    eps_pred = model(xt, t)                                 # εθ-预测 U-Net
    loss = (eps_pred - eps).pow(2).mean()                   # L_simple:就是 MSE
    opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()

@torch.no_grad()
def ddpm_sample(model, shape, beta, alpha, abar, T):       # 反向迭代去噪(回扣 11 章 §5.1)
    x = torch.randn(shape, device=beta.device)             # 从纯高斯起步 x_T
    for t in reversed(range(T)):
        tt = torch.full((shape[0],), t, device=beta.device)
        eps = model(x, tt)
        a, ab, b = alpha[t], abar[t], beta[t]
        mean = (x - b / (1 - ab).sqrt() * eps) / a.sqrt()  # 后验均值(ε 参数化)
        x = mean + (b.sqrt() * torch.randn_like(x) if t > 0 else 0)  # 末步不再加噪
    return x
```

网络用一个带时间步嵌入的小 U-Net(回扣第 11 章 §6:`timestep_embedding` + `ResBlock`,把时间嵌入加进每个残差块)。MNIST 上一个两三层下采样的 U-Net、T=1000、AdamW lr=2e-4、训几十个 epoch 就能生成清晰数字。**诊断信号**:loss 应稳定下降到一个低平台(不像 GAN 那样震荡);若采样出来是纯噪声,先 overfit 单张图验证 `q_sample`/采样公式的系数有没有抄错(系数错是这里的头号 bug)。

---

## 10. 设计权衡与常见坑

- **不 overfit 一个 batch 就开训** = 在沙地上盖楼。实现 bug 和超参问题混在一起,你永远调不出来。这是本章第一铁律,违反代价最大。
- **一次改多个变量做消融** = 收益互相抵消、归因失效。换增强的同时调了 lr,acc 涨了你都不知道是谁的功劳。**消融必须单变量。**
- **小数据从零训 ViT** = 撞第 07 章曲线最左端,过拟合到怀疑人生。小数据一律迁移学习。
- **迁移学习用错归一化/分辨率** = 输入分布对不上预训练,特征全废。永远用预训练模型自带的 transform 配置。
- **eval 时忘了 `model.eval()`** = BN 还在用 batch 统计、dropout 还在丢神经元,评估结果偏低且随 batch 抖动(回扣第 03 章"BN 训练/推理两套行为"的坑)。这是新手最高频低级错误。
- **scheduler 按 epoch 而非 step 更新** = cosine 曲线被拉伸 N 倍,warmup 形同虚设。`sched.step()` 放在内层 step 循环里。
- **batch size 改了不调 lr** = AdamW 对 batch size 不像 SGD 那样需要严格线性缩放,但大幅改变 batch 仍要重调 lr 和 warmup。
- **用 acc 自欺**:测试集和训练集同分布时 acc 高,不代表鲁棒。必须靠 Grad-CAM/分布偏移测试暴露 shortcut learning。
- **过早追求 SOTA 数字**:capstone 的目标是**走通工作流、能诊断**,不是刷榜。先要一条"能解释清楚每个数字怎么来的"管线,再谈涨点。

---

## 11. 动手练习

**练习 1(复现题,必做):跑通分类主线 + 验证 overfit 一个 batch。** 用 §3 的 `overfit_one_batch` 把 §5 的 CifarResNet 在 16 张 CIFAR 图上训到 train acc=100%、loss→0,并验证初始 loss ≈ ln 10 ≈ 2.30。然后用 §7 的训练循环在全量 CIFAR-10 上训到收敛,记录 best val top-1。
*提示:先关增强跑 overfit;过不了别往下走,按 §3 的二分排查表定位。初始 loss 若不在 2.3 附近,优先查输出层和标签对齐。*

**练习 2(消融题):去掉 BN / 换增强 / 冻结 backbone,各做一次单变量消融。**
(a) 把 §5 残差块里的 `BatchNorm2d` 全删掉(改 `nn.Identity()`),重训,观察:loss 还降吗?能训多深?用 §3 的梯度范数打印看是否梯度爆炸/消失(回扣第 03 章 BN 的作用)。
(b) 在"无增强 / 仅 RandAug / RandAug+Mixup"三档下各训一遍,画出三条 train/val 曲线,定位每档的 train-test gap,验证 §4"增强收窄 gap 但需更多 epoch"的论断。
(c) 对 §6 的 ViT,跑"线性探测 vs 全量微调"两档,对比 best val acc 和单 epoch 耗时/显存,验证 §6 的决策规则。
*提示:每次只改一个东西,其余配置完全冻结;否则归因失效(§10 第二坑)。(a) 中无 BN 时大概率要把 lr 调小才不崩,这本身就是 BN 价值的证据。*

**练习 3(分析+可视化题):用 Grad-CAM 抓 shortcut learning。** 用 §8 的 GradCAM 对你练习 1 训好的模型,分别可视化**答对**和**答错**的样本各 5 张。回答:(a) 答错的样本,热力图是否亮在了背景/无关区域?(b) 找出至少一类系统性错误(如某两类总混淆),用热力图解释模型把注意力放错在哪。(c) 若发现 shortcut,提出一个数据层面的修正方案。
*提示:CNN 的 target_layer 选 `model.layer3[-1].conv2`;务必专挑答错样本看(§8 诚实原则)。*

**练习 4(推导题):推导 cosine+warmup 的总"等效学习量"并解释 warmup 步数怎么选。**
(a) 对 §7 的调度,写出总训练步内学习率对步数的积分 ∫η(t)dt 的近似(把 warmup 段和 cosine 段分别积分),说明它近似正比于哪些量。(b) 论证:为什么 warmup 步数通常取总步数的 5%~10%,而不是固定步数——从"BN running stats 和 Adam 二阶矩需要多少 step 才稳定"的角度说。(c) 若把 epoch 从 100 改到 300,warmup 步数该不该跟着变?
*提示:(a) cosine 段 ∫½(1+cos)dt 在半周期上 = 周期长度的一半量级;(b) 稳定所需的是"见过足够多样本",故与数据规模/总步数挂钩;(c) 想清楚 warmup 是绝对步数需求还是相对比例需求。*

---

## 12. 源码 / 论文导读

- **timm(`pytorch-image-models`,本章迁移主线)**:`timm.create_model(..., pretrained=True, num_classes=N)` 是微调的入口;读 `timm/data/transforms_factory.py` 的 `create_transform`(RandAugment / Mixup / 归一化一条龙,对应本章 §4/§6),`timm/data/mixup.py` 的 `Mixup` 类(工业级 Mixup+CutMix 实现,对照本章 §4 的最小版),`timm/scheduler/cosine_lr.py`(cosine+warmup,对照 §7)。`timm/optim` 里有现成的 layer-wise lr decay 工具(对照 §6 分层 lr)。**这是把本章所有配方"调到生产级"的标准参照。**
- **torchvision**:`torchvision.models`(ResNet/ViT 官方实现)、`torchvision.transforms.v2`(新版增强,含 `MixUp`/`CutMix`/`RandAugment`,API 比 timm 更精简,适合初学通读)、官方 references/classification 目录有完整可跑的训练脚本(`train.py`),是"最小但完整训练管线"的权威范本。
- **小 ResNet for CIFAR**:He 等《Deep Residual Learning for Image Recognition》(CVPR 2016)的 **Section 4.2(CIFAR-10 实验)**——本章 §5 的 stem 改动、三 stage 16/32/64 通道、ResNet-20/32/56 的 n 取值,全部出自这一节;论文的网络配置表可直接对账你算的参数量。
- **数据增强论文**:Mixup(Zhang 等《mixup: Beyond Empirical Risk Minimization》,ICLR 2018,读凸组合训练目标的推导);CutMix(Yun 等,ICCV 2019);RandAugment(Cubuk 等,2020,读它如何把搜索空间压成 (N,M) 两个超参,对应 §4)。
- **AdamW**:Loshchilov & Hutter《Decoupled Weight Decay Regularization》(ICLR 2019)——读它把权重衰减从 Adam 梯度里解耦的那张对比(对应 §7 的两行更新式),理解为什么这对 Transformer 影响显著。
- **Grad-CAM**:Selvaraju 等《Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization》(ICCV 2017)——读梯度全局平均得到 αᵏ 的推导(对应 §8 的两个式子)。开源实现 **`jacobgil/pytorch-grad-cam`**:对 CNN 和 ViT 都有适配(ViT 的 reshape_transform 是 §8 代码注释里提到的"把 token 重排成 2D"的标准答案),还含 Grad-CAM++/ScoreCAM 等变体。
- **最小扩散实现(副线)**:`lucidrains/denoising-diffusion-pytorch`(单文件极简 DDPM,对照 §9 的 `q_sample`/`ddpm_sample` 系数);HuggingFace `diffusers` 的 `DDPMScheduler`/`UNet2DModel`(生产级,对照第 11 章 §5/§6)。
- **自监督预训练(回扣 08 章)**:微调 DINO/MAE 预训练的 ViT 权重(timm 里有 `vit_*` 的多种预训练来源),用 §8 的注意力可视化对比"监督预训练 vs 自监督预训练"的 ViT 注意力图差异,直观感受第 08 章学到的表征质量。

---

## 13. 小结与承上启下

这一章不教新模型,而是教你**把零件拼成项目、并在出问题时定位得了、修得动**——这是从"学过 CV"到"会做 CV"的分水岭。把整套工作流收束成一条可执行的清单:

1. **先 overfit 一个 batch**(§3):用初始 loss≈ln C 和"能否记住 16 张图"两个检查,在堆任何技巧前确认实现链路正确——**这是第一铁律**;
2. **数据增强是用训练难度换泛化**(§4):RandAug/Mixup/CutMix 各有数学本质,加了它训练 acc 降是正常的,看的是 train-test gap 收窄;
3. **能迁移就别从零训**(§5/§6):小 ResNet 适合从零搭小图分类,ViT 在小数据上必须迁移学习(回扣 07 章翻转曲线),迁移要配分层 lr、对齐归一化与分辨率;
4. **训练配方每个旋钮都有理由**(§7):AdamW 解耦衰减、warmup 给统计量预热、cosine 平滑精修,混合精度/梯度裁剪是稳定与效率的标配;
5. **诊断靠 train/val 两条曲线,理解靠可视化**(§8):过拟合/欠拟合从 gap 读,Grad-CAM 用梯度加权特征图暴露"模型在看哪"和 shortcut learning。

**在整门课里的位置**:这是计算机视觉专家课程的**收官实战章**。前面十几章把卷积(01)、CNN 演进(02)、ResNet+BN(03)、检测(04/05)、分割(06)、ViT(07)、自监督(08)、CLIP(09)、GAN(10)、扩散(11)逐个拆透;本章把分类与生成两条线上的核心零件**真正装到一条能跑的管线上**,并交给你一套通用的"实现验证 → 配方调优 → 诚实诊断"方法论。

**这套方法论是可迁移的资产。** 无论你接下来做检测、分割还是多模态,"先 overfit 一个 batch、单变量消融、能迁移就迁移、训完必可视化"这四条不变。你现在缺的不再是知识,而是**里程数**——拿一个真实数据集,把本章的练习从头跑一遍,你就完成了从读者到从业者的最后一跃。

---

### 附:训练报告模板(每个项目都填一份)

把下面这份表存成项目的 `REPORT.md`,每跑一次实验填一行/一节。**它的价值不在格式,在于强迫你把"这次改了什么、结果怎样、为什么"写清楚——这是消融纪律的载体。**

```markdown
# 训练报告:<任务名 / 数据集>

## 0. 健全性(做项目第一件事)
- [ ] overfit 一个 batch:final train_acc=____  loss=____(应≈0)
- [ ] 初始 loss=____  期望 ln(C)=____(应吻合)
- [ ] 数据反归一化后肉眼检查:通道/归一化正常? Y/N

## 1. 配置(改任何一项都另起一次实验)
- 模型 / 来源(from scratch | timm 预训练名):
- 输入分辨率 / 归一化(mean,std,来源):
- 增强(RandAug N,M / Mixup α / CutMix p):
- 优化器(AdamW lr_head / lr_backbone / wd):
- 调度(warmup steps / total / min_lr):
- batch size / epochs / 混合精度(fp16|bf16) / 梯度裁剪:

## 2. 结果
| 实验 | 单变量改动 | train_top1 | val_top1 | train-val gap | val_loss 最低点 |
|------|-----------|-----------|----------|---------------|----------------|
| base |     —     |           |          |               |                |
| abl1 |           |           |          |               |                |

## 3. 诊断
- 拟合状态:欠拟合 | 恰好 | 过拟合(依据:gap=____,val_loss 是否回升)
- Grad-CAM:答对样本看哪____;答错样本看哪____;是否发现 shortcut____
- 下一步单变量改动假设(改什么 + 预期效果 + 理由):
```
