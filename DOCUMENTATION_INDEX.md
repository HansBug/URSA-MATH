# 20260312 URSA-MATH论文+仓库阅读日志

## 1. 阅读背景

这份文档不再作为纯索引使用，而是改成我站在实习生角度整理的一份阅读日志。目标不是罗列仓库里有哪些 markdown，而是把我对论文、模型、训练流程、仓库结构和运行方式的理解尽量直接写清楚。

当前仓库 commit id：

- [3496410752fcc0e3e2f56461cbdd32d436d714ce](https://github.com/HansBug/URSA-MATH/commit/3496410752fcc0e3e2f56461cbdd32d436d714ce)

补充材料：

- 运行与输出字段说明：[RUN_GUIDE.md](https://github.com/HansBug/URSA-MATH/blob/main/RUN_GUIDE.md)
- 本次改动与验证记录：[report-2026-03-12.md](https://github.com/HansBug/URSA-MATH/blob/main/report-2026-03-12.md)
- 中文论文整理稿：[paper_zh.md](https://github.com/HansBug/URSA-MATH/blob/main/paper_zh.md)

## 2. 我对这篇论文的总体理解

URSA-MATH 的核心目标，不是单纯让模型“看懂图”，而是让一个 8B 级别的多模态模型具备更强的数学推理能力，尤其是：

- 能生成更完整、更可靠的 step-by-step reasoning
- 能判断一条推理轨迹中哪一步开始出错
- 能把这种过程层面的判断再用于测试时选择和强化学习

论文提出 URSA 的出发点，是先指出三类问题：

1. 高质量多模态数学 CoT 数据不够，基础模型上限低
2. 多模态场景缺少自动化过程监督数据
3. 过程奖励如果直接拿来做 RL，容易被模型 exploit，出现 reward hacking 和长度偏置

所以作者没有直接上 RL，而是设计成三阶段 pipeline：

1. Stage I：先训练出更强的基础生成模型 `URSA-8B`
2. Stage II：再训练过程奖励模型 `URSA-RM-8B`
3. Stage III：最后用 `URSA-RM-8B` 辅助在线 RL，得到 `URSA-8B-PS-GRPO`

我读完之后最认同的一点是：这篇论文真正的主线不是“用 PRM”，而是“先把基础生成模型做强，再把 PRM 用在它最可信的部分”。

## 3. 三阶段到底是怎么串起来的

我把这三阶段理解成一条很清晰的前后依赖链：

```text
Qwen2.5-Math-Instruct + Hybrid Vision Tower
    ↓
Stage I
    ↓
URSA-8B
    ├── Stage II → URSA-RM-8B
    └── Stage III + URSA-RM-8B → URSA-8B-PS-GRPO
```

也就是说：

- `URSA-8B` 是主干基础模型
- `URSA-RM-8B` 是从 `URSA-8B` 分出来的 verifier / PRM 分支
- `URSA-8B-PS-GRPO` 是 `URSA-8B` 在 `URSA-RM-8B` 辅助下继续优化得到的更强 policy

如果少了其中任何一步，后面的效果都会受限：

- 没有 Stage I，rollout 质量不够好，PRM 和 RL 上限都会低
- 没有 Stage II，就没有可靠的 step-level process signal
- 没有 Stage III，PRM 只能停留在 BoN verifier 层面，无法反过来提升 policy

## 4. Stage I：基础多模态数学生成模型是怎么来的

Stage I 的目标，是先得到一个更强的多模态数学推理模型 `URSA-8B`。

### 4.1 我对 Stage I 的理解

这一阶段其实是在解决两个问题：

1. 模型能不能把图片正确接入语言模型
2. 模型能不能围绕数学题生成高质量 reasoning trajectory

所以它本质上由两个部分组成：

- 视觉-语言对齐
- 基于数学 CoT 数据的 instruction tuning

### 4.2 Stage I 的主要技术

这一阶段最关键的技术点有四个：

1. 混合视觉塔
2. projector 对齐
3. 基于 `MMathCoT-1M` 的全参数 SFT
4. 高质量 CoT 数据合成

其中混合视觉塔由两部分构成：

- SigLIP-L：负责低分辨率全局理解
- SAM-B：负责高分辨率细节理解

这个设计很符合数学题场景，因为数学图像通常既需要看全局结构，也需要看局部细节，比如坐标值、角标、长度、角度、表格数字等。

### 4.3 Stage I 的输入

模型输入：

- hybrid vision tower
- 两层 MLP projector
- Qwen2.5-Math-Instruct 作为 LLM backbone

数据输入：

- 视觉-语言对齐数据
- `MMathCoT-1M`

`MMathCoT-1M` 是这一阶段最重要的数据资产。我的理解是，它不是普通问答数据，而是：

- 图像
- 问题
- 带 step-by-step reasoning 的解答
- 最终答案

### 4.4 Stage I 的数据是怎么来的

论文里对 `MMathCoT-1M` 的构造写得比较细。我整理后觉得可以记成三件事：

1. 把 answer-only 数学样本扩成完整 CoT
2. 把已有分析型解答重写成更清晰的步骤形式
3. 把风格统一成更自然、更稳定、更适合 SFT 的推理文本

也就是说，这个数据集不是简单抓来的，而是经过了较强的数据重写和质量控制。

### 4.5 Stage I 的输出

输出模型：

- `URSA-8B`

输出能力：

- 输入图片 + 问题
- 生成一步步 reasoning
- 最终给出答案

### 4.6 为什么 Stage I 很关键

我觉得论文里一个很重要的判断是：

- PRM 标注质量，依赖基础模型能不能生成足够有价值的 reasoning trajectories
- RL 的上限，也依赖策略模型本身能探索到什么样的轨迹

所以 Stage I 的意义并不只是“先做个 baseline”，而是给 Stage II 和 Stage III 抬上限。

## 5. Stage II：过程奖励模型是怎么来的

Stage II 的目标，是基于 `URSA-8B` 训练出一个多模态 PRM，也就是 `URSA-RM-8B`。

### 5.1 我对 Stage II 的理解

这一阶段不是让模型更会“解题”，而是让模型更会“判断一条解题过程是否可靠”。换句话说：

- Stage I 训练的是 answer generator
- Stage II 训练的是 process verifier

### 5.2 Stage II 的输入

模型输入：

- Stage I 得到的 `URSA-8B`

数据输入：

- `URSA-8B` 产生的 reasoning traces
- 针对这些 traces 生成的过程监督标签
- 最终合成的数据集 `DualMath-1.1M`

这里我最关注的一点是：Stage II 的数据不是凭空标出来的，而是建立在 `URSA-8B` 已经能生成不少正确/错误推理轨迹的基础上。

### 5.3 Stage II 的核心技术

论文里这一阶段最核心的是双视角数据合成：

1. BEL：Binary Error Localization
2. MIE：Misinterpretation Insertion Engine

BEL 的作用，我理解成：

- 给一条错误解答定位第一个真正错的 step

它的实现依赖 MCTS 思路：

- 从某个 prefix 出发继续采样
- 观察这些 rollout 有多少还能回到正确答案
- 用这个统计值估计该 step 之前的前缀是否仍然“有希望”

MIE 的作用，我理解成：

- 故意引入视觉误解、条件误读、图文不一致
- 让 PRM 不只会抓逻辑错误，还会抓多模态感知错误

这点很重要，因为多模态数学推理里，很多错误并不是计算错，而是“看图就看错了”。

### 5.4 Stage II 的训练对象到底是什么

我理解它训练的是一个 step-level binary classifier。

也就是说，对每个 step，模型要判断：

- 这个 step 是否仍然属于正确推理链的一部分

论文实现里在每个 step 后插入特殊标记 token，然后在这些位置做二分类。这使得 `URSA-RM-8B` 最后能输出：

- 每一步的 process reward / correctness score

### 5.5 Stage II 的输出

输出数据：

- `DualMath-1.1M`

输出模型：

- `URSA-RM-8B`

输出能力：

- 对每一步给出分数
- 判断哪里开始不可靠
- 做 BoN 排序
- 提供后续 drop-moment 信号

### 5.6 为什么 Stage II 很关键

如果没有 Stage II，后续只有两条路：

- 要么只做 self-consistency
- 要么只做 outcome reward RL

这样模型虽然能在最终答案层面被筛选，但看不到“过程到底哪里开始坏掉”。所以我认为 Stage II 的贡献是把 verifier 从 answer-level 提升到了 process-level。

## 6. Stage III：PS-GRPO 是怎么工作的

Stage III 的目标，是把已经训练好的 `URSA-RM-8B` 接到 RL 里，进一步优化 `URSA-8B`，得到 `URSA-8B-PS-GRPO`。

### 6.1 Stage III 的输入

模型输入：

- policy model：`URSA-8B`
- verifier / PRM：`URSA-RM-8B`

数据输入：

- 经过筛选的 RL 数据
- rollout groups
- outcome reward
- `URSA-RM-8B` 输出的 process reward sequence

论文里还提到，RL 数据不是直接全量使用，而是做了一次静态过滤：

- 对候选题做多次采样
- 去掉“全对”或“全错”的题
- 保留更有训练价值的样本

### 6.2 为什么作者不直接把过程奖励塞进 GRPO

论文先分析了两种看起来很自然，但效果不好的做法：

- Variant 1：`r^i = r_o^i + mean(r_s^i)`
- Variant 2：把 step-level process reward 直接并入 advantage

我读下来，作者的核心结论是：

- 直接拟合 process scalar reward，模型会学会迎合 verifier
- PRM 存在长度偏置，模型会越来越短、越来越保守
- 看起来“过程得分高”不等于真的更会解题

这一步分析非常关键，因为它解释了为什么 Stage III 不能简单写成“PRM + RL = 更强”。

### 6.3 PS-GRPO 的关键思想

我对 PS-GRPO 的理解是：

- 不直接相信 PRM 的绝对标量值
- 只相信 PRM 更稳定的内部信号

论文最终保留了两个相对可信的信号：

1. 哪条 rollout 的平均过程分数更高
2. 一条 rollout 的 reward sequence 是否出现明显下跌

这里第二点就是论文提出的 `drop-moment`。

对应公式 5：

- 如果相邻步骤之间出现明显相对下降，且超过阈值 `rho`，就认为发生了 drop-moment

对应公式 6：

- 正确且没有 drop-moment：reward = `1`
- 正确但有 drop-moment：reward = `1 - gamma`
- 错误：reward = `0`

### 6.4 我为什么觉得 PS-GRPO 是这篇论文最重要的部分

因为它没有把 PRM 当成一个完美 reward source，而是承认：

- PRM 有偏差
- PRM 会被 exploit
- PRM 的某些局部信号比整体标量更值得信任

所以这其实是一种更克制的 verifier-to-RL 接法：

- outcome 仍然是主目标
- process 只在暴露出不一致时提供惩罚

我觉得这比“把过程分直接加进 reward”更稳，也更符合这篇论文展示出来的实验证据。

### 6.5 Stage III 的输出

输出模型：

- `URSA-8B-PS-GRPO`

需要特别区分的一点是：

- `URSA-RM-8B` 是 verifier
- `URSA-8B-PS-GRPO` 仍然是 generator

也就是说，Stage III 不是再造一个奖励模型，而是在奖励模型辅助下把主干生成模型继续优化一轮。

## 7. 三个模型在训练时和推理时如何配合

### 7.1 训练时

训练时的关系可以简单写成：

1. 先训练 `URSA-8B`
2. 再用 `URSA-8B` 的轨迹和自动标注训练 `URSA-RM-8B`
3. 最后用 `URSA-RM-8B` 辅助 RL，继续更新 `URSA-8B`，得到 `URSA-8B-PS-GRPO`

这里最关键的是：

- Stage II 更新 verifier
- Stage III 更新 policy

### 7.2 推理时

推理时有两种模式：

1. 单模型直接答题
2. generator + verifier 联合推理

如果是单模型模式：

- `URSA-8B` 或 `URSA-8B-PS-GRPO` 直接生成答案

如果是联合模式：

- generator 先生成多条 rollout
- `URSA-RM-8B` 对每条 rollout 的每一步打分
- 再做 BoN 选优

所以从部署角度看：

- `URSA-8B` / `URSA-8B-PS-GRPO` 是答题模型
- `URSA-RM-8B` 是验题模型

## 8. 从仓库代码看，我会怎么对应这三阶段

### 8.1 生成模型相关

最直接的入口是：

- `models/`
- `examples/run_ursa_8b_torch_example.py`
- `examples/run_ursa_8b_torch_example_standalone.py`

这一部分体现的是：

- `URSA-8B` 怎么加载
- 图像怎么进模型
- 生成模型怎么输出 reasoning

### 8.2 PRM 相关

最直接的入口是：

- `inference/prm_infer_score.py`
- `examples/run_ursa_rm_8b_score_example.py`
- `examples/run_ursa_rm_8b_score_example_standalone.py`

这一部分体现的是：

- step-level 打分怎么实现
- 特殊标记 token 怎么对齐步骤
- process reward sequence 怎么取出来

### 8.3 Stage III 相关

虽然仓库当前不是一个完整的 RL 训练框架，但就复现论文机制来说，当前示例已经足够覆盖关键观测量：

- `process_reward_sequence`
- rollout 平均过程分数
- BoN 选优
- drop-moment
- PS-GRPO reward
- vanilla GRPO / Variant 1 / Variant 2 中间量

这部分对应的补充材料是：

- [RUN_GUIDE.md](https://github.com/HansBug/URSA-MATH/blob/main/RUN_GUIDE.md)
- [report-2026-03-12.md](https://github.com/HansBug/URSA-MATH/blob/main/report-2026-03-12.md)

## 9. 我当前的结论

站在实习生角度，我对这个项目目前的总结是：

- Stage I 决定了整个系统的上限
- Stage II 让多模态数学 verifier 第一次真正具备 step-level 过程判断能力
- Stage III 的关键不是“用 PRM”，而是“只用 PRM 中更可靠的部分”
- 三个模型不是并列关系，而是主干 policy + verifier 分支 + verifier 辅助后的强化 policy

如果后面还要继续扩展，我觉得最自然的方向有两个：

- 把当前 Stage III 的观测量接到更完整的训练脚本里
- 把仓库里的示例输出和论文公式、图表、算法伪代码做更严格的一一映射

## 10. 关联文档

- 更详细的运行命令与输出字段解释：[RUN_GUIDE.md](https://github.com/HansBug/URSA-MATH/blob/main/RUN_GUIDE.md)
- 本次环境验证与改动记录：[report-2026-03-12.md](https://github.com/HansBug/URSA-MATH/blob/main/report-2026-03-12.md)
- 中文论文整理稿：[paper_zh.md](https://github.com/HansBug/URSA-MATH/blob/main/paper_zh.md)
