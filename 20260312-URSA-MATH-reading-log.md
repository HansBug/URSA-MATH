# 20260312 URSA-MATH 论文+仓库阅读日志

## 1. 文档定位

这份日志用于把论文理解、仓库结构理解、运行路径和本次整理工作放到同一个入口里，避免信息分散在多份 markdown 里。

当前仓库 commit id：

- [b5b390d0fc9e4f332738e9a5dbb69a0ad98a52a2](https://github.com/HansBug/URSA-MATH/commit/b5b390d0fc9e4f332738e9a5dbb69a0ad98a52a2)

相关文档：

- 阅读工作与改动总结：[report-2026-03-12.md](https://github.com/HansBug/URSA-MATH/blob/main/report-2026-03-12.md)
- 系统性技术导读：[GUIDE.md](https://github.com/HansBug/URSA-MATH/blob/main/GUIDE.md)
- 实际运行路径：[RUN_GUIDE.md](https://github.com/HansBug/URSA-MATH/blob/main/RUN_GUIDE.md)
- 英文论文整理稿：[paper.md](https://github.com/HansBug/URSA-MATH/blob/main/paper.md)
- 中文论文整理稿：[paper_zh.md](https://github.com/HansBug/URSA-MATH/blob/main/paper_zh.md)

## 2. 论文主线：三阶段到底在做什么

URSA-MATH 要解决的是多模态数学推理，而不是一般图片描述。它面对的核心难点不是“看图”本身，而是：

- 基础模型的数学 CoT 能力不够强
- 多模态过程监督数据极少
- 过程奖励直接拿来做 RL 会出现 reward hacking 和长度偏置

所以论文不是直接从 RL 开始，而是先把能力地基打稳，再做 verifier，再做 RL。按论文结构，这三阶段可以概括成：

1. Stage I：做出一个更强的多模态数学生成模型 `URSA-8B`
2. Stage II：基于 `URSA-8B` 做出过程奖励模型 `URSA-RM-8B`
3. Stage III：用 `URSA-RM-8B` 去辅助在线 RL，得到 `URSA-8B-PS-GRPO`

我自己的理解是，这其实是一条非常明确的依赖链：

- 没有 Stage I，就没有足够强的 rollout 质量，后面 PRM 和 RL 上限都会低
- 没有 Stage II，就没有可靠的过程信号，Stage III 只能退回 outcome-only RL
- Stage III 不是再训练一个 verifier，而是把 Stage II 产出的 verifier 以更克制的方式接进策略优化

## 3. 三个 Stage 的详细拆解

### 3.1 Stage I：Math-Intensive Alignment and Instruction Tuning

这是整条链路的“基础生成模型阶段”，目标是得到一个强的多模态数学推理 policy model，也就是 `URSA-8B`。

#### Stage I 的技术本质

这一阶段在论文里虽然叫一个 stage，但从工程上看可以拆成两个子过程：

1. 视觉-语言对齐
2. 数学密集型 SFT

第一步解决“模型能不能把图像特征正确接到语言模型里”；第二步解决“模型能不能围绕数学问题生成高质量、逐步展开的 reasoning trajectory”。

#### Stage I 的输入

模型输入：

- 混合视觉塔
  - SigLIP-L：负责低分辨率全局语义
  - SAM-B：负责高分辨率细节感知
- 两层 MLP projector：把视觉特征投到语言空间
- Qwen2.5-Math-Instruct：作为 LLM backbone

数据输入：

- 视觉-语言对齐数据
- `MMathCoT-1M`

其中 `MMathCoT-1M` 是这一阶段的核心数据资产。它不是普通 QA，而是带完整推理过程的多模态数学 CoT 数据。

#### Stage I 用了什么技术

从 [GUIDE.md](https://github.com/HansBug/URSA-MATH/blob/main/GUIDE.md) 和 [paper_zh.md](https://github.com/HansBug/URSA-MATH/blob/main/paper_zh.md) 看，Stage I 的关键技术有四个：

1. 混合视觉塔
2. 视觉特征到语言空间的 projector 对齐
3. 基于 `MMathCoT-1M` 的全参数 instruction tuning
4. 数学 CoT 数据的三重合成策略

其中 `MMathCoT-1M` 的三重策略可以理解为：

- CoT expansion：把 answer-only 数据扩成 step-by-step 解答
- analysis rewriting：把分析型答案整理成更清晰的步骤
- format unify / naturalization：把风格统一成更自然、更适合 SFT 的推理文本

#### Stage I 的输出

直接输出：

- `URSA-8B`

这个模型的职责是：

- 输入图片 + 问题
- 输出带 reasoning steps 的答案

也就是说，Stage I 的输出是“会做题的多模态数学生成器”。

#### 我对 Stage I 的理解

Stage I 的价值不只是把 benchmark 分数做高一点，而是为后面两步提供更高质量的探索空间。

论文里有一个判断我认为非常重要：

- PRM 的标注质量，依赖基础模型能否产生足够有价值的 reasoning traces
- RL 的上限，也受限于策略模型本身能探索到什么样的轨迹

所以 `URSA-8B` 不是后续阶段的附属品，而是整个三阶段 pipeline 的核心地基。

### 3.2 Stage II：Dual-View Process Supervised Data Synthesis + PRM Training

这一阶段的目标是得到一个多模态过程奖励模型 `URSA-RM-8B`。换句话说，这一阶段不是让模型“更会答”，而是让模型“更会判断哪一步开始错了”。

#### Stage II 的输入

模型输入：

- Stage I 得到的 `URSA-8B`

数据输入：

- `URSA-8B` 在 `MMathCoT-1M` 上产生的正确/错误解答轨迹
- 自动化过程监督标注结果
- 最终整理成 `DualMath-1.1M`

可以把它理解成：

- Stage I 提供 reasoning generator
- Stage II 基于这些 reasoning traces 合成 PRM 训练数据

#### Stage II 用了什么技术

这一阶段有两个核心技术模块：

1. Binary Error Localization，简称 BEL
2. Misinterpretation Insertion Engine，简称 MIE

BEL 的目标是：定位一个错误解答里“第一个真正错误的步骤”。

MIE 的目标是：人为构造视觉误解、条件误读、图文不一致，让 PRM 不只学逻辑正确性，还学感知一致性。

从论文实现看，Stage II 里最关键的动作包括：

- 用 `URSA-8B` 的错误解答作为待标注对象
- 用 MCTS 估计某个 prefix 后续能否走到正确答案
- 给每一步打二元标签，形成 step-level supervision
- 在每个 step 后插入特殊标记 token，用二分类方式训练 PRM

所以 Stage II 的本质不是普通 reward modeling，而是“逐步 correctness classification”。

#### Stage II 的输入输出形式

输入样本更像这样：

- 图像
- 问题
- 一条 step-by-step response
- 每一步对应的 correctness label

输出模型：

- `URSA-RM-8B`

输出能力：

- 对每一步给出一个过程分数
- 识别 first-error tendency
- 支持 BoN 排序
- 支持 drop-moment 检测

#### Stage II 的输出为什么重要

这一步其实做了两件事：

1. 产出了一个 verifier 模型 `URSA-RM-8B`
2. 产出了一个新的多模态过程监督数据集 `DualMath-1.1M`

而 `URSA-RM-8B` 的价值有两个落地方向：

- inference 时做 Best-of-N verifier
- training 时给 Stage III 提供相对可信的过程信号

这也是为什么 Stage II 是整个论文里承上启下的阶段。

### 3.3 Stage III：Integrating Multimodal PRM into RL

这一阶段的目标不是训练 `URSA-RM-8B`，而是利用已经训练好的 `URSA-RM-8B` 去辅助 policy optimization，最终得到更强的生成模型 `URSA-8B-PS-GRPO`。

#### Stage III 的输入

模型输入：

- policy model：`URSA-8B`
- verifier / PRM：`URSA-RM-8B`

数据输入：

- 从 `MMathCoT-1M` 里筛出的 RL 数据
- rollout group
- outcome reward
- PRM 产生的 process reward sequence

论文里还明确提到，RL 数据不是直接全量上，而是先做一次静态过滤：

- 用 `URSA-8B` 对候选题目做多次采样
- 过滤掉“全对”或“全错”的题
- 保留那些仍有学习价值的样本

#### Stage III 用了什么技术

这一阶段的技术核心是：

- 先分析 vanilla GRPO + scalar process rewards 为什么失败
- 再提出 PS-GRPO

论文明确比较了两种失败变体：

- Variant 1：`r^i = r_o^i + mean(r_s^i)`
- Variant 2：把 step-level process reward 直接加进 step advantage

作者观察到的问题是：

- reward hacking
- PRM length bias
- 模型变得更保守、更短、更像在规避 verifier

于是 PS-GRPO 改成只相信 PRM 的“相对信号”：

- 平均过程分数是否足以做 BoN 排序
- reward sequence 是否出现明显下跌

然后用公式 5 定义 `drop-moment`，再用公式 6 定义新的 rollout reward：

- 正确且无明显下降：`1`
- 正确但出现下降：`1 - gamma`
- 错误：`0`

这相当于说：

- outcome reward 仍然是主信号
- process reward 不直接充当优化目标
- process 只在它显露出“不一致”时，用来惩罚错误方向

#### Stage III 的输出

最终输出模型：

- `URSA-8B-PS-GRPO`

这个模型仍然是“生成模型”，不是 verifier。也就是说，Stage III 输出的是一个更强的 policy model，而不是另一个 reward model。

#### 我对 Stage III 的理解

我认为 Stage III 的关键不是“把 PRM 用进 RL”，而是“把 PRM 用在它最可信的部分”。

这是整篇论文最有方法论价值的地方：它没有把 verifier 神化成完美 reward source，而是明确承认 PRM 有缺陷，再围绕这些缺陷设计更稳的接入方式。

## 4. 三个模型之间的前后置关系

如果只看最终产物，这篇论文最值得记住的三个模型是：

1. `URSA-8B`
2. `URSA-RM-8B`
3. `URSA-8B-PS-GRPO`

它们不是三条平行线，而是一条主干加一条支路：

```text
Qwen2.5-Math-Instruct + Hybrid Vision Tower
    ↓
Stage I
    ↓
URSA-8B
    ├── Stage II → URSA-RM-8B
    └── Stage III + URSA-RM-8B → URSA-8B-PS-GRPO
```

### 4.1 `URSA-8B`

角色：

- 基础生成模型
- Stage II 的数据来源
- Stage III 的初始 policy model

可以把它看成整篇论文的中心节点。

### 4.2 `URSA-RM-8B`

角色：

- verifier / PRM
- 从 `URSA-8B` 分支出来的奖励模型
- 不负责生成最终答案，而是负责打 step-level 分数

它与 `URSA-8B` 的关系，不是替代关系，而是辅助关系。

### 4.3 `URSA-8B-PS-GRPO`

角色：

- 经由 RL 后得到的更强 policy model
- 本质上还是生成模型

它不是直接从零训练，也不是直接从 `URSA-RM-8B` 派生，而是：

- 以 `URSA-8B` 为初始化
- 在 RL 过程中把 `URSA-RM-8B` 作为外部 verifier 使用
- 最终更新的是 policy model 本身

所以这三个模型之间的前后置关系可以总结成：

- `URSA-8B` 是基础
- `URSA-RM-8B` 是从 `URSA-8B` 出来的 verifier 支路
- `URSA-8B-PS-GRPO` 是 `URSA-8B` 在 `URSA-RM-8B` 辅助下继续优化后的主干结果

## 5. Stage III 复现时必须关注哪些输出

如果只是“调用 RM 打个分”，那只看 `min_score` / `avg_score` 就够了；但如果目标是复现论文里的 Stage III，就不够。

我认为最少需要保留这些量：

- 每一步的 `process_reward_sequence`
- rollout 级别的 `avg_process_reward`
- BoN 选优结果
- `drop_moment` 的相对下降序列与最大下降值
- PS-GRPO 的 `R^i`
- vanilla GRPO 的组内优势
- Variant 1 / Variant 2 的中间量

这正是本仓库当前 `rm_8b` 示例后来补齐的重点，详见：

- [report-2026-03-12.md](https://github.com/HansBug/URSA-MATH/blob/main/report-2026-03-12.md)
- [RUN_GUIDE.md](https://github.com/HansBug/URSA-MATH/blob/main/RUN_GUIDE.md)

## 6. 三个 Stage 的输入 / 输出 / 下游用途总表

| Stage | 目标 | 主要输入模型 | 主要输入数据 | 核心技术 | 主要输出 | 下游用途 |
|------|------|--------------|--------------|----------|----------|----------|
| Stage I | 训练强的多模态数学 policy | Qwen2.5-Math-Instruct + Hybrid Vision Tower | 对齐数据 + `MMathCoT-1M` | VL alignment + math-intensive SFT | `URSA-8B` | 生成 rollout；给 Stage II 提供轨迹；给 Stage III 提供初始策略 |
| Stage II | 训练 step-level verifier | `URSA-8B` | `URSA-8B` 生成的轨迹 + `DualMath-1.1M` | BEL + MIE + MCTS + step-level binary classification | `URSA-RM-8B` | 做 BoN 验证；给 Stage III 提供过程信号 |
| Stage III | 用 verifier 辅助在线 RL | `URSA-8B` + `URSA-RM-8B` | 过滤后的 RL 数据 + rollout groups | GRPO + drop-moment + process-as-outcome reward | `URSA-8B-PS-GRPO` | 最终更强的生成模型 |

如果只记一件事，我认为应该记这张表：Stage I 产生成熟 policy，Stage II 产生成熟 verifier，Stage III 再把二者组合起来优化 policy。

## 7. 训练时和推理时，这三个模型怎么配合

### 7.1 训练时的关系

训练链路可以拆成下面这条主线：

1. 先训练 `URSA-8B`
2. 再用 `URSA-8B` 产生轨迹、合成过程监督数据、训练 `URSA-RM-8B`
3. 最后把 `URSA-8B` 当 policy，把 `URSA-RM-8B` 当 verifier，用 PS-GRPO 做 RL，得到 `URSA-8B-PS-GRPO`

这里要注意一个细节：

- Stage II 更新的是 verifier 分支
- Stage III 更新的是 policy 分支

也就是说，在 Stage III 里，`URSA-RM-8B` 的职责主要是“提供过程判断”，不是继续跟着 policy 一起联合训练。

### 7.2 推理时的关系

推理侧其实有两种模式：

1. 单模型直接生成
2. 生成 + verifier 筛选

如果只用 `URSA-8B` 或 `URSA-8B-PS-GRPO`：

- 输入：图片 + 问题
- 输出：一条 reasoning trajectory + final answer

如果用 `URSA-8B` / `URSA-8B-PS-GRPO` 配 `URSA-RM-8B`：

- 先采样多条 rollout
- 再让 `URSA-RM-8B` 对每条 rollout 的每一步打分
- 用均值过程分数做 BoN 选优

所以从部署角度看：

- `URSA-8B` 和 `URSA-8B-PS-GRPO` 属于“答题模型”
- `URSA-RM-8B` 属于“验题模型”

## 8. 仓库里哪些代码分别对应三个 Stage

虽然仓库现在不是一个从头到尾一键重训的训练框架，但从目录职责看，还是能对上论文主线。

### 8.1 和 Stage I / 生成模型最相关的部分

- `models/`
- `examples/run_ursa_8b_torch_example.py`
- `examples/run_ursa_8b_torch_example_standalone.py`

这一部分体现的是：

- 多模态生成模型如何加载
- 图像特征如何进入语言模型
- 生成模型如何基于图片与问题输出 reasoning

### 8.2 和 Stage II / PRM 最相关的部分

- `inference/prm_infer_score.py`
- `examples/run_ursa_rm_8b_score_example.py`
- `examples/run_ursa_rm_8b_score_example_standalone.py`

这一部分体现的是：

- step-level 打分是怎么实现的
- 特殊标记 token 如何被用来对齐步骤
- PRM 最终怎样给出过程分数序列

### 8.3 和 Stage III / 复现信号最相关的部分

严格说，仓库示例脚本不是完整 RL trainer；但目前已经补齐了 Stage III 所需的关键观测量，集中体现在：

- `examples/run_ursa_rm_8b_score_example.py`
- `examples/run_ursa_rm_8b_score_example_standalone.py`
- [RUN_GUIDE.md](https://github.com/HansBug/URSA-MATH/blob/main/RUN_GUIDE.md)
- [report-2026-03-12.md](https://github.com/HansBug/URSA-MATH/blob/main/report-2026-03-12.md)

目前这部分已经能直接给出：

- `process_reward_sequence`
- BoN 选优结果
- drop-moment
- PS-GRPO reward
- vanilla GRPO / Variant 1 / Variant 2 中间量

也就是说，虽然这里不是完整训练器，但已经足够支持“算法机制复核”和“奖励信号复现”。

## 9. 这份阅读日志里，我最终想讲清楚的结论

截至当前 commit，我对 URSA-MATH 的结论是：

- Stage I 决定了后续 PRM 和 RL 的上限。
- Stage II 让验证器第一次具备了多模态数学过程监督能力。
- Stage III 的关键不是“让 PRM 直接当 reward”，而是“只提取 PRM 里相对可信的信号”。
- 三个模型不是平行关系，而是：
  - `URSA-8B` 是主干基础模型
  - `URSA-RM-8B` 是从主干分出来的 verifier
  - `URSA-8B-PS-GRPO` 是主干在 verifier 辅助下继续优化后的最终版本

如果后面还要继续深入，我建议直接沿着下面两个方向扩展：

- 把 Stage III 的观测量继续接到更完整的 RL 训练脚本
- 把当前 `rm_8b` 示例输出和论文公式、图表、算法伪代码做更严格的一一对应

## 10. 补充链接

这里只保留三个必要补充文档：

- 运行与输出字段说明：[RUN_GUIDE.md](https://github.com/HansBug/URSA-MATH/blob/main/RUN_GUIDE.md)
- 本次工作与验证记录：[report-2026-03-12.md](https://github.com/HansBug/URSA-MATH/blob/main/report-2026-03-12.md)
- 论文中文整理稿：[paper_zh.md](https://github.com/HansBug/URSA-MATH/blob/main/paper_zh.md)
