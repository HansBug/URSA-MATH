# 通过过程奖励模型解锁多模态数学推理

> arXiv:2501.04686v6 [cs.CL]，2025 年 10 月 5 日
> 第 39 届神经信息处理系统会议（NeurIPS 2025）

Ruilin Luo<sup>1,2,*</sup>, Zhuofan Zheng<sup>2,*</sup>, Yifan Wang<sup>1</sup>, Xinzhe Ni<sup>1</sup>, Zicheng Lin<sup>1</sup>, Songtao Jiang<sup>3</sup>, Yiyao Yu<sup>1</sup>, Chufan Shi<sup>1</sup>, Lei Wang<sup>4</sup>, Ruihang Chu<sup>1,†</sup>, Jin Zeng<sup>2,†</sup>, Yujiu Yang<sup>1</sup>

1. 清华大学  
2. 字节跳动  
3. 浙江大学  
4. 平安科技（深圳）有限公司

* 同等贡献。Ruilin 在字节跳动实习期间完成此项工作。  
† 通讯作者：ruihangchu@gmail.com，zengjin@bytedance.com

## 摘要

过程奖励模型（PRMs）通过测试时扩展（TTS）在增强大语言模型（LLMs）数学推理能力方面展现出潜力。然而，它们在多模态推理中的应用仍鲜有探索。本文迈出了将 PRM 引入多模态数学推理的第一步。我们识别出三个关键挑战：（i）高质量推理数据稀缺，限制了基础多模态大语言模型（MLLMs）的能力，也进一步压低了 TTS 和强化学习（RL）的上界；（ii）多模态场景中仍缺乏自动化的过程标注方法；（iii）在单模态 RL 中引入过程奖励会面临奖励黑客等问题，而这些问题可能延伸到多模态场景。为应对上述问题，我们提出 **URSA**，一个三阶段的多模态过程监督辅助训练框架。我们首先构建 MMathCoT-1M，这是一套高质量的大规模多模态思维链（CoT）推理数据集，用于打造更强的数学推理基础 MLLM：URSA-8B。随后，我们通过一套强调逻辑正确性与感知一致性的自动流程合成过程监督数据，并引入 DualMath-1.1M 以支持 URSA-8B-RM 的训练。最后，我们提出 **PS-GRPO**，一种优于普通 GRPO 的多模态 PRM 辅助在线 RL 方法。采用 PS-GRPO 后，URSA-8B-PS-GRPO 在 6 个基准上的平均表现分别超过 Gemma3-12B 和 GPT-4o 8.4% 与 2.7%。代码、数据和检查点见：https://github.com/URSA-MATH。

## 1 引言

继大语言模型（LLMs）在数学推理方面取得实质性进展[1, 2, 3, 4, 5, 6, 7, 8]之后，多模态大语言模型（MLLMs）的数学推理能力也日益受到关注[9, 10, 11, 12, 13]。以前的工作通常集中在数学推理数据整理[14, 15, 16, 17, 18]、训练数学密集型视觉编码器[19, 20]、增强视觉语言对齐[11, 21]或训练后技术[22, 23, 24, 13]的应用等方面。鉴于过程奖励模型 (PRMs) 通过测试时间缩放 (TTS) [25, 26] 和强化微调 (ReFT) [27, 28] 等方法在改进 LLM 推理方面取得了成功，PRMs 在多模态推理中的应用仍有待探索。

![图1](paper_assets/2501.04686v6/x1.png)

*图 1：与领先的开源 MLLMs 和 GPT-4o 的性能比较。*

在这项工作中，我们迈出了将 PRMs 集成到多模态数学推理中的第一步。我们确定了三个关键挑战：（i）由于TTS和RL都深受基础模型[29, 25]强度的影响，大规模、高质量推理数据的有限可用性限制了当前MLLMs的上限，并削弱了PRM集成的有效性； (ii) 尚未有足够的自动化过程标注技术合并到多模态环境中，应强调逻辑有效性和感知一致性[30, 31, 32]。 (iii) 虽然PRMs可以在TTS中有效使用，但直接在在线 RL 中应用它们会带来奖励黑客和奖励[33, 34]中的长度偏差等风险。

为应对这些挑战，我们提出 **URSA** 框架，一个三阶段的多模态过程监督辅助训练流程，覆盖多模态 PRM 的构建与应用。在第一阶段，我们整理得到 **MMathCoT-1M**，这是一套由 143 万个开源样本合成的大规模高质量多模态思维链数据集，可通过有针对性的指令微调增强基础模型的推理能力。在第二阶段，我们通过双视角过程监督数据合成策略构建 **DualMath-1.1M**，该策略结合二分错误定位引擎与误解插入引擎，为逻辑有效性和视觉对齐提供互补信号，并用于训练过程奖励模型。在第三阶段，我们分析了在线 RL 中标量过程奖励建模的局限性，并提出 **PS-GRPO**，通过在策略优化过程中隐式惩罚过程级不一致，缓解奖励黑客和 PRM 的长度偏差。

6 个多模态推理基准的结果表明，我们的 PRM 改进了 Best-of-N 验证，超越了自我一致性和基于结果的基线。当在 PS-GRPO 中使用时，生成的模型在类似规模的开源 MLLMs 中实现了最先进的性能。我们的贡献如下：

- 我们发布了两个大型开源数据集MMathCoT-1M和DualMath-1.1M，以解决高质量多模态CoT推理和过程监督数据的稀缺问题。
- 我们提出了 PS-GRPO，这是一种在线强化学习算法。它通过比较采样轨迹的相对质量来引入多模态 PRM，而不是依赖标量奖励建模，从而有效缓解 PRM 的奖励黑客问题和长度偏差。
- 实验结果表明，我们的奖励模型同时提升了测试时验证和在线训练效果。采用 PS-GRPO 后（图 1），URSA-8B-PS-GRPO 在 6 个基准上的平均表现分别超过 Gemma3-12B 和 GPT-4o 8.4% 与 2.7%。

## 2 第一阶段：数学密集型对齐与指令微调

### 2.1 视觉-语言对齐数据收集

我们采用类 LLaVA 架构，首先直接从现有开源数据集 [35, 36, 37, 38] 收集视觉语言对齐数据。如图 2 所示，我们从 Multimath [23]、MAVIS [19] 和 Geo170K [18] 收集 URSA-Alignment-860K。随后，我们过滤掉说明文字过于冗长的样本，得到一个 860K 的数学密集型对齐数据集。遵循先前工作的工程实践，我们仅在对齐阶段训练 MLP 投影器。

### 2.2 CoT 推理数据合成

为了构建强大的基础模型，我们从现有数学推理数据集中收集了 143 万个样本，用于支撑大规模 CoT 推理数据的构建。如图 2 所示，数据来源于 MathV360K [15]、Multimath [23]、MAVIS [19]、Geo170K [18] 和 VarsityTutors [11]。根据解答形式，我们将数据分为 *仅答案*、*分析格式* 和 *CoT 格式* 三类，并分别采用不同的合成策略来整理高质量的 CoT 推理轨迹。我们使用 Gemini-1.5-Flash-002（见下文的 $\mathcal{G}$）作为一种经济高效的数据整理工具，以避免昂贵的大规模人工标注。

![图2](paper_assets/2501.04686v6/x2.png)

*图2：URSA-Alignment-860K和MMathCoT-1M的统计数据。*

#### CoT 扩展。

对于 *仅答案* 数据 $\mathcal{D}_{1}=\{(x_{i},y_{i})\}_{i=1}^{N_{1}}$，例如 MathV360K [15]，每个样本都包含一个问题 $x_{i}$ 和一个真实答案 $y_{i}$。此类数据在以往的快思考推理模式工作中被大量使用[15, 11, 16]。然而，仅答案训练会限制模型对解题过程的完整把握，可能导致基于记忆的推理，从而削弱模型直接解答更复杂推理问题的能力[39]。我们为这类数据扩展生成了一部分 CoT 推理轨迹。给定扩展提示 $\mathcal{P}_{\mathcal{C}}$，我们提供 $x_{i}$ 和 $y_{i}$，然后提示 $\mathcal{G}$ 输出导向答案 $y_{i}$ 的推理轨迹，从而得到扩展解答 $\mathcal{S}_{Ao}=\mathcal{G}(\mathcal{P}_{\mathcal{C}};\{x_{i},y_{i}\}_{i=1}^{N_{1}})$。

#### 重写。

此策略专为 *分析格式* 样本设计，表示为 $\mathcal{D}_{2}=\{(x_{i},y_{i},a_{i})\}_{i=1}^{N_{2}}$。这包括 MAVIS-Geo、MAVIS-MetaGen [19]、VarsityTutors [11] 和 Geo170K-QA [40] 等数据集。每个样本包含一个问题$x_{i}$、一个答案$y_{i}$和文本分析$a_{i}$。虽然此类数据提供了演练，但它通常存在两个问题：（i）它缺乏严格的逐步逻辑，表现出语言或推理的跳跃。 (ii) 很大一部分答案相对简短，无法提供丰富的理由。给定重写提示$\mathcal{P}_{\mathcal{R}}$，我们利用$\mathcal{G}$来转录这些解决方案，从而增强它们的逐步推理轨迹和语言多样性，从而得到重写集$\mathcal{S}_{An}=\mathcal{G}(\mathcal{P}_{\mathcal{R}};\{x_{i},y_{i},a_{i}\}_{i=1}^{N_{2}})$。

#### 格式统一。

该策略用于*CoT 格式*数据，主要来源于Multimath-EN-300K[23]，它是从K-12教科书收集的，包含数学语言和符号式推理解决方案。这部分数据 $\mathcal{D}_{3}=\{(x_{i},y_{i},c_{i})\}_{i=1}^{N_{3}}$ 由问题 $x_{i}$、答案 $y_{i}$ 和解决方案 $c_{i}$ 组成。我们使用提示符 $\mathcal{P}_{\mathcal{F}}$ 和 $\mathcal{G}$ 通过自然语言风格化来统一格式，生成统一的集合 $\mathcal{S}_{C}=\mathcal{G}(\mathcal{P}_{\mathcal{F}};\{x_{i},y_{i},c_{i}\}_{i=1}^{N_{3}})$。

![图3](paper_assets/2501.04686v6/x3.png)

*图 3：URSA 的整体流程。第 1 阶段展示了第 2 节所述的数据整理工作流。第 2 阶段说明二分错误定位和误解插入如何推动过程监督数据的自动构建。第 3 阶段展示了 PS-GRPO 如何对受到 PRM 质疑的采样轨迹施加惩罚。*

#### MMathCoT-1M。

最后，我们过滤掉以下实例：(i) 违反正确性：生成的内容改变了原始答案，或 (ii) 一致性有问题：解决方案包含质疑原始答案或做出新假设来强制给定答案的文本。此过程产生MMathCoT-1M。完整的提示设计可以在附录 G 中找到。

我们基于对齐的模型，使用 MMathCoT-1M 进行全参数指令微调来训练 URSA-8B。 SFT 数据集 $\mathcal{D}_{SFT}$ 由策划的解决方案（即 $\mathcal{D}_{SFT}=\{(x_{i},y_{i})\mid(x_{i},y_{i})\in\mathcal{S}_{Ao}\cup\mathcal{S}_{An}\cup\mathcal{S}_{C}\}$）合并而成。训练目标如公式 1 所示。

$$
\mathcal{L}_{SFT} =-\mathbb{E}_{(x,y)\sim\mathcal{D}_{SFT}}\sum_{t=1}^{T}\log\mathcal{M}(y_{t}|x,y_{<t})
$$
<div align="right">(1)</div>

在这一阶段，我们构建了更强的推理基础模型 URSA-8B，期望在测试时获得更高上界，并能处理更多样的监督数据。

## 3 第二阶段：双视角过程监督数据合成

### 3.1 二分错误定位引擎

根据之前工作 [41, 42, 43] 的建议，我们训练了一个 PRM 来识别第一个错误步骤。我们从 URSA-8B 对 MMathCoT-1M 的零样本推理中收集了 $\sim$553K 个错误解答。这些解答中的错误步骤使用蒙特卡罗树搜索 (MCTS) 进行标记。对于 MCTS，操作 $\mathcal{F}(\{s_{1},\ldots,s_{i}\},N)$ 从推理前缀 $\{s_{1},\ldots,s_{i}\}$ 生成 $N$ 条采样轨迹。单步的蒙特卡罗估计值 $mc_{i}$ 是这些采样轨迹导向正确答案的比例：

$$
mc_{i}=\frac{|\text{来自 $\mathcal{F}(\{s_{1},s_{2},\ldots,s_{i}\},N)$ 的正确采样轨迹数}|}{|\text{来自 $\mathcal{F}(\{s_{1},s_{2},\ldots,s_{i}\},N)$ 的总采样轨迹数}|}
$$
<div align="right">(2)</div>

如果 $mc_{i}>0$ [43, 42]，则步骤 $s_{i}$ 被视为“可能正确”。我们使用二分错误定位引擎（BEL）来优化首个错误步骤的识别：如果中间步骤具有正的 $mc$（即 $mc_{mid}>0$），则错误位于后半段；否则位于前半段（见算法 1）。为减轻步骤级标签偏差并纳入正样本，我们额外加入了 $\sim$180K 条正确解答（约为错误解答数量的 1/3），其所有步骤都可直接标记为“True”。最终得到 $\mathcal{S}_{BEL}$，一个基于正确性潜力的 773K 过程标注数据集。

### 3.2 误解插入引擎

除了逻辑错误之外，推理步骤中图像和文本之间的感知不一致是多模态场景中的一个独特问题[30, 44, 45]。我们提出了一种误解插入引擎（MIE）来人为地插入幻觉信息，从插入点开始自动构建具有错误推理路径的过程监督数据。具体来说，MIE包括三个步骤。首先，我们提示$\mathcal{G}$执行字幕任务，尽可能从图像中提取数学范式信息。其次，模型$\mathcal{G}$需要关注现有正确解决方案中可能容易混淆的条件，并使用相邻或相似的条件对其进行修改。最后，提示模型$\mathcal{G}$根据插入错误的步骤继续推理。我们利用 $\mathcal{G}$ 强大的指令跟踪能力，指示它自动为错误插入后的每个后续步骤分配负标签。我们使用此策略生成 $\sim$302K 样本$\mathcal{S}_{MIE}$。 MIE的案例可以在附录H.2中找到。

### 3.3 PRM 训练

如公式 3 所示，我们合并两种类型的数据，提出一个名为 DualMath-1.1M 的 $\sim$1.1M 过程监督数据。在训练过程中，我们在每个步骤后添加一个特殊的标记来指示其预测的正确性。我们将 PRM 训练建模为二元分类任务，以确保每个步骤的正确性，如公式 4 所示，这里 $\pi_{p}$ 是基于 URSA-8B 训练的 PRM。 $e_{j}$ 和 $y_{j}$ 代表单步和相应的标签 ($y_{j}\in\{0,1\}$)。

$$
\mathcal{D}_{PRM}=\{(e,y_{e})\sim\mathcal{S}_{BEL}\cup\mathcal{S}_{MIE}\}
$$
<div align="right">(3)</div>

$$
\mathcal{L}_{PRM}=-\mathbb{E}_{(e,y)\sim\mathcal{D}_{PRM}}\sum_{j=1}^{|e|}\Big[y_{j}\log\pi_{p}(e_{j})+(1-y_{j})\log(1-\pi_{p}(e_{j}))\Big]
$$
<div align="right">(4)</div>

因此，阶段 II 产出了 URSA-8B-RM，这是一个基于 DualMath-1.1M 训练得到的强大 PRM，而 DualMath-1.1M 也是面向多模态推理过程监督的**首个**大规模自动标注数据集。虽然 BoN 评估证明了 PRM 在 TTS 中的价值，但一个关键问题随之出现：如何将其指导信号直接整合到 MLLM 的后训练中？这一问题在很大程度上仍属空白。阶段 III 则从以往标量过程奖励建模为何失效的经验中吸取教训，并通过“过程即结果”的奖励建模取得了有效进展。

## 4 第三阶段：将多模态 PRM 融入强化学习

受到 DeepSeek-R1 [46] 等成功的启发，最近的几项研究尝试将基于结果奖励的 GRPO 应用于多模态推理，展示了显着的进展 [47, 48, 49, 50]。基于结果奖励的 GRPO 通过标准化组内奖励来计算第 $i$ 个响应的优势。然而，基于结果奖励的GRPO忽略了推理过程[41, 51, 52]的质量。

![图4](paper_assets/2501.04686v6/x4.png)

*图 4：图 (a)-(d) 分别说明了普通 GRPO 和第 4 节中提出的两个变体的训练奖励、响应长度、响应步数和测试集准确性。测试集是从 MMathCoT-1M 中随机选择 500 个示例进行域内评估。*

遵循 RL [43, 28, 46, 13, 53] 中常见的响应级与步骤级奖励建模方式，我们考察了两种集成标量过程奖励的 GRPO 简单变体，以揭示训练过程中的失败模式 [54]。*变体 1*：对于第 $i$ 个采样轨迹，奖励是结果奖励与平均过程奖励之和，即 $r^{i}=r_{o}^{i}+\bar{r_{s}^{i}}$。*变体 2*：在结果奖励之外，再将标量过程奖励 $r_{s,t}^{i}$ 分配给第 $i$ 个采样轨迹的第 $t$ 个步骤。

![图5](paper_assets/2501.04686v6/x5.png)

*图 5：图 (a) 展示了 GRPO 训练过程中的 BoN 评估，我们使用过程奖励的均值选择最佳采样轨迹。图 (b) 展示了被 URSA-8B-RM 标记为“下降时刻”且最终答案确实错误的采样轨迹占比。图 (c) 和图 (d) 展示了 PS-GRPO 训练期间的响应长度与测试精度。*

我们从图 4 中观察到两个非常重要的结论：**(i)** *对奖励黑客的高度敏感*。两种变体的测试精度均低于普通 GRPO。这表明，当过程标量奖励被直接用作学习目标时，模型会迅速学会迎合“过程正确性”的策略。然而，过程看似正确并不必然与通向真实答案的启发式完全一致。**(ii)** *PRM 奖励的长度偏差*。我们观察到，随着训练推进，模型响应会变得更短、推理步骤更少。这一现象源于 PRM 训练标签中固有的长度偏差：对于答案错误的样本，第一次出错之后的步骤通常不再可能导向正确解。这会使 PRM 更保守地奖励推理采样轨迹的后期阶段，从而鼓励 MLLM 采取更被动的推理方式，依赖现有条件的模式识别或更简单的启发式。

#### PS-GRPO

上述发现进一步印证了这样一个判断：当标量过程奖励被直接作为优化目标时，奖励函数中的缺陷会被显著放大[55, 33]。因此，我们提出一个问题：“PRM 的哪些内部信号值得信任？”我们从两个视角考察 PRM 的可靠区域：其一是在线学习过程中的 BoN 表现，其二是 PRM 的错误识别能力。针对后者，我们在 PRM 的奖励序列中引入 *“下降时刻”* 的概念，用以表示 PRM 开始质疑前序步骤的有效性。具体来说，对于给定解答的 PRM 奖励序列 $\{r_{p1}^{i},r_{p2}^{i},\cdots,r_{pN}^{i}\}$，若相邻步骤之间的奖励出现显著下降，则认为发生了下降时刻。

$$
\delta_{p}^{i}=\max\left\{\frac{r_{p,j}^{i}-r_{p,j+1}^{i}}{r_{p,j}^{i}} \mid j=0,1,\dots,N-1\right\}>\rho
$$
<div align="right">(5)</div>

这里，$\rho$ 表示 PRM 的下降时刻阈值。如图 5 所示，PRM 在在线 RL 过程中用于 BoN 选择与错误识别的能力基本未受损，表现出稳定性能。这表明：*尽管 PRM 在在线 RL 中给出的标量奖励可能并不可靠，但它揭示出的解答相对质量仍然具有较高可信度*。

我们利用这一有益特性来缓解 GRPO [56, 57, 58] 中的奖励稀疏问题，使在线 RL 更聚焦于从结果正确且过程严谨的采样轨迹中学习。我们使用公式 5 中的 $\rho$ 作为“下降时刻”的触发阈值；当该现象出现时，我们会对结果正确的采样轨迹施加奖励惩罚 $\gamma$。这样既能区分不同结果正确采样轨迹的学习价值，又因其关注的是奖励序列的相对下降，而规避了 PRM 奖励长度偏差的影响。

$$
R^{i}=
\begin{cases}
1, & o^{i}\text{ 正确且 }\delta_{p}^{i}<\rho \\
1-\gamma, & o^{i}\text{ 正确且 }\delta_{p}^{i}\ge\rho \\
0, & \text{否则}
\end{cases}
$$
<div align="right">(6)</div>

我们利用等式 6 中的奖励建模来执行过程监督 GRPO，这有助于计算等式 7 中的组内优势。

*表 1：6 个数学推理基准的性能比较。我们使用 MathVerse、MathVision、MathVista 和 GeoQA 的准确率。我们在 WE-MATH 上使用分数（宽松）。 DYNAMATH 采用平均情况精度。闭源 MLLMs 的最佳结果以绿色突出显示。开源 MLLMs 的最佳和亚军成绩以红色和蓝色突出显示。*

<table>
<tbody>
<tr>
<th></th>
<td align="center" rowspan="2">尺寸</td>
<td align="center" rowspan="2">平均</td>
<td align="center">MathVerse</td>
<td align="center">MathVision</td>
<td align="center">MathVista</td>
<td align="center">WE-MATH</td>
<td align="center">DYNAMATH</td>
<td align="center">GeoQA</td>
</tr>
<tr>
<th></th>
<td align="center">testmini</td>
<td align="center">full set</td>
<td align="center">gps</td>
<td align="center">testmini</td>
<td align="center">testmini</td>
<td align="center">full set</td>
</tr>
<tr>
<th align="center" colspan="9">闭源 MLLMs</th>
</tr>
<tr>
<th>GPT-4o [59]</th>
<td align="center">-</td>
<td align="center">55.5</td>
<td align="center">50.2</td>
<td align="center">30.4</td>
<td align="center">64.7</td>
<td align="center">62.8</td>
<td align="center">64.9</td>
<td align="center">62.1</td>
</tr>
<tr>
<th>GPT-4o-mini[59]</th>
<td align="center">-</td>
<td align="center">49.2</td>
<td align="center">42.3</td>
<td align="center">22.8</td>
<td align="center">59.9</td>
<td align="center">56.3</td>
<td align="center">53.5</td>
<td align="center">60.1</td>
</tr>
<tr>
<th>Gemini-1.5-pro [60]</th>
<td align="center">-</td>
<td align="center">53.2</td>
<td align="center">35.3</td>
<td align="center">19.2</td>
<td align="center">81.7</td>
<td align="center">66.9</td>
<td align="center">60.5</td>
<td align="center">55.5</td>
</tr>
<tr>
<th align="center" colspan="9">开源通用 MLLMs</th>
</tr>
<tr>
<th>InternVL-Chat-V1.5 [61]</th>
<td align="center">26B</td>
<td align="center">33.6</td>
<td align="center">26.1</td>
<td align="center">15.4</td>
<td align="center">56.9</td>
<td align="center">32.7</td>
<td align="center">36.7</td>
<td align="center">33.5</td>
</tr>
<tr>
<th>Llama-3.2-11B-Vision-Instruct [62]</th>
<td align="center">11B</td>
<td align="center">28.0</td>
<td align="center">28.9</td>
<td align="center">16.9</td>
<td align="center">40.9</td>
<td align="center">12.0</td>
<td align="center">32.2</td>
<td align="center">36.9</td>
</tr>
<tr>
<th>Qwen2-VL [63]</th>
<td align="center">8B</td>
<td align="center">40.2</td>
<td align="center">33.6</td>
<td align="center">19.2</td>
<td align="center">51.0</td>
<td align="center">43.0</td>
<td align="center">42.1</td>
<td align="center">52.2</td>
</tr>
<tr>
<th>InternVL2-8B [64]</th>
<td align="center">8B</td>
<td align="center">41.8</td>
<td align="center">37.0</td>
<td align="center">18.4</td>
<td align="center">57.7</td>
<td align="center">44.9</td>
<td align="center">39.7</td>
<td align="center">52.8</td>
</tr>
<tr>
<th>InternVL2-8B-MPO [65]</th>
<td align="center">8B</td>
<td align="center">45.1</td>
<td align="center">38.2</td>
<td align="center">22.3</td>
<td align="center">69.2</td>
<td align="center">44.4</td>
<td align="center">40.5</td>
<td align="center">55.9</td>
</tr>
<tr>
<th>InternVL2.5-8B [66]</th>
<td align="center">8B</td>
<td align="center">45.2</td>
<td align="center">39.5</td>
<td align="center">19.7</td>
<td align="center">64.9</td>
<td align="center">44.7</td>
<td align="center">40.5</td>
<td align="center">61.6</td>
</tr>
<tr>
<th>LLaVA-OneVision [35]</th>
<td align="center">8B</td>
<td align="center">40.9</td>
<td align="center">28.9</td>
<td align="center">18.3</td>
<td align="center">71.6</td>
<td align="center">44.9</td>
<td align="center">37.5</td>
<td align="center">43.9</td>
</tr>
<tr>
<th>Points-Qwen2.5-Instruct [67]</th>
<td align="center">8B</td>
<td align="center">49.8</td>
<td align="center">41.1</td>
<td align="center">23.9</td>
<td align="center">76.0</td>
<td align="center">51.0</td>
<td align="center">42.8</td>
<td align="center">63.8</td>
</tr>
<tr>
<th>Gemma3-12B [68]</th>
<td align="center">12B</td>
<td align="center">49.8</td>
<td align="center">40.1</td>
<td align="center">29.1</td>
<td align="center">63.6</td>
<td align="center">51.7</td>
<td align="center">45.8</td>
<td align="center">67.7</td>
</tr>
<tr>
<th align="center" colspan="9">开源推理 MLLMs</th>
</tr>
<tr>
<th>数学-LLaVA [15]</th>
<td align="center">13B</td>
<td align="center">35.2</td>
<td align="center">22.9</td>
<td align="center">15.7</td>
<td align="center">57.7</td>
<td align="center">31.3</td>
<td align="center">35.5</td>
<td align="center">48.1</td>
</tr>
<tr>
<th>MathPUMA-Qwen2-7B [11]</th>
<td align="center">8B</td>
<td align="center">39.6</td>
<td align="center">33.6</td>
<td align="center">14.0</td>
<td align="center">48.1</td>
<td align="center">41.0</td>
<td align="center">37.3</td>
<td align="center">63.6</td>
</tr>
<tr>
<th>多元数学 [23]</th>
<td align="center">7B</td>
<td align="center">43.1</td>
<td align="center">27.7</td>
<td align="center">16.3</td>
<td align="center">66.8</td>
<td align="center">42.2</td>
<td align="center">37.9</td>
<td align="center">67.7</td>
</tr>
<tr>
<th>MAVIS [19]</th>
<td align="center">7B</td>
<td align="center">44.4</td>
<td align="center">35.2</td>
<td align="center">18.5</td>
<td align="center">64.1</td>
<td align="center">44.3</td>
<td align="center">36.2</td>
<td align="center">68.3</td>
</tr>
<tr>
<th>InfiMM-数学 [14]</th>
<td align="center">7B</td>
<td align="center">48.6</td>
<td align="center">40.5</td>
<td align="center">18.8</td>
<td align="center">77.3</td>
<td align="center">48.3</td>
<td align="center">38.2</td>
<td align="center">68.3</td>
</tr>
<tr>
<th>AtomThink-EMOVA [12]</th>
<td align="center">8B</td>
<td align="center">49.5</td>
<td align="center">42.5</td>
<td align="center">24.9</td>
<td align="center">75.9</td>
<td align="center">49.3</td>
<td align="center">40.9</td>
<td align="center">63.8</td>
</tr>
<tr>
<th>MathGLM-视觉 [9]</th>
<td align="center">9B</td>
<td align="center">47.6</td>
<td align="center">44.2</td>
<td align="center">19.2</td>
<td align="center">64.2</td>
<td align="center">45.2</td>
<td align="center">42.2</td>
<td align="center">70.4</td>
</tr>
<tr>
<th>LlamaV-o1 [69]</th>
<td align="center">11B</td>
<td align="center">38.4</td>
<td align="center">33.9</td>
<td align="center">17.9</td>
<td align="center">53.3</td>
<td align="center">42.6</td>
<td align="center">34.7</td>
<td align="center">43.1</td>
</tr>
<tr>
<th>OpenVLThinker [70]</th>
<td align="center">7B</td>
<td align="center">-</td>
<td align="center">47.9</td>
<td align="center">25.3</td>
<td align="center">76.4</td>
<td align="center">-</td>
<td align="center">-</td>
<td align="center">-</td>
</tr>
<tr>
<th>R1-Onevision [71]</th>
<td align="center">7B</td>
<td align="center">-</td>
<td align="center">47.4</td>
<td align="center">26.9</td>
<td align="center">72.4</td>
<td align="center">51.4</td>
<td align="center">-</td>
<td align="center">-</td>
</tr>
<tr>
<th>URSA-8B</th>
<td align="center">8B</td>
<td align="center">54.7</td>
<td align="center">45.7</td>
<td align="center">28.7</td>
<td align="center">81.7</td>
<td align="center">53.6</td>
<td align="center">44.7</td>
<td align="center">73.5</td>
</tr>
<tr>
<th>URSA-8B-PS-GRPO</th>
<td align="center">8B</td>
<td align="center">58.2</td>
<td align="center">50.9</td>
<td align="center">31.5</td>
<td align="center">83.2</td>
<td align="center">60.7</td>
<td align="center">47.4</td>
<td align="center">75.6</td>
</tr>
</tbody>
</table>

*表 2：URSA-8B 上的 TTS 和使用 BoN 的 AtomThink-EMOVA 性能比较。*

<table>
<thead>
<tr>
<th rowspan="2">模型</th>
<th rowspan="2">方法</th>
<th align="center" colspan="4">MathVerse</th>
<th align="center" colspan="4">MathVista-GPS</th>
<th align="center" colspan="4">MathVision</th>
</tr>
<tr>
<th align="center">N=4</th>
<th align="center">N=8</th>
<th align="center">N=16</th>
<th align="center">N=32</th>
<th align="center">N=4</th>
<th align="center">N=8</th>
<th align="center">N=16</th>
<th align="center">N=32</th>
<th align="center">N=4</th>
<th align="center">N=8</th>
<th align="center">N=16</th>
<th align="center">N=32</th>
</tr>
</thead>
<tbody>
<tr>
<th rowspan="3">URSA-8B</th>
<th>Self-Consistency</th>
<td align="center">49.3</td>
<td align="center">50.1</td>
<td align="center">50.7</td>
<td align="center">50.7</td>
<td align="center">82.7</td>
<td align="center">83.9</td>
<td align="center">84.8</td>
<td align="center">85.4</td>
<td align="center">29.4</td>
<td align="center">31.9</td>
<td align="center">32.8</td>
<td align="center">33.1</td>
</tr>
<tr>
<th>InternVL2.5-8B ORM</th>
<td align="center">48.6</td>
<td align="center">50.9</td>
<td align="center">51.8</td>
<td align="center">51.3</td>
<td align="center">82.5</td>
<td align="center">83.3</td>
<td align="center">84.3</td>
<td align="center">85.1</td>
<td align="center">29.9</td>
<td align="center">32.1</td>
<td align="center">32.8</td>
<td align="center">33.5</td>
</tr>
<tr>
<th>URSA-8B-RM</th>
<td align="center">53.3</td>
<td align="center">54.2</td>
<td align="center">54.7</td>
<td align="center">55.0</td>
<td align="center">83.2</td>
<td align="center">85.5</td>
<td align="center">86.5</td>
<td align="center">87.2</td>
<td align="center">31.6</td>
<td align="center">33.1</td>
<td align="center">34.0</td>
<td align="center">35.1</td>
</tr>
<tr>
<th rowspan="3">AtomThink-EMOVA</th>
<th>Self-Consistency</th>
<td align="center">45.9</td>
<td align="center">46.7</td>
<td align="center">47.1</td>
<td align="center">47.3</td>
<td align="center">76.8</td>
<td align="center">77.9</td>
<td align="center">78.6</td>
<td align="center">79.0</td>
<td align="center">25.3</td>
<td align="center">26.8</td>
<td align="center">27.6</td>
<td align="center">28.0</td>
</tr>
<tr>
<th>InternVL2.5-8B ORM</th>
<td align="center">45.7</td>
<td align="center">45.6</td>
<td align="center">46.4</td>
<td align="center">46.1</td>
<td align="center">76.6</td>
<td align="center">77.7</td>
<td align="center">78.3</td>
<td align="center">79.2</td>
<td align="center">26.0</td>
<td align="center">26.6</td>
<td align="center">27.2</td>
<td align="center">27.8</td>
</tr>
<tr>
<th>URSA-8B-RM</th>
<td align="center">48.0</td>
<td align="center">48.8</td>
<td align="center">49.3</td>
<td align="center">49.6</td>
<td align="center">78.0</td>
<td align="center">79.6</td>
<td align="center">80.5</td>
<td align="center">81.0</td>
<td align="center">27.5</td>
<td align="center">29.0</td>
<td align="center">30.2</td>
<td align="center">31.0</td>
</tr>
</tbody>
</table>

## 5 实验

### 5.1 实验设置

#### 基准测试

我们在 6 个广泛使用的推理基准上评估我们的 URSA 系列模型，包括 MathVerse [72]、DYNAMATH [73]、MathVista [74]、WE-MATH [75]、GeoQA [40] 和 MathVision [43]。详细描述和评价标准可参见附录F.3。我们一贯采用零样本推理进行比较。

#### 基线模型

我们纳入了一些领先的专有 MLLM，如 GPT-4o 和 GPT-4o-mini [59]。对于规模相当的开源 MLLM，我们选择 InternVL 系列 [64, 76]、LLaVA-OneVision [35]、Gemma3-12B [68]、Qwen2-VL [63] 等。对于专门面向数学推理的 MLLM，我们选择 AtomThink [12]、InfiMM-Math [14]、MAVIS [19]、MathGLM-Vision [9]、LlamaV-o1 [69]。这类工作主要关注 STEM 推理数据或类 o1 的慢思考训练。为保证公平性，我们不选择那些将 MathVision 作为训练集的基线，例如 Mulberry-Qwen2-VL-7B [77] 和 MAmooTH-VL [78]。在 PRM 的 TTS 性能比较中，我们选择 Self-Consistency [79] 和开源 MLLM 充当 ORM，例如 InternVL2.5-8B [64]。

#### 实现细节

URSA 使用 SAM-B+SigLIP-L 作为混合视觉编码器，使用 Qwen2.5-Math-Instruct 作为 LLM 主干。我们采用两层 MLP 连接进行视觉语言对齐训练。我们为 PS-GRPO 选择 MMathCoT-1M 中的 15K 数据。公式 6 中的 $\gamma$ 和 $\rho$ 分别设置为 0.5 和 0.3。有关模块选择、数据选择、超参数和时间成本的详细信息位于附录 D 和 F 中。

### 5.2 主要结果

#### 最先进性能

在表 1 中，我们展示了 URSA-8B 与 URSA-8B-PS-GRPO 的性能。首先，URSA-8B 提供了更强的推理基础能力，相比专注于“慢思考”训练的 AtomThink-EMOVA 提高了 5.2 分，也优于同等规模的领先通用 MLLM，如 Gemma3-12B 和 InternVL2.5-8B。URSA-8B-PS-GRPO 在 6 个基准上的平均表现优于 GPT-4o，并在 MathVista-GPS（83.2 vs 62.6）和 GeoQA（73.5 vs 62.1）上表现出显著优势，同时首次在 MathVision 上实现超越（31.5 vs 30.4）。不过，DynaMath 上仍存在显著差距，这表明小规模 MLLM 在更强的问题求解能力方面仍有不足。与领先的数学推理 MLLM AtomThink-EMOVA-8B 和通用 MLLM Gemma3-12B 相比，我们的模型平均性能分别高出 **8.5%** 和 **8.2%**。与近期受 R1 启发的方法 OpenVLThinker [70] 和 R1-OneVision [71] 相比，我们在 MathVision 和 WE-MATH 上仍表现出显著优势。

#### 有效的 Best-of-N 评估

在表 2 中，我们展示了 URSA-8B-RM 相对于自我一致性的优势，以及面向 TTS [43, 42] 的 ORM 基线表现。我们发现，自我一致性仍然是一个强基线，而 InternVL2.5-8B 作为 ORM 并不能稳定超越它。相比之下，URSA-8B-RM 展现出更有效的 BoN 评估能力，并体现出对 AtomThink-EMOVA-8B 的良好泛化。进一步地，仅使用 4 次采样并以 URSA-8B-RM 作为验证器，就能在 URSA-8B 的基础上带来显著提升。具体而言，它在 MathVerse 和 MathVision 上分别提升了 16.6% 和 10.1%。在 Best-of-32 设置下，URSA-8B 在 MathVision 和 MathVerse 上分别达到 35.1 和 55.0，表现出相较 GPT-4o 的明显优势。

#### PS-GRPO 与普通 GRPO 的对比

如图 6(a) 所示，在训练数据、超参数和采样轨迹数量相同的条件下，PS-GRPO 在平均性能上取得了更高提升（6.8% vs 3.1%）。PS-GRPO 在 WE-MATH 上的提升几乎是普通 GRPO 的两倍，在更具挑战性的 MathVision 上也表现出更明显的改进，体现了其有效性。我们注意到，RL 对 MathVista-GPS 和 GeoQA 的改进相对较小，这是因为 URSA-8B 在这两个基准上的固有能力已经接近上限。然而，PS-GRPO 仍然优于普通 GRPO。

![图6](paper_assets/2501.04686v6/x6.png)

*图 6：图 (a) 展示了 URSA-8B 的相对性能提升；图 (b) 展示了各训练阶段对总体性能的贡献。*

*表 3：DualMath-1.1M 的消融研究（BoN 评估）。 w/o $\mathcal{S}_{MIE}$ 和 w/o $\mathcal{S}_{BEL}$ 表示丢弃 DualMath-1.1M 的一部分来训练 PRM。*

<table>
<thead>
<tr>
<th rowspan="2">模型</th>
<th rowspan="2">数据集</th>
<th align="center" colspan="4">MathVerse</th>
<th align="center" colspan="4">MathVista-GPS</th>
<th align="center" colspan="4">MathVision</th>
</tr>
<tr>
<th align="center">N=4</th>
<th align="center">N=8</th>
<th align="center">N=16</th>
<th align="center">N=32</th>
<th align="center">N=4</th>
<th align="center">N=8</th>
<th align="center">N=16</th>
<th align="center">N=32</th>
<th align="center">N=4</th>
<th align="center">N=8</th>
<th align="center">N=16</th>
<th align="center">N=32</th>
</tr>
</thead>
<tbody>
<tr>
<th rowspan="3">URSA-8B</th>
<th>DualMath-1.1M</th>
<td align="center">53.3</td>
<td align="center">54.2</td>
<td align="center">54.7</td>
<td align="center">55.0</td>
<td align="center">83.2</td>
<td align="center">85.5</td>
<td align="center">86.5</td>
<td align="center">87.2</td>
<td align="center">31.6</td>
<td align="center">33.1</td>
<td align="center">34.0</td>
<td align="center">35.1</td>
</tr>
<tr>
<th>不含 𝒮MIE</th>
<td align="center">52.8</td>
<td align="center">52.6</td>
<td align="center">52.4</td>
<td align="center">53.9</td>
<td align="center">81.3</td>
<td align="center">83.8</td>
<td align="center">83.1</td>
<td align="center">83.2</td>
<td align="center">29.9</td>
<td align="center">30.5</td>
<td align="center">33.1</td>
<td align="center">34.5</td>
</tr>
<tr>
<th>不含𝒮BEL</th>
<td align="center">50.3</td>
<td align="center">51.4</td>
<td align="center">51.8</td>
<td align="center">53.0</td>
<td align="center">80.1</td>
<td align="center">83.1</td>
<td align="center">82.2</td>
<td align="center">83.0</td>
<td align="center">28.7</td>
<td align="center">29.8</td>
<td align="center">32.3</td>
<td align="center">34.2</td>
</tr>
<tr>
<th rowspan="3">AtomThink-EMOVA</th>
<th>DualMath-1.1M</th>
<td align="center">48.0</td>
<td align="center">48.8</td>
<td align="center">49.3</td>
<td align="center">49.6</td>
<td align="center">78.0</td>
<td align="center">79.6</td>
<td align="center">80.5</td>
<td align="center">81.0</td>
<td align="center">27.5</td>
<td align="center">29.0</td>
<td align="center">30.2</td>
<td align="center">31.0</td>
</tr>
<tr>
<th>不含 𝒮MIE</th>
<td align="center">47.5</td>
<td align="center">48.2</td>
<td align="center">47.8</td>
<td align="center">48.0</td>
<td align="center">76.8</td>
<td align="center">78.3</td>
<td align="center">79.1</td>
<td align="center">79.5</td>
<td align="center">26.0</td>
<td align="center">27.4</td>
<td align="center">28.5</td>
<td align="center">29.2</td>
</tr>
<tr>
<th>不含𝒮BEL</th>
<td align="center">46.8</td>
<td align="center">47.5</td>
<td align="center">47.9</td>
<td align="center">47.3</td>
<td align="center">76.0</td>
<td align="center">77.5</td>
<td align="center">78.3</td>
<td align="center">78.7</td>
<td align="center">25.4</td>
<td align="center">26.7</td>
<td align="center">27.8</td>
<td align="center">28.5</td>
</tr>
</tbody>
</table>

## 6 分析

### 6.1 各阶段如何提升性能

在本节中，我们将演示每个阶段如何对性能做出贡献。如图 6 (b) 所示，所有阶段都会对性能做出贡献。 MMathCoT-1M 贡献了最高的绝对性能增益。 Alignment-860K 的效果在 MathVerse 和 MathVision 上更为明显，可能是因为这两个数据集中的问题图像包含更丰富的文本模态信息，允许对齐资源（例如文本图像）更好地补充这种理解能力。 PS-GRPO则致力于突破大规模SFT之后的瓶颈，在WE-MATH和MathVerse上表现更为突出，与URSA-8B相比分别相对提升了13.2％和11.4％。我们在附录 C.4 中提供了对 InternVL2.5-8B 和 Multimath 的泛化验证。

*表4：奖励惩罚与PRM“下降时刻”判断的敏感性分析。*

<table>
<thead>
<tr>
<th rowspan="2">γ</th>
<th align="center" rowspan="2">ρ</th>
<th align="center">MathVerse</th>
<th align="center">MathVision</th>
<th align="center">MathVista</th>
<th align="center">WE-MATH</th>
<th align="center">DYNAMATH</th>
<th align="center">GeoQA</th>
<th align="center" rowspan="2">平均</th>
</tr>
<tr>
<th align="center">testmini</th>
<th align="center">full set</th>
<th align="center">gps</th>
<th align="center">testmini</th>
<th align="center">testmini</th>
<th align="center">full set</th>
</tr>
</thead>
<tbody>
<tr>
<th>0.5</th>
<td align="center">0.3</td>
<td align="center">50.9</td>
<td align="center">31.5</td>
<td align="center">83.2</td>
<td align="center">60.7</td>
<td align="center">47.4</td>
<td align="center">75.6</td>
<td align="center">58.2</td>
</tr>
<tr>
<th>0.5</th>
<td align="center">0.4</td>
<td align="center">49.9</td>
<td align="center">30.8</td>
<td align="center">81.2</td>
<td align="center">59.9</td>
<td align="center">46.9</td>
<td align="center">75.0</td>
<td align="center">57.3</td>
</tr>
<tr>
<th>0.5</th>
<td align="center">0.2</td>
<td align="center">49.6</td>
<td align="center">30.5</td>
<td align="center">80.9</td>
<td align="center">59.6</td>
<td align="center">46.6</td>
<td align="center">74.7</td>
<td align="center">57.0</td>
</tr>
<tr>
<th>1.0</th>
<td align="center">0.3</td>
<td align="center">49.0</td>
<td align="center">29.4</td>
<td align="center">79.8</td>
<td align="center">58.8</td>
<td align="center">45.3</td>
<td align="center">72.5</td>
<td align="center">56.3</td>
</tr>
<tr>
<th>0.7</th>
<td align="center">0.3</td>
<td align="center">52.0</td>
<td align="center">31.1</td>
<td align="center">81.7</td>
<td align="center">59.6</td>
<td align="center">47.0</td>
<td align="center">73.8</td>
<td align="center">57.5</td>
</tr>
<tr>
<th>0.3</th>
<td align="center">0.3</td>
<td align="center">51.5</td>
<td align="center">32.0</td>
<td align="center">82.1</td>
<td align="center">61.0</td>
<td align="center">46.3</td>
<td align="center">74.6</td>
<td align="center">57.9</td>
</tr>
</tbody>
</table>

### 6.2 自动过程标注的消融实验

我们对 DualMath-1.1M 的两个部分如何对 URSA-8B-RM 做出贡献进行了消融研究。如表3所示，我们可以看到基于BEL的方法（重点关注正确性的潜力）和基于MIE的方法（重点关注感知一致性）都对结果有积极的贡献。这进一步说明，在多模态数学推理过程中，图文不一致现象普遍存在，需要缓解。我们通过强制施加常见幻觉类别来增强过程监督训练数据来解决这个问题。具体来说，BEL生成的数据表现出更显着的影响，表明合成数据的质量仍有待提高。

### 6.3 奖励惩罚与下降时刻的敏感性分析

在本节中，我们对 PS-GRPO 的两个超参数 $\gamma$ 和 $\rho$ 进行敏感性分析。它们分别定义了对出现“下降时刻”的采样轨迹施加的奖励惩罚幅度，以及识别此类“下降时刻”的容忍阈值。如表 4 所示，我们的核心发现有两点：（i）$\gamma$ 不应设置得过高，因为这意味着对 PRM 的过度信任，可能导致组内奖励消失并使训练不稳定。当将 $\rho$ 固定为 0.3 时，我们发现将 $\gamma$ 设在一个合适区间内（本文测试了 0.3-0.7）通常更有利于平均性能；（ii）过大的 $\rho$ 会削弱奖励差异，使 RL 行为趋近于普通 GRPO。相反，过小的 $\rho$ 在设计上也不合理，因为它会对过程奖励变化过于敏感，往往导致过度惩罚。在所有正确采样轨迹都受到惩罚的极端情况下，PS-GRPO 会退化回普通 GRPO。

## 7 结论

在本研究中，我们迈出了探索 PRM 在多模态数学推理中应用的第一步。我们引入了三阶段训练流程 URSA，旨在解决三大核心挑战。首先，我们提供了大规模 CoT 推理数据集 MMathCoT-1M。该数据集构成了开发高推理能力模型 URSA-8B 的基础，并为后续的 TTS 与 RL 场景铺平了道路。接着，我们提出了一种双视角自动过程监督标注方法，覆盖多模态场景中的逻辑有效性与感知一致性，并据此构建了多模态推理领域首个大规模过程监督数据集 DualMath-1.1M。最后，我们通过“过程即结果”的建模方式缓解奖励黑客和长度偏差问题，并提出 PS-GRPO，这是一种超越普通 GRPO 的 PRM 辅助在线 RL 方法。最终得到的 URSA-8B-PS-GRPO 在平均性能上优于领先的开源 MLLM，例如 Gemma3-12B（8.4%），也超过专有模型 GPT-4o（2.7%）。

## 参考文献

- **Luo et al. [2023]** Haipeng Luo, Qingfeng Sun, Can Xu, Pu Zhao, Jianguang Lou, Chongyang Tao, Xiubo Geng, Qingwei Lin, Shifeng Chen, and Dongmei Zhang. Wizardmath: Empowering mathematical reasoning for large language models via reinforced evol-instruct. *arXiv preprint arXiv:2308.09583*, 2023.

- **Yang et al. [2024a]** An Yang, Beichen Zhang, Binyuan Hui, Bofei Gao, Bowen Yu, Chengpeng Li, Dayiheng Liu, Jianhong Tu, Jingren Zhou, Junyang Lin, Keming Lu, Mingfeng Xue, Runji Lin, Tianyu Liu, Xingzhang Ren, and Zhenru Zhang. Qwen2.5-math technical report: Toward mathematical expert model via self-improvement. *CoRR*, abs/2409.12122, 2024a. doi: 10.48550/ARXIV.2409.12122. URL https://doi.org/10.48550/arXiv.2409.12122.

- **Ying et al. [2024]** Huaiyuan Ying, Shuo Zhang, Linyang Li, Zhejian Zhou, Yunfan Shao, Zhaoye Fei, Yichuan Ma, Jiawei Hong, Kuikun Liu, Ziyi Wang, Yudong Wang, Zijian Wu, Shuaibin Li, Fengzhe Zhou, Hongwei Liu, Songyang Zhang, Wenwei Zhang, Hang Yan, Xipeng Qiu, Jiayu Wang, Kai Chen, and Dahua Lin. Internlm-math: Open math large language models toward verifiable reasoning. *CoRR*, abs/2402.06332, 2024. doi: 10.48550/ARXIV.2402.06332. URL https://doi.org/10.48550/arXiv.2402.06332.

- **Shao et al. [2024]** Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. *arXiv preprint arXiv:2402.03300*, 2024.

- **Yang et al. [2024b]** Zhen Yang, Jinhao Chen, Zhengxiao Du, Wenmeng Yu, Weihan Wang, Wenyi Hong, Zhihuan Jiang, Bin Xu, Yuxiao Dong, and Jie Tang. Mathglm-vision: Solving mathematical problems with multi-modal large language model. *arXiv preprint arXiv:2409.13729*, 2024b.

- **Yu et al. [2024]** Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James T. Kwok, Zhenguo Li, Adrian Weller, and Weiyang Liu. Metamath: Bootstrap your own mathematical questions for large language models. In *The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024*. OpenReview.net, 2024. URL https://openreview.net/forum?id=N8N0hgNDRt.

- **Ni et al. [2024]** Xinzhe Ni, Yeyun Gong, Zhibin Gou, Yelong Shen, Yujiu Yang, Nan Duan, and Weizhu Chen. Exploring the mystery of influential data for mathematical reasoning. *CoRR*, abs/2404.01067, 2024. doi: 10.48550/ARXIV.2404.01067. URL https://doi.org/10.48550/arXiv.2404.01067.

- **Yu et al. [2025a]** Yiyao Yu, Yuxiang Zhang, Dongdong Zhang, Xiao Liang, Hengyuan Zhang, Xingxing Zhang, Ziyi Yang, Mahmoud Khademi, Hany Awadalla, Junjie Wang, Yujiu Yang, and Furu Wei. Chain-of-reasoning: Towards unified mathematical reasoning in large language models via a multi-paradigm perspective. *CoRR*, abs/2501.11110, 2025a. doi: 10.48550/ARXIV.2501.11110. URL https://doi.org/10.48550/arXiv.2501.11110.

- **Yang et al. [2024c]** Zhen Yang, Jinhao Chen, Zhengxiao Du, Wenmeng Yu, Weihan Wang, Wenyi Hong, Zhihuan Jiang, Bin Xu, Yuxiao Dong, and Jie Tang. Mathglm-vision: Solving mathematical problems with multi-modal large language model. *CoRR*, abs/2409.13729, 2024c. doi: 10.48550/ARXIV.2409.13729. URL https://doi.org/10.48550/arXiv.2409.13729.

- **Yao et al. [2024a]** Huanjin Yao, Jiaxing Huang, Wenhao Wu, Jingyi Zhang, Yibo Wang, Shunyu Liu, Yingjie Wang, Yuxin Song, Haocheng Feng, Li Shen, and Dacheng Tao. Mulberry: Empowering MLLM with o1-like reasoning and reflection via collective monte carlo tree search. *CoRR*, abs/2412.18319, 2024a. doi: 10.48550/ARXIV.2412.18319. URL https://doi.org/10.48550/arXiv.2412.18319.

- **Zhuang et al. [2025]** Wenwen Zhuang, Xin Huang, Xiantao Zhang, and Jin Zeng. Math-puma: Progressive upward multimodal alignment to enhance mathematical reasoning. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 39, pages 26183–26191, 2025.

- **Xiang et al. [2024]** Kun Xiang, Zhili Liu, Zihao Jiang, Yunshuang Nie, Runhui Huang, Haoxiang Fan, Hanhui Li, Weiran Huang, Yihan Zeng, Jianhua Han, et al. Atomthink: A slow thinking framework for multimodal mathematical reasoning. *arXiv preprint arXiv:2411.11930*, 2024.

- **Huang et al. [2025a]** Wenxuan Huang, Bohan Jia, Zijie Zhai, Shaosheng Cao, Zheyu Ye, Fei Zhao, Zhe Xu, Yao Hu, and Shaohui Lin. Vision-r1: Incentivizing reasoning capability in multimodal large language models. *CoRR*, abs/2503.06749, 2025a. doi: 10.48550/ARXIV.2503.06749. URL https://doi.org/10.48550/arXiv.2503.06749.

- **Han et al. [2024]** Xiaotian Han, Yiren Jian, Xuefeng Hu, Haogeng Liu, Yiqi Wang, Qihang Fan, Yuang Ai, Huaibo Huang, Ran He, Zhenheng Yang, et al. Infimm-webmath-40b: Advancing multimodal pre-training for enhanced mathematical reasoning. In *The 4th Workshop on Mathematical Reasoning and AI at NeurIPS’24*, 2024.

- **Shi et al. [2024]** Wenhao Shi, Zhiqiang Hu, Yi Bin, Junhua Liu, Yang Yang, See-Kiong Ng, Lidong Bing, and Roy Ka-Wei Lee. Math-llava: Bootstrapping mathematical reasoning for multimodal large language models. *arXiv preprint arXiv:2406.17294*, 2024.

- **Cai et al. [2024]** Shihao Cai, Keqin Bao, Hangyu Guo, Jizhi Zhang, Jun Song, and Bo Zheng. Geogpt4v: Towards geometric multi-modal large language models with geometric image generation. *arXiv preprint arXiv:2406.11503*, 2024.

- **Deng et al. [2024]** Linger Deng, Yuliang Liu, Bohan Li, Dongliang Luo, Liang Wu, Chengquan Zhang, Pengyuan Lyu, Ziyang Zhang, Gang Zhang, Errui Ding, et al. R-cot: Reverse chain-of-thought problem generation for geometric reasoning in large multimodal models. *arXiv preprint arXiv:2410.17885*, 2024.

- **Gao et al. [2023a]** Jiahui Gao, Renjie Pi, Jipeng Zhang, Jiacheng Ye, Wanjun Zhong, Yufei Wang, Lanqing Hong, Jianhua Han, Hang Xu, Zhenguo Li, and Lingpeng Kong. G-llava: Solving geometric problem with multi-modal large language model. *CoRR*, abs/2312.11370, 2023a. doi: 10.48550/ARXIV.2312.11370. URL https://doi.org/10.48550/arXiv.2312.11370.

- **Zhang et al. [2024a]** Renrui Zhang, Xinyu Wei, Dongzhi Jiang, Ziyu Guo, Shicheng Li, Yichi Zhang, Chengzhuo Tong, Jiaming Liu, Aojun Zhou, Bin Wei, et al. Mavis: Mathematical visual instruction tuning with an automatic data engine. *arXiv preprint arXiv:2407.08739*, 2024a.

- **Xia et al. [2024a]** Renqiu Xia, Mingsheng Li, Hancheng Ye, Wenjie Wu, Hongbin Zhou, Jiakang Yuan, Tianshuo Peng, Xinyu Cai, Xiangchao Yan, Bin Wang, Conghui He, Botian Shi, Tao Chen, Junchi Yan, and Bo Zhang. Geox: Geometric problem solving through unified formalized vision-language pre-training. *CoRR*, abs/2412.11863, 2024a. doi: 10.48550/ARXIV.2412.11863. URL https://doi.org/10.48550/arXiv.2412.11863.

- **Xia et al. [2024b]** Renqiu Xia, Bo Zhang, Hancheng Ye, Xiangchao Yan, Qi Liu, Hongbin Zhou, Zijun Chen, Min Dou, Botian Shi, Junchi Yan, and Yu Qiao. Chartx & chartvlm: A versatile benchmark and foundation model for complicated chart reasoning. *CoRR*, abs/2402.12185, 2024b. doi: 10.48550/ARXIV.2402.12185. URL https://doi.org/10.48550/arXiv.2402.12185.

- **Zhang et al. [2024b]** Zhuosheng Zhang, Aston Zhang, Mu Li, Hai Zhao, George Karypis, and Alex Smola. Multimodal chain-of-thought reasoning in language models. *Trans. Mach. Learn. Res.*, 2024, 2024b. URL https://openreview.net/forum?id=y1pPWFVfvR.

- **Peng et al. [2024]** Shuai Peng, Di Fu, Liangcai Gao, Xiuqin Zhong, Hongguang Fu, and Zhi Tang. Multimath: Bridging visual and mathematical reasoning for large language models. *arXiv preprint arXiv:2409.00147*, 2024.

- **Zhang et al. [2024c]** Ruohong Zhang, Bowen Zhang, Yanghao Li, Haotian Zhang, Zhiqing Sun, Zhe Gan, Yinfei Yang, Ruoming Pang, and Yiming Yang. Improve vision language model chain-of-thought reasoning. *arXiv preprint arXiv:2410.16198*, 2024c.

- **Liu et al. [2025a]** Runze Liu, Junqi Gao, Jian Zhao, Kaiyan Zhang, Xiu Li, Biqing Qi, Wanli Ouyang, and Bowen Zhou. Can 1b LLM surpass 405b llm? rethinking compute-optimal test-time scaling. *CoRR*, abs/2502.06703, 2025a. doi: 10.48550/ARXIV.2502.06703. URL https://doi.org/10.48550/arXiv.2502.06703.

- **Zhang et al. [2024d]** Lunjun Zhang, Arian Hosseini, Hritik Bansal, Mehran Kazemi, Aviral Kumar, and Rishabh Agarwal. Generative verifiers: Reward modeling as next-token prediction. *arXiv preprint arXiv:2408.15240*, 2024d.

- **Zhang et al. [2024e]** Dan Zhang, Sining Zhoubian, Ziniu Hu, Yisong Yue, Yuxiao Dong, and Jie Tang. Rest-mcts*: LLM self-training via process reward guided tree search. In Amir Globersons, Lester Mackey, Danielle Belgrave, Angela Fan, Ulrich Paquet, Jakub M. Tomczak, and Cheng Zhang, editors, *Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024*, 2024e. URL http://papers.nips.cc/paper_files/paper/2024/hash/76ec4dc30e9faaf0e4b6093eaa377218-Abstract-Conference.html.

- **Liu et al. [2024a]** Wei Liu, Junlong Li, Xiwen Zhang, Fan Zhou, Yu Cheng, and Junxian He. Diving into self-evolving training for multimodal reasoning. *CoRR*, abs/2412.17451, 2024a. doi: 10.48550/ARXIV.2412.17451. URL https://doi.org/10.48550/arXiv.2412.17451.

- **Yue et al. [2025]** Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Yang Yue, Shiji Song, and Gao Huang. Does reinforcement learning really incentivize reasoning capacity in llms beyond the base model? *arXiv preprint arXiv:2504.13837*, 2025.

- **Yan et al. [2024]** Yibo Yan, Shen Wang, Jiahao Huo, Hang Li, Boyan Li, Jiamin Su, Xiong Gao, Yi-Fan Zhang, Tianlong Xu, Zhendong Chu, et al. Errorradar: Benchmarking complex mathematical reasoning of multimodal large language models via error detection. *arXiv preprint arXiv:2410.04509*, 2024.

- **Zhang et al. [2024f]** Di Zhang, Jingdi Lei, Junxian Li, Xunzhi Wang, Yujie Liu, Zonglin Yang, Jiatong Li, Weida Wang, Suorong Yang, Jianbo Wu, et al. Critic-v: Vlm critics help catch vlm errors in multimodal reasoning. *arXiv preprint arXiv:2411.18203*, 2024f.

- **Ai et al. [2025]** Jiaxin Ai, Pengfei Zhou, Zhaopan Xu, Ming Li, Fanrui Zhang, Zizhen Li, Jianwen Sun, Yukang Feng, Baojin Huang, Zhongyuan Wang, and Kaipeng Zhang. Projudge: A multi-modal multi-discipline benchmark and instruction-tuning dataset for mllm-based process judges. *CoRR*, abs/2503.06553, 2025. doi: 10.48550/ARXIV.2503.06553. URL https://doi.org/10.48550/arXiv.2503.06553.

- **Weng [2024]** Lilian Weng. Reward hacking and how to mitigate it. https://lilianweng.github.io/posts/2024-11-28-reward-hacking/, November 2024. [Accessed 11-28-2024].

- **Fu et al. [2025]** Jiayi Fu, Xuandong Zhao, Chengyuan Yao, Heng Wang, Qi Han, and Yanghua Xiao. Reward shaping to mitigate reward hacking in RLHF. *CoRR*, abs/2502.18770, 2025. doi: 10.48550/ARXIV.2502.18770. URL https://doi.org/10.48550/arXiv.2502.18770.

- **Li et al. [2024]** Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. *arXiv preprint arXiv:2408.03326*, 2024.

- **Zhai et al. [2023]** Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 11975–11986, 2023.

- **Kirillov et al. [2023]** Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 4015–4026, 2023.

- **Lu et al. [2024]** Haoyu Lu, Wen Liu, Bo Zhang, Bingxuan Wang, Kai Dong, Bo Liu, Jingxiang Sun, Tongzheng Ren, Zhuoshu Li, Hao Yang, et al. Deepseek-vl: towards real-world vision-language understanding. *arXiv preprint arXiv:2403.05525*, 2024.

- **Trinh et al. [2024]** Trieu H Trinh, Yuhuai Wu, Quoc V Le, He He, and Thang Luong. Solving olympiad geometry without human demonstrations. *Nature*, 625(7995):476–482, 2024.

- **Chen et al. [2021]** Jiaqi Chen, Jianheng Tang, Jinghui Qin, Xiaodan Liang, Lingbo Liu, Eric Xing, and Liang Lin. Geoqa: A geometric question answering benchmark towards multimodal numerical reasoning. In *Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021*, pages 513–523, 2021.

- **Lightman et al. [2023]** Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. *arXiv preprint arXiv:2305.20050*, 2023.

- **Luo et al. [2024]** Liangchen Luo, Yinxiao Liu, Rosanne Liu, Samrat Phatale, Harsh Lara, Yunxuan Li, Lei Shu, Yun Zhu, Lei Meng, Jiao Sun, et al. Improve mathematical reasoning in language models by automated process supervision. *arXiv preprint arXiv:2406.06592*, 2024.

- **Wang et al. [2024a]** Peiyi Wang, Lei Li, Zhihong Shao, Runxin Xu, Damai Dai, Yifei Li, Deli Chen, Yu Wu, and Zhifang Sui. Math-shepherd: Verify and reinforce llms step-by-step without human annotations. In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 9426–9439, 2024a.

- **Zheng et al. [2024]** Haojie Zheng, Tianyang Xu, Hanchi Sun, Shu Pu, Ruoxi Chen, and Lichao Sun. Thinking before looking: Improving multimodal llm reasoning via mitigating visual hallucination. *arXiv preprint arXiv:2411.12591*, 2024.

- **Gao et al. [2023b]** Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, and Graham Neubig. Pal: Program-aided language models. In *International Conference on Machine Learning*, pages 10764–10799. PMLR, 2023b.

- **Guo et al. [2025]** Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. *arXiv preprint arXiv:2501.12948*, 2025.

- **Pan et al. [2025]** Jiazhen Pan, Che Liu, Junde Wu, Fenglin Liu, Jiayuan Zhu, Hongwei Bran Li, Chen Chen, Cheng Ouyang, and Daniel Rueckert. Medvlm-r1: Incentivizing medical reasoning capability of vision-language models (vlms) via reinforcement learning. *CoRR*, abs/2502.19634, 2025. doi: 10.48550/ARXIV.2502.19634. URL https://doi.org/10.48550/arXiv.2502.19634.

- **Zhan et al. [2025]** Yufei Zhan, Yousong Zhu, Shurong Zheng, Hongyin Zhao, Fan Yang, Ming Tang, and Jinqiao Wang. Vision-r1: Evolving human-free alignment in large vision-language models via vision-guided reinforcement learning. *CoRR*, abs/2503.18013, 2025. doi: 10.48550/ARXIV.2503.18013. URL https://doi.org/10.48550/arXiv.2503.18013.

- **Huang et al. [2025b]** Wenxuan Huang, Bohan Jia, Zijie Zhai, Shaosheng Cao, Zheyu Ye, Fei Zhao, Zhe Xu, Yao Hu, and Shaohui Lin. Vision-r1: Incentivizing reasoning capability in multimodal large language models. *CoRR*, abs/2503.06749, 2025b. doi: 10.48550/ARXIV.2503.06749. URL https://doi.org/10.48550/arXiv.2503.06749.

- **Liu et al. [2025b]** Xiangyan Liu, Jinjie Ni, Zijian Wu, Chao Du, Longxu Dou, Haonan Wang, Tianyu Pang, and Michael Qizhe Shieh. Noisyrollout: Reinforcing visual reasoning with data augmentation, 2025b. URL https://arxiv.org/abs/2504.13055.

- **Li and Li [2024]** Wendi Li and Yixuan Li. Process reward model with q-value rankings. *CoRR*, abs/2410.11287, 2024. doi: 10.48550/ARXIV.2410.11287. URL https://doi.org/10.48550/arXiv.2410.11287.

- **Setlur et al. [2024]** Amrith Setlur, Chirag Nagpal, Adam Fisch, Xinyang Geng, Jacob Eisenstein, Rishabh Agarwal, Alekh Agarwal, Jonathan Berant, and Aviral Kumar. Rewarding progress: Scaling automated process verifiers for LLM reasoning. *CoRR*, abs/2410.08146, 2024. doi: 10.48550/ARXIV.2410.08146. URL https://doi.org/10.48550/arXiv.2410.08146.

- **Ma et al. [2025]** Yiran Ma, Zui Chen, Tianqiao Liu, Mi Tian, Zhuo Liu, Zitao Liu, and Weiqi Luo. What are step-level reward models rewarding? counterintuitive findings from mcts-boosted mathematical reasoning. In Toby Walsh, Julie Shah, and Zico Kolter, editors, *AAAI-25, Sponsored by the Association for the Advancement of Artificial Intelligence, February 25 - March 4, 2025, Philadelphia, PA, USA*, pages 24812–24820. AAAI Press, 2025. doi: 10.1609/AAAI.V39I23.34663. URL https://doi.org/10.1609/aaai.v39i23.34663.

- **Gao et al. [2024a]** Jiaxuan Gao, Shusheng Xu, Wenjie Ye, Weilin Liu, Chuyi He, Wei Fu, Zhiyu Mei, Guangju Wang, and Yi Wu. On designing effective RL reward at training time for LLM reasoning. *CoRR*, abs/2410.15115, 2024a. doi: 10.48550/ARXIV.2410.15115. URL https://doi.org/10.48550/arXiv.2410.15115.

- **Amodei et al. [2016]** Dario Amodei, Chris Olah, Jacob Steinhardt, Paul Christiano, John Schulman, and Dan Mané. Concrete problems in ai safety. *arXiv preprint arXiv:1606.06565*, 2016.

- **Zhang and Zuo [2025]** Jixiao Zhang and Chunsheng Zuo. Grpo-lead: A difficulty-aware reinforcement learning approach for concise mathematical reasoning in language models. *arXiv preprint arXiv:2504.09696*, 2025.

- **Zhang et al. [2025a]** Jingyi Zhang, Jiaxing Huang, Huanjin Yao, Shunyu Liu, Xikun Zhang, Shijian Lu, and Dacheng Tao. R1-vl: Learning to reason with multimodal large language models via step-wise group relative policy optimization. *arXiv preprint arXiv:2503.12937*, 2025a.

- **Yu et al. [2025b]** Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. *arXiv preprint arXiv:2503.14476*, 2025b.

- **OpenAI [2024]** OpenAI. GPT-4o system card, 2024. URL https://openai.com/research/gpt-4o-system-card.

- **Team et al. [2023]** Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. *arXiv preprint arXiv:2312.11805*, 2023.

- **Chen et al. [2024a]** Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. *arXiv preprint arXiv:2404.16821*, 2024a.

- **Meta [2024]** Meta. Llama 3.2: Revolutionizing edge AI and vision with open, customizable models — ai.meta.com. https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/, 2024. [Accessed 17-04-2025].

- **Wang et al. [2024b]** Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. *arXiv preprint arXiv:2409.12191*, 2024b.

- **Chen et al. [2024b]** Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 24185–24198, 2024b.

- **Wang et al. [2024c]** Weiyun Wang, Zhe Chen, Wenhai Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Jinguo Zhu, Xizhou Zhu, Lewei Lu, Yu Qiao, et al. Enhancing the reasoning ability of multimodal large language models via mixed preference optimization. *arXiv preprint arXiv:2411.10442*, 2024c.

- **Chen et al. [2024c]** Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. *arXiv preprint arXiv:2412.05271*, 2024c.

- **Liu et al. [2024b]** Yuan Liu, Zhongyin Zhao, Ziyuan Zhuang, Le Tian, Xiao Zhou, and Jie Zhou. Points: Improving your vision-language model with affordable strategies. *arXiv preprint arXiv:2409.04828*, 2024b.

- **Team et al. [2025]** Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin, Tatiana Matejovicova, Alexandre Ramé, Morgane Rivière, et al. Gemma 3 technical report. *arXiv preprint arXiv:2503.19786*, 2025.

- **Thawakar et al. [2025]** Omkar Thawakar, Dinura Dissanayake, Ketan More, Ritesh Thawkar, Ahmed Heakl, Noor Ahsan, Yuhao Li, Mohammed Zumri, Jean Lahoud, Rao Muhammad Anwer, Hisham Cholakkal, Ivan Laptev, Mubarak Shah, Fahad Shahbaz Khan, and Salman H. Khan. Llamav-o1: Rethinking step-by-step visual reasoning in llms. *CoRR*, abs/2501.06186, 2025. doi: 10.48550/ARXIV.2501.06186. URL https://doi.org/10.48550/arXiv.2501.06186.

- **Deng et al. [2025]** Yihe Deng, Hritik Bansal, Fan Yin, Nanyun Peng, Wei Wang, and Kai-Wei Chang. Openvlthinker: An early exploration to complex vision-language reasoning via iterative self-improvement. *CoRR*, abs/2503.17352, 2025. doi: 10.48550/ARXIV.2503.17352. URL https://doi.org/10.48550/arXiv.2503.17352.

- **Yang et al. [2025]** Yi Yang, Xiaoxuan He, Hongkun Pan, Xiyan Jiang, Yan Deng, Xingtao Yang, Haoyu Lu, Dacheng Yin, Fengyun Rao, Minfeng Zhu, Bo Zhang, and Wei Chen. R1-onevision: Advancing generalized multimodal reasoning through cross-modal formalization. *arXiv preprint arXiv:2503.10615*, 2025.

- **Zhang et al. [2025b]** Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Yu Qiao, et al. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? In *European Conference on Computer Vision*, pages 169–186. Springer, 2025b.

- **Zou et al. [2024]** Chengke Zou, Xingang Guo, Rui Yang, Junyu Zhang, Bin Hu, and Huan Zhang. Dynamath: A dynamic visual benchmark for evaluating mathematical reasoning robustness of vision language models. *arXiv preprint arXiv:2411.00836*, 2024.

- **Lu et al. [2023a]** Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. *arXiv preprint arXiv:2310.02255*, 2023a.

- **Qiao et al. [2024]** Runqi Qiao, Qiuna Tan, Guanting Dong, Minhui Wu, Chong Sun, Xiaoshuai Song, Zhuoma GongQue, Shanglin Lei, Zhe Wei, Miaoxuan Zhang, et al. We-math: Does your large multimodal model achieve human-like mathematical reasoning? *arXiv preprint arXiv:2407.01284*, 2024.

- **Dong et al. [2024a]** Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Bin Wang, Linke Ouyang, Xilin Wei, Songyang Zhang, Haodong Duan, Maosong Cao, et al. Internlm-xcomposer2: Mastering free-form text-image composition and comprehension in vision-language large model. *arXiv preprint arXiv:2401.16420*, 2024a.

- **Yao et al. [2024b]** Huanjin Yao, Jiaxing Huang, Wenhao Wu, Jingyi Zhang, Yibo Wang, Shunyu Liu, Yingjie Wang, Yuxin Song, Haocheng Feng, Li Shen, and Dacheng Tao. Mulberry: Empowering MLLM with o1-like reasoning and reflection via collective monte carlo tree search. *CoRR*, abs/2412.18319, 2024b. doi: 10.48550/ARXIV.2412.18319. URL https://doi.org/10.48550/arXiv.2412.18319.

- **Guo et al. [2024]** Jarvis Guo, Tuney Zheng, Yuelin Bai, Bo Li, Yubo Wang, King Zhu, Yizhi Li, Graham Neubig, Wenhu Chen, and Xiang Yue. Mammoth-vl: Eliciting multimodal reasoning with instruction tuning at scale. *arXiv preprint arXiv:2412.05237*, 2024.

- **Wang et al. [2022]** Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. *arXiv preprint arXiv:2203.11171*, 2022.

- **Gao et al. [2023c]** Jiahui Gao, Renjie Pi, Jipeng Zhang, Jiacheng Ye, Wanjun Zhong, Yufei Wang, Lanqing Hong, Jianhua Han, Hang Xu, Zhenguo Li, et al. G-llava: Solving geometric problem with multi-modal large language model. *arXiv preprint arXiv:2312.11370*, 2023c.

- **Dong et al. [2024b]** Yuhao Dong, Zuyan Liu, Hai-Long Sun, Jingkang Yang, Winston Hu, Yongming Rao, and Ziwei Liu. Insight-v: Exploring long-chain visual reasoning with multimodal large language models. *arXiv preprint arXiv:2411.14432*, 2024b.

- **Hu et al. [2024]** Yushi Hu, Weijia Shi, Xingyu Fu, Dan Roth, Mari Ostendorf, Luke Zettlemoyer, Noah A Smith, and Ranjay Krishna. Visual sketchpad: Sketching as a visual chain of thought for multimodal language models. *arXiv preprint arXiv:2406.09403*, 2024.

- **Yu et al. [2023]** Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James T Kwok, Zhenguo Li, Adrian Weller, and Weiyang Liu. Metamath: Bootstrap your own mathematical questions for large language models. *arXiv preprint arXiv:2309.12284*, 2023.

- **Liu et al. [2024c]** Dongyang Liu, Renrui Zhang, Longtian Qiu, Siyuan Huang, Weifeng Lin, Shitian Zhao, Shijie Geng, Ziyi Lin, Peng Jin, Kaipeng Zhang, et al. Sphinx-x: Scaling data and parameters for a family of multi-modal large language models. *arXiv preprint arXiv:2402.05935*, 2024c.

- **Sprague et al. [2024]** Zayne Sprague, Fangcong Yin, Juan Diego Rodriguez, Dongwei Jiang, Manya Wadhwa, Prasann Singhal, Xinyu Zhao, Xi Ye, Kyle Mahowald, and Greg Durrett. To cot or not to cot? chain-of-thought helps mainly on math and symbolic reasoning. *arXiv preprint arXiv:2409.12183*, 2024.

- **Lu et al. [2023b]** Yingzhou Lu, Minjie Shen, Huazheng Wang, Xiao Wang, Capucine van Rechem, Tianfan Fu, and Wenqi Wei. Machine learning for synthetic data generation: a review. *arXiv preprint arXiv:2302.04062*, 2023b.

- **Huang et al. [2024]** Yiming Huang, Xiao Liu, Yeyun Gong, Zhibin Gou, Yelong Shen, Nan Duan, and Weizhu Chen. Key-point-driven data synthesis with its enhancement on mathematical reasoning. *arXiv preprint arXiv:2403.02333*, 2024.

- **Fu et al. [2024]** Chaoyou Fu, Haojia Lin, Zuwei Long, Yunhang Shen, Meng Zhao, Yifan Zhang, Shaoqi Dong, Xiong Wang, Di Yin, Long Ma, et al. Vita: Towards open-source interactive omni multimodal llm. *arXiv preprint arXiv:2408.05211*, 2024.

- **Gou et al. [2023]** Zhibin Gou, Zhihong Shao, Yeyun Gong, Yelong Shen, Yujiu Yang, Nan Duan, and Weizhu Chen. Critic: Large language models can self-correct with tool-interactive critiquing. *arXiv preprint arXiv:2305.11738*, 2023.

- **Gao et al. [2024b]** Bofei Gao, Zefan Cai, Runxin Xu, Peiyi Wang, Ce Zheng, Runji Lin, Keming Lu, Junyang Lin, Chang Zhou, Wen Xiao, et al. Llm critics help catch bugs in mathematics: Towards a better mathematical verifier with natural language feedback. *CoRR*, 2024b.

- **Lin et al. [2024]** Zicheng Lin, Zhibin Gou, Tian Liang, Ruilin Luo, Haowei Liu, and Yujiu Yang. CriticBench: Benchmarking LLMs for critique-correct reasoning. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors, *Findings of the Association for Computational Linguistics: ACL 2024*, pages 1552–1587, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.findings-acl.91. URL https://aclanthology.org/2024.findings-acl.91.

- **Kumar et al. [2024]** Aviral Kumar, Vincent Zhuang, Rishabh Agarwal, Yi Su, John D Co-Reyes, Avi Singh, Kate Baumli, Shariq Iqbal, Colton Bishop, Rebecca Roelofs, et al. Training language models to self-correct via reinforcement learning. *arXiv preprint arXiv:2409.12917*, 2024.

- **Snell et al. [2024]** Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling llm test-time compute optimally can be more effective than scaling model parameters. *arXiv preprint arXiv:2408.03314*, 2024.

- **Tu et al. [2025]** Haoqin Tu, Weitao Feng, Hardy Chen, Hui Liu, Xianfeng Tang, and Cihang Xie. Vilbench: A suite for vision-language process reward modeling. *arXiv preprint arXiv:2503.20271*, 2025.

- **Wang et al. [2025]** Weiyun Wang, Zhangwei Gao, Lianjie Chen, Zhe Chen, Jinguo Zhu, Xiangyu Zhao, Yangzhou Liu, Yue Cao, Shenglong Ye, Xizhou Zhu, et al. Visualprm: An effective process reward model for multimodal reasoning. *arXiv preprint arXiv:2503.10291*, 2025.

- **Sun et al. [2025]** Linzhuang Sun, Hao Liang, Jingxuan Wei, Bihui Yu, Tianpeng Li, Fan Yang, Zenan Zhou, and Wentao Zhang. Mm-verify: Enhancing multimodal reasoning with chain-of-thought verification. *arXiv preprint arXiv:2502.13383*, 2025.

- **Gao et al. [2024c]** Bofei Gao, Zefan Cai, Runxin Xu, Peiyi Wang, Ce Zheng, Runji Lin, Keming Lu, Junyang Lin, Chang Zhou, Wen Xiao, Junjie Hu, Tianyu Liu, and Baobao Chang. LLM critics help catch bugs in mathematics: Towards a better mathematical verifier with natural language feedback. *CoRR*, abs/2406.14024, 2024c. doi: 10.48550/ARXIV.2406.14024. URL https://doi.org/10.48550/arXiv.2406.14024.

- **Zeng et al. [2024]** Weihao Zeng, Yuzhen Huang, Lulu Zhao, Yijun Wang, Zifei Shan, and Junxian He. B-star: Monitoring and balancing exploration and exploitation in self-taught reasoners. *arXiv preprint arXiv:2412.17256*, 2024.

- **Meng et al. [2025]** Fanqing Meng, Lingxiao Du, Zongkai Liu, Zhixiang Zhou, Quanfeng Lu, Daocheng Fu, Botian Shi, Wenhai Wang, Junjun He, Kaipeng Zhang, et al. Mm-eureka: Exploring visual aha moment with rule-based large-scale reinforcement learning. *arXiv preprint arXiv:2503.07365*, 2025.

- **von Werra et al. [2020]** Leandro von Werra, Younes Belkada, Lewis Tunstall, Edward Beeching, Tristan Thrush, Nathan Lambert, Shengyi Huang, Kashif Rasul, and Quentin Gallouédec. Trl: Transformer reinforcement learning. https://github.com/huggingface/trl, 2020.

- **Bai et al. [2023]** Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. *arXiv preprint arXiv:2308.12966*, 1(2):3, 2023.

- **Face [2025]** Hugging Face. Open r1: A fully open reproduction of deepseek-r1, January 2025. URL https://github.com/huggingface/open-r1.

- **Diederik [2014]** P Kingma Diederik. Adam: A method for stochastic optimization. *(No Title)*, 2014.

- **Zhao et al. [2023]** Yanli Zhao, Andrew Gu, Rohan Varma, Liang Luo, Chien-Chin Huang, Min Xu, Less Wright, Hamid Shojanazeri, Myle Ott, Sam Shleifer, Alban Desmaison, Can Balioglu, Pritam Damania, Bernard Nguyen, Geeta Chauhan, Yuchen Hao, Ajit Mathews, and Shen Li. Pytorch FSDP: experiences on scaling fully sharded data parallel. *Proc. VLDB Endow.*, 16(12):3848–3860, 2023. doi: 10.14778/3611540.3611569. URL https://www.vldb.org/pvldb/vol16/p3848-huang.pdf.

- **Kwon et al. [2023]** Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In *Proceedings of the 29th Symposium on Operating Systems Principles*, pages 611–626, 2023.

- **Hu et al. [2025]** Jingcheng Hu, Yinmin Zhang, Qi Han, Daxin Jiang, Xiangyu Zhang, and Heung-Yeung Shum. Open-reasoner-zero: An open source approach to scaling up reinforcement learning on the base model. *CoRR*, abs/2503.24290, 2025. doi: 10.48550/ARXIV.2503.24290. URL https://doi.org/10.48550/arXiv.2503.24290.

- **Wang et al. [2024d]** Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical reasoning with math-vision dataset. *arXiv preprint arXiv:2402.14804*, 2024d.

**附录内容**

## 附录 A 相关工作

#### 多模态数学推理

MLLMs的数学推理能力最近引起了极大关注[11, 80, 35, 81, 82, 5, 14, 78]。与语言模型（LLMs）[1, 83]中的传统数学推理任务不同，多模态数学推理需要MLLMs解释视觉信息并在图像和文本之间进行跨模态推理。解决几何问题和分析图形等任务尤其具有挑战性[40]。最近的进展集中在改进特定场景中的视觉数学输入专用编码器[19, 84, 61]。还非常重视综合多样化和复杂的训练数据。例如，Math-LLaVA [15] 引入了 MathV360K 数据集，该数据集按复杂性对图像进行分类并增强相关问题。 Multimath [23] 整理来自 K-12 教科书的高质量推理数据，并使用 GPT-4 进行 CoT 数据生成和验证。 R-CoT [17] 通过两阶段反向问答生成过程进一步使问题多样化。这些数据合成方法因其高效性而被学术界和工业界广泛采用[85, 86, 87, 88]。

#### 过程奖励模型

最近的研究探索了LLMs中的测试时间缩放法则，旨在从不同的思维轨迹[26, 89, 90, 91, 31, 92, 93]中识别最佳推理选择。最初的努力，例如自我一致性[79]，为测试时间扩展奠定了基础。 OpenAI引入了验证器来在推理过程中监督和选择推理路径[41]。 Math-Shepherd [43] 根据得出正确答案的可能性评估中间推理步骤，而 OmegaPRM [42] 构建 PRM 训练数据并使用 MCTS 进行训练。尽管取得了这些进步，但缺乏具有强大CoT推理能力的模型以及对多样化奖励模型训练数据的探索有限仍然是多模态数学推理的重大瓶颈。一些并发工作也开始关注PRM辅助视觉推理，例如构建和基准测试[94, 95, 96]。

## 附录 B 预备知识

### B.1 组相对策略优化

普通 GRPO 去除了 PPO 中的价值函数，并估计在线采样轨迹组内的优势。给定一个包含图像 $q$ 和真实值 $y$ 的问题，策略模型 $\pi_{\theta_{old}}$ 对一组 $G$ 个响应 $\{o^{i}\}_{i=1}^{G}$ 进行采样。GRPO 通过标准化组内奖励 $\{r^{j}\}_{j=1}^{G}$ 来计算第 $i$ 个响应的优势，并采用 PPO 的裁剪目标和 KL 惩罚项：

$$
A^{i}=\frac{r^{i}-\text{mean}(\{r^{j}\}_{j=1}^{G})}{\text{std}(\{r^{j}\}_{j=1}^{G})}
$$
<div align="right">(7)</div>

$$
\mathcal{J}_{GRPO}(\theta)=\mathbb{E}_{(q,y)\sim\mathcal{D},\{o^{i}\}_{i=1}^{G}\sim\pi_{\theta_{old}}(\cdot|q)} \\
\left[\frac{1}{G}\sum\limits_{i=1}^{G}\frac{1}{|o^{i}|}\sum\limits_{t=1}^{|o^{i}|}\big(\min(r_{t}^{i}(\theta)A^{i},\text{clip}(r_{t}^{i}(\theta),1-\epsilon,1+\epsilon)A^{i})-\beta D_{KL}^{i,t}(\pi_{\theta}||\pi_{ref})\big)\right]
$$
<div align="right">(8)</div>

我们介绍第 4 节讨论的两种 PRM 集成式 GRPO 变体。给定 PRM $\mathcal{M}_{p}$ 及其过程奖励序列 $r_{s}=\mathcal{M}_{p}(\{s_{1},s_{2},\cdots,s_{N}\})$，(i) *变体 1*：给定可验证的结果奖励 $r_{o}^{i}$，我们将单条采样轨迹的奖励设置为 $r^{i}=r_{o}^{i}+\bar{r_{s}^{i}}$；(ii) *变体 2*：我们利用步骤级奖励，以及由每条采样轨迹的过程奖励均值计算得到的多重相对优势：

$$
A_{t}^{i}=\underbrace{r_{s,t}^{i}\frac{\bar{r_{s}^{i}}-\text{mean}({\bar{\{r_{s}^{j}\}}_{j=1}^{G}})}{\text{std}({\bar{\{r_{s}^{j}\}}_{j=1}^{G}})}}_{\text{带过程奖励的 GRPO}}+\underbrace{\frac{r_{o}^{i}-\text{mean}(\{r_{o}^{j}\}_{j=1}^{G})}{\text{std}(\{r_{o}^{j}\}_{j=1}^{G})}}_{\text{带结果奖励的 GRPO}}
$$
<div align="right">(9)</div>

其中$\bar{r_{s}^{i}}=\text{mean}(\mathcal{M}_{p}(\{s_{1}^{i},s_{2}^{i},\cdots,s_{T_{i}}^{i}\}))$。

### B.2 通过 Best-of-N 评估进行测试时扩展

遵循先前工作[41, 43]，我们在 TTS 中采用 BoN 评估。对于给定问题 $q$，我们采样得到 $N$ 个响应。PRM 会为每次采样提供过程奖励，我们再使用过程奖励的平均值来选择最佳的单次采样：

$$
a_{\text{prm}}=\arg\max\limits_{s_{i}}\text{mean}\{\mathcal{M}_{p}(q,s_{i})\}
$$
<div align="right">(10)</div>

其他一些作品[97]合并了自洽性和PRM以采用基于投票的分数累积。但我们不选择这种方法是为了更简单的评估方式。

## 附录 C 补充结果

### C.1 已使用基准上的细粒度比较

在本节中，我们提供了一些细粒度的结果，以便进行更清晰的比较。如表 5 所示，我们提出的方法展示了显着的优势。与GPT-4o和GPT-4V等闭源模型相比，我们的URSA-8B和URSA-8B-PS-GRPO表现出很强的竞争力。在开源模型中，性能提升更为明显。我们的 URSA-8B 模型在大多数子任务中都优于其他开源模型，例如 InternLM-XComposer2-VL 和 Ovis1.6-Gemma2-9B。与 PS-GRPO 结合使用时，URSA-8B-PS-GRPO 模型可以获得更好的结果，在 Alg、AnaG、CombG 等子任务中显示出显着改进。我们的方法尤其擅长复杂的数学推理任务，展示了其强大的数学推理能力。这些结果凸显了我们提出的 MMathCoT-1M 和 PS-GRPO 方法在增强模型数学推理能力方面的有效性，特别是在视觉数学问题中。

在Dynamath中（表6），与开源的MLLMs相比，URSA系列在平面几何和代数方面具有明显的优势。令人惊讶的是，从知识水平分类来看，URSA系列模型在本科阶段表现出色，这部分归功于其数学密集型的对齐和大规模的教学微调。

在 MathVerse（表 7）中，URSA 系列模型的平均表现略高于 GPT-4o。此外，与其他开源 MLLM 相比，URSA-8B-PS-GRPO 相较领先的 AtomThink-EMOVA-8B 和 InternVL2.5-8B 分别高出 **8.4** 和 **11.4** 分。

*表 5：不同 MLLMs 在 MathVision 上的性能比较。*

<table>
<tbody>
<tr>
<th>模型</th>
<td align="center">尺寸</td>
<td align="center">ALL</td>
<td align="center">海藻糖</td>
<td align="center">AnaG</td>
<td align="center">阿里</td>
<td align="center">梳状G</td>
<td align="center">梳子</td>
<td align="center">碳数</td>
<td align="center">描述G</td>
<td align="center">图形T</td>
<td align="center">日志</td>
<td align="center">角度</td>
<td align="center">区域</td>
<td align="center">伦</td>
<td align="center">SoIG</td>
<td align="center">统计数据</td>
<td align="center">拓扑</td>
<td align="center">转G</td>
</tr>
<tr>
<th align="center" colspan="19">基线</th>
</tr>
<tr>
<th>人类</th>
<td align="center">-</td>
<td align="center">68.8</td>
<td align="center">55.1</td>
<td align="center">78.6</td>
<td align="center">99.6</td>
<td align="center">98.4</td>
<td align="center">43.5</td>
<td align="center">98.5</td>
<td align="center">91.3</td>
<td align="center">62.2</td>
<td align="center">61.3</td>
<td align="center">33.5</td>
<td align="center">47.2</td>
<td align="center">73.5</td>
<td align="center">87.3</td>
<td align="center">93.1</td>
<td align="center">99.8</td>
<td align="center">69.0</td>
</tr>
<tr>
<th align="center" colspan="19">闭源 MLLMs</th>
</tr>
<tr>
<th>GPT-4o</th>
<td align="center">-</td>
<td align="center">30.4</td>
<td align="center">42.0</td>
<td align="center">39.3</td>
<td align="center">49.3</td>
<td align="center">28.9</td>
<td align="center">25.6</td>
<td align="center">22.4</td>
<td align="center">24.0</td>
<td align="center">23.3</td>
<td align="center">29.4</td>
<td align="center">17.3</td>
<td align="center">29.8</td>
<td align="center">30.1</td>
<td align="center">29.1</td>
<td align="center">44.8</td>
<td align="center">34.8</td>
<td align="center">17.9</td>
</tr>
<tr>
<th>GPT-4V</th>
<td align="center">-</td>
<td align="center">22.8</td>
<td align="center">27.3</td>
<td align="center">32.1</td>
<td align="center">35.7</td>
<td align="center">21.1</td>
<td align="center">16.7</td>
<td align="center">13.4</td>
<td align="center">22.1</td>
<td align="center">14.4</td>
<td align="center">16.8</td>
<td align="center">22.0</td>
<td align="center">22.2</td>
<td align="center">20.9</td>
<td align="center">23.8</td>
<td align="center">24.1</td>
<td align="center">21.7</td>
<td align="center">25.6</td>
</tr>
<tr>
<th>CoT GPT-4V</th>
<td align="center">-</td>
<td align="center">24.0</td>
<td align="center">26.7</td>
<td align="center">26.2</td>
<td align="center">38.6</td>
<td align="center">22.1</td>
<td align="center">24.4</td>
<td align="center">19.4</td>
<td align="center">27.9</td>
<td align="center">23.3</td>
<td align="center">25.2</td>
<td align="center">17.3</td>
<td align="center">21.4</td>
<td align="center">23.4</td>
<td align="center">23.8</td>
<td align="center">25.9</td>
<td align="center">4.4</td>
<td align="center">25.6</td>
</tr>
<tr>
<th>Gemini-1.5-Pro</th>
<td align="center">-</td>
<td align="center">19.2</td>
<td align="center">20.3</td>
<td align="center">35.7</td>
<td align="center">34.3</td>
<td align="center">19.8</td>
<td align="center">15.5</td>
<td align="center">20.9</td>
<td align="center">26.0</td>
<td align="center">26.7</td>
<td align="center">22.7</td>
<td align="center">14.5</td>
<td align="center">14.4</td>
<td align="center">16.5</td>
<td align="center">18.9</td>
<td align="center">10.3</td>
<td align="center">26.1</td>
<td align="center">17.3</td>
</tr>
<tr>
<th align="center" colspan="19">开源 MLLMs</th>
</tr>
<tr>
<th>LLaVA-1.5</th>
<td align="center">7B</td>
<td align="center">8.5</td>
<td align="center">7.0</td>
<td align="center">7.1</td>
<td align="center">10.7</td>
<td align="center">7.1</td>
<td align="center">4.8</td>
<td align="center">10.5</td>
<td align="center">7.7</td>
<td align="center">10.0</td>
<td align="center">9.2</td>
<td align="center">15.6</td>
<td align="center">10.2</td>
<td align="center">9.8</td>
<td align="center">5.3</td>
<td align="center">8.6</td>
<td align="center">4.4</td>
<td align="center">4.8</td>
</tr>
<tr>
<th>LLaVA-1.5</th>
<td align="center">13B</td>
<td align="center">11.1</td>
<td align="center">7.0</td>
<td align="center">14.3</td>
<td align="center">14.3</td>
<td align="center">9.1</td>
<td align="center">6.6</td>
<td align="center">6.0</td>
<td align="center">13.5</td>
<td align="center">5.6</td>
<td align="center">13.5</td>
<td align="center">10.4</td>
<td align="center">12.6</td>
<td align="center">14.7</td>
<td align="center">11.5</td>
<td align="center">13.8</td>
<td align="center">13.0</td>
<td align="center">10.7</td>
</tr>
<tr>
<th>InternLM-XComposer2-VL</th>
<td align="center">7B</td>
<td align="center">14.5</td>
<td align="center">9.3</td>
<td align="center">15.5</td>
<td align="center">12.1</td>
<td align="center">15.3</td>
<td align="center">11.3</td>
<td align="center">10.5</td>
<td align="center">14.4</td>
<td align="center">22.2</td>
<td align="center">19.3</td>
<td align="center">19.7</td>
<td align="center">15.6</td>
<td align="center">15.0</td>
<td align="center">11.9</td>
<td align="center">15.5</td>
<td align="center">26.1</td>
<td align="center">15.5</td>
</tr>
<tr>
<th>Ovis1.6-Gemma2-9B</th>
<td align="center">9B</td>
<td align="center">18.8</td>
<td align="center">13.3</td>
<td align="center">15.5</td>
<td align="center">22.1</td>
<td align="center">17.9</td>
<td align="center">11.3</td>
<td align="center">22.4</td>
<td align="center">23.1</td>
<td align="center">20.0</td>
<td align="center">20.2</td>
<td align="center">20.8</td>
<td align="center">18.0</td>
<td align="center">24.7</td>
<td align="center">15.6</td>
<td align="center">20.7</td>
<td align="center">17.4</td>
<td align="center">20.8</td>
</tr>
<tr>
<th>MiniCPM-v2.6</th>
<td align="center">8B</td>
<td align="center">18.4</td>
<td align="center">9.9</td>
<td align="center">19.0</td>
<td align="center">18.6</td>
<td align="center">21.8</td>
<td align="center">13.1</td>
<td align="center">13.4</td>
<td align="center">17.3</td>
<td align="center">20.0</td>
<td align="center">16.0</td>
<td align="center">25.4</td>
<td align="center">19.4</td>
<td align="center">20.7</td>
<td align="center">15.2</td>
<td align="center">27.6</td>
<td align="center">30.4</td>
<td align="center">22.0</td>
</tr>
<tr>
<th>LLaVA-OneVision</th>
<td align="center">8B</td>
<td align="center">18.3</td>
<td align="center">11.6</td>
<td align="center">16.7</td>
<td align="center">20.7</td>
<td align="center">18.5</td>
<td align="center">11.9</td>
<td align="center">14.9</td>
<td align="center">19.2</td>
<td align="center">13.3</td>
<td align="center">20.2</td>
<td align="center">17.9</td>
<td align="center">21.6</td>
<td align="center">23.4</td>
<td align="center">12.3</td>
<td align="center">22.4</td>
<td align="center">13.0</td>
<td align="center">24.4</td>
</tr>
<tr>
<th>Qwen2-VL</th>
<td align="center">8B</td>
<td align="center">19.2</td>
<td align="center">15.4</td>
<td align="center">20.2</td>
<td align="center">19.3</td>
<td align="center">16.9</td>
<td align="center">16.7</td>
<td align="center">17.9</td>
<td align="center">22.1</td>
<td align="center">22.2</td>
<td align="center">16.0</td>
<td align="center">19.1</td>
<td align="center">22.4</td>
<td align="center">22.5</td>
<td align="center">14.8</td>
<td align="center">19.0</td>
<td align="center">4.3</td>
<td align="center">23.8</td>
</tr>
<tr>
<th>InternVL2-8B</th>
<td align="center">8B</td>
<td align="center">18.4</td>
<td align="center">18.6</td>
<td align="center">22.6</td>
<td align="center">28.6</td>
<td align="center">22.1</td>
<td align="center">13.7</td>
<td align="center">10.4</td>
<td align="center">11.5</td>
<td align="center">13.3</td>
<td align="center">21.0</td>
<td align="center">20.8</td>
<td align="center">22.4</td>
<td align="center">20.5</td>
<td align="center">16.8</td>
<td align="center">17.2</td>
<td align="center">26.1</td>
<td align="center">24.2</td>
</tr>
<tr>
<th>InternVL2.5-8B</th>
<td align="center">8B</td>
<td align="center">19.7</td>
<td align="center">15.1</td>
<td align="center">23.8</td>
<td align="center">29.3</td>
<td align="center">16.2</td>
<td align="center">8.9</td>
<td align="center">11.9</td>
<td align="center">10.6</td>
<td align="center">8.9</td>
<td align="center">18.5</td>
<td align="center">22.0</td>
<td align="center">19.4</td>
<td align="center">15.4</td>
<td align="center">13.9</td>
<td align="center">22.4</td>
<td align="center">21.7</td>
<td align="center">19.6</td>
</tr>
<tr>
<th align="center" colspan="19">开源数学 MLLMs</th>
</tr>
<tr>
<th>数学-LLaVA</th>
<td align="center">13B</td>
<td align="center">15.7</td>
<td align="center">9.0</td>
<td align="center">20.2</td>
<td align="center">15.7</td>
<td align="center">18.2</td>
<td align="center">10.1</td>
<td align="center">10.5</td>
<td align="center">16.4</td>
<td align="center">14.4</td>
<td align="center">16.0</td>
<td align="center">20.2</td>
<td align="center">18.4</td>
<td align="center">17.6</td>
<td align="center">9.4</td>
<td align="center">24.1</td>
<td align="center">21.7</td>
<td align="center">17.9</td>
</tr>
<tr>
<th>多数学</th>
<td align="center">7B</td>
<td align="center">16.3</td>
<td align="center">11.3</td>
<td align="center">21.1</td>
<td align="center">15.5</td>
<td align="center">15.9</td>
<td align="center">11.3</td>
<td align="center">12.1</td>
<td align="center">15.5</td>
<td align="center">15.9</td>
<td align="center">18.5</td>
<td align="center">20.1</td>
<td align="center">16.4</td>
<td align="center">21.3</td>
<td align="center">13.3</td>
<td align="center">14.6</td>
<td align="center">13.3</td>
<td align="center">20.8</td>
</tr>
<tr>
<th>Math-PUMA-Qwen2-7B</th>
<td align="center">8B</td>
<td align="center">14.0</td>
<td align="center">5.0</td>
<td align="center">21.1</td>
<td align="center">21.1</td>
<td align="center">21.1</td>
<td align="center">11.0</td>
<td align="center">5.6</td>
<td align="center">15.7</td>
<td align="center">10.5</td>
<td align="center">13.8</td>
<td align="center">11.7</td>
<td align="center">15.8</td>
<td align="center">12.2</td>
<td align="center">17.8</td>
<td align="center">19.2</td>
<td align="center">15.8</td>
<td align="center">12.2</td>
</tr>
<tr>
<th>MAVIS</th>
<td align="center">7B</td>
<td align="center">18.5</td>
<td align="center">17.5</td>
<td align="center">19.5</td>
<td align="center">21.5</td>
<td align="center">19.0</td>
<td align="center">12.0</td>
<td align="center">14.0</td>
<td align="center">18.0</td>
<td align="center">16.0</td>
<td align="center">19.0</td>
<td align="center">21.0</td>
<td align="center">18.5</td>
<td align="center">19.5</td>
<td align="center">15.0</td>
<td align="center">19.0</td>
<td align="center">20.0</td>
<td align="center">20.0</td>
</tr>
<tr>
<th>AtomThink-EMOVA</th>
<td align="center">8B</td>
<td align="center">24.9</td>
<td align="center">23.5</td>
<td align="center">25.5</td>
<td align="center">32.0</td>
<td align="center">21.0</td>
<td align="center">15.8</td>
<td align="center">19.5</td>
<td align="center">21.5</td>
<td align="center">22.5</td>
<td align="center">21.5</td>
<td align="center">26.5</td>
<td align="center">25.5</td>
<td align="center">26.5</td>
<td align="center">27.5</td>
<td align="center">28.0</td>
<td align="center">23.0</td>
<td align="center">22.5</td>
</tr>
<tr>
<th>URSA-8B</th>
<td align="center">8B</td>
<td align="center">28.7</td>
<td align="center">28.1</td>
<td align="center">26.2</td>
<td align="center">35.0</td>
<td align="center">22.1</td>
<td align="center">15.5</td>
<td align="center">19.4</td>
<td align="center">18.3</td>
<td align="center">22.2</td>
<td align="center">21.8</td>
<td align="center">37.0</td>
<td align="center">27.0</td>
<td align="center">26.5</td>
<td align="center">31.1</td>
<td align="center">27.6</td>
<td align="center">17.4</td>
<td align="center">23.8</td>
</tr>
<tr>
<th>URSA-8B-PS-GRPO</th>
<td align="center">8B</td>
<td align="center">31.5</td>
<td align="center">30.1</td>
<td align="center">28.6</td>
<td align="center">29.3</td>
<td align="center">31.5</td>
<td align="center">20.8</td>
<td align="center">20.9</td>
<td align="center">26.9</td>
<td align="center">17.8</td>
<td align="center">24.4</td>
<td align="center">35.8</td>
<td align="center">33.6</td>
<td align="center">37.2</td>
<td align="center">37.7</td>
<td align="center">25.9</td>
<td align="center">26.1</td>
<td align="center">35.1</td>
</tr>
</tbody>
</table>

在 WE-MATH 8 中，URSA 系列在三阶段精度方面优于领先的通用 MLLM 和数学推理 MLLM。另外，URSA系列在立体的图形、变换、位置、方向等方面也有着显着的优势。这主要归功于大规模的对齐和指令微调，这为理解数学元素奠定了基础。

*表 6：MLLMs 在 **DYNAMATH** *testmini* 数据集上的详细性能比较，按主题领域和知识水平细分。*

<table>
<tbody>
<tr>
<td>模型</td>
<td align="center">尺寸</td>
<td align="center">ALL</td>
<td align="center">PG</td>
<td align="center">SG</td>
<td align="center">AG</td>
<td align="center">AL</td>
<td align="center">PT</td>
<td align="center">GT</td>
<td align="center">AR</td>
<td align="center">埃莱姆。</td>
<td align="center">高的</td>
<td align="center">本科生。</td>
</tr>
<tr>
<td align="center" colspan="13">闭源 MLLMs</td>
</tr>
<tr>
<td>GPT-4o</td>
<td align="center">-</td>
<td align="center">64.9</td>
<td align="center">56.8</td>
<td align="center">52.0</td>
<td align="center">61.0</td>
<td align="center">76.9</td>
<td align="center">51.8</td>
<td align="center">58.1</td>
<td align="center">61.5</td>
<td align="center">68.6</td>
<td align="center">61.8</td>
<td align="center">36.8</td>
</tr>
<tr>
<td>Claude-3.5-Sonnet</td>
<td align="center">-</td>
<td align="center">64.8</td>
<td align="center">49.9</td>
<td align="center">49.3</td>
<td align="center">55.3</td>
<td align="center">81.0</td>
<td align="center">44.1</td>
<td align="center">69.4</td>
<td align="center">61.2</td>
<td align="center">66.7</td>
<td align="center">62.6</td>
<td align="center">33.3</td>
</tr>
<tr>
<td>Gemini-1.5-Pro</td>
<td align="center">-</td>
<td align="center">60.5</td>
<td align="center">52.7</td>
<td align="center">42.7</td>
<td align="center">61.6</td>
<td align="center">70.8</td>
<td align="center">20.6</td>
<td align="center">65.2</td>
<td align="center">54.2</td>
<td align="center">62.9</td>
<td align="center">59.2</td>
<td align="center">37.1</td>
</tr>
<tr>
<td align="center" colspan="13">开源 MLLMs</td>
</tr>
<tr>
<td>Llava-v1.5-7B</td>
<td align="center">7B</td>
<td align="center">16.6</td>
<td align="center">10.5</td>
<td align="center">7.3</td>
<td align="center">19.5</td>
<td align="center">6.5</td>
<td align="center">8.2</td>
<td align="center">32.3</td>
<td align="center">10.8</td>
<td align="center">18.9</td>
<td align="center">13.3</td>
<td align="center">11.7</td>
</tr>
<tr>
<td>Llava-v1.6-34B</td>
<td align="center">34B</td>
<td align="center">27.1</td>
<td align="center">21.4</td>
<td align="center">25.3</td>
<td align="center">27.6</td>
<td align="center">14.9</td>
<td align="center">7.6</td>
<td align="center">32.7</td>
<td align="center">23.1</td>
<td align="center">35.9</td>
<td align="center">23.8</td>
<td align="center">16.6</td>
</tr>
<tr>
<td>Deepseek-VL-7B-Chat</td>
<td align="center">7B</td>
<td align="center">21.5</td>
<td align="center">16.0</td>
<td align="center">13.3</td>
<td align="center">26.5</td>
<td align="center">12.9</td>
<td align="center">4.7</td>
<td align="center">32.3</td>
<td align="center">12.7</td>
<td align="center">28.3</td>
<td align="center">19.0</td>
<td align="center">16.0</td>
</tr>
<tr>
<td>InternVL2-8B</td>
<td align="center">8B</td>
<td align="center">39.7</td>
<td align="center">33.9</td>
<td align="center">37.3</td>
<td align="center">32.5</td>
<td align="center">46.9</td>
<td align="center">15.9</td>
<td align="center">42.1</td>
<td align="center">37.3</td>
<td align="center">51.1</td>
<td align="center">37.4</td>
<td align="center">19.6</td>
</tr>
<tr>
<td>Qwen2-VL</td>
<td align="center">8B</td>
<td align="center">42.1</td>
<td align="center">40.3</td>
<td align="center">38.7</td>
<td align="center">39.9</td>
<td align="center">37.1</td>
<td align="center">8.2</td>
<td align="center">44.8</td>
<td align="center">39.2</td>
<td align="center">47.6</td>
<td align="center">42.2</td>
<td align="center">24.4</td>
</tr>
<tr>
<td>AtomThink-EMOVA</td>
<td align="center">8B</td>
<td align="center">40.9</td>
<td align="center">42.0</td>
<td align="center">37.9</td>
<td align="center">33.6</td>
<td align="center">58.0</td>
<td align="center">23.0</td>
<td align="center">44.0</td>
<td align="center">38.4</td>
<td align="center">52.5</td>
<td align="center">43.5</td>
<td align="center">32.0</td>
</tr>
<tr>
<td>URSA-8B</td>
<td align="center">8B</td>
<td align="center">44.7</td>
<td align="center">48.1</td>
<td align="center">38.0</td>
<td align="center">33.7</td>
<td align="center">66.9</td>
<td align="center">24.7</td>
<td align="center">39.2</td>
<td align="center">38.5</td>
<td align="center">53.5</td>
<td align="center">44.3</td>
<td align="center">41.8</td>
</tr>
<tr>
<td>URSA-8B-PS-GRPO</td>
<td align="center">8B</td>
<td align="center">47.4</td>
<td align="center">49.7</td>
<td align="center">40.1</td>
<td align="center">35.2</td>
<td align="center">65.7</td>
<td align="center">24.7</td>
<td align="center">45.2</td>
<td align="center">41.1</td>
<td align="center">53.5</td>
<td align="center">46.7</td>
<td align="center">43.2</td>
</tr>
</tbody>
</table>

*表 7：**MATHVERSE** *testmini* 上与闭源 MLLMs 和开源 MLLMs 的比较。闭源 MLLMs 的最佳结果已突出显示。突出显示了开源 MLLMs 的最佳和第二佳结果。*

<table>
<tbody>
<tr>
<th>模型</th>
<td align="center">#参数</td>
<td align="center">ALL</td>
<td align="center">TD</td>
<td align="center">TL</td>
<td align="center">TO</td>
<td align="center">VI</td>
<td align="center">VD</td>
<td align="center">VO</td>
</tr>
<tr>
<th align="center" colspan="9">基线</th>
</tr>
<tr>
<th>随机的</th>
<td align="center">-</td>
<td align="center">12.4</td>
<td align="center">12.4</td>
<td align="center">12.4</td>
<td align="center">12.4</td>
<td align="center">12.4</td>
<td align="center">12.4</td>
<td align="center">12.4</td>
</tr>
<tr>
<th>人类</th>
<td align="center">-</td>
<td align="center">64.9</td>
<td align="center">71.2</td>
<td align="center">70.9</td>
<td align="center">41.7</td>
<td align="center">61.4</td>
<td align="center">68.3</td>
<td align="center">66.7</td>
</tr>
<tr>
<th align="center" colspan="9">闭源 MLLMs</th>
</tr>
<tr>
<th>GPT-4o</th>
<td align="center">-</td>
<td align="center">50.8</td>
<td align="center">59.8</td>
<td align="center">50.3</td>
<td align="center">52.4</td>
<td align="center">48.0</td>
<td align="center">46.5</td>
<td align="center">47.6</td>
</tr>
<tr>
<th>GPT-4V</th>
<td align="center">-</td>
<td align="center">39.4</td>
<td align="center">54.7</td>
<td align="center">41.4</td>
<td align="center">48.7</td>
<td align="center">34.9</td>
<td align="center">34.4</td>
<td align="center">31.6</td>
</tr>
<tr>
<th>Gemini-1.5-Flash-002</th>
<td align="center">-</td>
<td align="center">49.4</td>
<td align="center">57.2</td>
<td align="center">50.5</td>
<td align="center">50.3</td>
<td align="center">47.6</td>
<td align="center">45.1</td>
<td align="center">45.4</td>
</tr>
<tr>
<th>Gemini-1.5-Pro</th>
<td align="center">-</td>
<td align="center">35.3</td>
<td align="center">39.8</td>
<td align="center">34.7</td>
<td align="center">44.5</td>
<td align="center">32.0</td>
<td align="center">36.8</td>
<td align="center">33.3</td>
</tr>
<tr>
<th>Claude-3.5-Sonnet</th>
<td align="center">-</td>
<td align="center">-</td>
<td align="center">-</td>
<td align="center">-</td>
<td align="center">-</td>
<td align="center">-</td>
<td align="center">-</td>
<td align="center">-</td>
</tr>
<tr>
<th>Qwen-VL-Plus</th>
<td align="center">-</td>
<td align="center">21.3</td>
<td align="center">26.0</td>
<td align="center">21.2</td>
<td align="center">25.2</td>
<td align="center">18.5</td>
<td align="center">19.1</td>
<td align="center">21.8</td>
</tr>
<tr>
<th align="center" colspan="9">开源通用 MLLMs</th>
</tr>
<tr>
<th>mPLUG-Owl2-7B</th>
<td align="center">7B</td>
<td align="center">10.3</td>
<td align="center">11.6</td>
<td align="center">11.4</td>
<td align="center">13.8</td>
<td align="center">11.1</td>
<td align="center">9.4</td>
<td align="center">8.0</td>
</tr>
<tr>
<th>MiniGPT4-7B</th>
<td align="center">7B</td>
<td align="center">12.2</td>
<td align="center">12.3</td>
<td align="center">12.9</td>
<td align="center">13.4</td>
<td align="center">12.5</td>
<td align="center">14.8</td>
<td align="center">8.7</td>
</tr>
<tr>
<th>LLaVA-1.5-13B</th>
<td align="center">13B</td>
<td align="center">12.7</td>
<td align="center">17.1</td>
<td align="center">12.0</td>
<td align="center">22.6</td>
<td align="center">12.6</td>
<td align="center">12.7</td>
<td align="center">9.0</td>
</tr>
<tr>
<th>SPHINX-V2-13B</th>
<td align="center">13B</td>
<td align="center">16.1</td>
<td align="center">20.8</td>
<td align="center">14.1</td>
<td align="center">14.0</td>
<td align="center">16.4</td>
<td align="center">15.6</td>
<td align="center">16.2</td>
</tr>
<tr>
<th>LLaVA-NeXT-34B</th>
<td align="center">34B</td>
<td align="center">34.6</td>
<td align="center">49.0</td>
<td align="center">37.6</td>
<td align="center">30.1</td>
<td align="center">35.2</td>
<td align="center">28.9</td>
<td align="center">22.4</td>
</tr>
<tr>
<th>InternLM-XComposer2-VL</th>
<td align="center">7B</td>
<td align="center">25.9</td>
<td align="center">36.9</td>
<td align="center">28.3</td>
<td align="center">42.5</td>
<td align="center">20.1</td>
<td align="center">24.4</td>
<td align="center">19.8</td>
</tr>
<tr>
<th>Deepseek-VL</th>
<td align="center">8B</td>
<td align="center">19.3</td>
<td align="center">23.0</td>
<td align="center">23.2</td>
<td align="center">23.1</td>
<td align="center">20.2</td>
<td align="center">18.4</td>
<td align="center">11.8</td>
</tr>
<tr>
<th>LLaVA-OneVision (SI)</th>
<td align="center">8B</td>
<td align="center">28.9</td>
<td align="center">29.0</td>
<td align="center">31.5</td>
<td align="center">34.5</td>
<td align="center">30.1</td>
<td align="center">29.5</td>
<td align="center">26.9</td>
</tr>
<tr>
<th>Qwen2-VL</th>
<td align="center">8B</td>
<td align="center">33.6</td>
<td align="center">37.4</td>
<td align="center">33.5</td>
<td align="center">35.0</td>
<td align="center">31.3</td>
<td align="center">30.3</td>
<td align="center">28.1</td>
</tr>
<tr>
<th>InternVL2-8B</th>
<td align="center">8B</td>
<td align="center">35.9</td>
<td align="center">39.0</td>
<td align="center">33.8</td>
<td align="center">36.0</td>
<td align="center">32.2</td>
<td align="center">30.9</td>
<td align="center">27.7</td>
</tr>
<tr>
<th>InternVL2.5-8B</th>
<td align="center">8B</td>
<td align="center">39.5</td>
<td align="center">43.0</td>
<td align="center">43.0</td>
<td align="center">43.0</td>
<td align="center">43.0</td>
<td align="center">42.2</td>
<td align="center">22.8</td>
</tr>
<tr>
<th align="center" colspan="9">开源数学 MLLMs</th>
</tr>
<tr>
<th>G-LLaVA-7B</th>
<td align="center">7B</td>
<td align="center">16.6</td>
<td align="center">20.9</td>
<td align="center">20.7</td>
<td align="center">21.1</td>
<td align="center">17.2</td>
<td align="center">14.6</td>
<td align="center">9.4</td>
</tr>
<tr>
<th>Math-LLaVA-13B</th>
<td align="center">13B</td>
<td align="center">22.9</td>
<td align="center">27.3</td>
<td align="center">24.9</td>
<td align="center">27.0</td>
<td align="center">24.5</td>
<td align="center">21.7</td>
<td align="center">16.1</td>
</tr>
<tr>
<th>Math-PUMA-Qwen2-7B</th>
<td align="center">8B</td>
<td align="center">33.6</td>
<td align="center">42.1</td>
<td align="center">35.0</td>
<td align="center">39.8</td>
<td align="center">33.4</td>
<td align="center">31.6</td>
<td align="center">26.0</td>
</tr>
<tr>
<th>数学-PUMA-DeepSeek-Math</th>
<td align="center">7B</td>
<td align="center">31.8</td>
<td align="center">43.4</td>
<td align="center">35.4</td>
<td align="center">47.5</td>
<td align="center">33.6</td>
<td align="center">31.6</td>
<td align="center">14.7</td>
</tr>
<tr>
<th>MAVIS-7B</th>
<td align="center">7B</td>
<td align="center">35.2</td>
<td align="center">43.2</td>
<td align="center">37.2</td>
<td align="center">35.2</td>
<td align="center">34.1</td>
<td align="center">29.7</td>
<td align="center">31.8</td>
</tr>
<tr>
<th>InfiMM-数学</th>
<td align="center">7B</td>
<td align="center">40.5</td>
<td align="center">46.7</td>
<td align="center">39.4</td>
<td align="center">41.6</td>
<td align="center">38.1</td>
<td align="center">40.4</td>
<td align="center">27.8</td>
</tr>
<tr>
<th>Multimath-7B</th>
<td align="center">7B</td>
<td align="center">27.7</td>
<td align="center">34.8</td>
<td align="center">30.8</td>
<td align="center">35.3</td>
<td align="center">28.1</td>
<td align="center">25.9</td>
<td align="center">15.0</td>
</tr>
<tr>
<th>AtomThink-EMOVA</th>
<td align="center">8B</td>
<td align="center">42.5</td>
<td align="center">48.1</td>
<td align="center">47.7</td>
<td align="center">45.7</td>
<td align="center">44.0</td>
<td align="center">44.2</td>
<td align="center">26.8</td>
</tr>
<tr>
<th>URSA-8B</th>
<td align="center">8B</td>
<td align="center">45.7</td>
<td align="center">55.3</td>
<td align="center">48.3</td>
<td align="center">51.8</td>
<td align="center">46.4</td>
<td align="center">43.9</td>
<td align="center">28.6</td>
</tr>
<tr>
<th>URSA-8B-PS-GRPO</th>
<td align="center">8B</td>
<td align="center">50.9</td>
<td align="center">57.3</td>
<td align="center">52.2</td>
<td align="center">50.2</td>
<td align="center">48.7</td>
<td align="center">47.6</td>
<td align="center">31.5</td>
</tr>
</tbody>
</table>

*表 8：在 **WE-MATH** *testmini* 子集上与闭源 MLLMs 和开源 MLLMs 的准确度比较。前 3 列显示一步、两步和三步问题的整体性能。其他列用于展示不同问题策略的性能。红色表示开源模型中性能最好的，蓝色表示性能第二好的。*

<table>
<tbody>
<tr>
<th rowspan="2">模型</th>
<td align="center" rowspan="2">#参数</td>
<td align="center" rowspan="2">S1</td>
<td align="center" rowspan="2">S2</td>
<td align="center" rowspan="2">S3</td>
<td align="center" colspan="2">内存</td>
<td align="center" colspan="2">PF</td>
<td align="center" colspan="2">SF</td>
<td align="center" colspan="2">TMF</td>
<td align="center" colspan="4">PD</td>
</tr>
<tr>
<td align="center">UCU</td>
<td align="center">AL</td>
<td align="center">CPF</td>
<td align="center">UPF</td>
<td align="center">CSF</td>
<td align="center">USF</td>
<td align="center">BTF</td>
<td align="center">CCF</td>
<td align="center">目录</td>
<td align="center">位置</td>
<td align="center">RoM</td>
<td align="center">CCP</td>
</tr>
<tr>
<th align="center" colspan="17">闭源 MLLMs</th>
</tr>
<tr>
<th>GPT-4o</th>
<td align="center">-</td>
<td align="center">72.8</td>
<td align="center">58.1</td>
<td align="center">43.6</td>
<td align="center">86.6</td>
<td align="center">39.1</td>
<td align="center">77.4</td>
<td align="center">71.6</td>
<td align="center">84.5</td>
<td align="center">62.3</td>
<td align="center">58.7</td>
<td align="center">69.4</td>
<td align="center">93.1</td>
<td align="center">72.7</td>
<td align="center">47.5</td>
<td align="center">73.3</td>
</tr>
<tr>
<th>GPT-4V</th>
<td align="center">-</td>
<td align="center">65.5</td>
<td align="center">49.2</td>
<td align="center">38.2</td>
<td align="center">82.5</td>
<td align="center">38.4</td>
<td align="center">70.7</td>
<td align="center">60.2</td>
<td align="center">76.6</td>
<td align="center">56.3</td>
<td align="center">57.8</td>
<td align="center">67.7</td>
<td align="center">79.3</td>
<td align="center">57.5</td>
<td align="center">47.8</td>
<td align="center">63.3</td>
</tr>
<tr>
<th>Gemini-1.5-Pro</th>
<td align="center">-</td>
<td align="center">56.1</td>
<td align="center">51.4</td>
<td align="center">33.9</td>
<td align="center">51.0</td>
<td align="center">31.2</td>
<td align="center">61.8</td>
<td align="center">45.0</td>
<td align="center">70.0</td>
<td align="center">57.5</td>
<td align="center">39.2</td>
<td align="center">62.7</td>
<td align="center">68.8</td>
<td align="center">54.1</td>
<td align="center">40.7</td>
<td align="center">60.0</td>
</tr>
<tr>
<th>Qwen-VL-Max</th>
<td align="center">-</td>
<td align="center">40.8</td>
<td align="center">30.3</td>
<td align="center">20.6</td>
<td align="center">19.4</td>
<td align="center">25.3</td>
<td align="center">39.8</td>
<td align="center">41.4</td>
<td align="center">43.6</td>
<td align="center">48.0</td>
<td align="center">43.8</td>
<td align="center">43.4</td>
<td align="center">41.4</td>
<td align="center">35.1</td>
<td align="center">40.7</td>
<td align="center">26.7</td>
</tr>
<tr>
<th align="center" colspan="17">开源通用 MLLMs</th>
</tr>
<tr>
<th>LLaVA-1.6</th>
<td align="center">7B</td>
<td align="center">23.0</td>
<td align="center">20.8</td>
<td align="center">15.8</td>
<td align="center">18.5</td>
<td align="center">20.5</td>
<td align="center">16.9</td>
<td align="center">29.6</td>
<td align="center">15.6</td>
<td align="center">18.6</td>
<td align="center">42.7</td>
<td align="center">24.1</td>
<td align="center">17.6</td>
<td align="center">43.3</td>
<td align="center">28.9</td>
<td align="center">26.7</td>
</tr>
<tr>
<th>LLaVA-1.6</th>
<td align="center">13B</td>
<td align="center">29.4</td>
<td align="center">25.3</td>
<td align="center">32.7</td>
<td align="center">21.7</td>
<td align="center">23.2</td>
<td align="center">23.4</td>
<td align="center">34.7</td>
<td align="center">25.3</td>
<td align="center">26.4</td>
<td align="center">37.5</td>
<td align="center">41.7</td>
<td align="center">26.9</td>
<td align="center">28.9</td>
<td align="center">37.1</td>
<td align="center">30.0</td>
</tr>
<tr>
<th>GLM-4V-9B</th>
<td align="center">9B</td>
<td align="center">47.3</td>
<td align="center">37.2</td>
<td align="center">38.2</td>
<td align="center">53.4</td>
<td align="center">37.0</td>
<td align="center">51.3</td>
<td align="center">46.5</td>
<td align="center">50.6</td>
<td align="center">38.2</td>
<td align="center">44.1</td>
<td align="center">45.2</td>
<td align="center">41.0</td>
<td align="center">49.3</td>
<td align="center">36.8</td>
<td align="center">53.3</td>
</tr>
<tr>
<th>MiniCPM-LLaMA3-V2.5</th>
<td align="center">8B</td>
<td align="center">39.8</td>
<td align="center">31.1</td>
<td align="center">29.7</td>
<td align="center">28.6</td>
<td align="center">37.0</td>
<td align="center">40.8</td>
<td align="center">39.8</td>
<td align="center">41.0</td>
<td align="center">38.6</td>
<td align="center">32.0</td>
<td align="center">42.7</td>
<td align="center">41.0</td>
<td align="center">42.7</td>
<td align="center">44.0</td>
<td align="center">43.3</td>
</tr>
<tr>
<th>长VA</th>
<td align="center">7B</td>
<td align="center">43.5</td>
<td align="center">30.6</td>
<td align="center">28.5</td>
<td align="center">24.5</td>
<td align="center">39.8</td>
<td align="center">45.1</td>
<td align="center">40.8</td>
<td align="center">51.9</td>
<td align="center">42.5</td>
<td align="center">45.6</td>
<td align="center">44.6</td>
<td align="center">44.5</td>
<td align="center">40.7</td>
<td align="center">47.5</td>
<td align="center">20.0</td>
</tr>
<tr>
<th>InternLM-XComposer2-VL</th>
<td align="center">7B</td>
<td align="center">47.0</td>
<td align="center">33.1</td>
<td align="center">33.3</td>
<td align="center">31.3</td>
<td align="center">46.5</td>
<td align="center">47.7</td>
<td align="center">42.6</td>
<td align="center">51.4</td>
<td align="center">43.9</td>
<td align="center">41.1</td>
<td align="center">50.6</td>
<td align="center">65.5</td>
<td align="center">53.9</td>
<td align="center">55.2</td>
<td align="center">40.0</td>
</tr>
<tr>
<th>Phi3-Vision</th>
<td align="center">4.2B</td>
<td align="center">42.1</td>
<td align="center">34.2</td>
<td align="center">27.9</td>
<td align="center">28.7</td>
<td align="center">16.0</td>
<td align="center">47.2</td>
<td align="center">38.8</td>
<td align="center">50.0</td>
<td align="center">44.4</td>
<td align="center">28.8</td>
<td align="center">31.2</td>
<td align="center">48.6</td>
<td align="center">49.2</td>
<td align="center">26.4</td>
<td align="center">50.0</td>
</tr>
<tr>
<th>DeepSeek-VL</th>
<td align="center">7B</td>
<td align="center">32.6</td>
<td align="center">26.7</td>
<td align="center">25.5</td>
<td align="center">16.6</td>
<td align="center">35.1</td>
<td align="center">27.3</td>
<td align="center">38.0</td>
<td align="center">24.2</td>
<td align="center">38.7</td>
<td align="center">50.0</td>
<td align="center">23.3</td>
<td align="center">24.5</td>
<td align="center">41.0</td>
<td align="center">51.7</td>
<td align="center">23.3</td>
</tr>
<tr>
<th>InternVL2-8B</th>
<td align="center">8B</td>
<td align="center">59.4</td>
<td align="center">43.6</td>
<td align="center">35.2</td>
<td align="center">71.4</td>
<td align="center">20.5</td>
<td align="center">62.0</td>
<td align="center">55.5</td>
<td align="center">67.1</td>
<td align="center">57.3</td>
<td align="center">54.0</td>
<td align="center">60.5</td>
<td align="center">58.6</td>
<td align="center">63.6</td>
<td align="center">44.5</td>
<td align="center">50.0</td>
</tr>
<tr>
<th>InternVL2.5-8B</th>
<td align="center">8B</td>
<td align="center">58.7</td>
<td align="center">43.1</td>
<td align="center">38.8</td>
<td align="center">48.7</td>
<td align="center">35.8</td>
<td align="center">65.5</td>
<td align="center">54.5</td>
<td align="center">62.3</td>
<td align="center">61.5</td>
<td align="center">47.8</td>
<td align="center">60.3</td>
<td align="center">79.0</td>
<td align="center">64.0</td>
<td align="center">51.1</td>
<td align="center">63.3</td>
</tr>
<tr>
<th>Qwen2-VL</th>
<td align="center">8B</td>
<td align="center">59.1</td>
<td align="center">43.6</td>
<td align="center">26.7</td>
<td align="center">62.7</td>
<td align="center">37.2</td>
<td align="center">62.6</td>
<td align="center">60.8</td>
<td align="center">65.7</td>
<td align="center">49.2</td>
<td align="center">52.5</td>
<td align="center">49.2</td>
<td align="center">48.1</td>
<td align="center">68.2</td>
<td align="center">55.0</td>
<td align="center">56.7</td>
</tr>
<tr>
<th>Gemma3-12B</th>
<td align="center">12B</td>
<td align="center">64.3</td>
<td align="center">47.2</td>
<td align="center">42.1</td>
<td align="center">83.1</td>
<td align="center">33.9</td>
<td align="center">70.2</td>
<td align="center">58.2</td>
<td align="center">77.5</td>
<td align="center">61.1</td>
<td align="center">50.1</td>
<td align="center">63.7</td>
<td align="center">82.6</td>
<td align="center">58.4</td>
<td align="center">36.8</td>
<td align="center">60.0</td>
</tr>
<tr>
<th align="center" colspan="17">开源数学 MLLMs</th>
</tr>
<tr>
<th>G-LLaVA</th>
<td align="center">7B</td>
<td align="center">32.4</td>
<td align="center">30.6</td>
<td align="center">32.7</td>
<td align="center">33.3</td>
<td align="center">29.1</td>
<td align="center">32.0</td>
<td align="center">37.9</td>
<td align="center">19.6</td>
<td align="center">33.5</td>
<td align="center">37.1</td>
<td align="center">32.8</td>
<td align="center">31.2</td>
<td align="center">33.2</td>
<td align="center">25.6</td>
<td align="center">40.0</td>
</tr>
<tr>
<th>数学-LLaVA</th>
<td align="center">13B</td>
<td align="center">38.7</td>
<td align="center">34.2</td>
<td align="center">34.6</td>
<td align="center">30.3</td>
<td align="center">17.9</td>
<td align="center">39.2</td>
<td align="center">40.4</td>
<td align="center">37.1</td>
<td align="center">37.7</td>
<td align="center">53.0</td>
<td align="center">51.3</td>
<td align="center">30.8</td>
<td align="center">30.8</td>
<td align="center">40.9</td>
<td align="center">46.7</td>
</tr>
<tr>
<th>Math-PUMA-Qwen2-7B</th>
<td align="center">8B</td>
<td align="center">53.3</td>
<td align="center">39.4</td>
<td align="center">36.4</td>
<td align="center">63.5</td>
<td align="center">42.5</td>
<td align="center">60.2</td>
<td align="center">45.9</td>
<td align="center">66.2</td>
<td align="center">48.6</td>
<td align="center">42.3</td>
<td align="center">53.5</td>
<td align="center">31.2</td>
<td align="center">37.7</td>
<td align="center">40.4</td>
<td align="center">46.7</td>
</tr>
<tr>
<th>MAVIS 不带 DPO</th>
<td align="center">7B</td>
<td align="center">56.9</td>
<td align="center">37.1</td>
<td align="center">33.2</td>
<td align="center">-</td>
<td align="center">-</td>
<td align="center">-</td>
<td align="center">-</td>
<td align="center">-</td>
<td align="center">-</td>
<td align="center">-</td>
<td align="center">-</td>
<td align="center">-</td>
<td align="center">-</td>
<td align="center">-</td>
<td align="center">-</td>
</tr>
<tr>
<th>MAVIS</th>
<td align="center">7B</td>
<td align="center">57.2</td>
<td align="center">37.9</td>
<td align="center">34.6</td>
<td align="center">-</td>
<td align="center">-</td>
<td align="center">-</td>
<td align="center">-</td>
<td align="center">-</td>
<td align="center">-</td>
<td align="center">-</td>
<td align="center">-</td>
<td align="center">-</td>
<td align="center">-</td>
<td align="center">-</td>
<td align="center">-</td>
</tr>
<tr>
<th>URSA-8B</th>
<td align="center">8B</td>
<td align="center">63.1</td>
<td align="center">56.4</td>
<td align="center">41.8</td>
<td align="center">59.1</td>
<td align="center">32.5</td>
<td align="center">72.3</td>
<td align="center">60.3</td>
<td align="center">70.9</td>
<td align="center">66.0</td>
<td align="center">51.4</td>
<td align="center">59.8</td>
<td align="center">58.3</td>
<td align="center">39.5</td>
<td align="center">58.8</td>
<td align="center">53.3</td>
</tr>
<tr>
<th>URSA-8B-PS-GRPO</th>
<td align="center">8B</td>
<td align="center">68.6</td>
<td align="center">64.2</td>
<td align="center">52.7</td>
<td align="center">52.6</td>
<td align="center">63.5</td>
<td align="center">68.5</td>
<td align="center">64.1</td>
<td align="center">68.8</td>
<td align="center">73.6</td>
<td align="center">69.4</td>
<td align="center">75.8</td>
<td align="center">72.1</td>
<td align="center">72.6</td>
<td align="center">73.6</td>
<td align="center">63.3</td>
</tr>
</tbody>
</table>

### C.2 MMathCoT-1M 的缩放规律

为了更好地说明 MMathCoT-1M 的有效性，我们通过在代表完整数据集的各种比例的随机选择的样本上训练模型来检查 SFT 的缩放定律。

*表 9：使用 MMathCoT-1M 的不同比率对 URSA-8B 进行缩放定律验证。*

<table>
<thead>
<tr>
<th align="center">比率</th>
<th align="center">MathVerse</th>
<th align="center">MathVision</th>
<th align="center">MathVista-GPS</th>
<th align="center">WEMATH</th>
<th align="center">DYNAMATH</th>
</tr>
</thead>
<tbody>
<tr>
<th align="center">1/4</th>
<td align="center">34.7</td>
<td align="center">20.5</td>
<td align="center">68.5</td>
<td align="center">43.5</td>
<td align="center">36.6</td>
</tr>
<tr>
<th align="center">1/2</th>
<td align="center">40.5</td>
<td align="center">22.8</td>
<td align="center">72.3</td>
<td align="center">47.7</td>
<td align="center">38.8</td>
</tr>
<tr>
<th align="center">3/4</th>
<td align="center">42.0</td>
<td align="center">26.7</td>
<td align="center">77.9</td>
<td align="center">50.9</td>
<td align="center">42.2</td>
</tr>
<tr>
<th align="center">1</th>
<td align="center">45.7</td>
<td align="center">28.7</td>
<td align="center">81.7</td>
<td align="center">53.6</td>
<td align="center">44.7</td>
</tr>
</tbody>
</table>

如表9所示，我们可以看到MMathCoT-1M清楚地展示了训练时间缩放规律，进一步验证了合成数据的有效性。

### C.3 第一阶段带来的更高上界

在第一阶段，我们通过数学密集型视觉语言对齐与指令微调获得了更强的基础 MLLM 和更好的推理能力。除表 1 中的结果外，我们还进一步解释为何第一阶段能够更好地服务后续实验，重点关注测试时扩展和 PRM 应用。我们选择 MathVerse、MathVision 和 MathVista-GPS 来观察 **pass@N** 指标。如图 7 所示，URSA-8B 始终优于当前领先的通用 MLLM 和数学推理 MLLM。这表明，尽管当前趋势更偏向 RL 相关技术，但监督微调的缩放规律仍能帮助突破基础模型的局限，并自然带来 BoN 评估表现、在线 RL 中高价值采样轨迹比例等方面的优势。首先，URSA-8B 更高的上限使第二阶段能够生成更丰富、更可靠的过程标签。此外，近期研究表明 RL 只能在其自身探索路径内逼近最优解[29, 4, 98]，因此第一阶段也自然抬高了 RL 阶段的潜在上限。这为 URSA-PS-GRPO-8B 的性能提供了最根本的优势。

![图7](paper_assets/2501.04686v6/x7.png)

*图 7：**在三个基准上进行 Pass@N 评估。***

### C.4 泛化验证

为了进一步验证所提出的 MMathCoT-1M 和 PRM 辅助 PS-GRPO 的有效性，我们从通用 MLLM 中选择 InternVL2.5-8B，从数学推理 MLLM 中选择 MultiMath，开展泛化验证实验。我们没有额外进行超参数调优，而是几乎直接沿用了表 13 中的设置。InternVL2.5-8B 和 MultiMath 的实验实现基于 Meng 等人 [99] 与 TRL [100]。考虑到这两个模型在发布时已经在通用领域或特定垂直领域完成了较充分的对齐，我们只进行两个训练阶段：（i）使用 MMathCoT-1M 增强基础模型的数学推理能力；（ii）使用 URSA-8B-RM 参与 PS-GRPO 流程。结果列于表 8。

*图 8：MMathCoT-1.1M 和 URSA-8B-RM 的进展在 InternVL2.5-8B 和 MultiMath 上辅助了 PS-GRPO。*

所提出的 MMathCoT-1M 和 PRM 辅助 PS-GRPO 展示了跨不同模型和基准的卓越泛化能力。当应用于 InternVL2.5-8B 和 MultiMath 时，这两个模型都显示出显着的性能改进。对于 InternVL2.5-8B，添加 MMathCoT-1M 可以将平均得分从 45.2 提高到 51.7，与 PS-GRPO 结合使用时的提升更为显着，达到 54.7。同样，对于 MultiMath，MMathCoT-1M 的平均分数从 43.1 增加到 48.7，PS-GRPO 的平均分数进一步增加到 51.2。这些结果凸显了我们的方法在增强不同模型和任务的数学推理能力方面的有效性。性能改进在各种基准测试中都是一致的，包括 MathVerse、MathVision、MathVista、WE-MATH、DYNAMATH 和 GeoQA，这表明我们的方法不仅有效而且广泛适用。

### C.5 其他基准上的补充结果

在比较 BoN 选择时，我们提供了 WE-MATH、DYNAMATH 和 GeoQA 的补充结果。如表 10 所示，URSA-8B-RM 在自一致性和 InternVL2.5-8B ORM 方面仍然具有优势。当使用 URSA-8B 作为推理模型时，URSA-8B-RM 的性能优于自洽模型，Best-of-8 性能相对提高了 4.6%、4.5% 和 2.7%。

*表 10：使用 BoN 在 WE-MATH、DYNAMATH 和 GeoQA 上的性能比较 TTS 与不同模型。*

<table>
<thead>
<tr>
<th rowspan="2">模型</th>
<th rowspan="2">方法</th>
<th align="center" colspan="4">WE-MATH</th>
<th align="center" colspan="4">DYNAMATH</th>
<th align="center" colspan="4">GeoQA</th>
</tr>
<tr>
<th align="center">N=4</th>
<th align="center">N=8</th>
<th align="center">N=16</th>
<th align="center">N=32</th>
<th align="center">N=4</th>
<th align="center">N=8</th>
<th align="center">N=16</th>
<th align="center">N=32</th>
<th align="center">N=4</th>
<th align="center">N=8</th>
<th align="center">N=16</th>
<th align="center">N=32</th>
</tr>
</thead>
<tbody>
<tr>
<th rowspan="3">URSA-8B</th>
<th>Self-Consistency</th>
<td align="center">56.3</td>
<td align="center">57.0</td>
<td align="center">57.7</td>
<td align="center">58.0</td>
<td align="center">46.2</td>
<td align="center">46.7</td>
<td align="center">47.5</td>
<td align="center">48.0</td>
<td align="center">74.1</td>
<td align="center">75.3</td>
<td align="center">75.9</td>
<td align="center">75.9</td>
</tr>
<tr>
<th>InternVL2.5-8B ORM</th>
<td align="center">56.0</td>
<td align="center">56.8</td>
<td align="center">57.4</td>
<td align="center">57.7</td>
<td align="center">45.9</td>
<td align="center">46.5</td>
<td align="center">47.2</td>
<td align="center">47.7</td>
<td align="center">73.8</td>
<td align="center">75.0</td>
<td align="center">75.6</td>
<td align="center">75.6</td>
</tr>
<tr>
<th>URSA-8B-RM</th>
<td align="center">58.2</td>
<td align="center">59.0</td>
<td align="center">59.3</td>
<td align="center">59.7</td>
<td align="center">47.5</td>
<td align="center">48.4</td>
<td align="center">49.5</td>
<td align="center">50.5</td>
<td align="center">76.1</td>
<td align="center">77.3</td>
<td align="center">78.0</td>
<td align="center">78.1</td>
</tr>
<tr>
<th rowspan="3">AtomThink-EMOVA</th>
<th>Self-Consistency</th>
<td align="center">51.7</td>
<td align="center">52.4</td>
<td align="center">52.9</td>
<td align="center">53.6</td>
<td align="center">42.3</td>
<td align="center">43.0</td>
<td align="center">43.7</td>
<td align="center">44.0</td>
<td align="center">65.7</td>
<td align="center">66.5</td>
<td align="center">66.6</td>
<td align="center">66.8</td>
</tr>
<tr>
<th>InternVL2.5-8B ORM</th>
<td align="center">51.5</td>
<td align="center">52.2</td>
<td align="center">52.7</td>
<td align="center">53.3</td>
<td align="center">42.1</td>
<td align="center">42.8</td>
<td align="center">43.5</td>
<td align="center">43.7</td>
<td align="center">65.5</td>
<td align="center">66.3</td>
<td align="center">66.4</td>
<td align="center">66.6</td>
</tr>
<tr>
<th>URSA-8B-RM</th>
<td align="center">53.7</td>
<td align="center">54.5</td>
<td align="center">55.0</td>
<td align="center">55.8</td>
<td align="center">44.1</td>
<td align="center">44.9</td>
<td align="center">45.6</td>
<td align="center">46.0</td>
<td align="center">67.9</td>
<td align="center">68.8</td>
<td align="center">69.0</td>
<td align="center">69.3</td>
</tr>
</tbody>
</table>

## 附录 D 模块选择标准

在模块选择方面，我们主要考虑了视觉编码器和LLM主干网络的选择。

#### 视觉编码器

为了训练具有更高过程敏感性的推理模型并促进PRM训练，我们首先使用手动选择的数据集（约80个示例）对DeepSeek-VL、Qwen2-VL等开源模型进行字幕测试。这些示例主要包括容易造成视觉混乱的功能相关问题和几何问题。我们手动检查这些开源模型的输出，发现Qwen2-VL和LLaVA-OneVision表现不佳；尽管它们在标准基准上的表现很好，但它们无法确保足够准确的数学描述。然而，DeepSeekVL的原生混合视觉塔设计，集成了高分辨率和低分辨率处理，主观上表现出更好的识别精度。我们推测这是由于 QwenViT [101] 更偏向于一般的多模态任务，导致与更简单的视觉主干相比，数学描述不太精确。因此，我们选择SigLiP-L+SAM-B混合视觉塔设计。

#### LLM 骨干模型

考虑到 QwenLM 系列的开源影响力，我们通过使用 QwenLM 系列主干来遵循先前工作的选择，例如 MathPUMA [11] 和 Multimath [23]。然而，我们考虑是否可以通过利用经过单峰数学后训练的指令模型来获得更高的性能，从而比较 Qwen2.5-7B-Instruct[^1] 和 Qwen2.5-Math-7B-Instruct[^2]。完成VL对齐阶段后，我们对MMathCoT-1M进行小规模对比实验，对50K示例进行微调。最后，我们的结果表明，使用 Qwen2.5-Math-Instruct 作为骨干网在 MathVision 和 MathVerse 上产生了大约 1 个百分点的优势。因此，我们将 Qwen2.5-Math-7B-Instruct 作为后续实验的 LLM 主干。

## 附录 E 消融实验

### E.1 不同数据类别的有效性

![图9](paper_assets/2501.04686v6/x8.png)

*图 9：针对不同类型源数据的每种综合策略效果良好。*

第一阶段，我们主要合成大规模多类CoT数据。

- **w/o $\mathcal{S}_{Ao}$**：在此变体中，仅答案 数据将恢复为其原始格式。这直接模仿了 Math-LLaVA 和 Math-PUMA 等模型使用的训练模式，涉及直接答案（“快思考”）和 CoT 思维的混合训练。
- **w/o $\mathcal{S}_{An}$**：此数据将替换为其原始组织结构，其中分析和最终答案以自由格式文本格式提供。
- **w/o $\mathcal{S}_{C}$**：这批数据将被替换为数学形式语言表达的推理，更好地体现符号和‘计划推理’形式的推理。

结果如图9所示。首先，在所有数据集上都显示，使用完整的合成数据可以获得最佳结果，凸显了MMathCoT-1M数据的作用。更具体地说，我们发现： i) $\mathcal{S}_{Ao}$ 对 MathVerse 和 MathVision 的影响最大，表明扩展的 CoT 数据对于追求绝对解精度的问题非常重要； ii) 然而，在WE-MATH上，替换$\mathcal{S}_{An}$导致了最显着的性能下降，这表明内容重写更好地符合WE-MATH基准提出的端到端要求，并且将训练与缺乏清晰逻辑序列的数据混合可能会降低层次准确性； iii) DYNAMATH的结果表明，从文本多样性的角度来看，重写和自然语言表述有效增强了推理的稳健性。这表明文本形式的思维模式在涉及图像变换的场景下往往更有效地保持思维过程的稳定性。

### E.2 外部闭源 MLLM 的选择

在本节中，我们主要展示Gemini-1.5-Flash-002[^3]与其他流行的MLLMs之间的指标比较，以及部分训练数据的比较。

*表 11：分别由两个闭源 MLLMs 合成的 50K 数据的 SFT 性能。*

<table>
<thead>
<tr>
<th>模型</th>
<th align="center">MathVista-GPS</th>
<th align="center">MathVerse</th>
<th align="center">MathVision</th>
</tr>
</thead>
<tbody>
<tr>
<th>URSA-8B 带GPT-4o</th>
<td align="center">54.1</td>
<td align="center">33.3</td>
<td align="center">18.8</td>
</tr>
<tr>
<th>URSA-8B 带Gemini-1.5-Flash-002</th>
<td align="center">55.1</td>
<td align="center">32.5</td>
<td align="center">18.3</td>
</tr>
</tbody>
</table>

- *指标比较*：我们比较了 Gemini-1.5-Flash-002、GPT-4o 和 GPT-4o-mini 在一些数学相关任务上的性能，如表 12 所示。我们观察到 Gemini-1.5-Flash-002 是在单峰和多峰数学任务上都表现良好的 MLLM，而 GPT-4o 并没有比它有显着优势。这在一定程度上保证了数据合成的质量。
- *SFT性能*：为了最好地说明性能变化，我们从MMathCoT-1M中随机采样了50K数据源，使用GPT-4o应用了三种相应的策略，然后进行了SFT。性能结果如表 11 所示。我们观察到使用 GPT-4o 并没有提供明显的优势。然而，MMathCoT-1M和DualMath-1.1M的构建涉及大约270万次API调用。 Gemini-1.5-flash-002的输出代币成本与GPT-4o-mini相同，是GPT-4o[^4]的三十分之一。因此，Gemini-1.5-Flash-002成为一个高性价比的选择。

然而，我们必须说，如果社区研究人员能够承担访问更强大的闭源模型的成本，我们预计结果会更好。

*表 12：数学基准上的模型性能比较*

<table>
<thead>
<tr>
<th>模型</th>
<th align="center">平均</th>
<th align="center">MATH</th>
<th align="center">MathVista</th>
<th align="center">MathVerse</th>
<th align="center">MathVision</th>
</tr>
</thead>
<tbody>
<tr>
<th>GPT-4o</th>
<td align="center">55.4</td>
<td align="center">76.6</td>
<td align="center">63.8</td>
<td align="center">50.8</td>
<td align="center">30.4</td>
</tr>
<tr>
<th>Gemini-1.5-Flash-002</th>
<td align="center">53.6</td>
<td align="center">79.9</td>
<td align="center">58.4</td>
<td align="center">49.4</td>
<td align="center">26.3</td>
</tr>
<tr>
<th>GPT-4o-mini</th>
<td align="center">48.0</td>
<td align="center">70.2</td>
<td align="center">56.7</td>
<td align="center">42.3</td>
<td align="center">22.8</td>
</tr>
</tbody>
</table>

## 附录 F 实现细节

### F.1 强化学习数据整理

*图 10：普通 GRPO 和 PS-GRPO 的 RL 数据统计。*

经过MMathCoT-1M指令微调后，整体准确率不超过50%。因此，我们认为它仍然具有在RL阶段直接利用的潜力。我们以类似于指令微调的类型混合比例收集20K数据，并在RL之前进行一次性静态过滤。具体来说，我们使用 URSA-8B 对这 20K 数据执行 8 次采样，过滤掉所有 8 次采样结果都错误或正确的示例。这留下了大约 15K+ 数据用于训练普通 GRPO 和 PS-GRPO。我们使用 TRL [100, 102] 实现 PS-GRPO。 RL数据统计见表10。

### F.2 参数与时间成本

在本节中，我们提供三个阶段的具体参数设置和时间成本。我们的实验基于Python 3.10和PyTorch 2.4.0+cu124。我们使用 AdamW [103] 作为优化器。我们使用完全共享数据并行（FSDP）[104]作为分布式训练框架。除非另有说明，实验默认在 32$\times$ NVIDIA-H100-HBM3 GPUs 上进行。此外，我们还提供数据构建中使用的重要参数。在生成正例和负例对的过程中，我们将 $temperature$ 设置为 1.0，$n\_return\_sequences$ 设置为 16，$top\_p$ 设置为 0.95。在 *BinaryErrorLocating* 阶段，我们将 $temperature$ 设置为 0.3，$n\_return\_sequences$ 设置为 16，$top\_p$ 设置为 0.95。

我们针对 URSA-8B 的架构调整了 vLLM [105] 框架（VLLM 最初不支持混合视觉塔 + MLP + Qwen2.5-math-Instruct），并将其用作推理阶段的加速工具。在数据对生成阶段，我们使用 16$\times$ NVIDIA-H100-HBM3 GPUs 进行推理，大约需要 28 小时。在 *BinaryErrorLocating* 阶段，我们还使用 16$\times$ NVIDIA-H100-HBM3 GPUs 进行推理，大约需要 20 小时。

表13展示了阶段I和阶段 II使用的超参数和时间成本。由于阶段 III使用的参数有些不同，我们将它们单独列出在表14中。最近，很多工作为GRPO提供了许多优化技巧，例如训练时动态采样、裁剪更高的值、放弃KL损失等[106, 58]。但是，为了独立验证 PRM 引导的奖励建模的有效性，我们没有在普通 GRPO 或 PS-GRPO 中添加这些技巧，以确保公平有效的验证过程。在应用 RL 之前，我们仅进行**一次性**基于难度的数据选择。

*表 13：阶段 I 和 II 的超参数设置和训练时间成本。*

<table>
<tbody>
<tr>
<th>超参数和成本</th>
<td align="center">VL-对齐</td>
<td align="center">指令微调</td>
<td align="center">PRM 训练</td>
</tr>
<tr>
<th>学习率</th>
<td align="center">1e-4</td>
<td align="center">1e-5</td>
<td align="center">Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.</td>
</tr>
<tr>
<th>时代</th>
<td align="center">1</td>
<td align="center">2</td>
<td align="center">2</td>
</tr>
<tr>
<th>预热比率</th>
<td align="center">0.02</td>
<td align="center">0.02</td>
<td align="center">0.02</td>
</tr>
<tr>
<th>体重衰减</th>
<td align="center">0.02</td>
<td align="center">0.01</td>
<td align="center">0.02</td>
</tr>
<tr>
<th>批量大小</th>
<td align="center">64</td>
<td align="center">128</td>
<td align="center">128</td>
</tr>
<tr>
<th>可训练零件</th>
<td align="center">对准器</td>
<td align="center">视觉编码器、对准器、底座 LLM</td>
<td align="center">基础 LLM</td>
</tr>
<tr>
<th>数据大小</th>
<td align="center">860K</td>
<td align="center">1.0M</td>
<td align="center">1.1M</td>
</tr>
<tr>
<th>时间成本</th>
<td align="center">～3.5小时</td>
<td align="center">～11小时</td>
<td align="center">～12小时</td>
</tr>
</tbody>
</table>

### F.3 基准测试

在本节中，我们介绍了四个使用的基准的详细子任务和指标，以更准确地演示评估。

#### MathVerse

MathVerse [72]是测试MLLMs在文本和图像形式的信息内容不同时推理能力的基准。具体来说，这些模型重点关注六种场景下的性能：文本主导 (TD)、文本精简版 (TL)、纯文本 (TO)、视觉密集型 (VI)、视觉主导 (VD) 和仅视觉 (VO)。

*表 14：阶段 III 的超参数设置和训练时间成本。*

#### WE-MATH

WE-MATH [75] 是第一个根据所需知识概念将复合问题分解为子问题的基准。图8中缩写对应的实际内容如下。 Mem：测量、PF：平面图形、SF：立体图形、TMF：图形的变换和运动、PD：位置和方向、AL：角度和长度、UCU：单位的理解和转换、CPF：平面图形的计算、UPF：平面图形的理解、CSF：立体的计算图形、USF：立体图形的理解、BTF：图形的基本变换、CCF：图形的切割与组合、Dir：方向、Pos：位置、RoM：路线图、CCP：坐标与位置的对应关系。

#### DYNAMATH

DYNAMATH [73] 是一个基准，旨在评估 MLLMs 在数学推理方面的稳健性。具体来说，它包括跨多个维度的测试，包括立体几何（SG）、平面几何（PG）、解析几何（AG）、代数（AL）、拼图测试（PT）、图论（GT）、算术（AR）、科学图形（SF）和统计（ST）。它包括 501 个种子问题和 5010 个生成问题。

#### GeoQA

GeoQA [40]数据集是一个专门用于评估和训练地理问答领域模型的数据集。其测试集包括 734 个样本。

#### MathVista

MathVista [74] 总共包含 5 个子任务：几何问题解答 (GPS)、数学应用题 (MWP)、图形问答 (FQA)、教科书问答 (TQA) 和视觉问答 (VQA)。与之前的数学推理工作一样，我们的模型训练过程并没有过度关注知识密集型任务（例如 VQA 和 FQA），因此我们选择 GPS 作为主要任务。

#### MathVision

MathVision [107] 是一个大规模多模态数学推理数据集，拓宽了多模态数学领域的学科范围。该测试集包含3,040个示例，涵盖16个关键能力，并提供可靠的测试性能。具体来说，表5中各项纪律指标的具体含义如下。 Alg：代数、AnaG：解析几何、Ari：算术、CombG：组合几何、Comb：组合、Cnt：计数、DescG：画法几何、GrphT：图论、Log：逻辑、Angle：公制几何 - 角度、Area：公制几何 - 面积、Len：公制几何长度、SolG：立体几何、Stat：统计、Topo：拓扑、TransG：变换几何。

#### 评估标准

我们的比较基于以下标准：首先，我们从每个基准测试的官方排行榜中选择结果。其次，我们从每个模型的原始论文或技术报告中选择结果。最后，我们使用 vLLM [105] 进行自己的推理和评估。我们的评估遵循基准本身的规则，如下：

- **基于规则的匹配**：WEMATH，GeoQA。
- **LLM-as-a-Judge**：MathVista、MathVision、MathVerse、Dynamath。

LLM-as-a-Judge 的提示如图 11 所示。

![图11](paper_assets/2501.04686v6/x9.png)

*图 11：用于答案匹配的 LLM-as-a-Judge 提示。*

### F.4 算法

在本节中，我们将3.2节中BIE的具体过程放入算法1中。具体来说，输入是一个指向错误答案的解。我们设置每步采样超参数$N_{\text{mid}}$。最初，我们将搜索范围的起点和终点分别设置为步骤1和步骤N。我们首先考虑步骤(1+N)//2的$mc$值。如果为正，说明第一个完全错误的步骤发生在后半段；否则，我们看上半场。这会将搜索次数减少到 $\mathcal{O}(\log N)$。

此外，我们在算法 2 中给出了 PS-GRPO 的具体流程。该流程包括结果奖励与“过程即结果”奖励的合并，以及后续的相对优势计算。

## 附录 G 提示词设计

### G.1 用于 MMathCoT-1M 合成的提示词

在本节中，我们提供三模块数据合成的具体提示。另外，Gemini-1.5-Flash是一个在实际经验中对提示和参数非常敏感的模型，我们将分享详细的调整经验。

#### CoT 扩展

仅答案数据源的 CoT 扩展提示如图 12 所示。我们要求 Gemini-1.5-Flash 给出一条能够导向真实答案的合理推理过程。实际执行后，我们发现输出有时并不够清晰。该模型偶尔会给出诸如“我们必须相信答案”或“让我假设”等轨迹，我们将这些短语视为模型无法自然、独立地完成求解的信号，并据此过滤相关样本。

#### 分析重写

分析格式 数据合成的重写提示如图 13 所示。对于分析格式的解决方案，我们将它们转换为清晰的分步格式轨迹。在此过程中，Gemini-1.5-flash-002没有表现出明显的质疑或提出有条件的请求。我们通过重新组织和完善语言逻辑来提高数据质量。

#### 格式统一

通过采用图 14 所示的统一格式提示来修改计划推理和符号方法的推理风格，我们能够提取与预训练风格一致的更自然的语言过程。一个例子就足以引起感知上有利的反应。

#### 双重检查

完成上述三点后，我们应用LLM作为法官对合成数据进行双重检查，确保解决方案不包含不合理的过程，例如不合时宜的提问、条件请求或推理循环。具体的提示设计如图15所示。经过这一层过滤，我们得到最终的MMathCoT-1M。

![图12](paper_assets/2501.04686v6/x10.png)

*图 12：仅答案 数据的 CoT 扩展提示。*

![图13](paper_assets/2501.04686v6/x11.png)

*图 13：分析格式 数据的分析重写提示。*

![图14](paper_assets/2501.04686v6/x12.png)

*图 14：数学和符号推理风格数据的格式统一提示。*

![图15](paper_assets/2501.04686v6/x13.png)

*图 15：双重检查提示，以确保合成的 CoT 推理数据具有高质量且适当的轨迹。*

### G.2 用于 DualMath-1.1M 合成的提示词

在本节中，我们将演示 MIE 中使用的提示。

- **几何问题**：对于几何问题，我们提示Gemini-1.5-Flash-002首先识别图中的关键几何特征。然后我们命令它对这些要素进行误解。最后，使用错误的信息来执行误导性的解决方案。总体设计如图 16 所示。
- **图表和函数**：对于 ChartQA 和数学函数，我们提示 Gemini-1.5-Flash-002 首先检查细粒度数据点。然后，我们尝试插入空间相似的数据以引起误解。这随后会导致自动标记的解决方案不正确。
- **LLM-as-a-Judge**：对于图表推理和函数问题，我们在Gemini-1.5-Flash-002上执行类似的过程。我们将其放置在图 17 中。

![图16](paper_assets/2501.04686v6/x14.png)

*图 16：几何相关问题的误解插入。*

![图17](paper_assets/2501.04686v6/x15.png)

*图 17：函数和图表相关问题的误解插入。*

## 附录 H 案例研究

### H.1 Best-of-N 评估展示

为了更清楚地说明 URSA-8B-RM 在 BoN 评估中的有效性，我们以 MathVista-GPS 为例进行展示（图 18）。该案例表明，URSA-8B-RM 对错误的定理应用以及角度关系误判都较为敏感。这种良好特性不仅使 URSA-8B-RM 在 BoN 评估中表现出色，也赋予其在在线强化学习中识别更高价值学习样本的潜力。

![图18](paper_assets/2501.04686v6/x16.png)

*图 18：URSA-8B-RM 在 Best-of-N 评估中充当验证者的情况。*

### H.2 误解插入引擎流程

如图 19 所示，MIE 执行三个主要操作： 首先，它解释图像中的数学信息。然后，它会在选定的步骤替换关键信息。最后根据修改后的条件继续推理。

![图19](paper_assets/2501.04686v6/x17.png)

*图 19：来自 MIE 的案例。我们引入特定的步骤级感知错误并继续错误推理来构建正确性标记的解决方案。*

### H.3 GRPO 过程中的失败模式

在本节中，我们通过示例直观说明 PS-GRPO 为何有效。我们首先引入“误报采样轨迹”的概念：这类采样轨迹虽然给出了正确答案，却无法提供完整、严谨的中间推理过程。它们通常可分为两类：（i）缺乏视觉条件对齐。此类解答在边关系、坐标值和定理应用等基本视觉因素的推理上表现出不一致，暴露了预训练阶段的缺陷，如图 20 所示；（ii）利用捷径模式。由于预训练或 SFT 期间图像特征与解题模式之间存在较强相关性，这些采样轨迹会跳过关键步骤，在做出基础描述后直接导向正确答案，如图 21 所示。因此，PS-GRPO 借助 PRM 在在线 RL 中的敏感错误识别能力，抑制这些错误动作所带来的有利更新方向，从而形成一种将结果奖励与基于过程奖励的惩罚结合起来的更优范式。

![图20](paper_assets/2501.04686v6/x18.png)

*图 20：误报采样轨迹分析 I。*

![图21](paper_assets/2501.04686v6/x19.png)

*图 21：误报采样轨迹分析 II。*

### H.4 朴素过程奖励建模失败的案例

在本节中，我们详细阐述了第 4 节中提到的过程奖励引导 RL 的两个基本缺陷，并提供了一些案例进行说明。在在线 RL 中，模型可以轻松识别获得过程奖励的模式，从而在回避PRM审查时进行保守的分析和简洁的响应。如图 22 所示，我们观察到该模型通常遵循独特的推理模式。他们最初全面阅读和分析给定的条件，但随后根据这种分析做出错误的决定，从而导致错误的答案。这表明，当显式建模过程奖励时，模型可以轻松地关注孤立地看起来“正确”的过程。然而，这些过程可能对最终结果并没有真正的帮助，反而可能导致模型优先考虑高过程奖励而不是准确性。

![图22](paper_assets/2501.04686v6/x20.png)

*图 22：两种流程奖励建模变体的不良案例分析。*

## 脚注

[^1]: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct

[^2]: https://huggingface.co/Qwen/Qwen2.5-Math-7B-Instruct

[^3]: https://deepmind.google/technologies/gemini/flash/

[^4]: https://docsbot.ai/models/gemini-1-5-flash-002
