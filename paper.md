# Unlocking Multimodal Mathematical Reasoning via Process Reward Model

> arXiv:2501.04686v6 [cs.CL], October 5, 2025  
> 39th Conference on Neural Information Processing Systems (NeurIPS 2025)

Ruilin Luo<sup>1,2,*</sup>, Zhuofan Zheng<sup>2,*</sup>, Yifan Wang<sup>1</sup>, Xinzhe Ni<sup>1</sup>, Zicheng Lin<sup>1</sup>, Songtao Jiang<sup>3</sup>, Yiyao Yu<sup>1</sup>, Chufan Shi<sup>1</sup>, Lei Wang<sup>4</sup>, Ruihang Chu<sup>1,†</sup>, Jin Zeng<sup>2,†</sup>, Yujiu Yang<sup>1</sup>

1. Tsinghua University  
2. ByteDance  
3. Zhejiang University  
4. Ping An Technology (Shenzhen) Co., Ltd

* Equal contribution. Work done during Ruilin’s internship at ByteDance.  
† Corresponding authors: ruihangchu@gmail.com, zengjin@bytedance.com

## Abstract

Process Reward Models (PRMs) have shown promise in enhancing the mathematical reasoning capabilities of Large Language Models (LLMs) through Test-Time Scaling (TTS). However, their integration into multimodal reasoning remains largely unexplored. In this work, we take the first step toward unlocking the potential of PRMs in multimodal mathematical reasoning. We identify three key challenges: (i) the scarcity of high-quality reasoning data constrains the capabilities of foundation Multimodal Large Language Models (MLLMs), which imposes further limitations on the upper bounds of TTS and reinforcement learning (RL); (ii) a lack of automated methods for process labeling within multimodal contexts persists; (iii) the employment of process rewards in unimodal RL faces issues like reward hacking, which may extend to multimodal scenarios. To address these issues, we introduce **URSA**, a three-stage **U**nfolding multimodal p**R**ocess-**S**upervision **A**ided training framework. We first construct MMathCoT-1M, a high-quality large-scale multimodal Chain-of-Thought (CoT) reasoning dataset, to build a *stronger math reasoning foundation MLLM*, URSA-8B. Subsequently, we go through an automatic process to synthesize process supervision data, which emphasizes both logical correctness and perceptual consistency. We introduce DualMath-1.1M to facilitate the training of URSA-8B-RM. Finally, we propose **P**rocess-**S**upervised **G**roup-**R**elative-**P**olicy-**O**ptimization (**PS-GRPO**), pioneering a *multimodal PRM-aided online RL method* that outperforms vanilla GRPO. With PS-GRPO application, URSA-8B-PS-GRPO outperforms Gemma3-12B and GPT-4o by 8.4% and 2.7% on average across 6 benchmarks. Code, data and checkpoint can be found at https://github.com/URSA-MATH.

## 1 Introduction

Following the substantial progress of Large Language Models (LLMs) in math reasoning [1, 2, 3, 4, 5, 6, 7, 8], the math reasoning capabilities of Multimodal Large Language Models (MLLMs) have increasingly garnered attention [9, 10, 11, 12, 13]. Previous work has typically focused on aspects such as math reasoning data curation [14, 15, 16, 17, 18], training math-intensive vision encoders [19, 20], enhancing vision-language alignment [11, 21], or the application of post-training techniques [22, 23, 24, 13]. Given the success of Process Reward Models (PRMs) in improving LLM reasoning through methods like Test-Time Scaling (TTS) [25, 26] and Reinforcement Fine-Tuning (ReFT) [27, 28], the application of PRMs to multimodal reasoning remains unexplored.

![Figure 1](paper_assets/2501.04686v6/x1.png)

*Figure 1: Performance comparison with leading open-source MLLMs and GPT-4o.*

In this work, we take the first step toward integrating PRMs into multimodal math reasoning. We identify three key challenges: (i) Since both TTS and RL are heavily influenced by the strength of foundation models [29, 25], the limited availability of large-scale, high-quality reasoning data constrains the upper bounds of current MLLMs and weakens the effectiveness of PRM integration; (ii) There hasn’t yet been adequate automated process labeling techniques merged within multimodal contexts, where both logical validity and perceptual consistency should be emphasized [30, 31, 32]. (iii) While PRMs can be effectively used in TTS, applying them directly in online RL introduces risks such as reward hacking and length bias in rewarding [33, 34].

To address these challenges, we propose the **URSA** framework, a three-stage **U**nfolding multimodal p**R**ocess-**S**upervision **A**ided training pipeline that supports both the construction and application of multimodal PRMs. In Stage I, we curate **MMathCoT-1M**, a large-scale, high-quality multimodal Chain-of-Thought dataset synthesized from 1.43 million open-source examples, which enhances the foundation model’s reasoning capabilities through targeted instruction tuning. In Stage II, we construct **DualMath-1.1M** via a dual-view process supervised data synthesis strategy which combines a binary error locating engine and a misinterpretation insertion engine. It provides complementary signals for logical validity and visual grounding, and is used to train a process reward model. In Stage III, we analyze the limitations of scalar process reward modeling in online RL and propose **Process Supervision-GRPO (PS-GRPO)**, which mitigates reward hacking and PRM’s length bias in rewarding by implicitly penalizing process-level inconsistencies during policy optimization.

Results on 6 multimodal reasoning benchmarks show that our PRM improves Best-of-N verification, surpassing self-consistency and outcome-based baselines. When used in PS-GRPO, the resulting model achieves state-of-the-art performance among open-source MLLMs of similar size. Our contributions are as follows:

- We release two large-scale open-source datasets, MMathCoT-1M and DualMath-1.1M, to address the scarcity of high-quality multimodal CoT reasoning and process supervision data.
- We propose PS-GRPO, an online reinforcement learning algorithm that incorporates multimodal PRMs by comparing the relative quality of rollouts, rather than relying on scalar reward modeling. It effectively mitigates PRM’s reward hacking and length bias in rewarding.
- Experimental results show that our reward model improves both test-time verification and online training. With PS-GRPO application (Figure 1), URSA-8B-PS-GRPO outperforms Gemma3-12B and GPT-4o by 8.4% and 2.7% on average across 6 benchmarks.

## 2 Stage I: Math-Intensive Alignment and Instruction Tuning

### 2.1 Collection of Vision-Language Alignment Data

We employ a LLaVA-like architecture and first collect vision-language alignment data directly from existing open-source datasets [35, 36, 37, 38]. As demonstrated in Figure 2, we collect URSA-Alignment-860K from Multimath [23], MAVIS [19] and Geo170K [18]. We then filter out samples with overly verbose captions, to form an 860K math-intensive alignment dataset. Following the engineering practices of previous work, we only train the MLP projector in the alignment step.

### 2.2 CoT Reasoning Data Synthesis

For a powerful foundation building, we collect 1.43M samples from existing math reasoning datasets to support the construction of large-scale CoT reasoning data. As shown in Figure 2, data is sourced from MathV360K [15], Multimath [23], MAVIS [19], Geo170K [18] and VarsityTutors [11]. Based on the type of solution, we categorize the data into *answer-only*, *analysis-formatted*, and *CoT-formatted*. We adopt different synthesis strategies for them to curate high-quality CoT reasoning trajectories. We utilize Gemini-1.5-Flash-002 (refer to $\mathcal{G}$ below) as a cost-effective tool for data curation, avoiding expensive large-scale manual annotation.

![Figure 2](paper_assets/2501.04686v6/x2.png)

*Figure 2: Statistics of URSA-Alignment-860K and MMathCoT-1M.*

#### CoT Expansion.

For *answer-only* data $\mathcal{D}_{1}=\{(x_{i},y_{i})\}_{i=1}^{N_{1}}$, such as MathV360K [15], each sample contains a question $x_{i}$ and a ground-truth answer $y_{i}$. This type of data is heavily used in previous works for fast thinking reasoning mode [15, 11, 16]. However, answer-only training restricts the model from fully capturing the problem-solving process. It may lead to memory-based reasoning, hindering the model’s ability to directly provide answers to more complex reasoning problems [39]. We expand certain scale CoT reasoning trajectories for this category of data. Given a expansion prompt $\mathcal{P}_{\mathcal{C}}$, we provide $x_{i}$ and $y_{i}$, then prompt $\mathcal{G}$ to output the reasoning trajectory leading to the answer $y_{i}$, yielding the expanded solutions $\mathcal{S}_{Ao}=\mathcal{G}(\mathcal{P}_{\mathcal{C}};\{x_{i},y_{i}\}_{i=1}^{N_{1}})$.

#### Rewriting.

This strategy is designed for *analysis-formatted* samples, denoted as $\mathcal{D}_{2}=\{(x_{i},y_{i},a_{i})\}_{i=1}^{N_{2}}$. This includes datasets like MAVIS-Geo, MAVIS-MetaGen [19], VarsityTutors [11], and Geo170K-QA [40]. Each sample contains a question $x_{i}$, an answer $y_{i}$, and textual analysis $a_{i}$. While this type of data provides walkthroughs, it often suffers from two issues: (i) It lacks strict step-by-step logic, exhibiting jumps in language or reasoning. (ii) A significant portion of the answers are relatively brief and cannot provide rich rationale. Given a rewriting prompt $\mathcal{P}_{\mathcal{R}}$, we utilize $\mathcal{G}$ to transcribe these solutions, thereby enhancing their step-by-step reasoning trajectories and linguistic diversity, resulting in the rewritten set $\mathcal{S}_{An}=\mathcal{G}(\mathcal{P}_{\mathcal{R}};\{x_{i},y_{i},a_{i}\}_{i=1}^{N_{2}})$.

#### Format Unification.

This strategy is used for *CoT-formatted* data, primarily sourced from Multimath-EN-300K [23], which is collected from K-12 textbooks and contains mathematical language and symbolic-style reasoning solutions. This portion of the data, $\mathcal{D}_{3}=\{(x_{i},y_{i},c_{i})\}_{i=1}^{N_{3}}$, consists of a question $x_{i}$, an answer $y_{i}$, and a solution $c_{i}$. We unify the format through natural language stylization using a prompt $\mathcal{P}_{\mathcal{F}}$ with $\mathcal{G}$, producing the unified set $\mathcal{S}_{C}=\mathcal{G}(\mathcal{P}_{\mathcal{F}};\{x_{i},y_{i},c_{i}\}_{i=1}^{N_{3}})$.

![Figure 3](paper_assets/2501.04686v6/x3.png)

*Figure 3: Pipeline of URSA. Stage 1 depicts the workflow of data curation as described in Section 2. Stage 2 illustrates how binary error locating and misinterpretation insertion facilitate the automation of process supervision data. Stage 3 demonstrates how our PS-GRPO operates by imposing penalties on rollouts that are questioned by the PRM.*

#### MMathCoT-1M.

Finally, we filter out instances where: (i) Correctness is violated: the generated content altered the original answer, or (ii) Consistency is problematic: the solution includes text that questions the original answer or makes new assumptions to force the given answer. This process yields MMathCoT-1M. The complete prompt designs can be found in Appendix G.

We perform full-parameter instruction fine-tuning with MMathCoT-1M to train URSA-8B, based on the aligned model. The SFT dataset $\mathcal{D}_{SFT}$ is formed by the union of the curated solutions, i.e., $\mathcal{D}_{SFT}=\{(x_{i},y_{i})\mid(x_{i},y_{i})\in\mathcal{S}_{Ao}\cup\mathcal{S}_{An}\cup\mathcal{S}_{C}\}$. Training objective is demonstrated in Equation 1.

$$
\mathcal{L}_{SFT} =-\mathbb{E}_{(x,y)\sim\mathcal{D}_{SFT}}\sum_{t=1}^{T}\log\mathcal{M}(y_{t}|x,y_{<t})
$$
<div align="right">(1)</div>

In this phase, we construct a stronger reasoning foundation model, URSA-8B, with the expectation of achieving a higher bound at inference time and to process supervision data of greater diversity.

## 3 Stage II: Dual-View Process Supervised Data Synthesis

### 3.1 Binary Error Locating Engine

Following suggestions by previous work [41, 42, 43], we train a PRM for first error step identification. We collect $\sim$553K incorrect solutions from URSA-8B’s zero-shot inference on MMathCoT-1M. Erroneous steps in these solutions are labeled using Monte Carlo Tree Search (MCTS). For MCTS, an operation $\mathcal{F}(\{s_{1},\ldots,s_{i}\},N)$ generates $N$ rollouts from a reasoning prefix $\{s_{1},\ldots,s_{i}\}$. The single step’s Monte Carlo estimation value, $mc_{i}$, is the fraction of these rollouts leading to a correct answer:

$$
mc_{i}=\frac{|\text{Correct rollouts from $\mathcal{F}(\{s_{1},s_{2},\ldots,s_{i}\},N)$}|}{|\text{Total rollouts from $\mathcal{F}(\{s_{1},s_{2},\ldots,s_{i}\},N)$}|}
$$
<div align="right">(2)</div>

A step $s_{i}$ is deemed “potentially correct” if $mc_{i}>0$ [43, 42]. We optimize the identification of first error step using Binary Error Locating Engine (BEL): if the middle step has positive $mc$ (i.e. $mc_{mid}>0$), the error is in the latter half; otherwise, in the first (see Algorithm 1). To mitigate step-level label bias and include positive examples, we add $\sim$180K correct solutions (1/3 the number of incorrect ones), with all steps easily marked “True”. This yields $\mathcal{S}_{BEL}$, a 773K process annotation dataset based on correctness potential.

### 3.2 Misinterpretation Insertion Engine

Apart from logical errors, the perception inconsistency between images and text in reasoning steps is a unique problem in multimodal scenarios [30, 44, 45]. We propose a Misinterpretation Insertion Engine (MIE) to artificially insert hallucinatory information, automatically constructing process supervision data with incorrect reasoning paths starting from the insertion point. Specifically, MIE includes three steps. First, we prompt $\mathcal{G}$ to perform a captioning task, extracting mathematical paradigm information from the image as much as possible. Second, the model $\mathcal{G}$ is required to focus on potentially confusable conditions within the existing correct solution and modify them using adjacent or similar conditions. Finally, the model $\mathcal{G}$ is prompted to continue reasoning based on the step with the inserted error. We leverage strong instruction-following capability of $\mathcal{G}$, instructing it to automatically assign negative labels to every subsequent step following the erroneous insertion. We generate $\sim$302K samples $\mathcal{S}_{MIE}$ using this strategy. Cases from MIE can be found in the Appendix H.2.

### 3.3 PRM Training

As shown in Equation 3, we merge two types of data, proposing a $\sim$1.1M process supervision data called DualMath-1.1M. During training, we append a special token after each step to indicate its predicted correctness. We model the PRM training as a binary classification task for the correctness of each step, as shown in Equation 4, here $\pi_{p}$ is the trained PRM based on URSA-8B. $e_{j}$ and $y_{j}$ represent single step and corresponding label ($y_{j}\in\{0,1\}$).

$$
\mathcal{D}_{PRM}=\{(e,y_{e})\sim\mathcal{S}_{BEL}\cup\mathcal{S}_{MIE}\}
$$
<div align="right">(3)</div>

$$
\mathcal{L}_{PRM}=-\mathbb{E}_{(e,y)\sim\mathcal{D}_{PRM}}\sum_{j=1}^{|e|}\Big[y_{j}\log\pi_{p}(e_{j})+(1-y_{j})\log(1-\pi_{p}(e_{j}))\Big]
$$
<div align="right">(4)</div>

Thus, Stage II delivers URSA-8B-RM, a strong PRM trained on DualMath-1.1M—the **first** large-scale, automatically labeled dataset for multimodal reasoning process supervision. While BoN evaluation demonstrates PRM’s value in TTS, a critical question emerges: how can its guidance be directly integrated into MLLM post-training? This remains largely uncharted. Stage III draws a lesson about why previous scalar process reward modeling tends to fail, and then we achieve effective progress through process-as-outcome reward modeling.

## 4 Stage III: Integrating multimodal PRM into RL

Inspired by successes like DeepSeek-R1 [46], several recent studies have tried to adapt outcome reward-based GRPO for multimodal reasoning, demonstrating notable progress [47, 48, 49, 50]. Outcome reward-based GRPO computes the $i$-th response’s advantage through normalizing in-group rewards. However, outcome reward-based GRPO ignores the quality of reasoning processes [41, 51, 52].

![Figure 4](paper_assets/2501.04686v6/x4.png)

*Figure 4: Figure (a)-(d) respectively illustrate training rewards, response length, response step number and test set accuracy of vanilla GRPO and two variants proposed in Section 4. Test set is randomly selected 500 examples from MMathCoT-1M for an in-domain evaluation.*

Following most standard response-level and step-level reward modeling in RL [43, 28, 46, 13, 53], we examine two simple variants of GRPO with integrated scalar process rewards to reveal the failure patterns during the training process [54]. *Variant 1*: For $i$-th rollout, the reward is the sum of the outcome reward and the average process reward, i.e. $r^{i}=r_{o}^{i}+\bar{r_{s}^{i}}$. *Variant 2*: Despite the outcome reward, a scalar process reward $r_{s,t}^{i}$ is assigned to the $i$-th rollout’s $t$-th step.

![Figure 5](paper_assets/2501.04686v6/x5.png)

*Figure 5: Figure (a) shows the BoN evaluation during GRPO training. We select the best rollout using the mean value of process rewards. Figure (b) illustrates the proportion of rollouts where URSA-8B-RM identifies “drop-moment” and the final results are indeed incorrect. Figures (c) and (d) display the response length and test accuracy during PS-GRPO training.*

We observe two highly significant conclusions from Figure 4: **(i)** *High susceptibility to reward hacking*. The test accuracy of both variants is lower than vanilla GRPO. This indicates that when process scalar rewards are employed as learning objectives, the model quickly learns strategies that cater to process correctness. However, correctness in the process does not necessarily correlate fully with the heuristics leading to the ground-truth. **(ii)** *PRM’s length bias in rewarding*. We observe a trend where increased training leads to shorter model responses and fewer reasoning steps. This phenomenon stems from an inherent length bias in the PRM’s training labels; for examples with incorrect answers, steps taken after the first error are unlikely to yield a correct solution. This results in the PRM conservatively rewards the later stages of a reasoning rollout, thereby encouraging the MLLM towards more passive reasoning and a reliance on pattern recognition from existing conditions or simpler heuristics.

#### PS-GRPO

The findings above confirm the consideration that flaws in the reward function are amplified when scalar process rewards serve as the optimization target [55, 33]. We ask “Which internal signals of PRM can be trusted?” We employ two views to investigate the reliable region of the PRM: first, the BoN performance during online learning, and second, the PRM’s error identification capability. Regarding the latter, we introduce the concept of a *“drop-moment”* within the PRM’s reward sequence, which signifies that the PRM questions the validity of the preceding steps. Specifically, for a given solution’s PRM reward sequence $\{r_{p1}^{i},r_{p2}^{i},\cdots,r_{pN}^{i}\}$, a significant decrease in reward between consecutive steps indicates the occurrence of such a drop-moment.

$$
\delta_{p}^{i}=\max\left\{\frac{r_{p,j}^{i}-r_{p,j+1}^{i}}{r_{p,j}^{i}} \mid j=0,1,\dots,N-1\right\}>\rho
$$
<div align="right">(5)</div>

Here, $\rho$ represents PRM’s drop-moment threshold. As illustrated in Figure 5, the PRM’s ability for BoN selection and error identification remains largely unimpaired during the online RL process, exhibiting stable performance. This suggests that *although the scalar reward from the PRM in online RL might be unreliable, the relative quality of solutions it reveals is comparatively trustworthy*.

We leverage this beneficial property to address the reward sparsity problem in GRPO [56, 57, 58], aiming to make online RL focus more on learning from rollouts that have accurate results and rigorous processes. We use $\rho$ from Equation 5 as the occurrence threshold for a “drop-moment”; when it occurs, we apply a reward penalty $\gamma$ to rollouts with correct results. This both differentiates the learning value of outcome-correct rollouts and, due to its focus on relative drops in reward sequences, circumvents the impact of PRM’s length bias in rewarding.

$$
R^{i}=
\begin{cases}
1, & o^{i}\text{ is correct and }\delta_{p}^{i}<\rho \\
1-\gamma, & o^{i}\text{ is correct and }\delta_{p}^{i}\ge\rho \\
0, & \text{otherwise}
\end{cases}
$$
<div align="right">(6)</div>

We utilize reward modeling in Equation 6 to conduct a process-supervised GRPO, which facilitates the computation of in-group advantages in Equation 7.

*Table 1: Performance Comparison on 6 math reasoning benchmarks. We use accuracy for MathVerse, MathVision, MathVista and GeoQA. We use Score (Loose) on WE-MATH. And average-case accuracy is employed on DYNAMATH. Best results of Closed-source MLLMs are highlighted in green. Best and runner-up results of Open-source MLLMs are highlighted in red and blue.*

<table>
  <tbody>
    <tr>
      <th></th>
      <td rowspan="2" align="center">Size</td>
      <td rowspan="2" align="center">Avg</td>
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
      <th colspan="9" align="center">Closed-Source MLLMs</th>
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
      <th>GPT-4o-mini [59]</th>
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
      <th colspan="9" align="center">Open-Source General MLLMs</th>
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
      <th colspan="9" align="center">Open-Source Reasoning MLLMs</th>
    </tr>
    <tr>
      <th>Math-LLaVA [15]</th>
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
      <th>MultiMath [23]</th>
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
      <th>InfiMM-Math [14]</th>
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
      <th>MathGLM-Vision [9]</th>
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

*Table 2: Comparison of TTS on URSA-8B and AtomThink-EMOVA using BoN performance.*

<table>
  <thead>
    <tr>
      <th rowspan="2">Model</th>
      <th rowspan="2">Method</th>
      <th colspan="4" align="center">MathVerse</th>
      <th colspan="4" align="center">MathVista-GPS</th>
      <th colspan="4" align="center">MathVision</th>
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

## 5 Experiments

### 5.1 Experimental Setup

#### Benchmarks

We evaluate our URSA-series models on 6 widely used reasoning benchmarks, including MathVerse [72], DYNAMATH [73], MathVista [74], WE-MATH [75], GeoQA [40] and MathVision [43]. Detailed description and evaluation criteria can be found in Appendix F.3. We consistently employ zero-shot inference for comparison.

#### Baselines

We include some leading proprietary MLLMs, such as GPT-4o and GPT-4o-mini [59]. For open-source MLLMs with comparable size, we select InternVL-series [64, 76], LLaVA-OneVision [35], Gemma3-12B [68], Qwen2-VL [63], and so on. For MLLMs intended for math reasoning purposes, we select AtomThink [12], InfiMM-Math [14], MAVIS [19], MathGLM-Vision [9], LlamaV-o1 [69]. This kind of work focuses on the synthesis of STEM reasoning data or o1-like slow thinking. We do not select baselines that use MathVision as training set for fairness, such as Mulberry-Qwen2-VL-7B [77] and MAmooTH-VL [78]. For PRM’s TTS performance, we select Self-Consistency [79] and open-source MLLM as ORM for comparison, such as InternVL2.5-8B [64].

#### Implementation Details

URSA uses SAM-B+SigLIP-L as the hybrid vision encoder and Qwen2.5-Math-Instruct as the LLM backbone. We employ a two-layer MLP connection for vision-language alignment training. We select 15K data in MMathCoT-1M for PS-GRPO. $\gamma$ and $\rho$ in Equation 6 are set to 0.5 and 0.3, respectively. Details on module selection, data selection, hyperparameters, and time cost are placed in the Appendix D and F.

### 5.2 Main Results

#### SoTA Performance

In Table 1, we present the performance of URSA-8B and URSA-8B-PS-GRPO. First, URSA-8B provides a stronger reasoning foundation model. It demonstrates a 5.2 point advantage over AtomThink-EMOVA which focuses on “slow thinking” training. It also outperforms leading general-purpose MLLMs of comparable size, such as Gemma3-12B and InternVL2.5-8B. URSA-8B-PS-GRPO outperforms GPT-4o across 6 benchmarks on average and shows significant advantages on MathVista-GPS (83.2 vs 62.6), GeoQA (73.5 vs 62.1), and achieves the first surpassing performance on MathVision (31.5 vs 30.4). However, a significant performance gap on DynaMath suggests that smaller-scale MLLMs still lack more robust problem-solving capabilities. Compared to the leading math reasoning MLLM AtomThink-EMOVA-8B and general-purpose MLLM Gemma3-12B in terms of average performance, our model shows advantages of **8.5%** and **8.2%**, respectively. Compared with recent R1-inspired method OpenVLThinker [70] and R1-Onevision [71], we still show significant advantage on MathVision and WE-MATH.

#### Effective Best-of-N Evaluation

In Table 2, we demonstrate the advantages of URSA-8B-RM compared to self-consistency and the ORM baseline on serving TTS [43, 42]. We find that self-consistency remains a strong baseline, which InternVL2.5-8B (serving as the ORM) does not consistently surpasses. However, URSA-8B-RM exhibits more effective BoN evaluation and demonstrates its generalization on AtomThink-EMOVA-8B. In addition, using URSA-8B-RM as the verifier, only 4 samplings can achieve a huge improvement based on URSA-8B. Specifically, it provides a 16.6% and 10.1% relative improvement on MathVerse and MathVision. In Best-of-32 setting, URSA-8B achieve 35.1 and 55.0 in MathVision and MathVerse, showing clear advantage with GPT-4o.

#### PS-GRPO vs Vanilla GRPO

As shown in Figure 6 (a), given the same training data, hyperparameters, and rollout number PS-GRPO achieves a higher improvement on average performance (6.8% vs 3.1%). PS-GRPO demonstrates an improvement that is nearly double that of vanilla GRPO in WE-MATH and more challenging MathVision, suggesting its effectiveness. We notice that the improvement of RL on MathVista-GPS and GeoQA is relatively small. This is because URSA-8B’s inherent abilities have already achieved an effect close to the upper bound on these two benchmarks. However, PS-GRPO still has advantages over vanilla GRPO.

![Figure 6](paper_assets/2501.04686v6/x6.png)

*Figure 6: Figure(a) represents the comparison of relative improvements on URSA-8B; Figure(b) illustrates how each training stage contributes to the total performance.*

*Table 3: Ablation study on DualMath-1.1M (BoN evaluation). w/o $\mathcal{S}_{MIE}$ and w/o $\mathcal{S}_{BEL}$ represents dropping one part of DualMath-1.1M to train the PRM.*

<table>
  <thead>
    <tr>
      <th rowspan="2">Model</th>
      <th rowspan="2">Dataset</th>
      <th colspan="4" align="center">MathVerse</th>
      <th colspan="4" align="center">MathVista-GPS</th>
      <th colspan="4" align="center">MathVision</th>
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
      <th>w/o 𝒮MIE</th>
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
      <th>w/o 𝒮BEL</th>
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
      <th>w/o 𝒮MIE</th>
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
      <th>w/o 𝒮BEL</th>
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

## 6 Analysis

### 6.1 How Each Stage Contributes the Performance

In this section, we demonstrate how each stage contributes to the performance. As demonstrated in Figure 6 (b), all stages make a performance contribution. MMathCoT-1M contributes the highest absolute performance gain. The effect of Alignment-860K is more evident on MathVerse and MathVision, likely because the question images in these two datasets contain richer textual modality information, allowing alignment resources (such as textual images) to better supplement this comprehension capability. PS-GRPO, on the other hand, is dedicated to breaking the bottleneck after large-scale SFT, performing more prominently on WE-MATH and MathVerse with relative improvements of 13.2% and 11.4% respectively, compared to URSA-8B. We provide a generalization validation on InternVL2.5-8B and Multimath in Appendix C.4.

*Table 4: Sensitivity analysis on reward penalty and PRM’s “drop-moment” judgment.*

<table>
  <thead>
    <tr>
      <th rowspan="2">γ</th>
      <th rowspan="2" align="center">ρ</th>
      <th align="center">MathVerse</th>
      <th align="center">MathVision</th>
      <th align="center">MathVista</th>
      <th align="center">WE-MATH</th>
      <th align="center">DYNAMATH</th>
      <th align="center">GeoQA</th>
      <th rowspan="2" align="center">Avg</th>
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

### 6.2 Ablation Studies on Automatic Process Labeling

We give an ablation study on how two parts of DualMath-1.1M contribute to URSA-8B-RM. As shown in Table 3, we can see that the method based on BEL, which focuses on the potential to correctness, and the method based on MIE, which focuses on the perception consistency, both contribute positively to the outcome. This further illustrates that in the process of multimodal math reasoning, image-text inconsistency is widespread and needs to be mitigated. We address this issue by augmenting the process supervision training data through the enforced imposition of common hallucination categories. Specifically, the data generated by BEL demonstrates a more significant impact, indicating that the quality of synthesized data can still be improved.

### 6.3 Sensitivity Analysis on Reward Penalty and Drop-moment

In this section, we conduct a sensitivity analysis on two hyperparameters of PS-GRPO, $\gamma$ and $\rho$. These respectively define the magnitude of the reward penalty for rollouts exhibiting a “drop-moment” and the tolerance threshold for identifying such “drop-moments”. As shown in Table 4, our core findings are twofold: (i) The value of $\gamma$ should not be set too high, as this implies excessive trust in the PRM, which may cause the rewards of a group to vanish and lead to training instability. When fixing $\rho$ at 0.3, we find that setting $\gamma$ to a value within a certain appropriate range (we test 0.3-0.7) is generally beneficial for average performance. (ii) An excessively large $\rho$ diminishes reward differentiation, causing the RL behavior to approximate that of vanilla GRPO. Conversely, an excessively small $\rho$ is unreasonable by design, as it is overly sensitive to process reward changes and tends to result in an overly broad range of penalties. In an extreme case where all correct rollouts are penalized, PS-GRPO degenerates back to vanilla GRPO.

## 7 Conclusion

In this study, we take the first step to thoroughly explore the application of PRM in multimodal math reasoning. We introduce a three-stage training pipeline URSA designed to address three major challenges. Initially, we provide a large-scale CoT reasoning dataset MMathCoT-1M. This dataset forms the basis for developing URSA-8B, a MLLM with enhanced reasoning capabilities, and paves the way for further TTS or RL scenarios. Next, we present a dual-view automated process supervision annotation method, covering logical validity and perceptual consistency in multimodal scenarios. We introduce the first large-scale process supervision dataset in multimodal reasoning, DualMath-1.1M. Finally, we address reward hacking and rewarding length bias through process-as-outcome modeling, and put forward PS-GRPO, which is a PRM-aided online RL method that surpasses GRPO. The resulting URSA-8B-PS-GRPO model demonstrates superior average performance over leading open-source MLLM such as Gemma3-12B (8.4%) and proprietary GPT-4o (2.7%).

## References

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

**Appendices Content**

## Appendix A Related Work

#### Multimodal Math Reasoning

The mathematical reasoning capabilities of MLLMs have recently attracted significant attention [11, 80, 35, 81, 82, 5, 14, 78]. Unlike traditional mathematical reasoning tasks in Language Models (LLMs) [1, 83], multimodal mathematical reasoning requires MLLMs to interpret visual information and perform cross-modal reasoning between images and text. Tasks such as solving geometric problems and analyzing graphs are particularly challenging [40]. Recent advances have focused on improving visual mathematical input specialized encoders in specific scenarios [19, 84, 61]. A significant emphasis has also been placed on synthesizing diverse and complex training data. For instance, Math-LLaVA [15] introduces the MathV360K dataset, which categorizes images by complexity and enhances associated questions. Multimath [23] curates high-quality reasoning data from K-12 textbooks and employs GPT-4 for CoT data generation and validation. R-CoT [17] further diversifies problems through a two-stage reverse question-answer generation process. These data synthesis methods are widely adopted in academia and industry due to their demonstrated efficiency [85, 86, 87, 88].

#### Process Reward Model

Recent studies have explored test-time scaling laws in LLMs, aiming to identify optimal reasoning ecectories from diverse thinking trajectories [26, 89, 90, 91, 31, 92, 93]. Initial efforts, such as self-consistency [79], have laid the groundwork for test-time scaling. OpenAI has introduced verifiers to supervise and select reasoning paths during inference [41]. Math-Shepherd [43] evaluates intermediate reasoning steps based on their likelihood of leading to correct answers, while OmegaPRM [42] constructs PRM training data and employs MCTS for training. Despite these advancements, the lack of models with robust CoT reasoning capabilities and limited exploration into diverse reward model training data remain significant bottlenecks in multimodal mathematical reasoning. Some concurrent work also begins to pay attention to PRM-assisted visual reasoning, such as construction and benchmarking [94, 95, 96].

## Appendix B Preliminary

### B.1 Group Relative Policy Optimization

Vanilla GRPO eliminates value function in PPO and estimates the advantages within online rollout group. Given a question with image $q$ and ground-truth $y$, policy model $\pi_{\theta_{old}}$ samples a group of $G$ responses $\{o^{i}\}_{i=1}^{G}$. GRPO compute the $i$-th response’s advantage through normalizing in-group rewards $\{r^{j}\}_{j=1}^{G}$, and employs PPO’s clipped objective and KL penalty term:

$$
A^{i}=\frac{r^{i}-\text{mean}(\{r^{j}\}_{j=1}^{G})}{\text{std}(\{r^{j}\}_{j=1}^{G})}
$$
<div align="right">(7)</div>

$$
\mathcal{J}_{GRPO}(\theta)=\mathbb{E}_{(q,y)\sim\mathcal{D},\{o^{i}\}_{i=1}^{G}\sim\pi_{\theta_{old}}(\cdot|q)} \\
\left[\frac{1}{G}\sum\limits_{i=1}^{G}\frac{1}{|o^{i}|}\sum\limits_{t=1}^{|o^{i}|}\big(\min(r_{t}^{i}(\theta)A^{i},\text{clip}(r_{t}^{i}(\theta),1-\epsilon,1+\epsilon)A^{i})-\beta D_{KL}^{i,t}(\pi_{\theta}||\pi_{ref})\big)\right]
$$
<div align="right">(8)</div>

We introduce the two variants of PRM-integrated GRPO discussed in Section 4. Given PRM $\mathcal{M}_{p}$ and process reward sequences $r_{s}=\mathcal{M}_{p}(\{s_{1},s_{2},\cdots,s_{N}\})$ (i) *Variant 1*: Given verifiable outcome reward $r_{o}^{i}$, we set a single rollout’s reward as $r^{i}=r_{o}^{i}+\bar{r_{s}^{i}}$. (ii) *Variant 2*: We utilize a step-level reward and a multiple relative advantage calculated by the mean value of process rewards from each rollout:

$$
A_{t}^{i}=\underbrace{r_{s,t}^{i}\frac{\bar{r_{s}^{i}}-\text{mean}({\bar{\{r_{s}^{j}\}}_{j=1}^{G}})}{\text{std}({\bar{\{r_{s}^{j}\}}_{j=1}^{G}})}}_{\text{GRPO with process rewards}}+\underbrace{\frac{r_{o}^{i}-\text{mean}(\{r_{o}^{j}\}_{j=1}^{G})}{\text{std}(\{r_{o}^{j}\}_{j=1}^{G})}}_{\text{GRPO with outcome rewards}}
$$
<div align="right">(9)</div>

in which $\bar{r_{s}^{i}}=\text{mean}(\mathcal{M}_{p}(\{s_{1}^{i},s_{2}^{i},\cdots,s_{T_{i}}^{i}\}))$.

### B.2 Test-Time Scaling by Best-of-N evaluation

Following previous works [41, 43], we adopt BoN evaluation for TTS. Given $N$ response samplings for a question $q$. The PRM is used to give process reward for each sampling. We use mean value of process rewards to select the best single sampling:

$$
a_{\text{prm}}=\arg\max\limits_{s_{i}}\text{mean}\{\mathcal{M}_{p}(q,s_{i})\}
$$
<div align="right">(10)</div>

Some other works [97] merge self-consistency and PRM to employ a voting-based score cumulation. But we don’t select this method for a simpler evaluation manner.

## Appendix C Supplementary Results

### C.1 Fine-grained Comparison on Used benchmarks

In this section, we provide some fine-grained results for a clearer comparison. As demonstrated in Table 5, our proposed methods demonstrate significant advantages. Compared to closed-source models like GPT-4o and GPT-4V, our URSA-8B and URSA-8B-PS-GRPO show strong competitiveness. Among open-source models, the performance improvements are even more evident. Our URSA-8B model outperforms other open-source models such as InternLM-XComposer2-VL and Ovis1.6-Gemma2-9B in most subtasks. When combined with PS-GRPO, the URSA-8B-PS-GRPO model achieves even better results, showing significant improvements in subtasks like Alg, AnaG, CombG, and others. Our methods particularly excel in complex mathematical reasoning tasks, demonstrating their powerful mathematical reasoning capabilities. These results highlight the effectiveness of our proposed MMathCoT-1M and PS-GRPO methods in enhancing the mathematical reasoning abilities of models, especially in visual mathematical problems.

In Dynamath (Table 6), compared to open-source MLLMs, the URSA series has obvious advantages in plane geometry and algebra. Surprisingly, from the knowledge level classification, the URSA series model performs excellently at the undergraduate level, which is partly attributable to its math-intensive alignment and large-scale instruction fine-tuning.

In MathVerse (Table 7), we can see that URSA series model marginally surpass GPT-4o on average. Besides, compared with other open-source MLLMs, URSA-8B-PS-GRPO outperforms leading AtomThink-EMOVA-8B and InternVL2.5-8B with **8.4** and **11.4** points.

*Table 5: Performance comparison of different MLLMs on MathVision.*

<table>
  <tbody>
    <tr>
      <th>Model</th>
      <td align="center">Size</td>
      <td align="center">ALL</td>
      <td align="center">Alg</td>
      <td align="center">AnaG</td>
      <td align="center">Ari</td>
      <td align="center">CombG</td>
      <td align="center">Comb</td>
      <td align="center">Cnt</td>
      <td align="center">DescG</td>
      <td align="center">GrphT</td>
      <td align="center">Log</td>
      <td align="center">Angle</td>
      <td align="center">Area</td>
      <td align="center">Len</td>
      <td align="center">SoIG</td>
      <td align="center">Stat</td>
      <td align="center">Topo</td>
      <td align="center">TransG</td>
    </tr>
    <tr>
      <th colspan="19" align="center">Baselines</th>
    </tr>
    <tr>
      <th>Human</th>
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
      <th colspan="19" align="center">Closed-source MLLMs</th>
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
      <th colspan="19" align="center">Open-source MLLMs</th>
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
      <th colspan="19" align="center">Open-source Math MLLMs</th>
    </tr>
    <tr>
      <th>Math-LLaVA</th>
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
      <th>Multimath</th>
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

In WE-MATH 8, URSA-series outperforms leading general-purpose and math reasoning MLLMs in three-stage accuracy. Also, the URSA series has remarkable strengths in solid figures, transformations, positions, and directions. This is mainly due to large-scale alignment and instruction tuning, which builds its foundation in understanding mathematical elements.

*Table 6: Detailed performance comparison of MLLMs on **DYNAMATH** *testmini* dataset, broken down by subject area and knowledge level.*

<table>
  <tbody>
    <tr>
      <td>Model</td>
      <td align="center">Size</td>
      <td align="center">ALL</td>
      <td align="center">PG</td>
      <td align="center">SG</td>
      <td align="center">AG</td>
      <td align="center">AL</td>
      <td align="center">PT</td>
      <td align="center">GT</td>
      <td align="center">AR</td>
      <td align="center">Elem.</td>
      <td align="center">High</td>
      <td align="center">Undergrad.</td>
    </tr>
    <tr>
      <td colspan="13" align="center">Closed-source MLLMs</td>
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
      <td colspan="13" align="center">Open-source MLLMs</td>
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

*Table 7: Comparison with closed-source MLLMs and open-source MLLMs on **MATHVERSE** *testmini*. The best results of Closed-source MLLMs are highlighted. The best and second-best results of Open-source MLLMs are highlighted.*

<table>
  <tbody>
    <tr>
      <th>Model</th>
      <td align="center">#Params</td>
      <td align="center">ALL</td>
      <td align="center">TD</td>
      <td align="center">TL</td>
      <td align="center">TO</td>
      <td align="center">VI</td>
      <td align="center">VD</td>
      <td align="center">VO</td>
    </tr>
    <tr>
      <th colspan="9" align="center">Baselines</th>
    </tr>
    <tr>
      <th>Random</th>
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
      <th>Human</th>
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
      <th colspan="9" align="center">Closed-Source MLLMs</th>
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
      <th colspan="9" align="center">Open-Source General MLLMs</th>
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
      <th colspan="9" align="center">Open-Source Math MLLMs</th>
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
      <th>Math-PUMA-DeepSeek-Math</th>
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
      <th>InfiMM-Math</th>
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

*Table 8: Accuracy comparison with closed-source MLLMs and open-source MLLMs on **WE-MATH** *testmini* subset. First 3 columns show the overall performance on one-step, two-step and three-step problems. The other columns are used to demonstrate the performance in different problem strategies. Red indicates the best performance and Blue indicates the second best performance among open-source models.*

<table>
  <tbody>
    <tr>
      <th rowspan="2">Model</th>
      <td rowspan="2" align="center">#Params</td>
      <td rowspan="2" align="center">S1</td>
      <td rowspan="2" align="center">S2</td>
      <td rowspan="2" align="center">S3</td>
      <td colspan="2" align="center">Mem</td>
      <td colspan="2" align="center">PF</td>
      <td colspan="2" align="center">SF</td>
      <td colspan="2" align="center">TMF</td>
      <td colspan="4" align="center">PD</td>
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
      <td align="center">Dir</td>
      <td align="center">Pos</td>
      <td align="center">RoM</td>
      <td align="center">CCP</td>
    </tr>
    <tr>
      <th colspan="17" align="center">Closed-source MLLMs</th>
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
      <th colspan="17" align="center">Open-source General MLLMs</th>
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
      <th>LongVA</th>
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
      <th colspan="17" align="center">Open-source Math MLLMs</th>
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
      <th>Math-LLaVA</th>
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
      <th>MAVIS w/o DPO</th>
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

### C.2 Scaling Law of MMathCoT-1M

To better illustrate the effectiveness of MMathCoT-1M, we examine the scaling laws of SFT by training models on randomly selected samples representing various ratios of the full dataset.

*Table 9: Scaling law validation on URSA-8B using different ratios of the MMathCoT-1M.*

<table>
  <thead>
    <tr>
      <th align="center">Ratio</th>
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

As shown in Table 9, we can see that MMathCoT-1M clearly shows a training time scaling law, further validates the effectiveness of the synthesized data.

### C.3 Higher Upper Bound Taken from Stage I

In Stage I, we obtain a more powerful base MLLM with enhanced reasoning capabilities through math-intensive vision-language alignment and instruction fine-tuning. Beyond the results in Table 1, we explain why Stage I can better serve subsequent experiments, focusing on test-time scaling and PRM applications. We select MathVerse, MathVision, and MathVista-GPS to observe the **pass@N** metric. As demonstrated in Figure 7, we find that URSA-8B consistently outperforms current leading general MLLMs and math reasoning MLLMs. This indicates that while current trends favor RL-related techniques, the scaling law of supervised fine-tuning can still demonstrate its role in breaking through the base model’s limitations. This naturally brings advantages in areas such as BoN evaluation and the proportion of valuable rollouts in online RL. First, URSA-8B’s higher upper bound leads to richer and more reliable process label generation in Stage II. Furthermore, since recent works claims that RL can only approach the optimal solution within its own exploration path [29, 4, 98], Stage I naturally expands the potential upper limit of the RL stage. This provides the most fundamental advantage to the performance of URSA-PS-GRPO-8B.

![Figure 7](paper_assets/2501.04686v6/x7.png)

*Figure 7: **Pass@N evaluation on three benchmarks.***

### C.4 Generalization Validation

To further vallidate the effectiveness of proposed MMathCoT-1M and PRM aided PS-GRPO. We select InternVL2.5-8B from the general-purpose MLLMs and Multimath from the math reasoning MLLMs for a generalization validation experiment. We do not conduct additional hyperparameter tuning but almost directly adopt the settings from Table 13. The experiment on InternVL2.5-8B and Multimath are implemented on Meng et al. [99] and TRL [100]. Given that these two models have already undergone sufficient alignment for general domains or specific vertical domains upon their release, we only carry out two stages of training: (i) MMathCoT-1M is used to enhance the base model’s mathematical reasoning capabilities; (ii) URSA-8B-RM is involved in the PS-GRPO process. We present the results in Table 8.

*Figure 8: The progress of MMathCoT-1.1M and URSA-8B-RM aided PS-GRPO on InternVL2.5-8B and MultiMath.*

The proposed MMathCoT-1M and PRM aided PS-GRPO demonstrate remarkable generalization capabilities across different models and benchmarks. When applied to InternVL2.5-8B and MultiMath, both models show significant performance improvements. For InternVL2.5-8B, adding MMathCoT-1M boosts the average score from 45.2 to 51.7, with even more significant gains when combined with PS-GRPO, reaching 54.7. Similarly, for MultiMath, the average score increases from 43.1 to 48.7 with MMathCoT-1M and further to 51.2 with PS-GRPO. These results highlight the effectiveness of our approach in enhancing mathematical reasoning capabilities across diverse models and tasks. The performance improvements are consistent across various benchmarks, including MathVerse, MathVision, MathVista, WE-MATH, DYNAMATH, and GeoQA, indicating that our methods are not only effective but also broadly applicable.

### C.5 Implementary Results on Other Benchmarks

We provide supplementary results on WE-MATH, DYNAMATH and GeoQA when comparing BoN selection. As shown in table 10, URSA-8B-RM remains an advantage with Self-consistency and InternVL2.5-8B ORM. When employing URSA-8B as reasoning model, URSA-8B-RM outperforms Self-consistency with 4.6%, 4.5% and 2.7% relative improvements in Best-of-8 performance.

*Table 10: Comparison of TTS with different models using BoN performance on WE-MATH, DYNAMATH, and GeoQA.*

<table>
  <thead>
    <tr>
      <th rowspan="2">Model</th>
      <th rowspan="2">Method</th>
      <th colspan="4" align="center">WE-MATH</th>
      <th colspan="4" align="center">DYNAMATH</th>
      <th colspan="4" align="center">GeoQA</th>
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

## Appendix D Module Selection Criteria

As for module selection, we primarily considered the choice of the vision encoder and the LLM backbone.

#### Vision Encoder

To train a reasoning model with higher process sensibility and facilitate PRM training, we first conduct captioning tests on open-source models like DeepSeek-VL, Qwen2-VL, etc., using a manually selected dataset (approximately 80 examples). These examples primarily include function-related and geometry problems prone to visual confusion. We manually inspect the outputs of these open-source models and find that Qwen2-VL and LLaVA-OneVision performed poorly; even though their performance on standard benchmarks is good, they fail to ensure sufficiently accurate mathematical descriptions. However, DeepSeekVL’s native hybrid vision tower design, integrating high- and low-resolution processing, subjectively exhibit better recognition accuracy. We speculate that this is due to QwenViT [101] being more heavily biased towards general multimodal tasks, resulting in less precise mathematical descriptions compared to simpler vision backbones. Therefore, we choose the SigLiP-L+SAM-B hybrid vision tower design.

#### LLM Backbone

Considering the open-source influence of the QwenLM-Series, we follow the choice of prior work such as MathPUMA [11] and Multimath [23] by using the QwenLM-Series backbone. However, we consider whether we could achieve higher performance by leveraging instruction models that has undergone unimodal math post-training, and thus compare Qwen2.5-7B-Instruct[^1] and Qwen2.5-Math-7B-Instruct[^2]. After completing the VL alignment stage, we conduct a small-scale comparative experiment on MMathCoT-1M, fine-tuning on 50K examples. Finally, our results show that using Qwen2.5-Math-Instruct as the backbone yields an advantage of approximately 1 percentage point on MathVision and MathVerse. Therefore, we include Qwen2.5-Math-7B-Instruct as the LLM backbone for subsequent experiments.

## Appendix E Ablation Studies

### E.1 Effectiveness of Different Data Category

![Figure 9](paper_assets/2501.04686v6/x8.png)

*Figure 9: Each synthesis strategy towards different type of source data works well.*

In the first stage, we mainly synthesized large-scale multiclass CoT data.

- **w/o $\mathcal{S}_{Ao}$**: In this variant, the answer-only data is reverted to its original format. This directly mimics the training mode used by models such as Math-LLaVA and Math-PUMA, which involves hybrid training on both direct answers (’fast thinking’) and CoT thinking.
- **w/o $\mathcal{S}_{An}$**: This data will be replaced with its original organizational structure, where the analysis and final answer are provided in a free-form text format.
- **w/o $\mathcal{S}_{C}$**: This batch of data will be replaced with reasoning expressed in mathematical formal language, better reflecting symbolic and’plan and reasoning’ forms of reasoning.

The results are shown in Figure 9. Firstly, it is shown across all datasets that using the complete synthesized data achieves the best results, highlighting the role of MMathCoT-1M data. More specifically, we find: i) $\mathcal{S}_{Ao}$ demonstrates the greatest impact on MathVerse and MathVision, indicating that expanded CoT data is important for problems where absolute solution accuracy is pursued; ii) However, on WE-MATH, the replacement of $\mathcal{S}_{An}$ leads to the most significant performance drop, suggesting that content rewriting better aligns with the end-to-end requirements posed by the WE-MATH benchmark, and mixing training with data lacking clear logical sequences may reduce hierarchical accuracy; iii) The results on DYNAMATH indicate that rewriting and natural language formulation effectively enhance reasoning robustness from the perspective of textual diversity. This reveals that the thought pattern in textual form tends to maintain the stability of the thought process more effectively under scenarios involving image transformations.

### E.2 Selection of External Closed-source MLLM

In this section, we primarily present a comparison of metrics between Gemini-1.5-Flash-002[^3] and other popular MLLMs, as well as a comparison on partial training data.

*Table 11: SFT Performance with 50K data synthesized by two closed-source MLLMs, respectively.*

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th align="center">MathVista-GPS</th>
      <th align="center">MathVerse</th>
      <th align="center">MathVision</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>URSA-8B w/GPT-4o</th>
      <td align="center">54.1</td>
      <td align="center">33.3</td>
      <td align="center">18.8</td>
    </tr>
    <tr>
      <th>URSA-8B w/Gemini-1.5-Flash-002</th>
      <td align="center">55.1</td>
      <td align="center">32.5</td>
      <td align="center">18.3</td>
    </tr>
  </tbody>
</table>

- *Metrics Comparison*: We compare the performance of Gemini-1.5-Flash-002, GPT-4o, and GPT-4o-mini on some math-related tasks, as shown in Table 12. We observe that Gemini-1.5-Flash-002 is a MLLM that performs well on both unimodal and multimodal math tasks, and GPT-4o does not have a significant advantage over it. This, to some extent, ensures the quality of the data synthesis.
- *SFT Performance*: To best illustrate the performance variations, we randomly sampled 50K data sources from MMathCoT-1M, applied three corresponding strategies using GPT-4o, and subsequently conducted SFT. The performance results are shown in Table 11. We observe that using GPT-4o did not provide a clear advantage. However, the construction of MMathCoT-1M and DualMath-1.1M involves approximately 2.7 million API calls. The output token cost of Gemini-1.5-flash-002 is the same as that of GPT-4o-mini and is one-thirty-third of that of GPT-4o[^4]. Therefore, Gemini-1.5-Flash-002 becomes a cost-effective choice.

However, we must say that if community researchers can afford the cost of accessing more powerful closed-source models, we expect the results to be even better.

*Table 12: Comparison of Model Performance on Math Benchmarks*

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th align="center">Avg</th>
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

## Appendix F Implementation Details

### F.1 RL Data Curation

*Figure 10: Statistics of RL data for vanilla GRPO and PS-GRPO.*

After instruction fine-tuning on MMathCoT-1M, the overall accuracy did not exceed 50%. Therefore, we believe it still has the potential to be directly utilized in the RL phase. We collect 20K data with a types mixture ratio similar to that of instruction fine-tuning and conduct a one-time static filtering before RL. Specifically, we use URSA-8B to perform 8 samplings on this 20K data, filtering out examples where all 8 sampling results are either incorrect or correct. This left approximately 15K+ data for training vanilla GRPO and PS-GRPO. We implement PS-GRPO using TRL [100, 102]. The statistics of the RL data can be found in the table 10.

### F.2 Parameters and Time Cost

In this section, we provide the specific parameter settings and time costs for the three stages. Our experiments are based on Python 3.10 and PyTorch 2.4.0+cu124. We use AdamW [103] as the optimizer. We use Fully Shared Data Parallel (FSDP) [104] as the distributed training framework. Unless otherwise specified, experiments are conducted on 32$\times$ NVIDIA-H100-HBM3 GPUs by default. Additionally, we provide important parameters used in data construction. During the generation of positive and negative example pairs, we set the $temperature$ to 1.0, $n\_return\_sequences$ to 16, and $top\_p$ to 0.95. In the *BinaryErrorLocating* phase, we set the $temperature$ to 0.3, $n\_return\_sequences$ to 16, and $top\_p$ to 0.95.

We adapt the vLLM [105] framework for the URSA-8B’s architecture (hybrid vision tower + MLP + Qwen2.5-math-Instruct is not originally supported by VLLM) and use it as an acceleration tool during the inference phase. During the data pair generation phase, we use 16$\times$ NVIDIA-H100-HBM3 GPUs for inference, which takes approximately 28 hours. In the *BinaryErrorLocating* phase, we also use 16$\times$ NVIDIA-H100-HBM3 GPUs for inference, taking about 20 hours.

The hyperparameter and time cost used in Stage I and Stage II are demonstrated in Table 13. Since the parameters used in Stage III are somewhat different, we list them separately in Table 14. Recently, much work has provided numerous optimization tricks for GRPO, such as training-time dynamic sampling, clipping higher values, abandoning KL loss, etc [106, 58]. However, to independently verify the effectiveness of PRM-guided reward modeling, we have not added these tricks in either vanilla GRPO or PS-GRPO to ensure a fair and valid verification process. We only do **one-time** difficulty-based data selection before applying RL.

*Table 13: Hyperparameter setting and training time cost in Stage I and II.*

<table>
  <tbody>
    <tr>
      <th>Hyperparameters &amp; Cost</th>
      <td align="center">VL-alignment</td>
      <td align="center">Instruction Fine-tuning</td>
      <td align="center">PRM Training</td>
    </tr>
    <tr>
      <th>Learning Rate</th>
      <td align="center">1e-4</td>
      <td align="center">1e-5</td>
      <td align="center">5e-6</td>
    </tr>
    <tr>
      <th>Epoch</th>
      <td align="center">1</td>
      <td align="center">2</td>
      <td align="center">2</td>
    </tr>
    <tr>
      <th>Warm-up Ratio</th>
      <td align="center">0.02</td>
      <td align="center">0.02</td>
      <td align="center">0.02</td>
    </tr>
    <tr>
      <th>Weight Decay</th>
      <td align="center">0.02</td>
      <td align="center">0.01</td>
      <td align="center">0.02</td>
    </tr>
    <tr>
      <th>Batch Size</th>
      <td align="center">64</td>
      <td align="center">128</td>
      <td align="center">128</td>
    </tr>
    <tr>
      <th>Trainable Parts</th>
      <td align="center">Aligner</td>
      <td align="center">Vision Encoder, Aligner, Base LLM</td>
      <td align="center">Base LLM</td>
    </tr>
    <tr>
      <th>Data Size</th>
      <td align="center">860K</td>
      <td align="center">1.0M</td>
      <td align="center">1.1M</td>
    </tr>
    <tr>
      <th>Time Cost</th>
      <td align="center">∼3.5h</td>
      <td align="center">∼11h</td>
      <td align="center">∼12h</td>
    </tr>
  </tbody>
</table>

### F.3 Benchmarks

In this section, we introduce the detailed subtasks and metrics of four used benchmarks to more precisely demonstrate the evaluation.

#### MathVerse

MathVerse [72] is a benchmark for testing the reasoning abilities of MLLMs when the information content in text and image modalities varies. Specifically, the models focus on performance in six scenarios: Text-Dominant (TD), Text-Lite (TL), Text-Only (TO), Vision-Intensive (VI), Vision-Dominant (VD) and Vision-Only (VO).

*Table 14: Hyperparameter setting and training time cost in Stage III.*

#### WE-MATH

WE-MATH [75] is the first benchmark that decompose composite problems into sub-problems according to the required knowledge concepts. In figure 8, the actual content corresponding to the abbreviations is as follows. Mem: Measurement, PF: Plane Figures, SF: Solid Figures, TMF: Transformations and Motion of Figures, PD: Position and Direction, AL: Angles and Length, UCU: Understanding and Conversion of Units, CPF: Calculation of Plane Figures, UPF: Understanding of Plane Figures, CSF: Calculation of Solid Figures, USF: Understanding of Solid Figures, BTF: Basic Transformations of Figures, CCF: Cutting and Combining of Figures, Dir: Direction, Pos: Position, RoM: Route Map, CCP: Correspondence of Coordinates and Positions.

#### DYNAMATH

DYNAMATH [73] is a benchmark designed to evaluate the robustness of MLLMs in mathematical reasoning. Specifically, it includes tests across multiple dimensions, including Solid Geometry (SG), Plane Geometry (PG), Analytic Geometry (AG), Algebra (AL), Puzzle Test (PT), Graph Theory (GT), Arithmetic (AR), Scientific Figure (SF) and Statistics (ST). It includes 501 seed questions and 5010 generated questions.

#### GeoQA

The GeoQA [40] dataset is a specialized dataset designed for evaluating and training models in the field of geographic question answering. Its test set includes 734 samples.

#### MathVista

MathVista [74] comprises a total of 5 subtasks: Geometry Problem Solving (GPS), Math Word Problem (MWP), Figure Question Answering (FQA), Textbook Question Answering (TQA) and Visual Question Answering (VQA). Like the previous math reasoning works, our model training process does not overly focus on knowledge-intensive tasks (such as VQA and FQA), hence we choose GPS as the primary task.

#### MathVision

MathVision [107] is a large-scale multimodal math reasoning dataset that broadens the disciplinary scope of the multimodal mathematics field. The test set contains 3,040 examples, covering 16 key competencies, and provides reliable testing performance. Specifically, The specific meanings of the various disciplinary indicators in Table 5 are listed as following. Alg: algebra, AnaG: analytic geometry, Ari: arithmetic, CombG: combinatorial geometry, Comb: combinatorics, Cnt: counting, DescG: descriptive geometry, GrphT: graph theory, Log: logic, Angle: metric geometry - angle, Area: metric geometry - area, Len: metric geometry-length, SolG: solid geometry, Stat: statistics, Topo: topology, TransG: transformation geometry.

#### Evaluation Criteria

Our comparison is based on the following criteria: First, we select the results from the official leaderboards of each benchmark. Second, we choose the results from the original papers or technical reports of each model. Finally, we conduct our own inference and evaluation using vLLM [105]. Our evaluation adheres to the rules of the benchmarks themselves, which are as follows:

- **Rule-based Matching**: WEMATH, GeoQA.
- **LLM-as-a-Judge**: MathVista, MathVision, MathVerse, Dynamath.

The prompt for LLM-as-a-Judge is shown in Figure 11.

![Figure 11](paper_assets/2501.04686v6/x9.png)

*Figure 11: LLM-as-a-Judge prompt used for answer matching.*

### F.4 Algorithm

In this section, we place the specific process of BIE from Section 3.2 into Algorithm 1. Specifically, the input is a solution that points to an incorrect answer. We set a per-step sampling hyperparameter $N_{\text{mid}}$. Initially, we set the start and end points of the search range to Step 1 and Step N, respectively. We first consider the $mc$ value of Step (1+N)//2. If it is positive, it indicates that the first totally erroneous step occurs in the latter half; otherwise, we look in the first half. This reduces the number of searches to $\mathcal{O}(\log N)$.

Besides, we introduce the process of PS-GRPO in Algorithm 2. This process involves merging of outcome reward and process-as-outcome reward, and subsequent relative advantages calculation.

## Appendix G Prompt Design

### G.1 Prompt Utilized in MMathCoT-1M Synthesis

In this section, we provide the specific prompts for three-module data synthesis. Additionally, Gemini-1.5-Flash is a model that is very sensitive to prompts and parameters in practical experience, and we will share detailed adjustment experiences.

#### CoT Expansion

CoT expansion prompt for answer-only data source can be seen in Figure 12. We order the Gemini-1.5-flash to give a reasonable process directing to the ground-truth. After the execution, we find that the outputs is not so clear. The model sometimes will give trajectories that include “we must trust the answer” or “let me assume”. We identify these phrases as signals that the model can not solve the problem naturally and independently. We will filter these samples.

#### Analysis Rewriting

Rewriting prompt for analysis-formatted data synthesis is illustrated in Figure 13. For solutions in an analytical format, we transform them into clear step-by-step format trajectories. During this process, Gemini-1.5-flash-002 does not exhibit significant questioning or make conditional requests. We improve data quality through reorganization and polishing of the language logic.

#### Format Unified

By employing a unified format prompt shown in Figure 14 to modify the reasoning styles of plan-and-reasoning and symbolic approaches, we are able to extract a more natural language process aligned with the pre-training style. A single example is sufficient to elicit perceptually favorable responses.

#### Double-checking

After completing the above three points, we apply an LLM-as-a-judge for double-checking the synthesized data, ensuring that the solutions do not contain unreasonable processes, such as untimely questioning, conditional requests, or reasoning loops. The specific prompt design is shown in Figure 15. After this layer of filtering, we obtain the final MMathCoT-1M.

![Figure 12](paper_assets/2501.04686v6/x10.png)

*Figure 12: CoT expansion prompt for answer-only data.*

![Figure 13](paper_assets/2501.04686v6/x11.png)

*Figure 13: Analysis rewriting prompt for analysis-formatted data.*

![Figure 14](paper_assets/2501.04686v6/x12.png)

*Figure 14: Format unify prompt for mathematical and symbolic reasoning style data.*

![Figure 15](paper_assets/2501.04686v6/x13.png)

*Figure 15: Double-checking prompt for ensuring high-quality and appropriate trajectories in synthesized CoT reasoning data.*

### G.2 Prompt Utilized in DualMath-1.1M Synthesis

In this section, we demonstrate the prompts used in MIE.

- **Geometry Problem**: For geometry problem, we prompt the Gemini-1.5-Flash-002 to first identify key geometry features in the figure. We then order it to introduce a misinterpretation on these elements. Finally, use the wrong information to execute a misleading solution. The total design can be seen in Figure 16.
- **Charts & Function**: For ChartQA and math functions, we prompt Gemini-1.5-Flash-002 to first check the fine-grained data points. We then attempt to insert spatially similar data to induce a misinterpretation. This subsequently leads to incorrect solutions for automatic labeling.
- **LLM-as-a-Judge**: For chart reasoning and function problem, we execute similar process on Gemini-1.5-Flash-002. We place it in Figure 17.

![Figure 16](paper_assets/2501.04686v6/x14.png)

*Figure 16: Misinterpretation insertion for geometry-related problems.*

![Figure 17](paper_assets/2501.04686v6/x15.png)

*Figure 17: Misinterpretation insertion for function and chart-related problems.*

## Appendix H Case Study

### H.1 Showcase on Best-of-N Evaluation

To more clearly illustrate the effectiveness of URSA-8B-RM in BoN evaluation, a case on MathVista-GPS is demonstrated (Figure 18). This case shows that URSA-8B-RM is sensitive to false theorem application and misunderstandings of angle-number relations. The good property not only enables URSA-8B-RM to perform well in BoN evaluation but also endows it with the potential to identify more valuable learning samples in online reinforcement learning.

![Figure 18](paper_assets/2501.04686v6/x16.png)

*Figure 18: Case of URSA-8B-RM serving as a verifier in Best-of-N evaluation.*

### H.2 Process of Misinterpretation Insertion Engine

As shown in Figure 19, MIE performs three main actions: First, it interprets the mathematical information in the image. Then, it replaces key information at a selected step. Finally, it continues reasoning based on the modified conditions.

![Figure 19](paper_assets/2501.04686v6/x17.png)

*Figure 19: Case from MIE. We introduce specific step-level perception errors and continue false reasoning to construct a correctnesses-labeled solution.*

### H.3 Failure Pattern in Process During GRPO

In this section, we intuitively reveal through examples why PS-GRPO works effectively. We first introduce the concept of false-positive rollouts, which are rollouts that, despite reaching the correct answer, do not provide perfect intermediate actions to arrive at the solution. They can generally be divided into two categories: (i) the lack of visual condition alignment. Solutions in this category exhibit inconsistencies in reasoning regarding basic visual factors such as edge relationships, coordinate values, and theorem applications, revealing deficiencies in the pretraining phase, as shown in Figure 20. (ii) the exploitation of shortcut patterns. These rollouts do not go through key steps but are guided directly to the correct answer after basic descriptions due to the high correlation between image features and problem-solving patterns during pretraining or SFT, as shown in Figure 21. Therefore, PS-GRPO suppresses the advantageous direction brought by these erroneous actions through the sensitivity of the PRM in online RL for error identification. This leads to a more optimal paradigm that combines outcome rewards with process reward-based penalties.

![Figure 20](paper_assets/2501.04686v6/x18.png)

*Figure 20: False positive rollout analysis I.*

![Figure 21](paper_assets/2501.04686v6/x19.png)

*Figure 21: False positive rollout analysis II.*

### H.4 Cases on How Naive Process Reward Modeling Fails

In this section, we elaborate on the two fundamental flaws of process reward guided RL mentioned in Section 4 and present some cases for illustration. In online RL, models can easily recognize the patterns for obtaining process rewards, leading to conservative analyses and concise responses as they sidestep PRM scrutiny. As shown in Figure 22, we have observed that the model often follows a distinct reasoning pattern. They initially read and analyze the given conditions comprehensively, but then make incorrect decisions based on this analysis, leading to wrong answers. This indicates that when explicitly modeling process rewards, models can easily focus on processes that seem "correct" in isolation. However, these processes may not be genuinely helpful for the final outcome and instead may lead the model to prioritize high process rewards over accuracy.

![Figure 22](paper_assets/2501.04686v6/x20.png)

*Figure 22: Bad case analysis on two process reward modeling variants.*

## Footnotes

[^1]: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct

[^2]: https://huggingface.co/Qwen/Qwen2.5-Math-7B-Instruct

[^3]: https://deepmind.google/technologies/gemini/flash/

[^4]: https://docsbot.ai/models/gemini-1-5-flash-002
