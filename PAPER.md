                                             Unlocking Multimodal Mathematical Reasoning
                                                       via Process Reward Model


                                                        Ruilin Luo12∗ Zhuofan Zheng2∗ Yifan Wang1 Xinzhe Ni1
                                                           Zicheng Lin1 Songtao Jiang3 Yiyao Yu1 Chufan Shi1
                                                          Lei Wang4 Ruihang Chu1† Jin Zeng2† Yujiu Yang1
                                                                         1
                                                                          Tsinghua University 2 ByteDance
arXiv:2501.04686v6 [cs.CL] 5 Oct 2025




                                                        3
                                                            Zhejiang University 4 Ping An Technology (Shenzhen) Co., Ltd.



                                                                                       Abstract

                                                 Process Reward Models (PRMs) have shown promise in enhancing the mathemati-
                                                 cal reasoning capabilities of Large Language Models (LLMs) through Test-Time
                                                 Scaling (TTS). However, their integration into multimodal reasoning remains
                                                 largely unexplored. In this work, we take the first step toward unlocking the po-
                                                 tential of PRMs in multimodal mathematical reasoning. We identify three key
                                                 challenges: (i) the scarcity of high-quality reasoning data constrains the capabilities
                                                 of foundation Multimodal Large Language Models (MLLMs), which imposes
                                                 further limitations on the upper bounds of TTS and reinforcement learning (RL);
                                                 (ii) a lack of automated methods for process labeling within multimodal contexts
                                                 persists; (iii) the employment of process rewards in unimodal RL faces issues
                                                 like reward hacking, which may extend to multimodal scenarios. To address
                                                 these issues, we introduce URSA, a three-stage Unfolding multimodal pRocess-
                                                 Supervision Aided training framework. We first construct MMathCoT-1M, a
                                                 high-quality large-scale multimodal Chain-of-Thought (CoT) reasoning dataset,
                                                 to build a stronger math reasoning foundation MLLM, URSA-8B. Subsequently,
                                                 we go through an automatic process to synthesize process supervision data, which
                                                 emphasizes both logical correctness and perceptual consistency. We introduce
                                                 DualMath-1.1M to facilitate the training of URSA-8B-RM. Finally, we propose
                                                 Process-Supervised Group-Relative-Policy-Optimization (PS-GRPO), pioneering
                                                 a multimodal PRM-aided online RL method that outperforms vanilla GRPO. With
                                                 PS-GRPO application, URSA-8B-PS-GRPO outperforms Gemma3-12B and GPT-
                                                 4o by 8.4% and 2.7% on average across 6 benchmarks. Code, data and checkpoint
                                                 can be found at https://github.com/URSA-MATH.


                                        1   Introduction
                                        Following the substantial progress of Large Language Models (LLMs) in math reasoning [1–8], the
                                        math reasoning capabilities of Multimodal Large Language Models (MLLMs) have increasingly
                                        garnered attention [9–13]. Previous work has typically focused on aspects such as math reasoning
                                        data curation [14–18], training math-intensive vision encoders [19, 20], enhancing vision-language
                                        alignment [11, 21], or the application of post-training techniques [22–24, 13]. Given the success
                                        of Process Reward Models (PRMs) in improving LLM reasoning through methods like Test-Time
                                        Scaling (TTS) [25, 26] and Reinforcement Fine-Tuning (ReFT) [27, 28], the application of PRMs to
                                        multimodal reasoning remains unexplored.
                                            * Equal contribution. Work done during Ruilin’s internship at ByteDance. † Corresponding author.
                                             ruihangchu@gmail.com, zengjin@bytedance.com


                                        39th Conference on Neural Information Processing Systems (NeurIPS 2025).
                                              GPT-4o            Gemma3-12B                   InternVL2.5-8B                   AtomThink-EMOVA-8B                                URSA-8B-PS-GRPO (Ours)
                   90
                                                                                                                      83.2
                   80                                                                                          75.9                                                                                                          75.6

                   70                                                                                                                                                                                   67.7
                                                                                                        64.9                                                    64.9                                                  63.8
                                                                                            62.6 63.6                        62.8                                                                62.1          61.6
                                                                                                                                                         60.7
                   60
    Accuracy (%)


                          50.2                      50.9                                                                            51.7
                   50                                                                                                                             49.3
                                                                                                                                                                       45.8               47.4
                                                                                                                                           44.7
                                             42.5
                                 40.1 39.5                                                                                                                                    40.5 40.9
                   40
                                                           30.4 29.1                 31.5
                   30
                                                                              24.9
                                                                       19.7
                   20

                   10
                                 MathVerse                    MathVision                        MathVista                       WE-MATH                                DynaMath                          GeoQA
                                 (testmini)                    (full set)                         (gps)                         (testmini)                             (testmini)                        (full set)

                        Figure 1: Performance comparison with leading open-source MLLMs and GPT-4o.

In this work, we take the first step toward integrating PRMs into multimodal math reasoning. We
identify three key challenges: (i) Since both TTS and RL are heavily influenced by the strength
of foundation models [29, 25], the limited availability of large-scale, high-quality reasoning data
constrains the upper bounds of current MLLMs and weakens the effectiveness of PRM integration;
(ii) There hasn’t yet been adequate automated process labeling techniques merged within multimodal
contexts, where both logical validity and perceptual consistency should be emphasized [30–32]. (iii)
While PRMs can be effectively used in TTS, applying them directly in online RL introduces risks
such as reward hacking and length bias in rewarding [33, 34].
To address these challenges, we propose the URSA framework, a three-stage Unfolding multimodal
pRocess-Supervision Aided training pipeline that supports both the construction and application of
multimodal PRMs. In Stage I, we curate MMathCoT-1M, a large-scale, high-quality multimodal
Chain-of-Thought dataset synthesized from 1.43 million open-source examples, which enhances
the foundation model’s reasoning capabilities through targeted instruction tuning. In Stage II, we
construct DualMath-1.1M via a dual-view process supervised data synthesis strategy which combines
a binary error locating engine and a misinterpretation insertion engine. It provides complementary
signals for logical validity and visual grounding, and is used to train a process reward model. In
Stage III, we analyze the limitations of scalar process reward modeling in online RL and propose
Process Supervision-GRPO (PS-GRPO), which mitigates reward hacking and PRM’s length bias
in rewarding by implicitly penalizing process-level inconsistencies during policy optimization.
Results on 6 multimodal reasoning benchmarks show that our PRM improves Best-of-N verification,
surpassing self-consistency and outcome-based baselines. When used in PS-GRPO, the resulting
model achieves state-of-the-art performance among open-source MLLMs of similar size. Our
contributions are as follows:
     • We release two large-scale open-source datasets, MMathCoT-1M and DualMath-1.1M, to address
       the scarcity of high-quality multimodal CoT reasoning and process supervision data.
     • We propose PS-GRPO, an online reinforcement learning algorithm that incorporates multimodal
       PRMs by comparing the relative quality of rollouts, rather than relying on scalar reward modeling.
       It effectively mitigates PRM’s reward hacking and length bias in rewarding.
     • Experimental results show that our reward model improves both test-time verification and online
       training. With PS-GRPO application (Figure 1), URSA-8B-PS-GRPO outperforms Gemma3-
       12B and GPT-4o by 8.4% and 2.7% on average across 6 benchmarks.


2                  Stage I: Math-Intensive Alignment and Instruction Tuning

2.1                 Collection of Vision-Language Alignment Data

We employ a LLaVA-like architecture and first collect vision-language alignment data directly from
existing open-source datasets [35–38]. As demonstrated in Figure 2, we collect URSA-Alignment-
860K from Multimath [23], MAVIS [19] and Geo170K [18]. We then filter out samples with overly


                                                                                                                        2
                            Analytic
        Solid              Geometry
      Geometry                                                                                                         Arith-
                 Concept
                                                                                                                     A metic
                                                                                                               TQ
                                                                                                                          arsity




                           art
                                                    K                                                                  -V
                                           Geo170




                                                                                                 art
                                                                                                                    Geo k Tutors




                       Ch




                                                                                                                                              P la e t r
                                                                                                                                              Ge
                                                                                                 Ch
                                                              P la t r y
                                                              Ge




                                                                                                                                                  ne y
                                                                                                                       0




                                                                                                                                                  om
                                                         Mu
                                                                                                                     17




                                                                                                                                    Ma
                                                                  om
                                                                  ne
                                                          lt im




                                                                                                      -M e V I S
                                                                                                               en




                                                                                                                                      th V
                                                                     e




                                                                                       Algebra




                                                                                                          ta G
                                                                       a th




                                                                                                                                           36
                                                                                                         MA
                                                                            -E N




                                                                                                                                             0K
                                      URSA-Alignment-                                                                MMathCoT-
                                          860K                                                                          1M




                                                                                                          MA
                            M




                                                                                                          V IS
                             AV
                 Fun




                                                                                             Fu
                                 IS




                                                                                                            -G e
                                                                                                 nc
                  c t io



                                  Ca




                                                                                                  t io


                                                                                                               o
                                      pt
                    n




                                                                                                      n
                                      io n                                                                                 Multimath-EN          t ic
                                                                  t ic                                                                       a ly y
                                                              a ly t r y                                                                  A n m etr
                                                            An om e                                                   Solid                  e o
                                                Solid        Ge                                                     Geomet
                                                                                                                           ry
                                                                                                                                           G
                                              Geometry

                       Figure 2: Statistics of URSA-Alignment-860K and MMathCoT-1M.


verbose captions, to form an 860K math-intensive alignment dataset. Following the engineering
practices of previous work, we only train the MLP projector in the alignment step.

2.2   CoT Reasoning Data Synthesis

For a powerful foundation building, we collect 1.43M samples from existing math reasoning datasets
to support the construction of large-scale CoT reasoning data. As shown in Figure 2, data is sourced
from MathV360K [15], Multimath [23], MAVIS [19], Geo170K [18] and VarsityTutors [11]. Based
on the type of solution, we categorize the data into answer-only, analysis-formatted, and CoT-
formatted. We adopt different synthesis strategies for them to curate high-quality CoT reasoning
trajectories. We utilize Gemini-1.5-Flash-002 (refer to G below) as a cost-effective tool for data
curation, avoiding expensive large-scale manual annotation.

CoT Expansion. For answer-only data D1 = {(xi , yi )}N     1
                                                         i=1 , such as MathV360K [15], each sample
contains a question xi and a ground-truth answer yi . This type of data is heavily used in previous
works for fast thinking reasoning mode [15, 11, 16]. However, answer-only training restricts the
model from fully capturing the problem-solving process. It may lead to memory-based reasoning,
hindering the model’s ability to directly provide answers to more complex reasoning problems [39].
We expand certain scale CoT reasoning trajectories for this category of data. Given a expansion
prompt PC , we provide xi and yi , then prompt G to output the reasoning trajectory leading to the
answer yi , yielding the expanded solutions SAo = G(PC ; {xi , yi }N
                                                                   i=1 ).
                                                                     1




Rewriting. This strategy is designed for analysis-formatted samples, denoted as D2 =
{(xi , yi , ai )}N 2
                 i=1 . This includes datasets like MAVIS-Geo, MAVIS-MetaGen [19], VarsityTutors [11],
and Geo170K-QA [40]. Each sample contains a question xi , an answer yi , and textual analysis ai .
While this type of data provides walkthroughs, it often suffers from two issues: (i) It lacks strict
step-by-step logic, exhibiting jumps in language or reasoning. (ii) A significant portion of the answers
are relatively brief and cannot provide rich rationale. Given a rewriting prompt PR , we utilize G to
transcribe these solutions, thereby enhancing their step-by-step reasoning trajectories and linguistic
diversity, resulting in the rewritten set SAn = G(PR ; {xi , yi , ai }N
                                                                      i=1 ).
                                                                        2




Format Unification. This strategy is used for CoT-formatted data, primarily sourced from
Multimath-EN-300K [23], which is collected from K-12 textbooks and contains mathematical
language and symbolic-style reasoning solutions. This portion of the data, D3 = {(xi , yi , ci )}N        3
                                                                                                        i=1 ,
consists of a question xi , an answer yi , and a solution ci . We unify the format through natural language
stylization using a prompt PF with G, producing the unified set SC = G(PF ; {xi , yi , ci }N     i=1 ).
                                                                                                   3




MMathCoT-1M. Finally, we filter out instances where: (i) Correctness is violated: the generated
content altered the original answer, or (ii) Consistency is problematic: the solution includes text that
questions the original answer or makes new assumptions to force the given answer. This process
yields MMathCoT-1M. The complete prompt designs can be found in Appendix G.


                                                                                   3
Stage 1                                               Stage 2                                                   Stage 3
                    1.43M Source Data                  Binary Error Locating Engine                              PS-GRPO
                                                                       Image       Question
                             Classify                                                                                               In rhombus ABCD, M, N are respectively
                                                                                                                                    on AB, CD, and AM=CN, MN intersects AC
 ⋅ Answer-only (440K):
                                                        URSA-8B    Step 1   Step 2 Step 3 Step 4                                    at point O, connect BO. If ∠DAC=28°,
 The answer is A.                                                                                                                   then the degree of ∠OBC is ( )
 ⋅ Analysis (715K):                                                              Judge Step (1+4)//2=2
 The image shows a geometric figure with                                    … Step N                                                        URSA-8B
 points labeled A, B, C, D, and E. …                      Step 1   Step 2   … Step N            MC=0.33
                                                                            … Step N
 ⋅ Math & Symbilic CoT (275K):                                                                                                                                        𝑎$1
                                                                                                                        𝑎!1         …          𝑎#1
 Step 1 (Factorization): ∵ x² + 2x - 8 < 0,                                        Judge Step (2+4)//2=3
 ∴ x² + 2x - 8 = (x + 4)(x - 2),                                               …    Step N                                          …




                                                                                                                                                                      …
                                                                                                                                                …
                                                                                                                        …
 Step 2 (Determine inequality):                                                …
                                                          Step 1   Step 2           Step N         MC=0
  (x + 4)(x - 2) < 0,⇔ -4 < x < 2,
                                                                               …    Step N                              62          …          62                     59
 …
 Step 7 (Determine shaded region):                     Misinterpretation Insertion Engine
 Range of the shaded region = {x | 1 ≤ x < 2}
                                                                               Find the perimeter of △DEF if
                                                                               △DEF ~ △CBF, perimeter of          URSA-RM-8B
                                                                               △CBF=27, DF=6, and FC=8.
                      ⋅ CoT Expansion                                                                             Step 1: ABCD is a rhombus, the diagonal AC bisects ∠DAB. Therefore,
External                                                                                                          ∠CAB=∠DAC=28°. 0.95
MLLM                  ⋅ Rewriting                                                                                 Step 2: O is the intersection of BD and AC, so ∠AOB=∠ODC=90°.
                      ⋅ Format Unification                                                                        0.64
                                                                     ⋅ Information Extraction                     Step 3: In △AOB, ∠ABO+∠AOB+∠OAB=180°, we have ∠ABO=180°-
                                                                     ⋅ Mistake Insertion                          28°-90°=62°. 0.68
                                                                                                                  Step 4: BO is part of the diagonal BD. The diagonal BD bisects ∠ABC.
                                                                     ⋅ Wrong Reasoning                            Thus, the angle ∠CBO is the same as ∠ABD. Therefore, ∠OBC = 62°.
MLLM-As-               ⋅ Correctness                                                                              0.73
a-judge                ⋅ Consistency
                                                        Step 1: From statement △DEF~ △CBF, side DF
                                                        correspondes to CF;                                                          …                                0.0
                                                                                                               Rewards 0.5                       1.0
                                                        Step 2: Perimeter(△DEF)/Perimeter(△CBF)=4/3;
              MMathCoT-1M & URSA-8B                     Step 3: Perimeter(△DEF)=4/3*27=36;
                                                        Step 4: Thus, the perimeter of △DEF=36.                                     Group Advantage 𝐴(


Figure 3: Pipeline of URSA. Stage 1 depicts the workflow of data curation as described in Section 2.
Stage 2 illustrates how binary error locating and misinterpretation insertion facilitate the automation
of process supervision data. Stage 3 demonstrates how our PS-GRPO operates by imposing penalties
on rollouts that are questioned by the PRM.


We perform full-parameter instruction fine-tuning with MMathCoT-1M to train URSA-8B, based
on the aligned model. The SFT dataset DSF T is formed by the union of the curated solutions, i.e.,
DSF T = {(xi , yi ) | (xi , yi ) ∈ SAo ∪ SAn ∪ SC }. Training objective is demonstrated in Equation 1.
                                                                                    T
                                                                                    X
                                                LSF T = −E(x,y)∼DSF T                         log M(yt |x, y<t )                                                                  (1)
                                                                                     t=1

In this phase, we construct a stronger reasoning foundation model, URSA-8B, with the expectation
of achieving a higher bound at inference time and to process supervision data of greater diversity.


3      Stage II: Dual-View Process Supervised Data Synthesis

3.1        Binary Error Locating Engine

Following suggestions by previous work [41–43], we train a PRM for first error step identification.
We collect ∼553K incorrect solutions from URSA-8B’s zero-shot inference on MMathCoT-1M.
Erroneous steps in these solutions are labeled using Monte Carlo Tree Search (MCTS). For MCTS, an
operation F({s1 , . . . , si }, N ) generates N rollouts from a reasoning prefix {s1 , . . . , si }. The single
step’s Monte Carlo estimation value, mci , is the fraction of these rollouts leading to a correct answer:

                                                    |Correct rollouts from F({s1 , s2 , . . . , si }, N )|
                                        mci =                                                                                                                                     (2)
                                                     |Total rollouts from F({s1 , s2 , . . . , si }, N )|
A step si is deemed “potentially correct” if mci > 0 [43, 42]. We optimize the identification of
first error step using Binary Error Locating Engine (BEL): if the middle step has positive mc (i.e.
mcmid > 0), the error is in the latter half; otherwise, in the first (see Algorithm 1). To mitigate
step-level label bias and include positive examples, we add ∼180K correct solutions (1/3 the number
of incorrect ones), with all steps easily marked “True”. This yields SBEL , a 773K process annotation
dataset based on correctness potential.


                                                                                   4
            (a)                       (b)                            (c)                              (d)
Figure 4: Figure (a)-(d) respectively illustrate training rewards, response length, response step number
and test set accuracy of vanilla GRPO and two variants proposed in Section 4. Test set is randomly
selected 500 examples from MMathCoT-1M for an in-domain evaluation.


3.2   Misinterpretation Insertion Engine

Apart from logical errors, the perception inconsistency between images and text in reasoning steps is
a unique problem in multimodal scenarios [30, 44, 45]. We propose a Misinterpretation Insertion
Engine (MIE) to artificially insert hallucinatory information, automatically constructing process
supervision data with incorrect reasoning paths starting from the insertion point. Specifically, MIE
includes three steps. First, we prompt G to perform a captioning task, extracting mathematical
paradigm information from the image as much as possible. Second, the model G is required to focus
on potentially confusable conditions within the existing correct solution and modify them using
adjacent or similar conditions. Finally, the model G is prompted to continue reasoning based on the
step with the inserted error. We leverage strong instruction-following capability of G, instructing it
to automatically assign negative labels to every subsequent step following the erroneous insertion.
We generate ∼302K samples SM IE using this strategy. Cases from MIE can be found in the
Appendix H.2.

3.3   PRM Training

As shown in Equation 3, we merge two types of data, proposing a ∼1.1M process supervision data
called DualMath-1.1M. During training, we append a special token after each step to indicate its
predicted correctness. We model the PRM training as a binary classification task for the correctness
of each step, as shown in Equation 4, here πp is the trained PRM based on URSA-8B. ej and yj
represent single step and corresponding label (yj ∈ {0, 1}).
                                DP RM = {(e, ye ) ∼ SBEL ∪ SM IE }                                          (3)
                                            |e|
                                            Xh                                                    i
             LP RM = −E(e,y)∼DP RM                yj log πp (ej ) + (1 − yj ) log(1 − πp (ej ))             (4)
                                            j=1

Thus, Stage II delivers URSA-8B-RM, a strong PRM trained on DualMath-1.1M—the first large-
scale, automatically labeled dataset for multimodal reasoning process supervision. While BoN
evaluation demonstrates PRM’s value in TTS, a critical question emerges: how can its guidance be
directly integrated into MLLM post-training? This remains largely uncharted. Stage III draws a
lesson about why previous scalar process reward modeling tends to fail, and then we achieve effective
progress through process-as-outcome reward modeling.

4     Stage III: Integrating multimodal PRM into RL
Inspired by successes like DeepSeek-R1 [46], several recent studies have tried to adapt outcome
reward-based GRPO for multimodal reasoning, demonstrating notable progress [47–50]. Outcome
reward-based GRPO computes the i-th response’s advantage through normalizing in-group rewards.
However, outcome reward-based GRPO ignores the quality of reasoning processes [41, 51, 52].
Following most standard response-level and step-level reward modeling in RL [43, 28, 46, 13, 53],
we examine two simple variants of GRPO with integrated scalar process rewards to reveal the failure
patterns during the training process [54]. Variant 1: For i-th rollout, the reward is the sum of the


                                                       5
            (a)                          (b)                       (c)                       (d)
Figure 5: Figure (a) shows the BoN evaluation during GRPO training. We select the best rollout
using the mean value of process rewards. Figure (b) illustrates the proportion of rollouts where
URSA-8B-RM identifies “drop-moment” and the final results are indeed incorrect. Figures (c) and
(d) display the response length and test accuracy during PS-GRPO training.

outcome reward and the average process reward, i.e. ri = roi + r¯si . Variant 2: Despite the outcome
                                     i
reward, a scalar process reward rs,t    is assigned to the i-th rollout’s t-th step. We observe two highly
significant conclusions from Figure 4: (i) High susceptibility to reward hacking. The test accuracy
of both variants is lower than vanilla GRPO. This indicates that when process scalar rewards are
employed as learning objectives, the model quickly learns strategies that cater to process correctness.
However, correctness in the process does not necessarily correlate fully with the heuristics leading to
the ground-truth. (ii) PRM’s length bias in rewarding. We observe a trend where increased training
leads to shorter model responses and fewer reasoning steps. This phenomenon stems from an inherent
length bias in the PRM’s training labels; for examples with incorrect answers, steps taken after the
first error are unlikely to yield a correct solution. This results in the PRM conservatively rewards the
later stages of a reasoning rollout, thereby encouraging the MLLM towards more passive reasoning
and a reliance on pattern recognition from existing conditions or simpler heuristics.

PS-GRPO The findings above confirm the consideration that flaws in the reward function are
amplified when scalar process rewards serve as the optimization target [55, 33]. We ask “Which
internal signals of PRM can be trusted?” We employ two views to investigate the reliable region
of the PRM: first, the BoN performance during online learning, and second, the PRM’s error
identification capability. Regarding the latter, we introduce the concept of a “drop-moment” within
the PRM’s reward sequence, which signifies that the PRM questions the validity of the preceding
                                                                      i    i             i
steps. Specifically, for a given solution’s PRM reward sequence {rp1    , rp2 , · · · , rpN }, a significant
decrease in reward between consecutive steps indicates the occurrence of such a drop-moment.
                                    (                                           )
                                         i       i
                                        rp,j − rp,j+1
                        δpi = max             i
                                                      j = 0, 1, . . . , N − 1       >ρ                  (5)
                                             rp,j

Here, ρ represents PRM’s drop-moment threshold. As illustrated in Figure 5, the PRM’s ability for
BoN selection and error identification remains largely unimpaired during the online RL process,
exhibiting stable performance. This suggests that although the scalar reward from the PRM in online
RL might be unreliable, the relative quality of solutions it reveals is comparatively trustworthy.
We leverage this beneficial property to address the reward sparsity problem in GRPO [56–58], aiming
to make online RL focus more on learning from rollouts that have accurate results and rigorous
processes. We use ρ from Equation 5 as the occurrence threshold for a “drop-moment”; when it
occurs, we apply a reward penalty γ to rollouts with correct results. This both differentiates the
learning value of outcome-correct rollouts and, due to its focus on relative drops in reward sequences,
circumvents the impact of PRM’s length bias in rewarding.

                                          oi is correct and δpi < ρ
                                  
                                  1,
                                i
                               R = 1 − γ, oi is correct and δpi ≥ ρ                                     (6)
                                  
                                    0,    otherwise

We utilize reward modeling in Equation 6 to conduct a process-supervised GRPO, which facilitates
the computation of in-group advantages in Equation 7.


                                                      6
Table 1: Performance Comparison on 6 math reasoning benchmarks. We use accuracy for MathVerse,
MathVision, MathVista and GeoQA. We use Score (Loose) on WE-MATH. And average-case
accuracy is employed on DYNAMATH. Best results of Closed-source MLLMs are highlighted in
green. Best and runner-up results of Open-source MLLMs are highlighted in red and blue.
                                                     MathVerse        MathVision        MathVista   WE-MATH       DYNAMATH           GeoQA
                                      Size   Avg
                                                      testmini         full set           gps        testmini       testmini         full set
                                                        Closed-Source MLLMs
 GPT-4o [59]                           -     55.5       50.2              30.4            64.7           62.8          64.9           62.1
 GPT-4o-mini [59]                      -     49.2       42.3              22.8            59.9           56.3          53.5           60.1
 Gemini-1.5-pro [60]                   -     53.2       35.3              19.2            81.7           66.9          60.5           55.5
                                                     Open-Source General MLLMs
 InternVL-Chat-V1.5 [61]              26B    33.6       26.1              15.4            56.9           32.7          36.7           33.5
 Llama-3.2-11B-Vision-Instruct [62]   11B    28.0       28.9              16.9            40.9           12.0          32.2           36.9
 Qwen2-VL [63]                        8B     40.2       33.6              19.2            51.0           43.0          42.1           52.2
 InternVL2-8B [64]                    8B     41.8       37.0              18.4            57.7           44.9          39.7           52.8
 InternVL2-8B-MPO [65]                 8B    45.1       38.2              22.3            69.2           44.4          40.5           55.9
 InternVL2.5-8B [66]                  8B     45.2       39.5              19.7            64.9           44.7          40.5           61.6
 LLaVA-OneVision [35]                  8B    40.9       28.9              18.3            71.6           44.9          37.5           43.9
 Points-Qwen2.5-Instruct [67]          8B    49.8       41.1              23.9            76.0           51.0          42.8           63.8
 Gemma3-12B [68]                      12B    49.8       40.1              29.1            63.6           51.7          45.8           67.7
                                                     Open-Source Reasoning MLLMs
 Math-LLaVA [15]                      13B    35.2       22.9              15.7            57.7           31.3          35.5           48.1
 MathPUMA-Qwen2-7B [11]                8B    39.6       33.6              14.0            48.1           41.0          37.3           63.6
 MultiMath [23]                        7B    43.1       27.7              16.3            66.8           42.2          37.9           67.7
 MAVIS [19]                            7B    44.4       35.2              18.5            64.1           44.3          36.2           68.3
 InfiMM-Math [14]                      7B    48.6       40.5              18.8            77.3           48.3          38.2           68.3
 AtomThink-EMOVA [12]                  8B    49.5       42.5              24.9            75.9           49.3          40.9           63.8
 MathGLM-Vision [9]                    9B    47.6       44.2              19.2            64.2           45.2          42.2           70.4
 LlamaV-o1 [69]                       11B    38.4       33.9              17.9            53.3           42.6          34.7           43.1
 OpenVLThinker [70]                    7B     -         47.9              25.3            76.4            -             -              -
 R1-Onevision [71]                     7B     -         47.4              26.9            72.4           51.4           -              -
 URSA-8B                              8B     54.7       45.7              28.7            81.7           53.6          44.7           73.5
 URSA-8B-PS-GRPO                      8B     58.2       50.9              31.5            83.2           60.7          47.4           75.6

    Table 2: Comparison of TTS on URSA-8B and AtomThink-EMOVA using BoN performance.
                                                       MathVerse                         MathVista-GPS                 MathVision
 Model                 Method
                                              N=4     N=8      N=16       N=32   N=4      N=8    N=16    N=32   N=4    N=8    N=16     N=32
                       Self-Consistency       49.3    50.1     50.7       50.7   82.7     83.9   84.8    85.4   29.4   31.9   32.8      33.1
 URSA-8B               InternVL2.5-8B ORM     48.6    50.9     51.8       51.3   82.5     83.3   84.3    85.1   29.9   32.1   32.8      33.5
                       URSA-8B-RM             53.3    54.2     54.7       55.0   83.2     85.5   86.5    87.2   31.6   33.1   34.0      35.1
                       Self-Consistency       45.9    46.7     47.1       47.3   76.8     77.9   78.6    79.0   25.3   26.8   27.6      28.0
 AtomThink-EMOVA       InternVL2.5-8B ORM     45.7    45.6     46.4       46.1   76.6     77.7   78.3    79.2   26.0   26.6   27.2      27.8
                       URSA-8B-RM             48.0    48.8     49.3       49.6   78.0     79.6   80.5    81.0   27.5   29.0   30.2      31.0



5     Experiments
5.1      Experimental Setup

Benchmarks We evaluate our URSA-series models on 6 widely used reasoning benchmarks,
including MathVerse [72], DYNAMATH [73], MathVista [74], WE-MATH [75], GeoQA [40] and
MathVision [43]. Detailed description and evaluation criteria can be found in Appendix F.3. We
consistently employ zero-shot inference for comparison.

Baselines We include some leading proprietary MLLMs, such as GPT-4o and GPT-4o-mini [59].
For open-source MLLMs with comparable size, we select InternVL-series [64, 76], LLaVA-
OneVision [35], Gemma3-12B [68], Qwen2-VL [63], and so on. For MLLMs intended for math rea-
soning purposes, we select AtomThink [12], InfiMM-Math [14], MAVIS [19], MathGLM-Vision [9],
LlamaV-o1 [69]. This kind of work focuses on the synthesis of STEM reasoning data or o1-like
slow thinking. We do not select baselines that use MathVision as training set for fairness, such as
Mulberry-Qwen2-VL-7B [77] and MAmooTH-VL [78]. For PRM’s TTS performance, we select
Self-Consistency [79] and open-source MLLM as ORM for comparison, such as InternVL2.5-8B [64].

Implementation Details URSA uses SAM-B+SigLIP-L as the hybrid vision encoder and Qwen2.5-
Math-Instruct as the LLM backbone. We employ a two-layer MLP connection for vision-language
alignment training. We select 15K data in MMathCoT-1M for PS-GRPO. γ and ρ in Equation 6 are


                                                                      7
                                                                                                        15.0                                                                                                                                  75.6
                                                   Vanilla GRPO                                                                                              GeoQA              72.2                              73.5
                                       14          PS-GRPO                                                                                                   MathVerse
                                                                                                                                                     70      MathVision
                                                                                                                                                             WE-MATH


Percentage Increase over URSA-8B (%)
                                                                                                                                                                                                                                                                   60.7
                                       12                          11.4                                                                              60
                                                                                                                                                                                                     52.8                              53.6
                                                                                9.8                                                                                                                                                                  50.9
                                       10                                                                                                            50
                                                                                                                                                                                                                         45.7




                                                                                                                                       Performance
                                                                                                  8.5                                                                                  41.5
                                        8                    7.4                                                                                     40
                                                    6.8
                                                                                                                     6.0                                                                                                                                    31.5
                                        6                                                                                                            30                                                                         28.7
                                                                          4.9                                                                                                                 25.6

                                        4                                                                                                            20
                                             3.1                                                                                 2.9                                     13.9
                                                                                                               2.2                                        11.1 12.4 10.0
                                        2                                                   1.8                                                      10
                                                                                      0.9
                                                                                                                           0.0
                                        0                                                                                                             0
                                              Avg          MathVerse MathVision MathVista WE-MATH DYNAMATH GeoQA                                            VL-Alignment                 SFT                        VL-Alignment                VL-Alignment
                                                            testmini  full set    gps      testmini testmini full set                                                                                                  +SFT                        +SFT
                                                                                                                                                                                                                                                 +PS-GRPO
                                                                                        (a)                                                                                                                 (b)
Figure 6: Figure(a) represents the comparison of relative improvements on URSA-8B; Figure(b)
illustrates how each training stage contributes to the total performance.


set to 0.5 and 0.3, respectively. Details on module selection, data selection, hyperparameters, and
time cost are placed in the Appendix D and F.

5.2                                         Main Results

SoTA Performance In Table 1, we present the performance of URSA-8B and URSA-8B-PS-
GRPO. First, URSA-8B provides a stronger reasoning foundation model. It demonstrates a 5.2 point
advantage over AtomThink-EMOVA which focuses on “slow thinking” training. It also outperforms
leading general-purpose MLLMs of comparable size, such as Gemma3-12B and InternVL2.5-8B.
URSA-8B-PS-GRPO outperforms GPT-4o across 6 benchmarks on average and shows significant
advantages on MathVista-GPS (83.2 vs 62.6), GeoQA (73.5 vs 62.1), and achieves the first surpassing
performance on MathVision (31.5 vs 30.4). However, a significant performance gap on DynaMath
suggests that smaller-scale MLLMs still lack more robust problem-solving capabilities. Compared to
the leading math reasoning MLLM AtomThink-EMOVA-8B and general-purpose MLLM Gemma3-
12B in terms of average performance, our model shows advantages of 8.5% and 8.2%, respectively.
Compared with recent R1-inspired method OpenVLThinker [70] and R1-Onevision [71], we still
show significant advantage on MathVision and WE-MATH.

Effective Best-of-N Evaluation In Table 2, we demonstrate the advantages of URSA-8B-RM com-
pared to self-consistency and the ORM baseline on serving TTS [43, 42]. We find that self-consistency
remains a strong baseline, which InternVL2.5-8B (serving as the ORM) does not consistently sur-
passes. However, URSA-8B-RM exhibits more effective BoN evaluation and demonstrates its
generalization on AtomThink-EMOVA-8B. In addition, using URSA-8B-RM as the verifier, only 4
samplings can achieve a huge improvement based on URSA-8B. Specifically, it provides a 16.6%
and 10.1% relative improvement on MathVerse and MathVision. In Best-of-32 setting, URSA-8B
achieve 35.1 and 55.0 in MathVision and MathVerse, showing clear advantage with GPT-4o.

PS-GRPO vs Vanilla GRPO As shown in Figure 6 (a), given the same training data, hyperparame-
ters, and rollout number PS-GRPO achieves a higher improvement on average performance (6.8%
vs 3.1%). PS-GRPO demonstrates an improvement that is nearly double that of vanilla GRPO
in WE-MATH and more challenging MathVision, suggesting its effectiveness. We notice that the
improvement of RL on MathVista-GPS and GeoQA is relatively small. This is because URSA-8B’s
inherent abilities have already achieved an effect close to the upper bound on these two benchmarks.
However, PS-GRPO still has advantages over vanilla GRPO.

6                                           Analysis
6.1                                         How Each Stage Contributes the Performance

In this section, we demonstrate how each stage contributes to the performance. As demonstrated in
Figure 6 (b), all stages make a performance contribution. MMathCoT-1M contributes the highest abso-
lute performance gain. The effect of Alignment-860K is more evident on MathVerse and MathVision,


                                                                                                                                       8
Table 3: Ablation study on DualMath-1.1M (BoN evaluation). w/o SM IE and w/o SBEL represents
dropping one part of DualMath-1.1M to train the PRM.
                                             MathVerse                    MathVista-GPS                 MathVision
 Model              Dataset
                                     N=4    N=8     N=16    N=32   N=4    N=8     N=16    N=32   N=4    N=8     N=16    N=32
                    DualMath-1.1M    53.3   54.2    54.7    55.0   83.2    85.5   86.5    87.2   31.6   33.1    34.0     35.1
 URSA-8B            w/o SM IE        52.8   52.6    52.4    53.9   81.3    83.8   83.1    83.2   29.9   30.5    33.1     34.5
                    w/o SBEL         50.3   51.4    51.8    53.0   80.1    83.1   82.2    83.0   28.7   29.8    32.3     34.2
                    DualMath-1.1M    48.0   48.8    49.3    49.6   78.0    79.6   80.5    81.0   27.5   29.0    30.2     31.0
 AtomThink-EMOVA    w/o SM IE        47.5   48.2    47.8    48.0   76.8    78.3   79.1    79.5   26.0   27.4    28.5     29.2
                    w/o SBEL         46.8   47.5    47.9    47.3   76.0    77.5   78.3    78.7   25.4   26.7    27.8     28.5



          Table 4: Sensitivity analysis on reward penalty and PRM’s “drop-moment” judgment.
                 MathVerse    MathVision           MathVista       WE-MATH         DYNAMATH             GeoQA
    γ       ρ                                                                                                          Avg
                  testmini     full set              gps            testmini         testmini           full set
    0.5    0.3      50.9            31.5             83.2             60.7                47.4           75.6          58.2
    0.5    0.4      49.9            30.8             81.2             59.9                46.9           75.0          57.3
    0.5    0.2      49.6            30.5             80.9             59.6                46.6           74.7          57.0
    1.0    0.3      49.0            29.4             79.8             58.8                45.3           72.5          56.3
    0.7    0.3      52.0            31.1             81.7             59.6                47.0           73.8          57.5
    0.3    0.3      51.5            32.0             82.1             61.0                46.3           74.6          57.9



likely because the question images in these two datasets contain richer textual modality information,
allowing alignment resources (such as textual images) to better supplement this comprehension
capability. PS-GRPO, on the other hand, is dedicated to breaking the bottleneck after large-scale
SFT, performing more prominently on WE-MATH and MathVerse with relative improvements of
13.2% and 11.4% respectively, compared to URSA-8B. We provide a generalization validation on
InternVL2.5-8B and Multimath in Appendix C.4.

6.2     Ablation Studies on Automatic Process Labeling

We give an ablation study on how two parts of DualMath-1.1M contribute to URSA-8B-RM. As
shown in Table 3, we can see that the method based on BEL, which focuses on the potential to
correctness, and the method based on MIE, which focuses on the perception consistency, both
contribute positively to the outcome. This further illustrates that in the process of multimodal math
reasoning, image-text inconsistency is widespread and needs to be mitigated. We address this issue
by augmenting the process supervision training data through the enforced imposition of common
hallucination categories. Specifically, the data generated by BEL demonstrates a more significant
impact, indicating that the quality of synthesized data can still be improved.

6.3     Sensitivity Analysis on Reward Penalty and Drop-moment

In this section, we conduct a sensitivity analysis on two hyperparameters of PS-GRPO, γ and ρ. These
respectively define the magnitude of the reward penalty for rollouts exhibiting a “drop-moment” and
the tolerance threshold for identifying such “drop-moments”. As shown in Table 4, our core findings
are twofold: (i) The value of γ should not be set too high, as this implies excessive trust in the PRM,
which may cause the rewards of a group to vanish and lead to training instability. When fixing ρ at
0.3, we find that setting γ to a value within a certain appropriate range (we test 0.3-0.7) is generally
beneficial for average performance. (ii) An excessively large ρ diminishes reward differentiation,
causing the RL behavior to approximate that of vanilla GRPO. Conversely, an excessively small ρ
is unreasonable by design, as it is overly sensitive to process reward changes and tends to result
in an overly broad range of penalties. In an extreme case where all correct rollouts are penalized,
PS-GRPO degenerates back to vanilla GRPO.

7       Conclusion
In this study, we take the first step to thoroughly explore the application of PRM in multimodal
math reasoning. We introduce a three-stage training pipeline URSA designed to address three major


                                                            9
challenges. Initially, we provide a large-scale CoT reasoning dataset MMathCoT-1M. This dataset
forms the basis for developing URSA-8B, a MLLM with enhanced reasoning capabilities, and paves
the way for further TTS or RL scenarios. Next, we present a dual-view automated process supervision
annotation method, covering logical validity and perceptual consistency in multimodal scenarios.
We introduce the first large-scale process supervision dataset in multimodal reasoning, DualMath-
1.1M. Finally, we address reward hacking and rewarding length bias through process-as-outcome
modeling, and put forward PS-GRPO, which is a PRM-aided online RL method that surpasses GRPO.
The resulting URSA-8B-PS-GRPO model demonstrates superior average performance over leading
open-source MLLM such as Gemma3-12B (8.4%) and proprietary GPT-4o (2.7%).

References
  [1] Haipeng Luo, Qingfeng Sun, Can Xu, Pu Zhao, Jianguang Lou, Chongyang Tao, Xiubo Geng, Qingwei
      Lin, Shifeng Chen, and Dongmei Zhang. Wizardmath: Empowering mathematical reasoning for large
      language models via reinforced evol-instruct. arXiv preprint arXiv:2308.09583, 2023.

  [2] An Yang, Beichen Zhang, Binyuan Hui, Bofei Gao, Bowen Yu, Chengpeng Li, Dayiheng Liu, Jianhong
      Tu, Jingren Zhou, Junyang Lin, Keming Lu, Mingfeng Xue, Runji Lin, Tianyu Liu, Xingzhang Ren, and
      Zhenru Zhang. Qwen2.5-math technical report: Toward mathematical expert model via self-improvement.
      CoRR, abs/2409.12122, 2024. doi: 10.48550/ARXIV.2409.12122. URL https://doi.org/10.48550/
      arXiv.2409.12122.

  [3] Huaiyuan Ying, Shuo Zhang, Linyang Li, Zhejian Zhou, Yunfan Shao, Zhaoye Fei, Yichuan Ma, Jiawei
      Hong, Kuikun Liu, Ziyi Wang, Yudong Wang, Zijian Wu, Shuaibin Li, Fengzhe Zhou, Hongwei Liu,
      Songyang Zhang, Wenwei Zhang, Hang Yan, Xipeng Qiu, Jiayu Wang, Kai Chen, and Dahua Lin.
      Internlm-math: Open math large language models toward verifiable reasoning. CoRR, abs/2402.06332,
      2024. doi: 10.48550/ARXIV.2402.06332. URL https://doi.org/10.48550/arXiv.2402.06332.

  [4] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan
      Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open
      language models. arXiv preprint arXiv:2402.03300, 2024.

  [5] Zhen Yang, Jinhao Chen, Zhengxiao Du, Wenmeng Yu, Weihan Wang, Wenyi Hong, Zhihuan Jiang, Bin
      Xu, Yuxiao Dong, and Jie Tang. Mathglm-vision: Solving mathematical problems with multi-modal large
      language model. arXiv preprint arXiv:2409.13729, 2024.

  [6] Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James T. Kwok, Zhenguo
      Li, Adrian Weller, and Weiyang Liu. Metamath: Bootstrap your own mathematical questions for large
      language models. In The Twelfth International Conference on Learning Representations, ICLR 2024,
      Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024. URL https://openreview.net/forum?id=
      N8N0hgNDRt.

  [7] Xinzhe Ni, Yeyun Gong, Zhibin Gou, Yelong Shen, Yujiu Yang, Nan Duan, and Weizhu Chen. Exploring
      the mystery of influential data for mathematical reasoning. CoRR, abs/2404.01067, 2024. doi: 10.48550/
      ARXIV.2404.01067. URL https://doi.org/10.48550/arXiv.2404.01067.

  [8] Yiyao Yu, Yuxiang Zhang, Dongdong Zhang, Xiao Liang, Hengyuan Zhang, Xingxing Zhang, Ziyi
      Yang, Mahmoud Khademi, Hany Awadalla, Junjie Wang, Yujiu Yang, and Furu Wei. Chain-of-reasoning:
      Towards unified mathematical reasoning in large language models via a multi-paradigm perspective.
      CoRR, abs/2501.11110, 2025. doi: 10.48550/ARXIV.2501.11110. URL https://doi.org/10.48550/
      arXiv.2501.11110.

  [9] Zhen Yang, Jinhao Chen, Zhengxiao Du, Wenmeng Yu, Weihan Wang, Wenyi Hong, Zhihuan Jiang,
      Bin Xu, Yuxiao Dong, and Jie Tang. Mathglm-vision: Solving mathematical problems with multi-
      modal large language model. CoRR, abs/2409.13729, 2024. doi: 10.48550/ARXIV.2409.13729. URL
      https://doi.org/10.48550/arXiv.2409.13729.

 [10] Huanjin Yao, Jiaxing Huang, Wenhao Wu, Jingyi Zhang, Yibo Wang, Shunyu Liu, Yingjie Wang,
      Yuxin Song, Haocheng Feng, Li Shen, and Dacheng Tao. Mulberry: Empowering MLLM with o1-like
      reasoning and reflection via collective monte carlo tree search. CoRR, abs/2412.18319, 2024. doi:
      10.48550/ARXIV.2412.18319. URL https://doi.org/10.48550/arXiv.2412.18319.

 [11] Wenwen Zhuang, Xin Huang, Xiantao Zhang, and Jin Zeng. Math-puma: Progressive upward multimodal
      alignment to enhance mathematical reasoning. In Proceedings of the AAAI Conference on Artificial
      Intelligence, volume 39, pages 26183–26191, 2025.


                                                    10
[12] Kun Xiang, Zhili Liu, Zihao Jiang, Yunshuang Nie, Runhui Huang, Haoxiang Fan, Hanhui Li, Weiran
     Huang, Yihan Zeng, Jianhua Han, et al. Atomthink: A slow thinking framework for multimodal
     mathematical reasoning. arXiv preprint arXiv:2411.11930, 2024.
[13] Wenxuan Huang, Bohan Jia, Zijie Zhai, Shaosheng Cao, Zheyu Ye, Fei Zhao, Zhe Xu, Yao Hu, and
     Shaohui Lin. Vision-r1: Incentivizing reasoning capability in multimodal large language models. CoRR,
     abs/2503.06749, 2025. doi: 10.48550/ARXIV.2503.06749. URL https://doi.org/10.48550/arXiv.
     2503.06749.
[14] Xiaotian Han, Yiren Jian, Xuefeng Hu, Haogeng Liu, Yiqi Wang, Qihang Fan, Yuang Ai, Huaibo Huang,
     Ran He, Zhenheng Yang, et al. Infimm-webmath-40b: Advancing multimodal pre-training for enhanced
     mathematical reasoning. In The 4th Workshop on Mathematical Reasoning and AI at NeurIPS’24, 2024.
[15] Wenhao Shi, Zhiqiang Hu, Yi Bin, Junhua Liu, Yang Yang, See-Kiong Ng, Lidong Bing, and Roy Ka-Wei
     Lee. Math-llava: Bootstrapping mathematical reasoning for multimodal large language models. arXiv
     preprint arXiv:2406.17294, 2024.
[16] Shihao Cai, Keqin Bao, Hangyu Guo, Jizhi Zhang, Jun Song, and Bo Zheng. Geogpt4v: Towards geomet-
     ric multi-modal large language models with geometric image generation. arXiv preprint arXiv:2406.11503,
     2024.
[17] Linger Deng, Yuliang Liu, Bohan Li, Dongliang Luo, Liang Wu, Chengquan Zhang, Pengyuan Lyu,
     Ziyang Zhang, Gang Zhang, Errui Ding, et al. R-cot: Reverse chain-of-thought problem generation for
     geometric reasoning in large multimodal models. arXiv preprint arXiv:2410.17885, 2024.
[18] Jiahui Gao, Renjie Pi, Jipeng Zhang, Jiacheng Ye, Wanjun Zhong, Yufei Wang, Lanqing Hong, Jianhua
     Han, Hang Xu, Zhenguo Li, and Lingpeng Kong. G-llava: Solving geometric problem with multi-
     modal large language model. CoRR, abs/2312.11370, 2023. doi: 10.48550/ARXIV.2312.11370. URL
     https://doi.org/10.48550/arXiv.2312.11370.
[19] Renrui Zhang, Xinyu Wei, Dongzhi Jiang, Ziyu Guo, Shicheng Li, Yichi Zhang, Chengzhuo Tong, Jiaming
     Liu, Aojun Zhou, Bin Wei, et al. Mavis: Mathematical visual instruction tuning with an automatic data
     engine. arXiv preprint arXiv:2407.08739, 2024.
[20] Renqiu Xia, Mingsheng Li, Hancheng Ye, Wenjie Wu, Hongbin Zhou, Jiakang Yuan, Tianshuo Peng,
     Xinyu Cai, Xiangchao Yan, Bin Wang, Conghui He, Botian Shi, Tao Chen, Junchi Yan, and Bo Zhang.
     Geox: Geometric problem solving through unified formalized vision-language pre-training. CoRR,
     abs/2412.11863, 2024. doi: 10.48550/ARXIV.2412.11863. URL https://doi.org/10.48550/arXiv.
     2412.11863.
[21] Renqiu Xia, Bo Zhang, Hancheng Ye, Xiangchao Yan, Qi Liu, Hongbin Zhou, Zijun Chen, Min Dou,
     Botian Shi, Junchi Yan, and Yu Qiao. Chartx & chartvlm: A versatile benchmark and foundation model
     for complicated chart reasoning. CoRR, abs/2402.12185, 2024. doi: 10.48550/ARXIV.2402.12185. URL
     https://doi.org/10.48550/arXiv.2402.12185.
[22] Zhuosheng Zhang, Aston Zhang, Mu Li, Hai Zhao, George Karypis, and Alex Smola. Multimodal
     chain-of-thought reasoning in language models. Trans. Mach. Learn. Res., 2024, 2024. URL https:
     //openreview.net/forum?id=y1pPWFVfvR.
[23] Shuai Peng, Di Fu, Liangcai Gao, Xiuqin Zhong, Hongguang Fu, and Zhi Tang. Multimath: Bridging
     visual and mathematical reasoning for large language models. arXiv preprint arXiv:2409.00147, 2024.
[24] Ruohong Zhang, Bowen Zhang, Yanghao Li, Haotian Zhang, Zhiqing Sun, Zhe Gan, Yinfei Yang,
     Ruoming Pang, and Yiming Yang. Improve vision language model chain-of-thought reasoning. arXiv
     preprint arXiv:2410.16198, 2024.
[25] Runze Liu, Junqi Gao, Jian Zhao, Kaiyan Zhang, Xiu Li, Biqing Qi, Wanli Ouyang, and Bowen Zhou.
     Can 1b LLM surpass 405b llm? rethinking compute-optimal test-time scaling. CoRR, abs/2502.06703,
     2025. doi: 10.48550/ARXIV.2502.06703. URL https://doi.org/10.48550/arXiv.2502.06703.
[26] Lunjun Zhang, Arian Hosseini, Hritik Bansal, Mehran Kazemi, Aviral Kumar, and Rishabh Agarwal.
     Generative verifiers: Reward modeling as next-token prediction. arXiv preprint arXiv:2408.15240, 2024.
[27] Dan Zhang, Sining Zhoubian, Ziniu Hu, Yisong Yue, Yuxiao Dong, and Jie Tang. Rest-mcts*: LLM self-
     training via process reward guided tree search. In Amir Globersons, Lester Mackey, Danielle Belgrave,
     Angela Fan, Ulrich Paquet, Jakub M. Tomczak, and Cheng Zhang, editors, Advances in Neural Information
     Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS
     2024, Vancouver, BC, Canada, December 10 - 15, 2024, 2024. URL http://papers.nips.cc/paper_
     files/paper/2024/hash/76ec4dc30e9faaf0e4b6093eaa377218-Abstract-Conference.html.


                                                   11
[28] Wei Liu, Junlong Li, Xiwen Zhang, Fan Zhou, Yu Cheng, and Junxian He. Diving into self-evolving
     training for multimodal reasoning. CoRR, abs/2412.17451, 2024. doi: 10.48550/ARXIV.2412.17451.
     URL https://doi.org/10.48550/arXiv.2412.17451.

[29] Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Yang Yue, Shiji Song, and Gao Huang.
     Does reinforcement learning really incentivize reasoning capacity in llms beyond the base model? arXiv
     preprint arXiv:2504.13837, 2025.

[30] Yibo Yan, Shen Wang, Jiahao Huo, Hang Li, Boyan Li, Jiamin Su, Xiong Gao, Yi-Fan Zhang, Tianlong
     Xu, Zhendong Chu, et al. Errorradar: Benchmarking complex mathematical reasoning of multimodal
     large language models via error detection. arXiv preprint arXiv:2410.04509, 2024.

[31] Di Zhang, Jingdi Lei, Junxian Li, Xunzhi Wang, Yujie Liu, Zonglin Yang, Jiatong Li, Weida Wang,
     Suorong Yang, Jianbo Wu, et al. Critic-v: Vlm critics help catch vlm errors in multimodal reasoning.
     arXiv preprint arXiv:2411.18203, 2024.

[32] Jiaxin Ai, Pengfei Zhou, Zhaopan Xu, Ming Li, Fanrui Zhang, Zizhen Li, Jianwen Sun, Yukang Feng,
     Baojin Huang, Zhongyuan Wang, and Kaipeng Zhang. Projudge: A multi-modal multi-discipline
     benchmark and instruction-tuning dataset for mllm-based process judges. CoRR, abs/2503.06553, 2025.
     doi: 10.48550/ARXIV.2503.06553. URL https://doi.org/10.48550/arXiv.2503.06553.

[33] Lilian Weng. Reward hacking and how to mitigate it. https://lilianweng.github.io/posts/
     2024-11-28-reward-hacking/, November 2024. [Accessed 11-28-2024].

[34] Jiayi Fu, Xuandong Zhao, Chengyuan Yao, Heng Wang, Qi Han, and Yanghua Xiao. Reward shaping to
     mitigate reward hacking in RLHF. CoRR, abs/2502.18770, 2025. doi: 10.48550/ARXIV.2502.18770.
     URL https://doi.org/10.48550/arXiv.2502.18770.

[35] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang,
     Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326,
     2024.

[36] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image
     pre-training. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages
     11975–11986, 2023.

[37] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao,
     Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the
     IEEE/CVF International Conference on Computer Vision, pages 4015–4026, 2023.

[38] Haoyu Lu, Wen Liu, Bo Zhang, Bingxuan Wang, Kai Dong, Bo Liu, Jingxiang Sun, Tongzheng Ren,
     Zhuoshu Li, Hao Yang, et al. Deepseek-vl: towards real-world vision-language understanding. arXiv
     preprint arXiv:2403.05525, 2024.

[39] Trieu H Trinh, Yuhuai Wu, Quoc V Le, He He, and Thang Luong. Solving olympiad geometry without
     human demonstrations. Nature, 625(7995):476–482, 2024.

[40] Jiaqi Chen, Jianheng Tang, Jinghui Qin, Xiaodan Liang, Lingbo Liu, Eric Xing, and Liang Lin. Geoqa: A
     geometric question answering benchmark towards multimodal numerical reasoning. In Findings of the
     Association for Computational Linguistics: ACL-IJCNLP 2021, pages 513–523, 2021.

[41] Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John
     Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. arXiv preprint arXiv:2305.20050,
     2023.

[42] Liangchen Luo, Yinxiao Liu, Rosanne Liu, Samrat Phatale, Harsh Lara, Yunxuan Li, Lei Shu, Yun Zhu,
     Lei Meng, Jiao Sun, et al. Improve mathematical reasoning in language models by automated process
     supervision. arXiv preprint arXiv:2406.06592, 2024.

[43] Peiyi Wang, Lei Li, Zhihong Shao, Runxin Xu, Damai Dai, Yifei Li, Deli Chen, Yu Wu, and Zhifang Sui.
     Math-shepherd: Verify and reinforce llms step-by-step without human annotations. In Proceedings of the
     62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages
     9426–9439, 2024.

[44] Haojie Zheng, Tianyang Xu, Hanchi Sun, Shu Pu, Ruoxi Chen, and Lichao Sun. Thinking before looking:
     Improving multimodal llm reasoning via mitigating visual hallucination. arXiv preprint arXiv:2411.12591,
     2024.


                                                    12
[45] Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, and Graham
     Neubig. Pal: Program-aided language models. In International Conference on Machine Learning, pages
     10764–10799. PMLR, 2023.
[46] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong
     Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement
     learning. arXiv preprint arXiv:2501.12948, 2025.
[47] Jiazhen Pan, Che Liu, Junde Wu, Fenglin Liu, Jiayuan Zhu, Hongwei Bran Li, Chen Chen, Cheng Ouyang,
     and Daniel Rueckert. Medvlm-r1: Incentivizing medical reasoning capability of vision-language models
     (vlms) via reinforcement learning. CoRR, abs/2502.19634, 2025. doi: 10.48550/ARXIV.2502.19634.
     URL https://doi.org/10.48550/arXiv.2502.19634.
[48] Yufei Zhan, Yousong Zhu, Shurong Zheng, Hongyin Zhao, Fan Yang, Ming Tang, and Jinqiao
     Wang. Vision-r1: Evolving human-free alignment in large vision-language models via vision-guided
     reinforcement learning. CoRR, abs/2503.18013, 2025. doi: 10.48550/ARXIV.2503.18013. URL
     https://doi.org/10.48550/arXiv.2503.18013.
[49] Wenxuan Huang, Bohan Jia, Zijie Zhai, Shaosheng Cao, Zheyu Ye, Fei Zhao, Zhe Xu, Yao Hu, and
     Shaohui Lin. Vision-r1: Incentivizing reasoning capability in multimodal large language models. CoRR,
     abs/2503.06749, 2025. doi: 10.48550/ARXIV.2503.06749. URL https://doi.org/10.48550/arXiv.
     2503.06749.
[50] Xiangyan Liu, Jinjie Ni, Zijian Wu, Chao Du, Longxu Dou, Haonan Wang, Tianyu Pang, and
     Michael Qizhe Shieh. Noisyrollout: Reinforcing visual reasoning with data augmentation, 2025. URL
     https://arxiv.org/abs/2504.13055.
[51] Wendi Li and Yixuan Li. Process reward model with q-value rankings. CoRR, abs/2410.11287, 2024.
     doi: 10.48550/ARXIV.2410.11287. URL https://doi.org/10.48550/arXiv.2410.11287.
[52] Amrith Setlur, Chirag Nagpal, Adam Fisch, Xinyang Geng, Jacob Eisenstein, Rishabh Agarwal, Alekh
     Agarwal, Jonathan Berant, and Aviral Kumar. Rewarding progress: Scaling automated process verifiers
     for LLM reasoning. CoRR, abs/2410.08146, 2024. doi: 10.48550/ARXIV.2410.08146. URL https:
     //doi.org/10.48550/arXiv.2410.08146.
[53] Yiran Ma, Zui Chen, Tianqiao Liu, Mi Tian, Zhuo Liu, Zitao Liu, and Weiqi Luo. What are step-
     level reward models rewarding? counterintuitive findings from mcts-boosted mathematical reasoning.
     In Toby Walsh, Julie Shah, and Zico Kolter, editors, AAAI-25, Sponsored by the Association for the
     Advancement of Artificial Intelligence, February 25 - March 4, 2025, Philadelphia, PA, USA, pages
     24812–24820. AAAI Press, 2025. doi: 10.1609/AAAI.V39I23.34663. URL https://doi.org/10.
     1609/aaai.v39i23.34663.
[54] Jiaxuan Gao, Shusheng Xu, Wenjie Ye, Weilin Liu, Chuyi He, Wei Fu, Zhiyu Mei, Guangju Wang, and
     Yi Wu. On designing effective RL reward at training time for LLM reasoning. CoRR, abs/2410.15115,
     2024. doi: 10.48550/ARXIV.2410.15115. URL https://doi.org/10.48550/arXiv.2410.15115.
[55] Dario Amodei, Chris Olah, Jacob Steinhardt, Paul Christiano, John Schulman, and Dan Mané. Concrete
     problems in ai safety. arXiv preprint arXiv:1606.06565, 2016.
[56] Jixiao Zhang and Chunsheng Zuo. Grpo-lead: A difficulty-aware reinforcement learning approach for
     concise mathematical reasoning in language models. arXiv preprint arXiv:2504.09696, 2025.
[57] Jingyi Zhang, Jiaxing Huang, Huanjin Yao, Shunyu Liu, Xikun Zhang, Shijian Lu, and Dacheng Tao.
     R1-vl: Learning to reason with multimodal large language models via step-wise group relative policy
     optimization. arXiv preprint arXiv:2503.12937, 2025.
[58] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Tiantian Fan, Gaohong Liu,
     Lingjun Liu, Xin Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv
     preprint arXiv:2503.14476, 2025.
[59] OpenAI. GPT-4o system card, 2024. URL https://openai.com/research/gpt-4o-system-card.
[60] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan
     Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable
     multimodal models. arXiv preprint arXiv:2312.11805, 2023.
[61] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi
     Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal
     models with open-source suites. arXiv preprint arXiv:2404.16821, 2024.


                                                    13
[62] Meta. Llama 3.2: Revolutionizing edge AI and vision with open, customizable models — ai.meta.com.
     https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/, 2024.
     [Accessed 17-04-2025].

[63] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin
     Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any
     resolution. arXiv preprint arXiv:2409.12191, 2024.

[64] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang,
     Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic
     visual-linguistic tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
     Recognition, pages 24185–24198, 2024.

[65] Weiyun Wang, Zhe Chen, Wenhai Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Jinguo Zhu, Xizhou
     Zhu, Lewei Lu, Yu Qiao, et al. Enhancing the reasoning ability of multimodal large language models via
     mixed preference optimization. arXiv preprint arXiv:2411.10442, 2024.

[66] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye,
     Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models
     with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271, 2024.

[67] Yuan Liu, Zhongyin Zhao, Ziyuan Zhuang, Le Tian, Xiao Zhou, and Jie Zhou. Points: Improving your
     vision-language model with affordable strategies. arXiv preprint arXiv:2409.04828, 2024.

[68] Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah
     Perrin, Tatiana Matejovicova, Alexandre Ramé, Morgane Rivière, et al. Gemma 3 technical report. arXiv
     preprint arXiv:2503.19786, 2025.

[69] Omkar Thawakar, Dinura Dissanayake, Ketan More, Ritesh Thawkar, Ahmed Heakl, Noor Ahsan,
     Yuhao Li, Mohammed Zumri, Jean Lahoud, Rao Muhammad Anwer, Hisham Cholakkal, Ivan Laptev,
     Mubarak Shah, Fahad Shahbaz Khan, and Salman H. Khan. Llamav-o1: Rethinking step-by-step
     visual reasoning in llms. CoRR, abs/2501.06186, 2025. doi: 10.48550/ARXIV.2501.06186. URL
     https://doi.org/10.48550/arXiv.2501.06186.

[70] Yihe Deng, Hritik Bansal, Fan Yin, Nanyun Peng, Wei Wang, and Kai-Wei Chang. Openvlthinker:
     An early exploration to complex vision-language reasoning via iterative self-improvement. CoRR,
     abs/2503.17352, 2025. doi: 10.48550/ARXIV.2503.17352. URL https://doi.org/10.48550/arXiv.
     2503.17352.

[71] Yi Yang, Xiaoxuan He, Hongkun Pan, Xiyan Jiang, Yan Deng, Xingtao Yang, Haoyu Lu, Dacheng Yin,
     Fengyun Rao, Minfeng Zhu, Bo Zhang, and Wei Chen. R1-onevision: Advancing generalized multimodal
     reasoning through cross-modal formalization. arXiv preprint arXiv:2503.10615, 2025.

[72] Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu,
     Kai-Wei Chang, Yu Qiao, et al. Mathverse: Does your multi-modal llm truly see the diagrams in visual
     math problems? In European Conference on Computer Vision, pages 169–186. Springer, 2025.

[73] Chengke Zou, Xingang Guo, Rui Yang, Junyu Zhang, Bin Hu, and Huan Zhang. Dynamath: A dynamic
     visual benchmark for evaluating mathematical reasoning robustness of vision language models. arXiv
     preprint arXiv:2411.00836, 2024.

[74] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei
     Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation
     models in visual contexts. arXiv preprint arXiv:2310.02255, 2023.

[75] Runqi Qiao, Qiuna Tan, Guanting Dong, Minhui Wu, Chong Sun, Xiaoshuai Song, Zhuoma GongQue,
     Shanglin Lei, Zhe Wei, Miaoxuan Zhang, et al. We-math: Does your large multimodal model achieve
     human-like mathematical reasoning? arXiv preprint arXiv:2407.01284, 2024.

[76] Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Bin Wang, Linke Ouyang, Xilin Wei, Songyang
     Zhang, Haodong Duan, Maosong Cao, et al. Internlm-xcomposer2: Mastering free-form text-image
     composition and comprehension in vision-language large model. arXiv preprint arXiv:2401.16420, 2024.

[77] Huanjin Yao, Jiaxing Huang, Wenhao Wu, Jingyi Zhang, Yibo Wang, Shunyu Liu, Yingjie Wang,
     Yuxin Song, Haocheng Feng, Li Shen, and Dacheng Tao. Mulberry: Empowering MLLM with o1-like
     reasoning and reflection via collective monte carlo tree search. CoRR, abs/2412.18319, 2024. doi:
     10.48550/ARXIV.2412.18319. URL https://doi.org/10.48550/arXiv.2412.18319.


                                                   14
[78] Jarvis Guo, Tuney Zheng, Yuelin Bai, Bo Li, Yubo Wang, King Zhu, Yizhi Li, Graham Neubig, Wenhu
     Chen, and Xiang Yue. Mammoth-vl: Eliciting multimodal reasoning with instruction tuning at scale.
     arXiv preprint arXiv:2412.05237, 2024.

[79] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery,
     and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. arXiv
     preprint arXiv:2203.11171, 2022.

[80] Jiahui Gao, Renjie Pi, Jipeng Zhang, Jiacheng Ye, Wanjun Zhong, Yufei Wang, Lanqing Hong, Jianhua
     Han, Hang Xu, Zhenguo Li, et al. G-llava: Solving geometric problem with multi-modal large language
     model. arXiv preprint arXiv:2312.11370, 2023.

[81] Yuhao Dong, Zuyan Liu, Hai-Long Sun, Jingkang Yang, Winston Hu, Yongming Rao, and Ziwei Liu.
     Insight-v: Exploring long-chain visual reasoning with multimodal large language models. arXiv preprint
     arXiv:2411.14432, 2024.

[82] Yushi Hu, Weijia Shi, Xingyu Fu, Dan Roth, Mari Ostendorf, Luke Zettlemoyer, Noah A Smith, and
     Ranjay Krishna. Visual sketchpad: Sketching as a visual chain of thought for multimodal language
     models. arXiv preprint arXiv:2406.09403, 2024.

[83] Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James T Kwok, Zhenguo
     Li, Adrian Weller, and Weiyang Liu. Metamath: Bootstrap your own mathematical questions for large
     language models. arXiv preprint arXiv:2309.12284, 2023.

[84] Dongyang Liu, Renrui Zhang, Longtian Qiu, Siyuan Huang, Weifeng Lin, Shitian Zhao, Shijie Geng, Ziyi
     Lin, Peng Jin, Kaipeng Zhang, et al. Sphinx-x: Scaling data and parameters for a family of multi-modal
     large language models. arXiv preprint arXiv:2402.05935, 2024.

[85] Zayne Sprague, Fangcong Yin, Juan Diego Rodriguez, Dongwei Jiang, Manya Wadhwa, Prasann Singhal,
     Xinyu Zhao, Xi Ye, Kyle Mahowald, and Greg Durrett. To cot or not to cot? chain-of-thought helps
     mainly on math and symbolic reasoning. arXiv preprint arXiv:2409.12183, 2024.

[86] Yingzhou Lu, Minjie Shen, Huazheng Wang, Xiao Wang, Capucine van Rechem, Tianfan Fu, and Wenqi
     Wei. Machine learning for synthetic data generation: a review. arXiv preprint arXiv:2302.04062, 2023.

[87] Yiming Huang, Xiao Liu, Yeyun Gong, Zhibin Gou, Yelong Shen, Nan Duan, and Weizhu Chen.
     Key-point-driven data synthesis with its enhancement on mathematical reasoning. arXiv preprint
     arXiv:2403.02333, 2024.

[88] Chaoyou Fu, Haojia Lin, Zuwei Long, Yunhang Shen, Meng Zhao, Yifan Zhang, Shaoqi Dong, Xiong
     Wang, Di Yin, Long Ma, et al. Vita: Towards open-source interactive omni multimodal llm. arXiv
     preprint arXiv:2408.05211, 2024.

[89] Zhibin Gou, Zhihong Shao, Yeyun Gong, Yelong Shen, Yujiu Yang, Nan Duan, and Weizhu Chen. Critic:
     Large language models can self-correct with tool-interactive critiquing. arXiv preprint arXiv:2305.11738,
     2023.

[90] Bofei Gao, Zefan Cai, Runxin Xu, Peiyi Wang, Ce Zheng, Runji Lin, Keming Lu, Junyang Lin, Chang
     Zhou, Wen Xiao, et al. Llm critics help catch bugs in mathematics: Towards a better mathematical verifier
     with natural language feedback. CoRR, 2024.

[91] Zicheng Lin, Zhibin Gou, Tian Liang, Ruilin Luo, Haowei Liu, and Yujiu Yang. CriticBench: Benchmark-
     ing LLMs for critique-correct reasoning. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors,
     Findings of the Association for Computational Linguistics: ACL 2024, pages 1552–1587, Bangkok, Thai-
     land, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.findings-acl.91.
     URL https://aclanthology.org/2024.findings-acl.91.

[92] Aviral Kumar, Vincent Zhuang, Rishabh Agarwal, Yi Su, John D Co-Reyes, Avi Singh, Kate Baumli,
     Shariq Iqbal, Colton Bishop, Rebecca Roelofs, et al. Training language models to self-correct via
     reinforcement learning. arXiv preprint arXiv:2409.12917, 2024.

[93] Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling llm test-time compute optimally can
     be more effective than scaling model parameters. arXiv preprint arXiv:2408.03314, 2024.

[94] Haoqin Tu, Weitao Feng, Hardy Chen, Hui Liu, Xianfeng Tang, and Cihang Xie. Vilbench: A suite for
     vision-language process reward modeling. arXiv preprint arXiv:2503.20271, 2025.


                                                    15
 [95] Weiyun Wang, Zhangwei Gao, Lianjie Chen, Zhe Chen, Jinguo Zhu, Xiangyu Zhao, Yangzhou Liu, Yue
      Cao, Shenglong Ye, Xizhou Zhu, et al. Visualprm: An effective process reward model for multimodal
      reasoning. arXiv preprint arXiv:2503.10291, 2025.

 [96] Linzhuang Sun, Hao Liang, Jingxuan Wei, Bihui Yu, Tianpeng Li, Fan Yang, Zenan Zhou, and Wentao
      Zhang. Mm-verify: Enhancing multimodal reasoning with chain-of-thought verification. arXiv preprint
      arXiv:2502.13383, 2025.

 [97] Bofei Gao, Zefan Cai, Runxin Xu, Peiyi Wang, Ce Zheng, Runji Lin, Keming Lu, Junyang Lin, Chang
      Zhou, Wen Xiao, Junjie Hu, Tianyu Liu, and Baobao Chang. LLM critics help catch bugs in mathematics:
      Towards a better mathematical verifier with natural language feedback. CoRR, abs/2406.14024, 2024.
      doi: 10.48550/ARXIV.2406.14024. URL https://doi.org/10.48550/arXiv.2406.14024.
 [98] Weihao Zeng, Yuzhen Huang, Lulu Zhao, Yijun Wang, Zifei Shan, and Junxian He. B-star: Monitoring
      and balancing exploration and exploitation in self-taught reasoners. arXiv preprint arXiv:2412.17256,
      2024.
 [99] Fanqing Meng, Lingxiao Du, Zongkai Liu, Zhixiang Zhou, Quanfeng Lu, Daocheng Fu, Botian Shi,
      Wenhai Wang, Junjun He, Kaipeng Zhang, et al. Mm-eureka: Exploring visual aha moment with
      rule-based large-scale reinforcement learning. arXiv preprint arXiv:2503.07365, 2025.
[100] Leandro von Werra, Younes Belkada, Lewis Tunstall, Edward Beeching, Tristan Thrush, Nathan Lambert,
      Shengyi Huang, Kashif Rasul, and Quentin Gallouédec. Trl: Transformer reinforcement learning.
      https://github.com/huggingface/trl, 2020.
[101] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou,
      and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text
      reading, and beyond. arXiv preprint arXiv:2308.12966, 1(2):3, 2023.

[102] Hugging Face. Open r1: A fully open reproduction of deepseek-r1, January 2025. URL https:
      //github.com/huggingface/open-r1.

[103] P Kingma Diederik. Adam: A method for stochastic optimization. (No Title), 2014.

[104] Yanli Zhao, Andrew Gu, Rohan Varma, Liang Luo, Chien-Chin Huang, Min Xu, Less Wright, Hamid
      Shojanazeri, Myle Ott, Sam Shleifer, Alban Desmaison, Can Balioglu, Pritam Damania, Bernard Nguyen,
      Geeta Chauhan, Yuchen Hao, Ajit Mathews, and Shen Li. Pytorch FSDP: experiences on scaling fully
      sharded data parallel. Proc. VLDB Endow., 16(12):3848–3860, 2023. doi: 10.14778/3611540.3611569.
      URL https://www.vldb.org/pvldb/vol16/p3848-huang.pdf.

[105] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph
      Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving
      with pagedattention. In Proceedings of the 29th Symposium on Operating Systems Principles, pages
      611–626, 2023.

[106] Jingcheng Hu, Yinmin Zhang, Qi Han, Daxin Jiang, Xiangyu Zhang, and Heung-Yeung Shum. Open-
      reasoner-zero: An open source approach to scaling up reinforcement learning on the base model. CoRR,
      abs/2503.24290, 2025. doi: 10.48550/ARXIV.2503.24290. URL https://doi.org/10.48550/arXiv.
      2503.24290.
[107] Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Mingjie Zhan, and Hongsheng Li. Measuring multimodal
      mathematical reasoning with math-vision dataset. arXiv preprint arXiv:2402.14804, 2024.
