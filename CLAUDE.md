# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

URSA (Understanding and Verifying Chain-of-thought Reasoning in Multimodal Mathematics) 是首个专注于多模态数学推理的过程奖励模型(PRM)框架。该项目实现了论文《Unlocking Multimodal Mathematical Reasoning via Process Reward Model》中提出的三阶段训练策略。

## 论文简述

### 研究背景与动机

过程奖励模型(PRM)在通过测试时扩展(TTS)增强大语言模型(LLM)的数学推理能力方面已显示出潜力，但其在多模态推理中的应用仍未被充分探索。本研究首次尝试将PRM整合到多模态数学推理中。

### 三个核心挑战

1. **高质量推理数据稀缺**: TTS和强化学习都严重依赖基础模型的强度，大规模高质量推理数据的有限可用性限制了当前MLLM的上限，削弱了PRM集成的有效性。

2. **缺乏自动化过程标注方法**: 多模态场景中尚未有充分的自动化过程标注技术，需要同时强调逻辑有效性和感知一致性，图像-文本不一致是多模态推理中的独特问题。

3. **过程奖励在在线RL中的问题**: 虽然PRM可以有效用于TTS，但直接应用于在线RL会引入奖励欺骗(reward hacking)和奖励中的长度偏差(length bias)问题。

### URSA三阶段框架详解

#### 阶段一：数学密集型对齐和指令微调

**视觉-语言对齐 (URSA-Alignment-860K)**
- 从Multimath、MAVIS和Geo170K等开源数据集收集860K对齐数据
- 仅训练MLP投影器，保持视觉和语言模型冻结

**CoT推理数据合成 (MMathCoT-1M)**

从1.43M样本中构建，采用三种策略：

1. **CoT扩展** (440K样本): 针对仅有答案的数据，使用Gemini-1.5-Flash-002扩展推理轨迹，避免基于记忆的推理
2. **重写** (715K样本): 针对分析格式数据，增强逐步推理轨迹，提高语言多样性，解决推理跳跃问题
3. **格式统一** (275K样本): 将数学语言和符号风格的解决方案统一为自然语言风格

**质量过滤**: 过滤违反正确性和一致性的实例，确保生成内容不改变原始答案。

#### 阶段二：双视角过程监督数据合成

**二分错误定位引擎 (BEL)**
- 从URSA-8B对MMathCoT-1M的零样本推理中收集约553K错误解决方案
- 使用蒙特卡洛树搜索(MCTS)标注错误步骤
- 二分搜索优化：通过MC估计值判断错误位置
- 添加约180K正确解决方案作为正例
- 生成773K基于逻辑正确性的过程标注数据集SBEL

**误解插入引擎 (MIE)**

针对多模态场景中图像-文本感知不一致问题：
1. 提示Gemini提取图像中的数学范式信息
2. 关注现有正确解决方案中可能混淆的条件，使用相邻或相似条件修改
3. 基于插入错误的步骤继续推理
4. 自动为错误插入后的每个后续步骤分配负标签
5. 生成约302K样本SMIE

**PRM训练**: 合并两类数据形成DualMath-1.1M，在每个步骤后附加特殊token指示其预测正确性，建模为二分类任务。

#### 阶段三：PS-GRPO算法

**问题分析**

研究发现标量过程奖励建模的失败模式：
- 高度易受奖励欺骗影响：模型快速学习迎合过程正确性的策略
- PRM奖励中的长度偏差：导致更短的模型响应和更少的推理步骤

**PS-GRPO核心创新**

核心思想：将过程作为结果奖励建模(process-as-outcome reward modeling)

**关键概念："drop-moment"**
- PRM奖励序列中的显著下降点
- 表示PRM质疑前面步骤的有效性
- 用于识别被质疑的推理轨迹

**算法优势**
- 不依赖标量过程奖励作为优化目标
- 通过比较rollouts的相对质量进行优化
- 对被PRM质疑的rollouts施加隐式惩罚
- 有效缓解奖励欺骗和长度偏差

**参数设置**
- γ=0.5: 控制奖励惩罚的强度
- ρ=0.3: 控制PRM的drop-moment判断阈值

### 主要实验结果

**整体性能**
- URSA-8B-PS-GRPO在6个基准上平均超越Gemma3-12B 8.4%
- 平均超越GPT-4o 2.7%
- MathVista-GPS: 83.2 vs GPT-4o的64.7
- GeoQA: 75.6 vs GPT-4o的62.1
- MathVision: 31.5 vs GPT-4o的30.4 (首次超越)

**Best-of-N评估**
- URSA-8B-RM显著优于自洽性和ORM基线
- N=4时在MathVerse上相对提升16.6%
- N=32时达到MathVision 35.1、MathVerse 55.0

**PS-GRPO vs Vanilla GRPO**
- 平均性能提升：6.8% vs 3.1% (提升翻倍)
- WE-MATH提升：11.4% vs 4.9%
- MathVision提升：9.8% vs 1.8%

### 关键技术贡献

1. **首个多模态PRM框架**: 首次系统性地将PRM整合到多模态数学推理中

2. **两个大规模开源数据集**:
   - MMathCoT-1M: 高质量多模态CoT推理数据
   - DualMath-1.1M: 首个大规模自动标注的多模态过程监督数据

3. **双视角过程监督**: BEL关注逻辑正确性，MIE关注感知一致性，互补信号提升PRM质量

4. **PS-GRPO算法**: 首个多模态PRM辅助的在线RL方法，有效缓解奖励欺骗和长度偏差

5. **自动化数据合成流程**: 避免昂贵的人工标注，利用强大的MLLM进行数据增强，二分搜索优化MCTS标注效率

6. **实用性验证**: 在TTS和在线RL场景下均验证有效，具有跨模型泛化能力

### 核心组件

- **URSA-8B**: 首个专注于链式思维(CoT)多模态数学推理的小型MLLM
- **URSA-RM-8B**: 首个开源的小型多模态数学奖励模型
- **MMathCoT-1M**: 高质量大规模多模态CoT推理数据集
- **DualMath-1.1M**: 双视角过程监督数据集

### 三阶段训练框架详解

#### Stage I: 视觉-语言对齐和数学指令微调

**训练数据**:
- **对齐阶段**: URSA-Alignment-860K
  - 来源: Multimath、MAVIS、Geo170K等开源数据集
  - 包含图像-文本对和详细描述
- **指令微调阶段**: MMathCoT-1M
  - 1.43M源数据经过CoT扩展、重写、格式统一后得到
  - 包含440K CoT扩展、715K重写、275K格式统一样本

**基础模型**:
- **视觉编码器**: SAM-B + SigLIP-L (混合视觉塔)
- **语言模型**: Qwen2.5-Math-Instruct (8B参数，纯文本模型)
- **投影器**: 两层MLP连接视觉和语言模态

**为什么需要图文对齐训练？**

Qwen2.5-Math-Instruct是**纯文本语言模型**，完全不具备视觉能力。对齐训练的必要性：

1. **建立视觉-语言桥梁**: 语言模型本身无法处理图像，必须通过MLP投影器将视觉特征映射到语言模型的表示空间
2. **保护预训练知识**: 只训练投影器，冻结视觉编码器和语言模型，避免破坏已有的数学推理能力和视觉理解能力
3. **学习模态对齐**: 使用860K图文对数据让模型学习如何将视觉信息转换为语言模型可理解的表示

这是标准的**LLaVA架构范式**：纯文本LLM + 视觉编码器 + 投影器 = 多模态模型

**为什么同时使用SigLIP和SAM？**

混合视觉塔设计源于数学推理任务的双重需求：

**SigLIP-L的作用**：
- **语义理解**: 基于对比学习训练，擅长捕捉图像的高层语义信息
- **符号识别**: 对数学公式、文本、符号等有较好的识别能力
- **全局特征**: 提供整体的视觉-语言对齐能力

**SAM-B的作用**：
- **细粒度分割**: Segment Anything Model专注于精确的空间分割
- **几何细节**: 对几何图形的边界、形状、角度、线段有精确感知
- **局部特征**: 捕捉图像中的细节信息，如点的位置、线段关系

**互补性与必要性**：
- 数学推理任务（尤其是几何问题）需要**同时理解语义和几何**
- SigLIP擅长"这是什么"（语义识别），SAM擅长"在哪里、什么形状"（空间关系）
- 单一视觉编码器难以同时满足语义理解和几何细节捕捉的需求
- 论文提到前人工作关注"training math-intensive vision encoders"，说明这是多模态数学推理的关键挑战

**训练策略**:
- **对齐训练**:
  - 只训练MLP投影器
  - 冻结视觉编码器和语言模型
- **指令微调**:
  - 全参数微调 (full-parameter fine-tuning)
  - 使用标准监督学习目标 (交叉熵损失)
  - 训练目标: L_SFT = -E_{(x,y)~D_SFT} Σ log M(y_t|x, y_<t)

**训练产物**:
- **URSA-8B**: 具有强大多模态数学推理能力的基础模型

**辅助工具**:
- 使用Gemini-1.5-Flash-002进行数据合成和质量过滤

---

#### Stage II: 过程奖励模型训练

**训练数据**: DualMath-1.1M (约1.1M过程监督样本)
- **SBEL (Binary Error Locating)**: ~773K样本
  - ~553K错误解决方案 (从URSA-8B零样本推理MMathCoT-1M得到)
  - ~180K正确解决方案 (作为正例)
  - 使用MCTS进行错误步骤标注
  - 二分搜索优化标注效率
- **SMIE (Misinterpretation Insertion Engine)**: ~302K样本
  - 针对图像-文本感知不一致问题
  - 使用Gemini提取图像信息并插入误解

**基础模型**:
- 基于**URSA-8B**继续训练 (Stage I的产物)

**训练策略**:
- 在每个推理步骤后附加特殊token `и`
- 建模为二分类任务 (每步正确/错误)
- 训练目标: L_PRM = -E_{(e,y)~D_PRM} Σ [y_j log π_p(e_j) + (1-y_j) log(1-π_p(e_j))]
- 同时关注逻辑正确性和感知一致性

**训练产物**:
- **URSA-RM-8B**: 首个开源的小型多模态数学过程奖励模型

**数据标注工具**:
- MCTS (蒙特卡洛树搜索) 用于错误定位
- Gemini-1.5-Flash-002 用于误解插入

---

#### Stage III: PS-GRPO在线强化学习

**训练数据**:
- 从MMathCoT-1M中选择15K样本用于在线RL训练

**基础模型**:
- **策略模型**: URSA-8B (Stage I的产物)
- **奖励模型**: URSA-RM-8B (Stage II的产物)

**训练策略**: Process-Supervised GRPO
- **核心思想**: 过程作为结果奖励建模 (process-as-outcome reward modeling)
- **关键机制**:
  - 检测"drop-moment" (PRM分数显著下降点)
  - 对被PRM质疑的rollouts施加惩罚
  - 通过组内相对比较优化策略
- **奖励计算**:
  ```
  如果检测到drop-moment:
    reward = outcome_reward × γ  (γ=0.5)
  否则:
    reward = outcome_reward
  ```
- **Drop-moment判断**: 使用阈值ρ=0.3判断PRM分数下降

**训练目标**:
- 不直接优化标量过程奖励
- 通过GRPO算法最大化组内相对优势
- 隐式惩罚过程级不一致性

**训练产物**:
- **URSA-8B-PS-GRPO**: 最终的强化学习优化模型

**对比方法**:
- **Vanilla GRPO**: 只使用结果奖励的标准GRPO
- **PS-GRPO**: 集成过程监督的改进版GRPO
- 性能提升: PS-GRPO平均提升6.8% vs Vanilla GRPO的3.1%

---

### 三阶段训练流程总结

| 阶段 | 输入模型 | 训练数据 | 训练方法 | 输出模型 | 数据规模 |
|------|---------|---------|---------|---------|---------|
| **Stage I** | Qwen2.5-Math-Instruct + SAM-B + SigLIP-L | URSA-Alignment-860K + MMathCoT-1M | 对齐训练 + 全参数SFT | URSA-8B | 860K + 1M |
| **Stage II** | URSA-8B | DualMath-1.1M (SBEL + SMIE) | 二分类PRM训练 | URSA-RM-8B | 1.1M |
| **Stage III** | URSA-8B (策略) + URSA-RM-8B (奖励) | MMathCoT-1M (15K子集) | PS-GRPO在线RL | URSA-8B-PS-GRPO | 15K |

**关键特点**:
- **Stage I**: 标准监督学习，构建强大基础模型
- **Stage II**: 判别式训练，学习评估推理过程质量
- **Stage III**: 强化学习，通过相对比较优化策略
- **无RFT**: 三个阶段均未使用Rejection Fine-Tuning方法

---

### 三阶段在代码库中的对应关系

#### Stage I 相关代码

**模型架构实现**:
- **主模型类**: `models/ursa_model/modeling_ursa.py`
  - `UrsaForConditionalGeneration` (第78-387行): Stage I训练的生成模型类
  - 包含视觉编码器、投影器、语言模型的完整架构

**核心组件**:
- **视觉编码器**: `models/ursa_model/clip_encoder.py` (242行)
  - `HybridVisionTower`: 混合视觉塔实现，结合SAM-B和SigLIP-L

- **投影器**: `models/ursa_model/projector.py` (100行)
  - `MlpProjector`: 两层MLP投影器，连接视觉和语言模态

- **SigLIP视觉模型**: `models/ursa_model/siglip_vit.py` (681行)
  - SigLIP Vision Transformer实现

- **SAM组件**: `models/ursa_model/sam.py` (593行)
  - Segment Anything Model相关组件

**配置和处理**:
- **配置类**: `models/ursa_model/configuration_ursa.py` (143行)
  - `UrsaConfig`: 模型配置
  - `VisionConfig`: 视觉编码器配置
  - `AlignerConfig`: 投影器配置

- **处理器**: `models/ursa_model/processing_ursa.py` (60行)
  - `UrsaProcessor`: 多模态输入处理器

- **图像处理**: `models/ursa_model/image_processing_vlm.py` (208行)
  - `VLMImageProcessor`: 图像预处理

**训练数据**:
- 数据集位置: HuggingFace
  - URSA-Alignment-860K: 对齐数据
  - MMathCoT-1M: 指令微调数据

**训练产物**: URSA-8B模型

---

#### Stage II 相关代码

**模型架构实现**:
- **奖励模型类**: `models/ursa_model/modeling_ursa.py`
  - `UrsaForTokenClassification` (第389-705行): Stage II训练的PRM模型类
  - 基于URSA-8B架构，添加token分类头用于步骤级评分

**推理和评分**:
- **PRM推理脚本**: `inference/prm_infer_score.py` (8629字节)
  - `single_inference`: 单样本推理函数
  - `return_score`: 分数聚合函数（min/avg）
  - `replace_specific_plus_minus_with_ki`: 插入特殊token `и` 标记推理步骤
  - `prepare_input`: 准备PRM输入格式
  - 支持Best-of-N验证策略

- **PRM推理启动脚本**: `inference/start_score_infer.sh`
  - 配置PRM评分推理的参数

**关键机制**:
- 特殊token `и` 用于分隔推理步骤
- 每个步骤后的token位置输出logit，通过sigmoid转换为概率分数
- 支持min和avg两种分数聚合策略

**训练数据**:
- 数据集位置: HuggingFace
  - DualMath-1.1M: 过程监督数据
    - SBEL: ~773K样本（逻辑正确性）
    - SMIE: ~302K样本（感知一致性）

**训练产物**: URSA-RM-8B模型

---

#### Stage III 相关代码

**说明**:
本代码库**不包含Stage III的训练代码**，仅包含推理代码。Stage III (PS-GRPO)是在线强化学习阶段，训练代码未开源。

**相关推理代码**:
- **基础推理**: `inference/vllm_infer.py` (7677字节)
  - 使用vLLM加速推理
  - 支持多数据集格式（mathvista/dynamath/wemath/mathverse/mathvision）
  - `extract_answer_try_all_methods`: 答案提取函数

- **推理启动脚本**: `inference/start_vllm_infer.sh`
  - 配置推理参数（温度、数据集、生成数量等）

**vLLM集成**:
- **定制vLLM**: `vllm/` 目录
  - 特定commit版本: `0b8bb86bf19d68950b4d92a99350e07a26ec0d2c`
  - 支持URSA模型架构的推理加速

**环境配置**:
- **配置脚本**: `start.sh` (318字节)
  - 卸载现有vLLM
  - 安装特定版本vLLM
  - 配置flash attention

**训练数据**:
- MMathCoT-1M的15K子集（用于在线RL）

**训练产物**: URSA-8B-PS-GRPO模型（未开源训练代码）

---

### 代码库文件结构总结

```
URSA-MATH/
├── models/ursa_model/              # Stage I & II 模型实现
│   ├── modeling_ursa.py            # 主模型类 (705行)
│   │   ├── UrsaForConditionalGeneration (78-387行)   # Stage I 生成模型
│   │   └── UrsaForTokenClassification (389-705行)    # Stage II 奖励模型
│   ├── clip_encoder.py             # 混合视觉塔 (242行)
│   ├── projector.py                # MLP投影器 (100行)
│   ├── siglip_vit.py              # SigLIP视觉模型 (681行)
│   ├── sam.py                      # SAM组件 (593行)
│   ├── configuration_ursa.py       # 配置类 (143行)
│   ├── processing_ursa.py          # 处理器 (60行)
│   ├── image_processing_vlm.py     # 图像处理 (208行)
│   └── __init__.py                 # 模块导出 (29行)
│
├── inference/                      # 推理脚本
│   ├── vllm_infer.py              # vLLM推理 (Stage I & III产物)
│   ├── prm_infer_score.py         # PRM评分推理 (Stage II产物)
│   ├── start_vllm_infer.sh        # vLLM推理启动脚本
│   └── start_score_infer.sh       # PRM评分启动脚本
│
├── vllm/                           # 定制vLLM实现
│   └── [vLLM源码]                 # 支持URSA架构的推理加速
│
├── start.sh                        # 环境配置脚本
├── PAPER.md                        # 论文完整文本
├── CLAUDE.md                       # 项目指南（本文件）
└── README.md                       # 项目说明
```

**注意事项**:
1. **训练代码未开源**: 本repo仅包含模型架构和推理代码，三个阶段的训练脚本均未开源
2. **模型权重**: 训练产物（URSA-8B、URSA-RM-8B）可从HuggingFace下载
3. **数据集**: 训练数据集（MMathCoT-1M、DualMath-1.1M）可从HuggingFace获取
4. **推理优先**: 代码库重点在于展示如何使用训练好的模型进行推理和评估

## 架构设计

### 模型架构

URSA采用混合视觉塔架构：
- **视觉编码器**: 混合视觉塔(HybridVisionTower)，结合CLIP和SigLIP
  - 位置: `models/ursa_model/clip_encoder.py`
  - 支持多种视觉编码方案
- **投影器**: MLP投影器连接视觉和语言模态
  - 位置: `models/ursa_model/projector.py`
- **语言模型**: 基于Qwen-2.5-Math-Instruct
  - 主模型定义: `models/ursa_model/modeling_ursa.py`

### 关键模块

- `UrsaForConditionalGeneration`: 主生成模型类
- `UrsaForTokenClassification`: 用于奖励模型的token分类
- `UrsaProcessor`: 处理多模态输入的处理器
- `UrsaConfig`: 模型配置类

## 常用命令

### 环境配置

```bash
# 配置vLLM环境（必须先执行）
bash start.sh
```

该脚本会：
1. 卸载现有vLLM
2. 安装特定commit的vLLM版本
3. 配置vLLM flash attention
4. 运行python_only_dev.py进行开发环境设置

### 推理

#### 基础推理（使用vLLM）

```bash
# 编辑配置文件
vim inference/start_vllm_infer.sh

# 运行推理
bash inference/start_vllm_infer.sh
```

配置参数说明：
- `TEMPERATURE`: 采样温度（默认0.2）
- `DATASET`: 数据集名称（mathvista/dynamath/wemath/mathverse/mathvision）
- `IMAGE_ROOT`: 图像根目录路径
- `GENERATE_NUM`: 生成序列数量
- `MODEL_PATH`: 模型路径
- `DATA_PATH`: 数据文件路径
- `OUTPUT_FILE`: 输出文件路径

#### 直接使用Python脚本

```bash
CUDA_VISIBLE_DEVICES=0 python3 inference/vllm_infer.py \
  --model ./URSA-8B \
  --dataset mathvista \
  --temperature 0.2 \
  --data_path ./data/mathvista/mathvista_testmini.jsonl \
  --output_file ./output.jsonl \
  --image_root /path/to/images \
  --num_return_sequences 1
```

#### 奖励模型评分推理

```bash
python3 inference/prm_infer_score.py \
  --dataset_name mathverse \
  --data_path ./data/mathverse/testmini.json \
  --model_path ./URSA-RM-8B \
  --dtype bfloat16 \
  --output_path ./scores.pt \
  --image_root /path/to/images \
  --cuda_device 0 \
  --cuda_sum 1 \
  --best_of_n 32
```

奖励模型推理说明：
- 支持Best-of-N验证策略
- 计算min和avg两种聚合分数
- 使用特殊token `и` 标记推理步骤边界
- 输出包含不同采样数量(8/16/32/64)的结果

## 数据格式

### 推理输入格式

支持多种数据集格式：

**MathVista/MathVision/DynaMath格式** (JSONL):
```json
{
  "question": "问题文本",
  "image": "图像文件名",
  "answer": "答案"
}
```

**MathVerse格式** (JSON):
```json
{
  "query_cot": "CoT查询文本",
  "image": "图像路径",
  "answer": "答案"
}
```

**WE-MATH格式** (JSON):
```json
{
  "question": "问题文本",
  "option": "选项文本",
  "image_path": "图像路径",
  "answer": "答案"
}
```

### 推理输出格式

```json
{
  "question": "原始问题",
  "image": "图像路径",
  "model_answer": ["生成的答案1", "生成的答案2"],
  "extraction": ["提取的答案1", "提取的答案2"]
}
```

## 重要实现细节

### 答案提取逻辑

答案提取使用多种模式匹配策略（`vllm_infer.py:extract_answer_try_all_methods`）：
- 支持`\boxed{}`格式
- 支持`†Answer:`标记
- 支持多种自然语言答案格式
- 按优先级顺序尝试所有模式

### 过程奖励计算

奖励模型使用特殊标记`и`来分隔推理步骤：
1. 在每个推理步骤后插入`и`标记
2. 模型对每个标记位置输出logit
3. 通过sigmoid转换为概率分数
4. 支持min和avg两种聚合策略

### vLLM集成

项目包含定制的vLLM集成：
- 位置: `vllm/` 目录
- 使用特定commit版本: `0b8bb86bf19d68950b4d92a99350e07a26ec0d2c`
- 支持URSA模型架构的推理加速

## 评估基准

项目在6个多模态数学基准上进行评估：
- **MathVerse**: 多样化数学推理
- **MathVista-GPS**: 几何、物理、科学问题
- **WE-MATH**: 中文数学问题（使用strict metrics）
- **DynaMath**: 动态数学问题（报告average accuracy）
- **Math-Vision**: 数学视觉理解（使用full set）
- **GeoQA**: 几何问答

## 论文资源

项目根目录包含完整的论文资源：
- **PAPER.md**: 论文的Markdown文本版本，包含完整的论文内容（约1210行）
- **2501.04686v6.pdf**: 论文的PDF原文

这些文件包含了URSA框架的完整技术细节、实验设置、消融研究和理论分析。建议在深入开发前阅读这些资源以全面理解项目背景。

## 数据集

所有训练数据集可在HuggingFace获取：
- [URSA-MATH组织页面](https://huggingface.co/URSA-MATH)
- [MMathCoT-1M](https://huggingface.co/datasets/URSA-MATH/MMathCoT-1M)
- [DualMath-1.1M](https://huggingface.co/datasets/URSA-MATH/DualMath-1.1M)

## 模型检查点

- [URSA-8B](https://huggingface.co/URSA-MATH/URSA-8B): 基础生成模型
- [URSA-RM-8B](https://huggingface.co/URSA-MATH/URSA-RM-8B): 过程奖励模型

## 代码组织

```
URSA-MATH/
├── models/ursa_model/          # 模型实现
│   ├── modeling_ursa.py        # 主模型类
│   ├── configuration_ursa.py   # 配置类
│   ├── processing_ursa.py      # 处理器
│   ├── clip_encoder.py         # 视觉编码器
│   ├── projector.py            # 投影器
│   ├── siglip_vit.py          # SigLIP视觉transformer
│   └── sam.py                  # SAM相关组件
├── inference/                  # 推理脚本
│   ├── vllm_infer.py          # vLLM推理
│   ├── prm_infer_score.py     # 奖励模型评分
│   ├── start_vllm_infer.sh    # vLLM推理启动脚本
│   └── start_score_infer.sh   # 评分推理启动脚本
├── data/                       # 数据目录
│   ├── mathverse/
│   ├── mathvista/
│   ├── we-math/
│   └── ...
├── vllm/                       # 定制vLLM实现
└── start.sh                    # 环境配置脚本
```

## 注意事项

1. **vLLM版本**: 必须使用项目指定的vLLM commit版本，否则可能导致模型加载失败
2. **图像路径**: 确保`IMAGE_ROOT`正确配置，所有图像路径都是相对于该根目录
3. **GPU内存**: URSA-8B需要至少24GB GPU内存进行推理
4. **温度设置**: 对于数学推理任务，建议使用较低温度(0.2)以获得更确定性的输出
5. **Best-of-N采样**: 使用奖励模型时，建议生成32个候选答案以获得最佳性能
6. **数据格式**: 不同数据集使用不同的输入格式，确保使用正确的`--dataset`参数

## 性能指标

URSA-8B在各基准上的表现：
- MathVerse (testmini): 45.7% → 55.0% (with RM)
- MathVista-GPS: 79.3% → 86.4% (with RM)
- WE-MATH (testmini): 32.2%
- DynaMath (testmini): 44.7%
- Math-Vision (full): 26.2% → 35.2% (with RM)
- GeoQA (full): 73.5%

使用URSA-RM-8B进行Best-of-32验证可显著提升性能。
