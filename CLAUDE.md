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

### 三阶段训练框架

1. **Stage I**: 视觉-语言对齐和数学指令微调
   - 使用URSA-Alignment-860K进行视觉-语言对齐
   - 使用MMathCoT-1M进行指令微调

2. **Stage II**: 过程奖励模型训练
   - 基于URSA-8B继续训练
   - 使用DualMath-1.1M数据集，包含逻辑正确性和视觉一致性的双视角监督

3. **Stage III**: PS-GRPO在线强化学习
   - Process-Supervised Group-Relative-Policy-Optimization
   - 缓解奖励欺骗和长度偏差问题

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
