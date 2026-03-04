# URSA Stage 3 复现计划 - 在LightRFT框架下

## 执行摘要

本文档详细规划如何在LightRFT框架下复现URSA论文的Stage 3 (PS-GRPO)实验。

**好消息**: LightRFT已经实现了URSA-8B-RM的完整支持！在`examples/math_prm/`目录下已有完整的PRM集成代码。

**核心发现**:
- LightRFT的`MathPRMReward`类已完全实现URSA-8B-RM的推理逻辑
- 已支持特殊token `и` 的插入和步骤级评分
- 已有完整的GRPO训练脚本和配置

---

## 一、现状分析

### 1.1 LightRFT已有的URSA支持

**已实现的关键组件**:

1. **MathPRMReward类** (`examples/math_prm/reward_models.py:2094`)
   - 完全复刻URSA-MATH的`prm_infer_score.py`逻辑
   - 支持`UrsaForTokenClassification`模型加载
   - 实现特殊token `и` (U+0438)的插入和识别
   - 支持min/avg/last三种分数聚合策略
   - 处理575个图像占位符token的对齐

2. **步骤标记插入** (`MathPRMReward.replace_specific_plus_minus_with_ki`)
   - 直接移植自URSA的`replace_specific_plus_minus_with_ki`函数
   - 在每个"Step N:"后插入` и`标记
   - 在"†Answer:"前插入` и`标记

3. **GRPO训练脚本** (`examples/math_prm/run_grpo_math_prm_qwen2.5_7b.sh`)
   - 完整的训练配置
   - 支持URSA-8B-RM作为奖励模型
   - 包含详细的使用说明

4. **训练主程序** (`examples/math_prm/train_colocate.py`)
   - 650行完整训练逻辑
   - 支持colocate模式（奖励模型与策略模型共享GPU）
   - 集成vLLM/SGLang推理引擎

### 1.2 URSA原始实现

**URSA-MATH代码库提供**:
- `models/ursa_model/modeling_ursa.py`: 模型架构定义
  - `UrsaForConditionalGeneration` (生成模型)
  - `UrsaForTokenClassification` (奖励模型)
- `inference/prm_infer_score.py`: PRM推理脚本
- `inference/vllm_infer.py`: vLLM推理脚本

**缺失的部分**:
- ❌ Stage 3训练代码（PS-GRPO实现）
- ❌ Drop-moment检测逻辑
- ❌ 在线RL训练循环

---

## 二、需要迁移的核心内容

### 2.1 从URSA迁移到LightRFT

#### ✅ 已完成（无需迁移）

1. **URSA-8B-RM模型加载** - LightRFT已支持
2. **特殊token处理** - LightRFT已实现
3. **步骤级评分** - LightRFT已实现
4. **基础GRPO训练** - LightRFT已支持

#### 🔧 需要实现的部分

**核心任务：实现PS-GRPO算法**

LightRFT目前只支持vanilla GRPO，需要添加PS-GRPO的核心创新：

1. **Drop-moment检测**
   - 检测PRM奖励序列的显著下降点
   - 公式：`δ_i = max{(r_p,j - r_p,j+1) / r_p,j} > ρ`
   - 阈值：ρ = 0.3

2. **过程监督奖励计算**
   - 三级奖励系统：
     - 答案正确 + 无drop-moment → R = 1.0
     - 答案正确 + 有drop-moment → R = 0.5 (γ=0.5)
     - 答案错误 → R = 0.0

3. **集成到GRPO训练循环**
   - 在经验生成阶段调用PRM评分
   - 检测drop-moment
   - 调整奖励值
   - 传递给GRPO优化器

### 2.2 需要从URSA复制的文件

**模型定义** (可选，如果需要自定义):
```
URSA-MATH/models/ursa_model/
├── modeling_ursa.py          → 已有HuggingFace模型，可直接加载
├── configuration_ursa.py     → 已有HuggingFace配置
├── processing_ursa.py        → 已有HuggingFace处理器
└── (其他视觉编码器文件)      → 已包含在模型checkpoint中
```

**推理逻辑** (已在LightRFT中实现):
```
URSA-MATH/inference/
├── prm_infer_score.py        → 已移植到MathPRMReward
└── vllm_infer.py             → LightRFT已有vLLM集成
```

---

## 三、详细实施步骤

### Step 1: 环境准备

```bash
cd /home/hansbug/sensetime-projects/LightRFT

# 1. 确认LightRFT环境已安装
pip install -e .

# 2. 下载URSA模型（如果还没有）
# URSA-8B (策略模型)
huggingface-cli download URSA-MATH/URSA-8B --local-dir ./models/URSA-8B

# URSA-RM-8B (奖励模型)
huggingface-cli download URSA-MATH/URSA-RM-8B --local-dir ./models/URSA-RM-8B

# 3. 准备数据集
# 从MMathCoT-1M中选择15K样本用于RL训练
```

### Step 2: 实现PS-GRPO核心逻辑

**创建新文件**: `lightrft/trainer/ps_grpo_utils.py`

```python
"""
PS-GRPO (Process-Supervised GRPO) Utilities

Implements the drop-moment detection and process-supervised reward computation
from URSA paper (arXiv:2501.04686).
"""

import torch
from typing import List, Tuple

def detect_drop_moment(
    step_scores: torch.Tensor,
    threshold: float = 0.3
) -> Tuple[bool, float]:
    """
    Detect drop-moment in PRM reward sequence.

    Drop-moment: significant decrease in reward between consecutive steps,
    indicating PRM questions the validity of preceding steps.

    Args:
        step_scores: Tensor of shape (n_steps,) with per-step PRM scores
        threshold: Drop-moment threshold ρ (default: 0.3)

    Returns:
        (has_drop_moment, max_relative_drop)

    Formula:
        δ_i = max{(r_p,j - r_p,j+1) / r_p,j | j = 0,1,...,N-1} > ρ
    """
    if len(step_scores) < 2:
        return False, 0.0

    # Compute relative drops between consecutive steps
    relative_drops = []
    for j in range(len(step_scores) - 1):
        if step_scores[j] > 1e-6:  # Avoid division by zero
            relative_drop = (step_scores[j] - step_scores[j+1]) / step_scores[j]
            relative_drops.append(relative_drop.item())

    if not relative_drops:
        return False, 0.0

    max_drop = max(relative_drops)
    has_drop_moment = max_drop > threshold

    return has_drop_moment, max_drop


def compute_ps_grpo_reward(
    outcome_correct: bool,
    has_drop_moment: bool,
    gamma: float = 0.5
) -> float:
    """
    Compute process-supervised GRPO reward.

    Three-level reward system:
        - Correct answer + no drop-moment → R = 1.0
        - Correct answer + drop-moment → R = 1 - γ (default: 0.5)
        - Incorrect answer → R = 0.0

    Args:
        outcome_correct: Whether final answer is correct
        has_drop_moment: Whether drop-moment detected in PRM sequence
        gamma: Penalty coefficient (default: 0.5)

    Returns:
        Final reward value
    """
    if not outcome_correct:
        return 0.0

    if has_drop_moment:
        return 1.0 - gamma  # Penalize questioned rollouts

    return 1.0


def process_prm_scores_for_ps_grpo(
    prm_scores: List[torch.Tensor],
    outcome_rewards: torch.Tensor,
    rho: float = 0.3,
    gamma: float = 0.5
) -> torch.Tensor:
    """
    Process PRM scores and apply PS-GRPO reward adjustment.

    Args:
        prm_scores: List of per-step PRM score tensors, one per rollout
        outcome_rewards: Tensor of shape (batch_size,) with outcome rewards
        rho: Drop-moment threshold (default: 0.3)
        gamma: Penalty coefficient (default: 0.5)

    Returns:
        Adjusted rewards tensor of shape (batch_size,)
    """
    adjusted_rewards = []

    for i, (step_scores, outcome_reward) in enumerate(zip(prm_scores, outcome_rewards)):
        outcome_correct = outcome_reward > 0.5  # Assume binary outcome reward

        # Detect drop-moment
        has_drop_moment, max_drop = detect_drop_moment(step_scores, threshold=rho)

        # Compute PS-GRPO reward
        final_reward = compute_ps_grpo_reward(
            outcome_correct=outcome_correct,
            has_drop_moment=has_drop_moment,
            gamma=gamma
        )

        adjusted_rewards.append(final_reward)

    return torch.tensor(adjusted_rewards, dtype=outcome_rewards.dtype, device=outcome_rewards.device)
```

### Step 3: 修改奖励模型集成

**修改文件**: `examples/math_prm/reward_models.py`

在`MathPRMReward`类中添加方法：

```python
def get_step_scores(
    self,
    prompts: List[str],
    responses: List[str],
    image_data: List[List] | None = None,
) -> List[torch.Tensor]:
    """
    Get per-step PRM scores for PS-GRPO.

    Returns:
        List of tensors, each containing per-step scores for one response
    """
    # ... (使用现有的forward逻辑，但返回完整的step_scores而非聚合值)
    pass
```

### Step 4: 修改训练脚本

**修改文件**: `examples/math_prm/train_colocate.py`

在经验生成后、优势计算前添加PS-GRPO逻辑：

```python
# 在make_experience之后
if args.use_ps_grpo:
    from lightrft.trainer.ps_grpo_utils import process_prm_scores_for_ps_grpo

    # 获取PRM的步骤级分数
    prm_step_scores = reward_model.get_step_scores(
        prompts=experience.prompts,
        responses=experience.sequences,
        image_data=experience.images
    )

    # 应用PS-GRPO奖励调整
    experience.rewards = process_prm_scores_for_ps_grpo(
        prm_scores=prm_step_scores,
        outcome_rewards=experience.rewards,
        rho=args.ps_grpo_rho,
        gamma=args.ps_grpo_gamma
    )
```

### Step 5: 创建PS-GRPO训练脚本

**创建文件**: `examples/math_prm/run_ps_grpo_ursa_8b.sh`

```bash
#!/bin/bash
#
# PS-GRPO Training Script for URSA-8B
# Implements Process-Supervised GRPO from URSA paper
#

# Model paths
ACTOR_MODEL="./models/URSA-8B"
REWARD_MODEL="./models/URSA-RM-8B"

# Dataset (15K subset from MMathCoT-1M)
DATASET="./data/mmathcot_15k_rl.jsonl"

# PS-GRPO parameters
PS_GRPO_RHO=0.3      # Drop-moment threshold
PS_GRPO_GAMMA=0.5    # Penalty coefficient

# GRPO parameters
N_SAMPLES=8
EPISODE=10
KL=0.001
LR=1e-6

# Batch sizes
RBS=128
TBS=128

# Launch training
torchrun \
    --nnodes 1 \
    --nproc-per-node 8 \
    examples/math_prm/train_colocate.py \
    --pretrain "${ACTOR_MODEL}" \
    --reward_pretrain "{\"math_prm\":\"${REWARD_MODEL}\"}" \
    --prompt_data "${DATASET}" \
    --use_ps_grpo \
    --ps_grpo_rho ${PS_GRPO_RHO} \
    --ps_grpo_gamma ${PS_GRPO_GAMMA} \
    --n_samples_per_prompt ${N_SAMPLES} \
    --num_episodes ${EPISODE} \
    --init_kl_coef ${KL} \
    --actor_learning_rate ${LR} \
    --rollout_batch_size ${RBS} \
    --train_batch_size ${TBS} \
    --advantage_estimator "group_norm" \
    --fsdp \
    --bf16 \
    --flash_attn \
    --gradient_checkpointing \
    --system_prompt 'A conversation between the User and Assistant. The User asks a math question, and the Assistant solves it step by step. Each step MUST begin with "Step N:" (e.g. "Step 1:", "Step 2:") on its own line. After all steps, the final answer MUST be on its own line prefixed with "†Answer:" (e.g. "†Answer: 42").' \
    2>&1 | tee "logs/ps_grpo_ursa_8b_$(date +%Y%m%d_%H%M%S).log"
```

### Step 6: 数据准备

**创建脚本**: `scripts/prepare_mmathcot_15k.py`

```python
"""
Prepare 15K subset from MMathCoT-1M for PS-GRPO training.

Format required:
{
    "prompt": "math question",
    "label": "math_prm",  # Use PRM reward
    "reference": "ground truth answer" (optional)
}
"""

import json
import random
from pathlib import Path

def prepare_dataset(
    input_path: str,
    output_path: str,
    num_samples: int = 15000,
    seed: int = 42
):
    """Sample 15K examples from MMathCoT-1M."""
    random.seed(seed)

    # Load full dataset
    with open(input_path, 'r') as f:
        data = [json.loads(line) for line in f]

    # Sample 15K
    sampled = random.sample(data, min(num_samples, len(data)))

    # Convert to LightRFT format
    output_data = []
    for item in sampled:
        output_data.append({
            "prompt": item["question"],
            "label": "math_prm",  # Use PRM reward
            "reference": item.get("answer", "")
        })

    # Save
    with open(output_path, 'w') as f:
        for item in output_data:
            f.write(json.dumps(item) + '\n')

    print(f"Saved {len(output_data)} samples to {output_path}")

if __name__ == "__main__":
    prepare_dataset(
        input_path="./data/MMathCoT-1M/train.jsonl",
        output_path="./data/mmathcot_15k_rl.jsonl",
        num_samples=15000
    )
```

---

## 四、关键技术对比

### 4.1 URSA原始实现 vs LightRFT实现

| 组件 | URSA原始 | LightRFT | 状态 |
|------|---------|----------|------|
| **模型加载** | `UrsaForTokenClassification.from_pretrained()` | `MathPRMReward(base_model, processor)` | ✅ 已实现 |
| **特殊token** | ` и` (U+0438) | ` и` (U+0438) | ✅ 一致 |
| **步骤标记** | `replace_specific_plus_minus_with_ki` | `MathPRMReward.replace_specific_plus_minus_with_ki` | ✅ 已移植 |
| **图像占位符** | 575个dummy tokens | 575个dummy tokens | ✅ 一致 |
| **分数聚合** | min/avg | min/avg/last | ✅ 已实现 |
| **Drop-moment** | 论文描述 | ❌ 需实现 | 🔧 待添加 |
| **PS-GRPO** | 论文描述 | ❌ 需实现 | 🔧 待添加 |

### 4.2 训练流程对比

**URSA论文描述的Stage 3**:
```
1. 策略模型生成N个候选答案
2. PRM对每个答案进行步骤级评分
3. 检测drop-moment
4. 计算PS-GRPO奖励
5. GRPO优化策略模型
```

**LightRFT当前流程**:
```
1. 策略模型生成N个候选答案
2. PRM对每个答案进行步骤级评分
3. 使用min/avg聚合分数作为奖励  ← 这里需要改为PS-GRPO
4. GRPO优化策略模型
```

**需要修改的部分**:
- 在步骤3添加drop-moment检测
- 将聚合分数替换为PS-GRPO三级奖励

---

## 五、实施优先级

### 🔴 P0 - 核心功能（必须实现）

1. **实现drop-moment检测** (`ps_grpo_utils.py`)
   - 时间估计：关键算法实现
   - 依赖：无
   - 输出：`detect_drop_moment()`函数

2. **实现PS-GRPO奖励计算** (`ps_grpo_utils.py`)
   - 时间估计：核心逻辑实现
   - 依赖：drop-moment检测
   - 输出：`compute_ps_grpo_reward()`函数

3. **集成到训练循环** (`train_colocate.py`)
   - 时间估计：集成工作
   - 依赖：PS-GRPO工具函数
   - 输出：修改后的训练脚本

### 🟡 P1 - 数据和配置

4. **准备15K训练数据**
   - 时间估计：数据处理
   - 依赖：MMathCoT-1M数据集
   - 输出：`mmathcot_15k_rl.jsonl`

5. **创建训练脚本**
   - 时间估计：配置编写
   - 依赖：无
   - 输出：`run_ps_grpo_ursa_8b.sh`

### 🟢 P2 - 验证和优化

6. **单元测试**
   - 测试drop-moment检测准确性
   - 测试奖励计算正确性

7. **小规模训练验证**
   - 100样本快速验证
   - 检查训练稳定性

8. **完整训练**
   - 15K样本完整训练
   - 10个episodes

---

## 六、预期结果

### 6.1 性能指标

根据URSA论文，PS-GRPO相比vanilla GRPO应该有以下提升：

| 指标 | Vanilla GRPO | PS-GRPO | 提升 |
|------|--------------|---------|------|
| **平均性能** | +3.1% | +6.8% | **2.2x** |
| **WE-MATH** | +4.9% | +11.4% | **2.3x** |
| **MathVision** | +1.8% | +9.8% | **5.4x** |
| **MathVerse** | - | +11.4% | - |

### 6.2 验证方法

1. **定量验证**:
   - 在6个基准上评估（MathVerse, MathVision, MathVista, WE-MATH, DynaMath, GeoQA）
   - 对比vanilla GRPO和PS-GRPO的性能提升
   - 验证是否达到论文报告的提升幅度

2. **定性验证**:
   - 检查drop-moment检测是否合理
   - 分析被惩罚的rollouts是否确实存在推理问题
   - 观察训练过程中的响应长度变化

---

## 七、潜在问题和解决方案

### 问题1: Drop-moment阈值敏感性

**问题**: ρ=0.3可能不是最优值

**解决方案**:
- 实现阈值扫描实验（ρ ∈ {0.2, 0.25, 0.3, 0.35, 0.4}）
- 在验证集上选择最优阈值

### 问题2: 奖励稀疏性

**问题**: 三级奖励可能导致梯度稀疏

**解决方案**:
- 监控训练过程中的奖励分布
- 如果过于稀疏，考虑使用软化版本（连续奖励）

### 问题3: 计算开销

**问题**: PRM推理增加计算成本

**解决方案**:
- 使用LightRFT的colocate模式共享GPU
- 启用engine sleep机制节省内存
- 考虑使用FP8推理加速

### 问题4: 数据格式不匹配

**问题**: MMathCoT-1M格式可能与LightRFT不完全兼容

**解决方案**:
- 编写数据转换脚本
- 确保"Step N:"格式严格遵守
- 验证特殊token插入正确性

---

## 八、时间线和里程碑

### 阶段1: 核心实现
- [ ] 实现`ps_grpo_utils.py`
- [ ] 修改`MathPRMReward`添加`get_step_scores()`
- [ ] 修改`train_colocate.py`集成PS-GRPO

### 阶段2: 数据准备
- [ ] 下载URSA-8B和URSA-RM-8B模型
- [ ] 准备15K训练数据
- [ ] 创建训练脚本

### 阶段3: 验证
- [ ] 单元测试
- [ ] 小规模训练（100样本）
- [ ] 验证drop-moment检测

### 阶段4: 完整训练
- [ ] 15K样本训练（10 episodes）
- [ ] 6个基准评估
- [ ] 结果分析和对比

---

## 九、参考资源

### 代码参考

**URSA-MATH**:
- 模型定义: `models/ursa_model/modeling_ursa.py`
- PRM推理: `inference/prm_infer_score.py`
- 文档: `CLAUDE.md`

**LightRFT**:
- PRM实现: `examples/math_prm/reward_models.py`
- 训练脚本: `examples/math_prm/train_colocate.py`
- 配置示例: `examples/math_prm/run_grpo_math_prm_qwen2.5_7b.sh`

### 论文参考

- URSA论文: arXiv:2501.04686
- GRPO论文: arXiv:2402.03300
- DeepSeek-R1: arXiv:2501.12948

---

## 十、总结

### 关键优势

1. **LightRFT已有完整的PRM支持** - 大部分基础设施已就绪
2. **清晰的实现路径** - 只需添加PS-GRPO核心逻辑
3. **完善的训练框架** - FSDP、DeepSpeed、vLLM集成完备

### 核心工作量

- **新增代码**: ~300行（ps_grpo_utils.py + 集成代码）
- **修改代码**: ~100行（train_colocate.py + reward_models.py）
- **配置脚本**: ~200行（训练脚本 + 数据准备）

### 成功标准

1. ✅ Drop-moment检测准确率 > 90%
2. ✅ PS-GRPO相比vanilla GRPO提升 > 2x
3. ✅ 在至少3个基准上达到论文报告的性能
4. ✅ 训练稳定，无奖励欺骗或长度偏差

---

## 附录A: 快速开始命令

```bash
# 1. 克隆并安装LightRFT
cd /home/hansbug/sensetime-projects/LightRFT
pip install -e .

# 2. 下载模型
huggingface-cli download URSA-MATH/URSA-8B --local-dir ./models/URSA-8B
huggingface-cli download URSA-MATH/URSA-RM-8B --local-dir ./models/URSA-RM-8B

# 3. 准备数据
python scripts/prepare_mmathcot_15k.py

# 4. 实现PS-GRPO（按照Step 2-4）

# 5. 启动训练
bash examples/math_prm/run_ps_grpo_ursa_8b.sh
```

---

**文档版本**: v1.0
**创建日期**: 2026-03-04
**作者**: Claude Sonnet 4.6
**基于**: URSA论文 (arXiv:2501.04686) + LightRFT v0.1.1
