# URSA-MATH 运行指南

本文档给出一套从零开始的中文流程，包含：

1. 克隆仓库
2. 创建并激活 conda 环境
3. 安装依赖
4. 下载 URSA-8B 和 URSA-RM-8B 模型到 `checkpoints/`
5. 运行两个依赖仓库本地代码的示例脚本
6. 运行两个不依赖仓库原始 `models/` / `inference/` 代码的 standalone 示例脚本

本文默认你在 Linux 机器上操作，并且机器已经安装好 NVIDIA 驱动、CUDA 对应运行环境，以及 `conda`。

## 1. 克隆仓库

```bash
git clone https://github.com/HansBug/URSA-MATH.git
cd URSA-MATH
```

## 2. 检查 GPU

先确认机器 GPU 正常。

```bash
nvidia-smi
```

如果你和当前验证环境一致，应该能看到 8 张 `NVIDIA A100-SXM4-80GB`。

也可以用更简洁的命令查看：

```bash
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
```

## 3. 创建 conda 环境

如果你还没有 `ursa` 环境，执行：

```bash
conda create -n ursa python=3.12 -y
conda activate ursa
```

如果你已经有 `ursa` 环境，只需要：

```bash
conda activate ursa
```

## 4. 安装依赖

仓库根目录已经提供了经过实际验证的 [requirements.txt](/home/ubuntu/URSA-MATH/requirements.txt#L1)。

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

说明：

- 这份 `requirements.txt` 已验证可支持：
  - `URSA-8B` 作为完整 torch 模型加载到 GPU 并生成
  - `URSA-RM-8B` 加载到 GPU 并做打分
- 其中已经包含 `torch==2.5.1+cu124`、`torchvision==0.20.1+cu124`、`transformers==4.45.2` 等关键版本

## 5. 可选：初始化仓库自带 vLLM

如果你后续还想跑仓库里的 `inference/vllm_infer.py`，建议额外执行一次：

```bash
bash start.sh
```

说明：

- 本文后面两个示例脚本不依赖 vLLM
- 这一步主要是给仓库内嵌的 `vllm/` 做本地源码链接和扩展库复制

## 6. 下载模型到 checkpoints

模型必须下载到仓库根目录下的 `checkpoints/`，当前仓库已经把这个目录加入了 git ignore。

先创建目录：

```bash
mkdir -p checkpoints
```

然后下载两个模型：

```bash
huggingface-cli download URSA-MATH/URSA-8B --local-dir ./checkpoints/URSA-8B
huggingface-cli download URSA-MATH/URSA-RM-8B --local-dir ./checkpoints/URSA-RM-8B
```

下载完成后，目录结构应该类似：

```text
checkpoints/
├── URSA-8B/
│   ├── config.json
│   ├── model-00001-of-00007.safetensors
│   ├── ...
│   └── tokenizer_config.json
└── URSA-RM-8B/
    ├── config.json
    ├── model-00001-of-00007.safetensors
    ├── ...
    └── tokenizer_config.json
```

## 7. 示例一：URSA-8B 作为完整 torch 模型加载到 GPU，并做一次短生成

示例脚本：

- [examples/run_ursa_8b_torch_example.py](/home/ubuntu/URSA-MATH/examples/run_ursa_8b_torch_example.py)

这个脚本做的事情：

- 使用本仓库本地的 `UrsaForConditionalGeneration`
- 从 `checkpoints/URSA-8B` 加载完整权重
- 把模型放到 GPU
- 用 `figures/framework.png` 跑一次短生成

### 运行命令

如果你想用第 0 张卡：

```bash
python examples/run_ursa_8b_torch_example.py --device cuda:0
```

如果你的机器上某些卡已经被占用，建议先显式挑一张空闲卡。例如只暴露物理 2 号卡：

```bash
CUDA_VISIBLE_DEVICES=2 python examples/run_ursa_8b_torch_example.py --device cuda:0
```

### 默认输入

- 模型路径：`checkpoints/URSA-8B`
- 图片路径：`figures/framework.png`
- 问题：

```text
How many training stages are shown in this diagram? Answer briefly.
```

### 已验证成功的参考输出

我在当前机器上实际跑通时，输出类似：

```json
{
  "model_class": "UrsaForConditionalGeneration",
  "device": "cuda:0",
  "dtype": "torch.bfloat16",
  "load_seconds": 18.63,
  "peak_mem_gb": 16.24,
  "generated_text": "Step 1: The diagram shows four distinct training stages.\n\nStep 2: These stages are: Pre- and Post-Exposure, Prophylactic, Therapeutic, and Research Training.\n\n†Answer: 4"
}
```

### 自定义参数

例如换图片、换问题、限制生成长度：

```bash
python examples/run_ursa_8b_torch_example.py \
  --device cuda:0 \
  --model-path ./checkpoints/URSA-8B \
  --image-path ./figures/framework.png \
  --question "How many training stages are shown in this diagram? Answer briefly." \
  --max-new-tokens 48
```

## 8. 示例二：URSA-RM-8B 加载到 GPU，并逐 step 输出 PRM 分数

示例脚本：

- [examples/run_ursa_rm_8b_score_example.py](/home/ubuntu/URSA-MATH/examples/run_ursa_rm_8b_score_example.py)

这个脚本做的事情：

- 使用本仓库本地的 `UrsaForTokenClassification`
- 从 `checkpoints/URSA-RM-8B` 加载完整权重
- 把模型放到 GPU
- 复用 [inference/prm_infer_score.py](/home/ubuntu/URSA-MATH/inference/prm_infer_score.py) 里的 `prepare_input` 和 `return_score`
- 对输入 response 里的每个 `Step N:` 逐个打分
- 只输出 PRM 相关字段，不再输出 BoN / GRPO / drop-moment / paper metrics 相关内容

### 运行命令

如果你想用第 0 张卡：

```bash
python examples/run_ursa_rm_8b_score_example.py --device cuda:0
```

如果你要指定一张空闲卡，例如物理 2 号卡：

```bash
CUDA_VISIBLE_DEVICES=2 python examples/run_ursa_rm_8b_score_example.py --device cuda:0
```

### 默认输入

- 模型路径：`checkpoints/URSA-RM-8B`
- 图片路径：`figures/framework.png`
- 问题：

```text
How many numbered training stages are shown in this diagram?
```

- 默认 response：

```text
Step 1: The figure explicitly labels Stage 1 as VL Alignment.
Step 2: The top-right panel is labeled Stage 2 Math SFT.
Step 3: The bottom-left panel is labeled Stage 3 PRM Training and Verifying.
†Answer: 3
```

- 默认 `ground_truth_answer=3`

### 输出字段说明

- `prepared_input`
  - 这是送进 RM 的最终文本，会保留 `prepare_input(...)` 插入的 `и` 分隔 token
- `steps[].prm_score`
  - 每个 step 的逐步 PRM 分数
- `min_prm_score`
  - 当前 response 的最小 step 分数
- `avg_prm_score`
  - 当前 response 的平均 step 分数
- `step_reward_alignment_ok`
  - 用来检查脚本提取到的 step 数量和实际打分数量是否一致

说明：

- 顶层只保留以下元信息：
  - `model_class`
  - `device`
  - `dtype`
  - `load_seconds`
  - `peak_mem_gb`
  - `image_path`
  - `question`
  - `ground_truth_answer`
- 其余只保留 PRM 逐步打分结果

### 4 个可直接运行的 PRM 案例

下面 4 组命令都已经在当前机器上实际跑通。为了让文档聚焦 PRM 本身，下面的“运行输出结果”保留脚本最终打印的 JSON，省略了 `transformers` warning 和 checkpoint shard 进度条。`load_seconds` 会随着机器负载略有波动。

如果你的机器只有 1 张可用 GPU，把下面命令里的 `CUDA_VISIBLE_DEVICES=0/1/2/3` 统一替换成同一张空闲卡即可。

### 8.1 每一步完全正确

运行命令：

```bash
CUDA_VISIBLE_DEVICES=0 python examples/run_ursa_rm_8b_score_example.py \
  --device cuda:0 \
  --response "Step 1: The figure explicitly labels Stage 1 as VL Alignment. Step 2: It explicitly labels Stage 2 as Math SFT. Step 3: It explicitly labels Stage 3 as PRM Training and Verifying. Step 4: Only three numbered stages appear in the figure. †Answer: 3"
```

运行输出结果：

```json
{
  "model_class": "UrsaForTokenClassification",
  "device": "cuda:0",
  "dtype": "torch.bfloat16",
  "load_seconds": 24.5,
  "peak_mem_gb": 16.24,
  "image_path": "/home/ubuntu/URSA-MATH/figures/framework.png",
  "question": "How many numbered training stages are shown in this diagram?",
  "ground_truth_answer": "3",
  "responses": [
    {
      "response_id": 0,
      "response": "Step 1: The figure explicitly labels Stage 1 as VL Alignment. Step 2: It explicitly labels Stage 2 as Math SFT. Step 3: It explicitly labels Stage 3 as PRM Training and Verifying. Step 4: Only three numbered stages appear in the figure. †Answer: 3",
      "prepared_input": "You are given a problem and a step-by-step solution. You need to check the correctness of each step.\nQuestion:How many numbered training stages are shown in this diagram?\nStep 1: The figure explicitly labels Stage 1 as VL Alignment. и Step 2: It explicitly labels Stage 2 as Math SFT. и Step 3: It explicitly labels Stage 3 as PRM Training and Verifying. и Step 4: Only three numbered stages appear in the figure. и †Answer: 3",
      "step_count_in_text": 4,
      "scored_step_count": 4,
      "step_reward_alignment_ok": true,
      "steps": [
        {
          "step_index": 1,
          "step_text": "Step 1: The figure explicitly labels Stage 1 as VL Alignment.",
          "prm_score": 0.984375
        },
        {
          "step_index": 2,
          "step_text": "Step 2: It explicitly labels Stage 2 as Math SFT.",
          "prm_score": 0.976562
        },
        {
          "step_index": 3,
          "step_text": "Step 3: It explicitly labels Stage 3 as PRM Training and Verifying.",
          "prm_score": 0.902344
        },
        {
          "step_index": 4,
          "step_text": "Step 4: Only three numbered stages appear in the figure.",
          "prm_score": 0.671875
        }
      ],
      "min_prm_score": 0.671875,
      "avg_prm_score": 0.883789
    }
  ]
}
```

### 8.2 前面几步正确，后半段错误

运行命令：

```bash
CUDA_VISIBLE_DEVICES=1 python examples/run_ursa_rm_8b_score_example.py \
  --device cuda:0 \
  --response "Step 1: The figure explicitly labels Stage 1 as VL Alignment. Step 2: It also labels Stage 2 as Math SFT. Step 3: After that, I will incorrectly assume the lower-left block is not a stage and that there is a hidden Stage 4 elsewhere. Step 4: Based on that wrong reading, I conclude there are four numbered stages. †Answer: 4"
```

运行输出结果：

```json
{
  "model_class": "UrsaForTokenClassification",
  "device": "cuda:0",
  "dtype": "torch.bfloat16",
  "load_seconds": 25.82,
  "peak_mem_gb": 16.24,
  "image_path": "/home/ubuntu/URSA-MATH/figures/framework.png",
  "question": "How many numbered training stages are shown in this diagram?",
  "ground_truth_answer": "3",
  "responses": [
    {
      "response_id": 0,
      "response": "Step 1: The figure explicitly labels Stage 1 as VL Alignment. Step 2: It also labels Stage 2 as Math SFT. Step 3: After that, I will incorrectly assume the lower-left block is not a stage and that there is a hidden Stage 4 elsewhere. Step 4: Based on that wrong reading, I conclude there are four numbered stages. †Answer: 4",
      "prepared_input": "You are given a problem and a step-by-step solution. You need to check the correctness of each step.\nQuestion:How many numbered training stages are shown in this diagram?\nStep 1: The figure explicitly labels Stage 1 as VL Alignment. и Step 2: It also labels Stage 2 as Math SFT. и Step 3: After that, I will incorrectly assume the lower-left block is not a stage and that there is a hidden Stage 4 elsewhere. и Step 4: Based on that wrong reading, I conclude there are four numbered stages. и †Answer: 4",
      "step_count_in_text": 4,
      "scored_step_count": 4,
      "step_reward_alignment_ok": true,
      "steps": [
        {
          "step_index": 1,
          "step_text": "Step 1: The figure explicitly labels Stage 1 as VL Alignment.",
          "prm_score": 0.984375
        },
        {
          "step_index": 2,
          "step_text": "Step 2: It also labels Stage 2 as Math SFT.",
          "prm_score": 0.96875
        },
        {
          "step_index": 3,
          "step_text": "Step 3: After that, I will incorrectly assume the lower-left block is not a stage and that there is a hidden Stage 4 elsewhere.",
          "prm_score": 0.863281
        },
        {
          "step_index": 4,
          "step_text": "Step 4: Based on that wrong reading, I conclude there are four numbered stages.",
          "prm_score": 0.316406
        }
      ],
      "min_prm_score": 0.316406,
      "avg_prm_score": 0.783203
    }
  ]
}
```

### 8.3 前面几步正确，中间几步明显错误，但最终结果偏偏正确

运行命令：

```bash
CUDA_VISIBLE_DEVICES=2 python examples/run_ursa_rm_8b_score_example.py \
  --device cuda:0 \
  --response "Step 1: The figure explicitly labels Stage 1 as VL Alignment. Step 2: It also labels Stage 2 as Math SFT. Step 3: I now make a wrong detour and claim the lower-left Stage 3 block should be ignored because it is only evaluation. Step 4: I also wrongly claim the right-bottom inference-time scaling panel is Stage 4. Step 5: Rechecking the labels, only Stage 1, Stage 2, and Stage 3 are numbered training stages. †Answer: 3"
```

运行输出结果：

```json
{
  "model_class": "UrsaForTokenClassification",
  "device": "cuda:0",
  "dtype": "torch.bfloat16",
  "load_seconds": 24.87,
  "peak_mem_gb": 16.24,
  "image_path": "/home/ubuntu/URSA-MATH/figures/framework.png",
  "question": "How many numbered training stages are shown in this diagram?",
  "ground_truth_answer": "3",
  "responses": [
    {
      "response_id": 0,
      "response": "Step 1: The figure explicitly labels Stage 1 as VL Alignment. Step 2: It also labels Stage 2 as Math SFT. Step 3: I now make a wrong detour and claim the lower-left Stage 3 block should be ignored because it is only evaluation. Step 4: I also wrongly claim the right-bottom inference-time scaling panel is Stage 4. Step 5: Rechecking the labels, only Stage 1, Stage 2, and Stage 3 are numbered training stages. †Answer: 3",
      "prepared_input": "You are given a problem and a step-by-step solution. You need to check the correctness of each step.\nQuestion:How many numbered training stages are shown in this diagram?\nStep 1: The figure explicitly labels Stage 1 as VL Alignment. и Step 2: It also labels Stage 2 as Math SFT. и Step 3: I now make a wrong detour and claim the lower-left Stage 3 block should be ignored because it is only evaluation. и Step 4: I also wrongly claim the right-bottom inference-time scaling panel is Stage 4. и Step 5: Rechecking the labels, only Stage 1, Stage 2, and Stage 3 are numbered training stages. и †Answer: 3",
      "step_count_in_text": 5,
      "scored_step_count": 5,
      "step_reward_alignment_ok": true,
      "steps": [
        {
          "step_index": 1,
          "step_text": "Step 1: The figure explicitly labels Stage 1 as VL Alignment.",
          "prm_score": 0.984375
        },
        {
          "step_index": 2,
          "step_text": "Step 2: It also labels Stage 2 as Math SFT.",
          "prm_score": 0.96875
        },
        {
          "step_index": 3,
          "step_text": "Step 3: I now make a wrong detour and claim the lower-left Stage 3 block should be ignored because it is only evaluation.",
          "prm_score": 0.90625
        },
        {
          "step_index": 4,
          "step_text": "Step 4: I also wrongly claim the right-bottom inference-time scaling panel is Stage 4.",
          "prm_score": 0.773438
        },
        {
          "step_index": 5,
          "step_text": "Step 5: Rechecking the labels, only Stage 1, Stage 2, and Stage 3 are numbered training stages.",
          "prm_score": 0.269531
        }
      ],
      "min_prm_score": 0.269531,
      "avg_prm_score": 0.780469
    }
  ]
}
```

### 8.4 完全错误

运行命令：

```bash
CUDA_VISIBLE_DEVICES=3 python examples/run_ursa_rm_8b_score_example.py \
  --device cuda:0 \
  --response "Step 1: The diagram has nothing to do with numbered stages. Step 2: I will say it shows five numbered training stages without using the labels. Step 3: Therefore the answer is 5. †Answer: 5"
```

运行输出结果：

```json
{
  "model_class": "UrsaForTokenClassification",
  "device": "cuda:0",
  "dtype": "torch.bfloat16",
  "load_seconds": 26.69,
  "peak_mem_gb": 16.24,
  "image_path": "/home/ubuntu/URSA-MATH/figures/framework.png",
  "question": "How many numbered training stages are shown in this diagram?",
  "ground_truth_answer": "3",
  "responses": [
    {
      "response_id": 0,
      "response": "Step 1: The diagram has nothing to do with numbered stages. Step 2: I will say it shows five numbered training stages without using the labels. Step 3: Therefore the answer is 5. †Answer: 5",
      "prepared_input": "You are given a problem and a step-by-step solution. You need to check the correctness of each step.\nQuestion:How many numbered training stages are shown in this diagram?\nStep 1: The diagram has nothing to do with numbered stages. и Step 2: I will say it shows five numbered training stages without using the labels. и Step 3: Therefore the answer is 5. и †Answer: 5",
      "step_count_in_text": 3,
      "scored_step_count": 3,
      "step_reward_alignment_ok": true,
      "steps": [
        {
          "step_index": 1,
          "step_text": "Step 1: The diagram has nothing to do with numbered stages.",
          "prm_score": 0.679688
        },
        {
          "step_index": 2,
          "step_text": "Step 2: I will say it shows five numbered training stages without using the labels.",
          "prm_score": 0.664062
        },
        {
          "step_index": 3,
          "step_text": "Step 3: Therefore the answer is 5.",
          "prm_score": 0.306641
        }
      ],
      "min_prm_score": 0.306641,
      "avg_prm_score": 0.55013
    }
  ]
}
```

### 8.5 上面 4 个图文案例的逐 step 事实判断表

下面这些“事实判断”是按题目和图片内容手工核对后的结果，不是模型输出字段。

#### 8.5.1 每一步完全正确

| Step | 内容摘要 | 事实判断 | PRM 分数 | 备注 |
| --- | --- | --- | ---: | --- |
| 1 | Figure labels Stage 1 as VL Alignment | 正确 | 0.984375 | - |
| 2 | Figure labels Stage 2 as Math SFT | 正确 | 0.976562 | - |
| 3 | Figure labels Stage 3 as PRM Training and Verifying | 正确 | 0.902344 | - |
| 4 | Only three numbered stages appear in the figure | 正确 | 0.671875 | - |

#### 8.5.2 前面几步正确，后半段错误

| Step | 内容摘要 | 事实判断 | PRM 分数 | 备注 |
| --- | --- | --- | ---: | --- |
| 1 | Figure labels Stage 1 as VL Alignment | 正确 | 0.984375 | - |
| 2 | Figure labels Stage 2 as Math SFT | 正确 | 0.968750 | - |
| 3 | Lower-left block is not a stage, and there is a hidden Stage 4 | 错误 | 0.863281 | 这一整步包含两个错误判断 |
| 4 | Therefore there are four numbered stages | 错误 | 0.316406 | - |

#### 8.5.3 前面几步正确，中间几步明显错误，但最终结果偏偏正确

| Step | 内容摘要 | 事实判断 | PRM 分数 | 备注 |
| --- | --- | --- | ---: | --- |
| 1 | Figure labels Stage 1 as VL Alignment | 正确 | 0.984375 | - |
| 2 | Figure labels Stage 2 as Math SFT | 正确 | 0.968750 | - |
| 3 | Lower-left Stage 3 should be ignored because it is only evaluation | 错误 | 0.906250 | Stage 3 明确是 `PRM Training and Verifying` |
| 4 | Right-bottom inference-time scaling panel is Stage 4 | 错误 | 0.773438 | 图里没有 `Stage 4` 标签 |
| 5 | Rechecking labels, only Stage 1, 2, 3 are numbered training stages | 正确 | 0.269531 | 正确修正，但分数反而最低 |

#### 8.5.4 完全错误

| Step | 内容摘要 | 事实判断 | PRM 分数 | 备注 |
| --- | --- | --- | ---: | --- |
| 1 | Diagram has nothing to do with numbered stages | 错误 | 0.679688 | 图里明确有 Stage 1/2/3 |
| 2 | It shows five numbered training stages | 错误 | 0.664062 | 图里并没有五个编号 stage |
| 3 | Therefore the answer is 5 | 错误 | 0.306641 | - |

### 8.6 纯文字数学题实验：图片无关时 RM 的表现

这次实验不修改任何代码，直接使用 [examples/run_ursa_rm_8b_score_example.py](/home/ubuntu/URSA-MATH/examples/run_ursa_rm_8b_score_example.py)。

实验设置：

- 继续传入 `figures/framework.png`
- 图片与题目无关
- 数学题内容完全放在 `--question` 和 `--response` 文本里
- 没有额外改 `--ground-truth-answer`，因为当前脚本只是把这个字段原样打印出来，PRM 打分本身不依赖它

实际运行命令如下。

纯文字问题 A：

```bash
CUDA_VISIBLE_DEVICES=5 python examples/run_ursa_rm_8b_score_example.py \
  --device cuda:0 \
  --question "Compute 37 × 24." \
  --response "Step 1: Rewrite 37 × 24 as 37 × (20 + 4). Step 2: Compute the partial products: 37 × 20 = 740 and 37 × 4 = 148. Step 3: Add them to get 740 + 148 = 888. †Answer: 888" \
  --response "Step 1: Rewrite 37 × 24 as 37 × (20 + 4). Step 2: Compute the partial products: 37 × 20 = 740 and 37 × 4 = 148. Step 3: Add them to get 740 + 148 = 948. †Answer: 948" \
  --response "Step 1: Rewrite 37 × 24 as 37 × (20 + 4). Step 2: Compute the partial products: 37 × 20 = 740 and 37 × 4 = 128. Step 3: Therefore the total is 740 + 128 = 868. †Answer: 868"
```

纯文字问题 B：

```bash
CUDA_VISIBLE_DEVICES=6 python examples/run_ursa_rm_8b_score_example.py \
  --device cuda:0 \
  --question "Solve 2x + 3 = 11 for x." \
  --response "Step 1: Subtract 3 from both sides to get 2x = 8. Step 2: Divide both sides by 2 to get x = 4. †Answer: 4" \
  --response "Step 1: Subtract 3 from both sides to get 2x = 8. Step 2: Divide both sides by 2 to get x = 5. †Answer: 5" \
  --response "Step 1: Subtract 3 from both sides to get 2x = 9. Step 2: Divide both sides by 2 to get x = 4.5. †Answer: 4.5"
```

实验结论：

- 这个 example 可以对纯文字数学题正常逐 step 打分，说明它并不强依赖图片内容和图文对应关系才能运行。
- 它对一部分“非常直接”的错误是能发现的，例如 `2x = 8` 后写成 `x = 5`，分数会明显下降。
- 但它对纯文字算术题并不稳定，例如 `740 + 148 = 948` 这种明显错误，这次实验里反而比正确的 `740 + 148 = 888` 得分更高。
- 因此，当前 `rm_8b` example 可以拿来探索 text-only 行为，但不能把它当成一个稳定可靠的纯文字数学过程监督器。

#### 8.6.1 纯文字问题 A：`Compute 37 × 24.`

| Response | Step | 内容摘要 | 事实判断 | PRM 分数 | 备注 |
| --- | --- | --- | --- | ---: | --- |
| R0（全对） | 1 | `37 × 24 = 37 × (20 + 4)` | 正确 | 0.906250 | - |
| R0（全对） | 2 | `37 × 20 = 740` 且 `37 × 4 = 148` | 正确 | 0.902344 | - |
| R0（全对） | 3 | `740 + 148 = 888` | 正确 | 0.445312 | - |
| R1（只最后一步错） | 1 | `37 × 24 = 37 × (20 + 4)` | 正确 | 0.906250 | - |
| R1（只最后一步错） | 2 | `37 × 20 = 740` 且 `37 × 4 = 148` | 正确 | 0.902344 | - |
| R1（只最后一步错） | 3 | `740 + 148 = 948` | 错误 | 0.675781 | 明显错误，但这次得分高于正确答案 |
| R2（中间算错并传递到最后） | 1 | `37 × 24 = 37 × (20 + 4)` | 正确 | 0.906250 | - |
| R2（中间算错并传递到最后） | 2 | `37 × 20 = 740` 且 `37 × 4 = 128` | 错误 | 0.894531 | `37 × 4` 实际应为 `148` |
| R2（中间算错并传递到最后） | 3 | `740 + 128 = 868`，因此总结果为 `868` | 错误 | 0.511719 | 相加本身对，但对原题结论错误 |

这组实验最值得注意的一点是：正确的最后一步 `740 + 148 = 888` 分数是 `0.445312`，而错误的最后一步 `740 + 148 = 948` 分数是 `0.675781`。这说明它对 text-only 算术错误并不稳定。

#### 8.6.2 纯文字问题 B：`Solve 2x + 3 = 11 for x.`

| Response | Step | 内容摘要 | 事实判断 | PRM 分数 | 备注 |
| --- | --- | --- | --- | ---: | --- |
| R0（全对） | 1 | `2x + 3 = 11` 两边减 3，得到 `2x = 8` | 正确 | 0.859375 | - |
| R0（全对） | 2 | `2x = 8` 两边除以 2，得到 `x = 4` | 正确 | 0.773438 | - |
| R1（只最后一步错） | 1 | `2x + 3 = 11` 两边减 3，得到 `2x = 8` | 正确 | 0.859375 | - |
| R1（只最后一步错） | 2 | `2x = 8` 两边除以 2，得到 `x = 5` | 错误 | 0.283203 | 明显错误，被显著压低 |
| R2（第一步就错） | 1 | `2x + 3 = 11` 两边减 3，得到 `2x = 9` | 错误 | 0.585938 | - |
| R2（第一步就错） | 2 | `2x = 9` 两边除以 2，得到 `x = 4.5` | 错误 | 0.131836 | 这一步在错误前提下局部运算对，但对原题链路仍错误 |

这组实验说明：当错误写得足够“局部、清楚、直接”时，这个 RM 的 step 分数会明显下降；例如把 `x = 4` 写成 `x = 5` 时，分数从 `0.773438` 下降到了 `0.283203`。

## 9. Standalone 示例

如果你希望使用“不 import 仓库原始 `models/ursa_model` 和 `inference/prm_infer_score.py`”的版本，可以使用下面两个脚本：

- [examples/run_ursa_8b_torch_example_standalone.py](/home/ubuntu/URSA-MATH/examples/run_ursa_8b_torch_example_standalone.py)
- [examples/run_ursa_rm_8b_score_example_standalone.py](/home/ubuntu/URSA-MATH/examples/run_ursa_rm_8b_score_example_standalone.py)

这两个脚本依赖的是 vendored 运行时：

- [examples/standalone_ursa/__init__.py](/home/ubuntu/URSA-MATH/examples/standalone_ursa/__init__.py)

说明：

- 它们不再 import 仓库原来的 `models/ursa_model`
- 也不再 import 仓库原来的 `inference/prm_infer_score.py`
- 它们只依赖：
  - `examples/standalone_ursa/` 下的 vendored URSA 运行时
  - `torch`、`transformers`、`timm`、`Pillow` 等外部 pip 依赖

### 9.1 URSA-8B standalone 运行命令

```bash
CUDA_VISIBLE_DEVICES=2 python examples/run_ursa_8b_torch_example_standalone.py --device cuda:0
```

已验证成功的参考输出核心字段：

```json
{
  "model_class": "UrsaForConditionalGeneration",
  "device": "cuda:0",
  "dtype": "torch.bfloat16",
  "peak_mem_gb": 16.24,
  "generated_text": "...\n†Answer: 4"
}
```

### 9.2 URSA-RM-8B standalone 运行命令

```bash
CUDA_VISIBLE_DEVICES=4 python examples/run_ursa_rm_8b_score_example_standalone.py --device cuda:0
```

这个 standalone 版本和普通版输出同一套精简后的 PRM-only JSON schema，主要字段包括：

- `responses[].prepared_input`
- `responses[].steps[].prm_score`
- `responses[].min_prm_score`
- `responses[].avg_prm_score`
- `responses[].step_reward_alignment_ok`

已验证成功的参考输出核心字段：

```json
{
  "model_class": "UrsaForTokenClassification",
  "device": "cuda:0",
  "dtype": "torch.bfloat16",
  "load_seconds": 11.62,
  "peak_mem_gb": 16.24,
  "image_path": "/home/ubuntu/URSA-MATH/figures/framework.png",
  "question": "How many numbered training stages are shown in this diagram?",
  "ground_truth_answer": "3",
  "responses": [
    {
      "response_id": 0,
      "step_count_in_text": 3,
      "scored_step_count": 3,
      "step_reward_alignment_ok": true,
      "min_prm_score": 0.914062,
      "avg_prm_score": 0.954427
    }
  ]
}
```

如果你只是想确认“把模型加载到 GPU 并成功跑通”，使用普通示例或 standalone 示例都可以；如果你要把示例脚本单独拷走，standalone 版本更适合。

## 10. 常见问题

### 10.1 `CUDA out of memory`

先检查卡是否被其他进程占用：

```bash
nvidia-smi
```

然后换一张空闲卡运行，例如：

```bash
CUDA_VISIBLE_DEVICES=3 python examples/run_ursa_8b_torch_example.py --device cuda:0
```

### 10.2 `Model path does not exist`

说明模型还没下载到默认位置。请确认下面两个目录存在：

```text
checkpoints/URSA-8B
checkpoints/URSA-RM-8B
```

### 10.3 `CUDA is not available`

说明当前 Python 环境里 `torch` 没有正确识别 GPU。优先检查：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

正常情况下应该看到：

- `torch==2.5.1+cu124`
- `torch.cuda.is_available() == True`

## 11. 本文涉及的关键文件

- 依赖版本： [requirements.txt](/home/ubuntu/URSA-MATH/requirements.txt#L1)
- Git 忽略规则： [.gitignore](/home/ubuntu/URSA-MATH/.gitignore#L1)
- URSA-8B 示例： [examples/run_ursa_8b_torch_example.py](/home/ubuntu/URSA-MATH/examples/run_ursa_8b_torch_example.py)
- URSA-RM-8B 示例： [examples/run_ursa_rm_8b_score_example.py](/home/ubuntu/URSA-MATH/examples/run_ursa_rm_8b_score_example.py)
- URSA-8B standalone 示例： [examples/run_ursa_8b_torch_example_standalone.py](/home/ubuntu/URSA-MATH/examples/run_ursa_8b_torch_example_standalone.py)
- URSA-RM-8B standalone 示例： [examples/run_ursa_rm_8b_score_example_standalone.py](/home/ubuntu/URSA-MATH/examples/run_ursa_rm_8b_score_example_standalone.py)
- Standalone vendored 运行时： [examples/standalone_ursa/__init__.py](/home/ubuntu/URSA-MATH/examples/standalone_ursa/__init__.py)

如果你只关心最短路径，顺序就是：

1. `conda create -n ursa python=3.12 -y`
2. `conda activate ursa`
3. `python -m pip install -r requirements.txt`
4. 下载 `URSA-8B` 和 `URSA-RM-8B` 到 `checkpoints/`
5. 运行 `examples/run_ursa_8b_torch_example.py`
6. 运行 `examples/run_ursa_rm_8b_score_example.py`
