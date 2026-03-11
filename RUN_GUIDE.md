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

## 8. 示例二：URSA-RM-8B 加载到 GPU，并对一步步答案打分

示例脚本：

- [examples/run_ursa_rm_8b_score_example.py](/home/ubuntu/URSA-MATH/examples/run_ursa_rm_8b_score_example.py)

这个脚本做的事情：

- 使用本仓库本地的 `UrsaForTokenClassification`
- 从 `checkpoints/URSA-RM-8B` 加载完整权重
- 把模型放到 GPU
- 用 `figures/framework.png` 和一段分步答案做一次打分

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
How many training stages are shown in this diagram?
```

- 待打分回答：

```text
Step 1: The figure is a training framework diagram. Step 2: I can identify four training stages in the diagram. †Answer: 4
```

### 已验证成功的参考输出

我在当前机器上实际跑通时，输出类似：

```json
{
  "model_class": "UrsaForTokenClassification",
  "device": "cuda:0",
  "dtype": "torch.bfloat16",
  "load_seconds": 18.8,
  "peak_mem_gb": 16.24,
  "min_score": 0.910156,
  "avg_score": 0.917969
}
```

### 自定义参数

例如换成你自己的问题、答案和图片：

```bash
python examples/run_ursa_rm_8b_score_example.py \
  --device cuda:0 \
  --model-path ./checkpoints/URSA-RM-8B \
  --image-path ./figures/framework.png \
  --question "How many training stages are shown in this diagram?" \
  --response "Step 1: This is a training diagram. Step 2: It contains four stages. †Answer: 4"
```

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
CUDA_VISIBLE_DEVICES=3 python examples/run_ursa_rm_8b_score_example_standalone.py --device cuda:0
```

已验证成功的参考输出核心字段：

```json
{
  "model_class": "UrsaForTokenClassification",
  "device": "cuda:0",
  "dtype": "torch.bfloat16",
  "peak_mem_gb": 16.24,
  "min_score": 0.910156,
  "avg_score": 0.917969
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
