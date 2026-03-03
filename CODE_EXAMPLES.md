# URSA-MATH 代码示例与出处

本文档提供独立可运行的代码示例，以及它们在URSA-MATH仓库中的出处。

## 目录

1. [加载URSA-8B模型](#加载ursa-8b模型)
2. [加载URSA-RM-8B模型](#加载ursa-rm-8b模型)
3. [加载MMathCoT-1M数据集](#加载mmathcot-1m数据集)
4. [加载DualMath-1.1M数据集](#加载dualmath-11m数据集)
5. [完整推理示例](#完整推理示例)
6. [Best-of-N采样示例](#best-of-n采样示例)

---

## 加载URSA-8B模型

### 代码出处

- **文件**: `inference/vllm_infer.py`
- **行数**: 14-25, 主函数中的模型加载部分
- **关键代码段**:
  ```python
  from vllm import LLM, SamplingParams
  llm = LLM(model=model_path, ...)
  ```

### 独立可运行代码

```python
"""
URSA-8B 模型加载示例
独立运行，不依赖URSA仓库的import
"""

from vllm import LLM, SamplingParams
from PIL import Image
import torch

def load_ursa_8b_model(model_path: str, num_gpus: int = 1):
    """
    加载URSA-8B生成模型

    Args:
        model_path: 模型路径，例如 "URSA-MATH/URSA-8B" 或本地路径
        num_gpus: 使用的GPU数量

    Returns:
        llm: vLLM的LLM对象
    """
    llm = LLM(
        model=model_path,
        trust_remote_code=True,  # 必须，因为URSA使用自定义模型代码
        tensor_parallel_size=num_gpus,
        max_model_len=4096,  # 最大序列长度
        gpu_memory_utilization=0.9,  # GPU内存利用率
    )
    return llm

def generate_answer(llm, question: str, image_path: str):
    """
    使用URSA-8B生成答案

    Args:
        llm: 加载的模型
        question: 问题文本
        image_path: 图片路径

    Returns:
        generated_text: 生成的答案
    """
    # 构造prompt（来自 inference/vllm_infer.py 第28-33行）
    template = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<|image|>{}<|im_end|>
<|im_start|>assistant
"""

    prompt = template.format(question)
    image = Image.open(image_path)

    # 采样参数
    sampling_params = SamplingParams(
        temperature=0.2,
        top_p=0.95,
        max_tokens=2048,
        stop=["<|im_end|>"]
    )

    # 生成
    outputs = llm.generate(
        {
            'prompt': prompt,
            'multi_modal_data': {'image': image}
        },
        sampling_params
    )

    return outputs[0].outputs[0].text

# 使用示例
if __name__ == "__main__":
    # 1. 加载模型
    model_path = "URSA-MATH/URSA-8B"  # 或本地路径
    llm = load_ursa_8b_model(model_path, num_gpus=1)

    # 2. 生成答案
    question = "What is the area of the triangle in the figure?"
    image_path = "path/to/triangle.png"

    answer = generate_answer(llm, question, image_path)
    print("Generated Answer:")
    print(answer)
```

**依赖安装**:
```bash
pip install vllm pillow torch
```

---

## 加载URSA-RM-8B模型

### 代码出处

- **文件**: `inference/prm_infer_score.py`
- **行数**: 27, 91-100 (模型加载和推理函数)
- **关键代码段**:
  ```python
  from models.ursa_model import UrsaProcessor, UrsaForTokenClassification
  model = UrsaForTokenClassification.from_pretrained(...)
  processor = UrsaProcessor.from_pretrained(...)
  ```

### 独立可运行代码

```python
"""
URSA-RM-8B 奖励模型加载示例
独立运行，不依赖URSA仓库的import
"""

import torch
from transformers import AutoModel, AutoProcessor
from PIL import Image
import re

def load_ursa_rm_8b_model(model_path: str, device: str = "cuda:0"):
    """
    加载URSA-RM-8B奖励模型

    Args:
        model_path: 模型路径，例如 "URSA-MATH/URSA-RM-8B" 或本地路径
        device: 设备，例如 "cuda:0"

    Returns:
        model: 奖励模型
        processor: 处理器
    """
    # 加载模型（使用AutoModel会自动识别UrsaForTokenClassification）
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,  # 必须，因为使用自定义模型
        torch_dtype=torch.bfloat16,  # 使用bf16节省显存
    ).to(device).eval()

    # 加载处理器
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    return model, processor

def insert_step_markers(text: str) -> str:
    """
    在推理步骤间插入特殊标记 'и'

    代码出处: inference/prm_infer_score.py 第43-66行

    Args:
        text: 原始推理文本

    Returns:
        带标记的文本
    """
    pattern = r'Step \d+'
    matches = list(re.finditer(pattern, text))
    positions = [(match.start(), match.end()) for match in matches]

    text_list = list(text)
    insert_pos = []

    try:
        # 在每个Step之间插入标记
        for i in range(1, len(positions)):
            for j in range(positions[i][0] - 1, positions[i - 1][1], -1):
                if text_list[j] != ' ' and text_list[j] != '\n':
                    insert_pos.append(j + 1)
                    break

        # 在Answer之前插入标记
        answer_start = text.find('†Answer:')
        if answer_start != -1:
            for j in range(answer_start - 1, positions[-1][1], -1):
                if text_list[j] != ' ' and text_list[j] != '\n':
                    insert_pos.append(j + 1)
                    break

        # 插入标记
        for index in sorted(insert_pos, reverse=True):
            text = text[:index] + ' и' + text[index:]

        return text
    except:
        return text + ' и'

def score_response(
    model,
    processor,
    question: str,
    response: str,
    image_path: str,
    device: str = "cuda:0"
) -> dict:
    """
    使用URSA-RM-8B给推理过程打分

    代码出处: inference/prm_infer_score.py 第91-100行 (single_inference函数)

    Args:
        model: 奖励模型
        processor: 处理器
        question: 问题文本
        response: 推理过程
        image_path: 图片路径
        device: 设备

    Returns:
        scores: 包含min_score, avg_score, step_scores的字典
    """
    # 1. 插入标记
    response_with_markers = insert_step_markers(response)

    # 2. 构造输入
    prompt = f"You are given a problem and a step-by-step solution. You need to check the correctness of each step.\nQuestion:{question}\n{response_with_markers}"

    # 构造对话格式
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"<|image|>{prompt}"}
    ]

    # 处理输入
    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    image = Image.open(image_path)

    inputs = processor(
        text=[text],
        images=[image],
        return_tensors='pt',
        padding=True
    ).to(device)

    # 3. 前向传播
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # [1, seq_len, 1]

    # 4. 找到 'и' 的位置
    marker_token_id = processor.tokenizer.encode(' и', add_special_tokens=False)[0]
    marker_positions = (inputs['input_ids'] == marker_token_id)

    # 5. 提取这些位置的分数
    marker_logits = logits[marker_positions]
    marker_scores = torch.sigmoid(marker_logits).squeeze(-1)  # 转换为概率

    # 6. 计算总分
    if marker_scores.numel() == 0:
        return {
            'min_score': 0.0,
            'avg_score': 0.0,
            'step_scores': []
        }

    min_score = torch.min(marker_scores).item()
    avg_score = torch.mean(marker_scores).item()
    step_scores = marker_scores.cpu().tolist()

    return {
        'min_score': min_score,
        'avg_score': avg_score,
        'step_scores': step_scores
    }

# 使用示例
if __name__ == "__main__":
    # 1. 加载模型
    model_path = "URSA-MATH/URSA-RM-8B"  # 或本地路径
    model, processor = load_ursa_rm_8b_model(model_path, device="cuda:0")

    # 2. 准备数据
    question = "What is the area of the triangle?"
    response = """Step 1: From the figure, the base is 5 cm and height is 3 cm.

Step 2: Using the formula A = 1/2 × base × height

Step 3: A = 1/2 × 5 × 3 = 7.5 cm²

†Answer: 7.5"""

    image_path = "path/to/triangle.png"

    # 3. 评分
    scores = score_response(model, processor, question, response, image_path)

    print(f"Minimum Score: {scores['min_score']:.3f}")
    print(f"Average Score: {scores['avg_score']:.3f}")
    print(f"Step Scores: {scores['step_scores']}")
```

**依赖安装**:
```bash
pip install transformers torch pillow
```

---

## 加载MMathCoT-1M数据集

### 代码出处

- **文件**: `inference/vllm_infer.py`
- **行数**: 92-150 (prepare_data函数)
- **数据集**: HuggingFace `URSA-MATH/MMathCoT-1M`

### 独立可运行代码

```python
"""
MMathCoT-1M 数据集加载示例
独立运行，不依赖URSA仓库
"""

from datasets import load_dataset
from PIL import Image
import json

def load_mmathcot_dataset(split: str = "train", streaming: bool = False):
    """
    从HuggingFace加载MMathCoT-1M数据集

    Args:
        split: 数据集分割，"train" 或 "test"
        streaming: 是否使用流式加载（节省内存）

    Returns:
        dataset: HuggingFace Dataset对象
    """
    dataset = load_dataset(
        "URSA-MATH/MMathCoT-1M",
        split=split,
        streaming=streaming
    )
    return dataset

def parse_mmathcot_sample(sample: dict) -> dict:
    """
    解析MMathCoT数据样本

    Args:
        sample: 原始数据样本

    Returns:
        parsed: 解析后的数据，包含question, solution, answer, image等
    """
    return {
        'id': sample.get('id', ''),
        'question': sample['question'],
        'solution': sample['solution'],  # 完整的推理过程
        'answer': sample['answer'],      # 最终答案
        'image': sample['image'],        # PIL Image对象
        'source': sample.get('source', ''),  # 来源数据集
        'difficulty': sample.get('difficulty', ''),  # 难度
        'topic': sample.get('topic', '')  # 主题
    }

# 使用示例
if __name__ == "__main__":
    # 1. 加载数据集
    print("Loading MMathCoT-1M dataset...")
    dataset = load_mmathcot_dataset(split="train", streaming=False)

    # 2. 查看数据集信息
    print(f"Dataset size: {len(dataset)}")
    print(f"Features: {dataset.features}")

    # 3. 读取第一个样本
    sample = dataset[0]
    parsed = parse_mmathcot_sample(sample)

    print("\n=== Sample Data ===")
    print(f"ID: {parsed['id']}")
    print(f"Question: {parsed['question']}")
    print(f"\nSolution:\n{parsed['solution']}")
    print(f"\nAnswer: {parsed['answer']}")
    print(f"Source: {parsed['source']}")
    print(f"Difficulty: {parsed['difficulty']}")

    # 4. 保存图片
    if parsed['image'] is not None:
        parsed['image'].save("sample_image.png")
        print("\nImage saved to sample_image.png")

    # 5. 统计数据分布
    print("\n=== Dataset Statistics ===")

    # 按来源统计
    sources = {}
    for sample in dataset:
        source = sample.get('source', 'unknown')
        sources[source] = sources.get(source, 0) + 1

    print("By Source:")
    for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
        print(f"  {source}: {count}")

    # 按难度统计
    difficulties = {}
    for sample in dataset:
        diff = sample.get('difficulty', 'unknown')
        difficulties[diff] = difficulties.get(diff, 0) + 1

    print("\nBy Difficulty:")
    for diff, count in sorted(difficulties.items()):
        print(f"  {diff}: {count}")
```

**依赖安装**:
```bash
pip install datasets pillow
```

**数据格式说明**:

MMathCoT-1M的每个样本包含：
- `id`: 样本唯一标识
- `question`: 问题文本
- `solution`: 完整的逐步推理过程（包含Step 1, Step 2, ..., †Answer:）
- `answer`: 最终答案
- `image`: PIL Image对象
- `source`: 来源数据集（mathvista, geoqa, math, tabmwp, chartqa等）
- `difficulty`: 难度级别（easy, medium, hard）
- `topic`: 主题（geometry, algebra, statistics等）

---

## 加载DualMath-1.1M数据集

### 代码出处

- **文件**: `inference/prm_infer_score.py`
- **行数**: 数据加载逻辑分散在主函数中
- **数据集**: HuggingFace `URSA-MATH/DualMath-1.1M`

### 独立可运行代码

```python
"""
DualMath-1.1M 数据集加载示例
独立运行，不依赖URSA仓库
"""

from datasets import load_dataset
from PIL import Image
import json

def load_dualmath_dataset(split: str = "train", streaming: bool = False):
    """
    从HuggingFace加载DualMath-1.1M数据集

    Args:
        split: 数据集分割，"train" 或 "test"
        streaming: 是否使用流式加载

    Returns:
        dataset: HuggingFace Dataset对象
    """
    dataset = load_dataset(
        "URSA-MATH/DualMath-1.1M",
        split=split,
        streaming=streaming
    )
    return dataset

def parse_dualmath_sample(sample: dict) -> dict:
    """
    解析DualMath数据样本

    Args:
        sample: 原始数据样本

    Returns:
        parsed: 解析后的数据
    """
    return {
        'id': sample.get('id', ''),
        'question': sample['question'],
        'response': sample['response'],  # 带标记 'и' 的推理过程
        'labels': sample['labels'],      # 每个标记位置的标签 [1, 1, 0, ...]
        'image': sample['image'],
        'error_type': sample.get('error_type', None),  # 错误类型
        'error_step': sample.get('error_step', None),  # 错误步骤
        'data_type': sample.get('data_type', '')  # logic_positive, logic_negative, visual_positive, visual_negative
    }

def count_step_markers(response: str) -> int:
    """
    统计推理过程中的标记数量

    Args:
        response: 带标记的推理文本

    Returns:
        count: 标记数量
    """
    return response.count(' и')

# 使用示例
if __name__ == "__main__":
    # 1. 加载数据集
    print("Loading DualMath-1.1M dataset...")
    dataset = load_dualmath_dataset(split="train", streaming=False)

    # 2. 查看数据集信息
    print(f"Dataset size: {len(dataset)}")
    print(f"Features: {dataset.features}")

    # 3. 读取样本
    # 正样本示例
    positive_sample = None
    negative_sample = None

    for sample in dataset:
        if all(label == 1 for label in sample['labels']):
            if positive_sample is None:
                positive_sample = parse_dualmath_sample(sample)
        elif any(label == 0 for label in sample['labels']):
            if negative_sample is None:
                negative_sample = parse_dualmath_sample(sample)

        if positive_sample and negative_sample:
            break

    # 4. 展示正样本
    print("\n=== Positive Sample (All Correct) ===")
    print(f"Question: {positive_sample['question']}")
    print(f"\nResponse:\n{positive_sample['response']}")
    print(f"\nLabels: {positive_sample['labels']}")
    print(f"Number of steps: {count_step_markers(positive_sample['response'])}")

    # 5. 展示负样本
    print("\n=== Negative Sample (Contains Error) ===")
    print(f"Question: {negative_sample['question']}")
    print(f"\nResponse:\n{negative_sample['response']}")
    print(f"\nLabels: {negative_sample['labels']}")
    print(f"Error Type: {negative_sample['error_type']}")
    print(f"Error Step: {negative_sample['error_step']}")
    print(f"Number of steps: {count_step_markers(negative_sample['response'])}")

    # 6. 统计数据分布
    print("\n=== Dataset Statistics ===")

    # 按数据类型统计
    data_types = {}
    error_types = {}
    label_distribution = {'all_correct': 0, 'has_error': 0}

    for sample in dataset:
        # 数据类型
        dtype = sample.get('data_type', 'unknown')
        data_types[dtype] = data_types.get(dtype, 0) + 1

        # 错误类型
        if sample.get('error_type'):
            etype = sample['error_type']
            error_types[etype] = error_types.get(etype, 0) + 1

        # 标签分布
        if all(label == 1 for label in sample['labels']):
            label_distribution['all_correct'] += 1
        else:
            label_distribution['has_error'] += 1

    print("By Data Type:")
    for dtype, count in sorted(data_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {dtype}: {count}")

    print("\nBy Error Type:")
    for etype, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {etype}: {count}")

    print("\nLabel Distribution:")
    print(f"  All Correct: {label_distribution['all_correct']}")
    print(f"  Has Error: {label_distribution['has_error']}")
    print(f"  Positive Ratio: {label_distribution['all_correct'] / len(dataset):.2%}")
```

**依赖安装**:
```bash
pip install datasets pillow
```

**数据格式说明**:

DualMath-1.1M的每个样本包含：
- `id`: 样本唯一标识
- `question`: 问题文本
- `response`: 带标记 'и' 的推理过程
- `labels`: 每个 'и' 位置的标签列表，1表示正确，0表示错误
- `image`: PIL Image对象
- `error_type`: 错误类型（calculation_error, logic_error, visual_misread等）
- `error_step`: 第几步出错（从1开始）
- `data_type`: 数据类型
  - `logic_positive`: 逻辑推理正样本
  - `logic_negative`: 逻辑推理负样本
  - `visual_positive`: 视觉理解正样本
  - `visual_negative`: 视觉理解负样本

**标记 'и' 的含义**:
- 这是一个俄语字符，用于标记每个推理步骤的结束位置
- 模型在这些位置输出0或1的分数
- 例如: `"Step 1: ... и Step 2: ... и †Answer: ... и"`

---

## 完整推理示例

### 代码出处

- **文件**: `inference/vllm_infer.py` (完整推理流程)
- **行数**: 主函数部分

### 独立可运行代码

```python
"""
URSA-8B 完整推理示例
从加载模型到生成答案的完整流程
"""

from vllm import LLM, SamplingParams
from PIL import Image
import re

# ========== 模型加载 ==========
def load_model(model_path: str, num_gpus: int = 1):
    """加载URSA-8B模型"""
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=num_gpus,
        max_model_len=4096,
        gpu_memory_utilization=0.9,
    )
    return llm

# ========== Prompt模板 ==========
# 代码出处: inference/vllm_infer.py 第27-33行
SYSTEM_PROMPT = "You are a helpful assistant."
TEMPLATE = """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
<|image|>{question}<|im_end|>
<|im_start|>assistant
"""

# ========== 答案提取 ==========
# 代码出处: inference/vllm_infer.py 第38-79行
def extract_answer(text: str) -> str:
    """
    从生成的文本中提取最终答案

    尝试多种模式匹配答案格式
    """
    patterns = [
        r'Answer: \\boxed\{(.*?)\}',
        r'\\boxed\{(.*?)\}',
        r'†Answer:(.*?)(?:\n|$)',
        r'†Answer: (.*?)(?:\n|$)',
        r'Answer:(.*?)(?:\n|$)',
        r'Answer: (.*?)(?:\n|$)',
        r'The answer is (.*?)(?:\n|$)'
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            valid_matches = [
                match.strip() for match in matches
                if match.strip() and (len(match.strip()) > 1 or match.strip().isalnum())
            ]
            if valid_matches:
                return valid_matches[0]

    return text.strip()

# ========== 推理函数 ==========
def generate_answer(
    llm,
    question: str,
    image_path: str,
    temperature: float = 0.2,
    max_tokens: int = 2048
) -> dict:
    """
    生成答案

    Args:
        llm: 加载的模型
        question: 问题文本
        image_path: 图片路径
        temperature: 采样温度
        max_tokens: 最大生成token数

    Returns:
        result: 包含完整响应和提取的答案
    """
    # 1. 构造prompt
    prompt = TEMPLATE.format(system=SYSTEM_PROMPT, question=question)

    # 2. 加载图片
    image = Image.open(image_path)

    # 3. 设置采样参数
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=0.95,
        max_tokens=max_tokens,
        stop=["<|im_end|>"]
    )

    # 4. 生成
    outputs = llm.generate(
        {
            'prompt': prompt,
            'multi_modal_data': {'image': image}
        },
        sampling_params
    )

    # 5. 提取结果
    generated_text = outputs[0].outputs[0].text
    extracted_answer = extract_answer(generated_text)

    return {
        'full_response': generated_text,
        'extracted_answer': extracted_answer
    }

# ========== 批量推理 ==========
def batch_inference(
    llm,
    questions: list,
    image_paths: list,
    batch_size: int = 8
) -> list:
    """
    批量推理

    Args:
        llm: 加载的模型
        questions: 问题列表
        image_paths: 图片路径列表
        batch_size: 批次大小

    Returns:
        results: 结果列表
    """
    results = []

    for i in range(0, len(questions), batch_size):
        batch_questions = questions[i:i+batch_size]
        batch_images = image_paths[i:i+batch_size]

        # 准备批次输入
        batch_inputs = []
        for question, image_path in zip(batch_questions, batch_images):
            prompt = TEMPLATE.format(system=SYSTEM_PROMPT, question=question)
            image = Image.open(image_path)
            batch_inputs.append({
                'prompt': prompt,
                'multi_modal_data': {'image': image}
            })

        # 批量生成
        sampling_params = SamplingParams(
            temperature=0.2,
            top_p=0.95,
            max_tokens=2048,
            stop=["<|im_end|>"]
        )

        outputs = llm.generate(batch_inputs, sampling_params)

        # 提取结果
        for output in outputs:
            generated_text = output.outputs[0].text
            extracted_answer = extract_answer(generated_text)
            results.append({
                'full_response': generated_text,
                'extracted_answer': extracted_answer
            })

    return results

# ========== 使用示例 ==========
if __name__ == "__main__":
    # 1. 加载模型
    print("Loading URSA-8B model...")
    model_path = "URSA-MATH/URSA-8B"  # 或本地路径
    llm = load_model(model_path, num_gpus=1)

    # 2. 单个样本推理
    print("\n=== Single Sample Inference ===")
    question = "What is the area of the triangle in the figure?"
    image_path = "path/to/triangle.png"

    result = generate_answer(llm, question, image_path)

    print(f"Question: {question}")
    print(f"\nFull Response:\n{result['full_response']}")
    print(f"\nExtracted Answer: {result['extracted_answer']}")

    # 3. 批量推理
    print("\n=== Batch Inference ===")
    questions = [
        "What is the area of the triangle?",
        "What is the slope of the line?",
        "Calculate the volume of the cylinder."
    ]
    image_paths = [
        "path/to/triangle.png",
        "path/to/line.png",
        "path/to/cylinder.png"
    ]

    results = batch_inference(llm, questions, image_paths, batch_size=3)

    for i, result in enumerate(results):
        print(f"\nSample {i+1}:")
        print(f"Answer: {result['extracted_answer']}")
```

**依赖安装**:
```bash
pip install vllm pillow
```

---

## Best-of-N采样示例

### 代码出处

- **生成部分**: `inference/vllm_infer.py`
- **评分部分**: `inference/prm_infer_score.py`
- **组合逻辑**: 需要结合两个文件

### 独立可运行代码

```python
"""
Best-of-N 采样完整示例
结合URSA-8B生成和URSA-RM-8B评分
"""

from vllm import LLM, SamplingParams
from transformers import AutoModel, AutoProcessor
from PIL import Image
import torch
import re

# ========== 加载模型 ==========
def load_models(
    generator_path: str,
    reward_model_path: str,
    num_gpus: int = 1
):
    """
    加载生成模型和奖励模型

    Args:
        generator_path: URSA-8B路径
        reward_model_path: URSA-RM-8B路径
        num_gpus: GPU数量

    Returns:
        generator: 生成模型
        reward_model: 奖励模型
        processor: 处理器
    """
    # 加载生成模型
    generator = LLM(
        model=generator_path,
        trust_remote_code=True,
        tensor_parallel_size=num_gpus,
        max_model_len=4096,
        gpu_memory_utilization=0.85,  # 留一些显存给RM
    )

    # 加载奖励模型
    reward_model = AutoModel.from_pretrained(
        reward_model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to("cuda").eval()

    processor = AutoProcessor.from_pretrained(
        reward_model_path,
        trust_remote_code=True
    )

    return generator, reward_model, processor

# ========== 插入标记 ==========
# 代码出处: inference/prm_infer_score.py 第43-66行
def insert_step_markers(text: str) -> str:
    """在推理步骤间插入特殊标记 'и'"""
    pattern = r'Step \d+'
    matches = list(re.finditer(pattern, text))
    positions = [(match.start(), match.end()) for match in matches]

    text_list = list(text)
    insert_pos = []

    try:
        for i in range(1, len(positions)):
            for j in range(positions[i][0] - 1, positions[i - 1][1], -1):
                if text_list[j] != ' ' and text_list[j] != '\n':
                    insert_pos.append(j + 1)
                    break

        answer_start = text.find('†Answer:')
        if answer_start != -1:
            for j in range(answer_start - 1, positions[-1][1], -1):
                if text_list[j] != ' ' and text_list[j] != '\n':
                    insert_pos.append(j + 1)
                    break

        for index in sorted(insert_pos, reverse=True):
            text = text[:index] + ' и' + text[index:]

        return text
    except:
        return text + ' и'

# ========== 生成N个候选 ==========
def generate_n_candidates(
    generator,
    question: str,
    image_path: str,
    n: int = 32,
    temperature: float = 0.7
) -> list:
    """
    生成N个候选答案

    Args:
        generator: 生成模型
        question: 问题
        image_path: 图片路径
        n: 候选数量
        temperature: 采样温度（较高以增加多样性）

    Returns:
        candidates: 候选答案列表
    """
    # 构造prompt
    template = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<|image|>{}<|im_end|>
<|im_start|>assistant
"""
    prompt = template.format(question)
    image = Image.open(image_path)

    # 采样参数
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=0.95,
        max_tokens=2048,
        n=n,  # 生成N个候选
        stop=["<|im_end|>"]
    )

    # 生成
    outputs = generator.generate(
        {
            'prompt': prompt,
            'multi_modal_data': {'image': image}
        },
        sampling_params
    )

    # 提取所有候选
    candidates = [output.text for output in outputs[0].outputs]

    return candidates

# ========== 评分函数 ==========
def score_candidate(
    reward_model,
    processor,
    question: str,
    response: str,
    image_path: str,
    strategy: str = "min"
) -> float:
    """
    给单个候选答案打分

    Args:
        reward_model: 奖励模型
        processor: 处理器
        question: 问题
        response: 候选答案
        image_path: 图片路径
        strategy: 评分策略，"min"（最保守）或"avg"（平均）

    Returns:
        score: 分数
    """
    # 插入标记
    response_with_markers = insert_step_markers(response)

    # 构造输入
    prompt = f"You are given a problem and a step-by-step solution. You need to check the correctness of each step.\nQuestion:{question}\n{response_with_markers}"

    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"<|image|>{prompt}"}
    ]

    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    image = Image.open(image_path)

    inputs = processor(
        text=[text],
        images=[image],
        return_tensors='pt',
        padding=True
    ).to("cuda")

    # 前向传播
    with torch.no_grad():
        outputs = reward_model(**inputs)
        logits = outputs.logits

    # 提取标记位置的分数
    marker_token_id = processor.tokenizer.encode(' и', add_special_tokens=False)[0]
    marker_positions = (inputs['input_ids'] == marker_token_id)
    marker_scores = torch.sigmoid(logits[marker_positions]).squeeze(-1)

    if marker_scores.numel() == 0:
        return 0.0

    # 根据策略返回分数
    if strategy == "min":
        return torch.min(marker_scores).item()
    elif strategy == "avg":
        return torch.mean(marker_scores).item()
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

# ========== Best-of-N选择 ==========
def best_of_n_inference(
    generator,
    reward_model,
    processor,
    question: str,
    image_path: str,
    n: int = 32,
    temperature: float = 0.7,
    strategy: str = "min"
) -> dict:
    """
    Best-of-N推理完整流程

    Args:
        generator: 生成模型
        reward_model: 奖励模型
        processor: 处理器
        question: 问题
        image_path: 图片路径
        n: 候选数量
        temperature: 采样温度
        strategy: 评分策略

    Returns:
        result: 包含最佳答案、所有候选、所有分数
    """
    print(f"Generating {n} candidates...")
    candidates = generate_n_candidates(
        generator, question, image_path, n, temperature
    )

    print(f"Scoring {n} candidates...")
    scores = []
    for i, candidate in enumerate(candidates):
        score = score_candidate(
            reward_model, processor, question, candidate, image_path, strategy
        )
        scores.append(score)
        print(f"  Candidate {i+1}/{n}: score = {score:.3f}")

    # 选择最佳
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    best_candidate = candidates[best_idx]
    best_score = scores[best_idx]

    return {
        'best_answer': best_candidate,
        'best_score': best_score,
        'best_index': best_idx,
        'all_candidates': candidates,
        'all_scores': scores
    }

# ========== 使用示例 ==========
if __name__ == "__main__":
    # 1. 加载模型
    print("Loading models...")
    generator, reward_model, processor = load_models(
        generator_path="URSA-MATH/URSA-8B",
        reward_model_path="URSA-MATH/URSA-RM-8B",
        num_gpus=1
    )

    # 2. Best-of-N推理
    question = "What is the area of the triangle in the figure?"
    image_path = "path/to/triangle.png"

    result = best_of_n_inference(
        generator=generator,
        reward_model=reward_model,
        processor=processor,
        question=question,
        image_path=image_path,
        n=32,  # 生成32个候选
        temperature=0.7,  # 较高温度增加多样性
        strategy="min"  # 使用最保守策略
    )

    # 3. 输出结果
    print("\n" + "="*50)
    print("BEST-OF-N RESULTS")
    print("="*50)
    print(f"\nBest Answer (Index: {result['best_index']}, Score: {result['best_score']:.3f}):")
    print(result['best_answer'])

    print(f"\n\nScore Distribution:")
    print(f"  Min: {min(result['all_scores']):.3f}")
    print(f"  Max: {max(result['all_scores']):.3f}")
    print(f"  Avg: {sum(result['all_scores'])/len(result['all_scores']):.3f}")

    # 4. 显示top-3候选
    print(f"\n\nTop-3 Candidates:")
    sorted_indices = sorted(range(len(result['all_scores'])),
                          key=lambda i: result['all_scores'][i],
                          reverse=True)

    for rank, idx in enumerate(sorted_indices[:3], 1):
        print(f"\n--- Rank {rank} (Score: {result['all_scores'][idx]:.3f}) ---")
        print(result['all_candidates'][idx][:200] + "...")  # 只显示前200字符
```

**依赖安装**:
```bash
pip install vllm transformers torch pillow
```

**性能提示**:
- Best-of-32需要较大显存（建议8×A100或4×A100）
- 可以先用Best-of-8测试（n=8）
- 使用`strategy="min"`更保守，`strategy="avg"`更激进

---

## 代码出处索引

### 核心模型文件

| 文件路径 | 主要内容 | 关键类/函数 |
|---------|---------|-----------|
| `models/ursa_model/modeling_ursa.py` | 模型定义 | `UrsaForConditionalGeneration`, `UrsaForTokenClassification` |
| `models/ursa_model/configuration_ursa.py` | 配置类 | `UrsaConfig`, `VisionConfig`, `AlignerConfig` |
| `models/ursa_model/clip_encoder.py` | 视觉编码器 | `HybridVisionTower`, `CLIPVisionTower` |
| `models/ursa_model/projector.py` | 投影器 | `MlpProjector` |
| `models/ursa_model/processing_ursa.py` | 数据处理 | `UrsaProcessor` |

### 推理脚本

| 文件路径 | 主要内容 | 关键函数 |
|---------|---------|---------|
| `inference/vllm_infer.py` | 生成推理 | `prepare_data`, `extract_answer_try_all_methods` |
| `inference/prm_infer_score.py` | PRM评分 | `replace_specific_plus_minus_with_ki`, `single_inference` |

### 关键代码段位置

#### modeling_ursa.py

- **第78-96行**: `UrsaForConditionalGeneration.__init__` - 模型初始化
- **第125-201行**: `_merge_input_ids_with_image_features` - 视觉文本合并
- **第203-330行**: `UrsaForConditionalGeneration.forward` - 前向传播
- **第389-650行**: `UrsaForTokenClassification` - 奖励模型定义

#### clip_encoder.py

- **HybridVisionTower类**: 混合视觉塔实现
- **forward方法**: 双分支视觉编码

#### vllm_infer.py

- **第27-33行**: Prompt模板定义
- **第38-79行**: `extract_answer_try_all_methods` - 答案提取
- **第92-150行**: `prepare_data` - 数据准备

#### prm_infer_score.py

- **第29行**: PRM的Prompt定义
- **第31-41行**: `return_score` - 分数计算策略
- **第43-66行**: `replace_specific_plus_minus_with_ki` - 插入标记
- **第91-100行**: `single_inference` - 单样本推理

---

## 常见问题

### Q1: 如何处理显存不足？

**方法1**: 使用量化
```python
llm = LLM(
    model=model_path,
    quantization="awq",  # 或 "gptq"
    ...
)
```

**方法2**: 减少batch size
```python
sampling_params = SamplingParams(
    n=8,  # 从32减少到8
    ...
)
```

**方法3**: 使用CPU offload
```python
reward_model = AutoModel.from_pretrained(
    model_path,
    device_map="auto",  # 自动分配到CPU/GPU
    ...
)
```

### Q2: 如何加速推理？

**方法1**: 使用多GPU
```python
llm = LLM(
    model=model_path,
    tensor_parallel_size=4,  # 使用4个GPU
    ...
)
```

**方法2**: 批量推理
```python
# 一次处理多个样本
outputs = llm.generate(batch_inputs, sampling_params)
```

**方法3**: 降低精度
```python
reward_model = AutoModel.from_pretrained(
    model_path,
    torch_dtype=torch.float16,  # 使用fp16
    ...
)
```

### Q3: 如何自定义Prompt？

修改模板即可：
```python
CUSTOM_TEMPLATE = """<|im_start|>system
You are a math expert. Solve problems step by step.<|im_end|>
<|im_start|>user
<|image|>
Problem: {}
Please provide detailed reasoning.<|im_end|>
<|im_start|>assistant
"""
```

### Q4: 如何处理没有图片的纯文本题？

```python
# 不提供multi_modal_data
outputs = llm.generate(
    {'prompt': prompt},  # 不包含 multi_modal_data
    sampling_params
)
```

### Q5: 如何保存和加载中间结果？

```python
import json

# 保存候选答案
with open('candidates.json', 'w') as f:
    json.dump({
        'candidates': candidates,
        'scores': scores
    }, f, indent=2)

# 加载
with open('candidates.json', 'r') as f:
    data = json.load(f)
    candidates = data['candidates']
    scores = data['scores']
```

---

## 总结

本文档提供了URSA-MATH项目的所有关键代码示例，包括：

1. **模型加载**: URSA-8B和URSA-RM-8B的独立加载代码
2. **数据集加载**: MMathCoT-1M和DualMath-1.1M的使用方法
3. **完整推理**: 从输入到输出的完整流程
4. **Best-of-N采样**: 结合生成和评分的高级推理
5. **代码出处**: 每个代码段在原仓库中的位置

所有代码都是独立可运行的，不依赖URSA仓库的内部import，可以直接复制到GPU环境运行。

**下一步建议**:
1. 先运行单样本推理，熟悉基本流程
2. 尝试批量推理，提高效率
3. 实验Best-of-N采样，观察性能提升
4. 在自己的数据集上测试模型效果
