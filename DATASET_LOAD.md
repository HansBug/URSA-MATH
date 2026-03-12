# Dataset Load Guide

这份文档只做两件事：

1. 把 URSA 相关数据下到仓库里的固定位置。
2. 在 **不修改** [`inference/`](/home/ubuntu/URSA-MATH/inference) 和 [`models/`](/home/ubuntu/URSA-MATH/models) 的前提下，让现有加载逻辑直接吃到兼容格式的数据。

## 0. 先说结论

- stage3 的 policy / RL 数据源是 `MMathCoT-1M`。
- `DualMath-1.1M` 是 PRM 训练数据，不是 stage3 policy rollout 的直接输入。
- 现有 [`inference/vllm_infer.py`](/home/ubuntu/URSA-MATH/inference/vllm_infer.py) 不认识 URSA raw 的 `image_url / instruction / output` 三列。
- 想复用现有 inference loader，最省事的办法是先生成一层兼容 manifest：
  - policy 侧复用 `mathvista` 分支，只需要 `question` 和 `image`
  - PRM 侧复用 [`inference/prm_infer_score.py`](/home/ubuntu/URSA-MATH/inference/prm_infer_score.py) 的 `.jsonl` 分支，需要 `input`、`image`、`label`

## 1. 目录约定

推荐固定成下面这个结构：

```text
datasets/URSA-MATH/
  MMathCoT-1M/
    README.md
    train.jsonl
    DataEngine_Geometry.7z
    Geo170K.7z
    Mavis_Extra.7z
    VarsityTutors.7z
  DualMath-1.1M/
    README.md
    train.jsonl
  images/
    MathV-360k/...
    Multimath/...
    DataEngine_Geometry/...
    Geo170K/...
    Mavis_Extra/...
    VarsityTutors/...
  _example_manifests/
    mmathcot_mathvista_compat.jsonl
    mmathcot_prm_compat.jsonl
```

关键点只有一个：`images/` 下面最终暴露出来的路径必须和 raw `image_url` 一致，也就是下面这 6 个前缀都要能直接命中：

- `MathV-360k`
- `Multimath`
- `DataEngine_Geometry`
- `Geo170K`
- `Mavis_Extra`
- `VarsityTutors`

## 2. 下载 raw 标注

下面的命令会把 URSA 自己放在 Hugging Face 上的两个 raw jsonl 拉到仓库里：

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="URSA-MATH/MMathCoT-1M",
    repo_type="dataset",
    local_dir="datasets/URSA-MATH/MMathCoT-1M",
    allow_patterns=["README.md", "train.jsonl"],
)

snapshot_download(
    repo_id="URSA-MATH/DualMath-1.1M",
    repo_type="dataset",
    local_dir="datasets/URSA-MATH/DualMath-1.1M",
    allow_patterns=["README.md", "train.jsonl"],
)
PY
```

执行完后，本地应该至少有：

- [datasets/URSA-MATH/MMathCoT-1M/train.jsonl](/home/ubuntu/URSA-MATH/datasets/URSA-MATH/MMathCoT-1M/train.jsonl)
- [datasets/URSA-MATH/DualMath-1.1M/train.jsonl](/home/ubuntu/URSA-MATH/datasets/URSA-MATH/DualMath-1.1M/train.jsonl)

## 3. 下载图片资产

### 3.1 先下 URSA 已经打包好的 4 份图片归档

`MMathCoT-1M` 仓库里已经给了 4 份可直接复用的图片包：

- `DataEngine_Geometry.7z`
- `Geo170K.7z`
- `Mavis_Extra.7z`
- `VarsityTutors.7z`

下载命令：

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="URSA-MATH/MMathCoT-1M",
    repo_type="dataset",
    local_dir="datasets/URSA-MATH/MMathCoT-1M",
    allow_patterns=[
        "DataEngine_Geometry.7z",
        "Geo170K.7z",
        "Mavis_Extra.7z",
        "VarsityTutors.7z",
    ],
)
PY
```

### 3.2 解开 4 份 `.7z` 到 `datasets/URSA-MATH/images/`

仓库环境里不要求你装系统 `7z`，直接用 Python 包就行：

```bash
python -m pip install py7zr
python - <<'PY'
from pathlib import Path
import py7zr

src_root = Path("datasets/URSA-MATH/MMathCoT-1M")
dst_root = Path("datasets/URSA-MATH/images")
dst_root.mkdir(parents=True, exist_ok=True)

for name in [
    "DataEngine_Geometry.7z",
    "Geo170K.7z",
    "Mavis_Extra.7z",
    "VarsityTutors.7z",
]:
    archive = src_root / name
    print(f"extracting {archive} -> {dst_root}")
    with py7zr.SevenZipFile(archive, mode="r") as zf:
        zf.extractall(path=dst_root)
PY
```

### 3.3 `MathV-360k` 和 `Multimath` 需要从上游单独下载

URSA raw 里另外两类大头前缀是：

- `MathV-360k/...`
- `Multimath/RGB_images/...`

它们都不在 URSA 自己的 4 个 `.7z` 里，需要从上游数据集补齐：

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Zhiqiang007/MathV360K",
    repo_type="dataset",
    local_dir="datasets/_sources/MathV360K",
    allow_patterns=["data_images.zip"],
)

snapshot_download(
    repo_id="pengshuai-rin/multimath-300k",
    repo_type="dataset",
    local_dir="datasets/_sources/multimath-300k",
    allow_patterns=["images.zip"],
)
PY
```

解压：

```bash
python - <<'PY'
from pathlib import Path
import zipfile

dst_root = Path("datasets/URSA-MATH/images")
dst_root.mkdir(parents=True, exist_ok=True)

for zip_path in [
    Path("datasets/_sources/MathV360K/data_images.zip"),
    Path("datasets/_sources/multimath-300k/images.zip"),
]:
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dst_root)
    print(f"extracted {zip_path} -> {dst_root}")
PY
```

这两个上游 zip 的原始顶层目录名和 URSA raw `image_url` 并不一致：

- `MathV360K` 解出来是 `data_images/...`
- `multimath-300k` 解出来是 `RGB_images/...`

所以还要补一层映射，让最终路径满足：

```text
datasets/URSA-MATH/images/MathV-360k/...
datasets/URSA-MATH/images/Multimath/RGB_images/...
```

直接执行下面这段即可：

```bash
python - <<'PY'
from pathlib import Path

root = Path("datasets/URSA-MATH/images")

mathv_link = root / "MathV-360k"
if not mathv_link.exists():
    mathv_link.symlink_to("data_images")
    print(f"created symlink: {mathv_link} -> data_images")

multimath_dir = root / "Multimath"
multimath_dir.mkdir(exist_ok=True)
multimath_link = multimath_dir / "RGB_images"
if not multimath_link.exists():
    multimath_link.symlink_to("../RGB_images")
    print(f"created symlink: {multimath_link} -> ../RGB_images")
PY
```

### 3.4 完整的 images 路径配置

执行完上面的解压和软链命令后，`datasets/URSA-MATH/images` 的最终结构应该理解成两层：

1. 实际解压出来的物理目录
2. 为了兼容 URSA raw `image_url` 而补的逻辑入口目录

推荐你最终保持下面这个结构：

```text
datasets/URSA-MATH/images/
  data_images/                      # MathV360K 上游 zip 解出来的真实目录
  RGB_images/                       # multimath-300k 上游 zip 解出来的真实目录
  DataEngine_Geometry/
  Geo170K/
  Mavis_Extra/
  VarsityTutors/
  MathV-360k -> data_images
  Multimath/
    RGB_images -> ../RGB_images
```

这里有两个容易踩坑的点：

- `MathV-360k` 在 raw 数据里是顶层前缀，但上游 zip 解出来的真实目录名是 `data_images`
- `Multimath` 在 raw 数据里是顶层前缀，但真正的图片又在它下面一层 `RGB_images`

所以你后续传给现有脚本的 `--image_root` 必须是：

```text
datasets/URSA-MATH/images
```

而不是下面这些错误写法：

- `datasets/URSA-MATH/images/data_images`
- `datasets/URSA-MATH/images/MathV-360k`
- `datasets/URSA-MATH/images/RGB_images`
- `datasets/URSA-MATH/images/Multimath`

原因很简单：现有脚本是把 jsonl 里的 `image` 或 raw `image_url` 直接拼到 `image_root` 后面，所以它期望的是：

```text
image_root / MathV-360k/...
image_root / Multimath/RGB_images/...
image_root / Mavis_Extra/...
```

如果你想一次性把完整目录和软链都补好，可以直接跑下面这段：

```bash
python - <<'PY'
from pathlib import Path

root = Path("datasets/URSA-MATH/images")
root.mkdir(parents=True, exist_ok=True)

# MathV-360k raw 前缀映射到 MathV360K 上游解压目录 data_images/
mathv_link = root / "MathV-360k"
if not mathv_link.exists():
    mathv_link.symlink_to("data_images")
    print(f"created symlink: {mathv_link} -> data_images")

# Multimath raw 前缀映射到 multimath-300k 上游解压目录 RGB_images/
multimath_dir = root / "Multimath"
multimath_dir.mkdir(exist_ok=True)
multimath_link = multimath_dir / "RGB_images"
if not multimath_link.exists():
    multimath_link.symlink_to("../RGB_images")
    print(f"created symlink: {multimath_link} -> ../RGB_images")
PY
```

如果你不想用软链，也可以直接改成实体目录搬运：

```text
mv datasets/URSA-MATH/images/data_images datasets/URSA-MATH/images/MathV-360k
mkdir -p datasets/URSA-MATH/images/Multimath
mv datasets/URSA-MATH/images/RGB_images datasets/URSA-MATH/images/Multimath/RGB_images
```

但这种做法会让你失去上游原始目录名，不如软链清晰，所以默认还是建议软链。

## 4. 验证图片目录和 raw `image_url` 是否对齐

先看 raw 里到底有哪些前缀：

```bash
python - <<'PY'
import jsonlines
from collections import Counter

counter = Counter()
with open("datasets/URSA-MATH/MMathCoT-1M/train.jsonl", "r", encoding="utf-8") as fp:
    for i, item in enumerate(jsonlines.Reader(fp)):
        counter[item["image_url"].split("/")[0]] += 1
        if i >= 20000:
            break

print(counter)
PY
```

再检查这些前缀在本地是不是都存在：

```bash
python - <<'PY'
from pathlib import Path

root = Path("datasets/URSA-MATH/images")
for name in [
    "MathV-360k",
    "Multimath",
    "DataEngine_Geometry",
    "Geo170K",
    "Mavis_Extra",
    "VarsityTutors",
]:
    print(name, (root / name).exists(), root / name)
PY
```

再做一次真正有意义的检查，不是只看顶层目录，而是直接验证 `image_root / image_url` 能不能命中真实文件：

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("datasets/URSA-MATH/images")
jsonl_path = Path("datasets/URSA-MATH/MMathCoT-1M/train.jsonl")
want = {
    "MathV-360k",
    "Multimath",
    "DataEngine_Geometry",
    "Geo170K",
    "Mavis_Extra",
    "VarsityTutors",
}

seen = {}
with jsonl_path.open("r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        item = json.loads(line)
        image_url = item["image_url"]
        prefix = image_url.split("/")[0]
        if prefix in want and prefix not in seen:
            seen[prefix] = {
                "index": idx,
                "image_url": image_url,
                "exists": (root / image_url).exists(),
            }
            if len(seen) == len(want):
                break

print(json.dumps(seen, ensure_ascii=False, indent=2))
PY
```

只要这里 6 个前缀的 `exists` 都是 `true`，就说明你的 images 路径已经配到可以直接给现有脚本用的程度。

## 5. 不改 inference/model，怎么让现有 loader 直接吃到数据

### 5.1 policy 侧：借用 `mathvista` loader

现有 [`inference/vllm_infer.py`](/home/ubuntu/URSA-MATH/inference/vllm_infer.py) 的 `mathvista` 分支只要求每条数据至少有：

- `question`
- `image`

所以最稳妥的做法是把 `MMathCoT-1M` raw 转成一个 `mathvista` 兼容 jsonl。每条最少保留：

```json
{"question":"...","image":"MathV-360k/Geometry3K/images/454.png","ground_truth":"A","policy_instruction":"you are given a math problem image, please solve the problem step by step.\nQuestion:...","reference_output":"Step 1: ...\n†Answer: A"}
```

然后直接走现有 loader：

```python
from inference.vllm_infer import prepare_data
input_data, origin_data = prepare_data(
    dataset="mathvista",
    data_path="datasets/URSA-MATH/_example_manifests/mmathcot_mathvista_compat.jsonl",
    image_root="datasets/URSA-MATH/images",
)
```

真实推理命令也不用改：

```bash
CUDA_VISIBLE_DEVICES=0 python inference/vllm_infer.py \
  --model ./checkpoints/URSA-8B \
  --dataset mathvista \
  --data_path datasets/URSA-MATH/_example_manifests/mmathcot_mathvista_compat.jsonl \
  --image_root datasets/URSA-MATH/images \
  --output_file outputs/mmathcot_demo.jsonl \
  --num_return_sequences 1
```

### 5.2 PRM 侧：直接复用 `prm_infer_score.py` 的 `.jsonl` 输入

现有 [`inference/prm_infer_score.py`](/home/ubuntu/URSA-MATH/inference/prm_infer_score.py) 的 `.jsonl` 路径要求每条记录至少有：

- `input`
- `image`
- `label`

其中 `input` 的构造方式不要自己重写，直接调用现有的 `prepare_input(question, response)`。

兼容格式示例：

```json
{"input":"You are given a problem and a step-by-step solution. You need to check the correctness of each step.\nQuestion:...\nStep 1: ... и Step 2: ... и †Answer: A","image":"MathV-360k/Geometry3K/images/454.png","label":1}
```

真实打分命令：

```bash
CUDA_VISIBLE_DEVICES=0 python inference/prm_infer_score.py \
  --dataset_name custom \
  --data_path datasets/URSA-MATH/_example_manifests/mmathcot_prm_compat.jsonl \
  --model_path ./checkpoints/URSA-RM-8B \
  --output_path outputs/mmathcot_prm_scores.pt \
  --image_root datasets/URSA-MATH/images \
  --cuda_device 0 \
  --cuda_sum 1 \
  --best_of_n 1
```

## 6. 可运行 example

仓库里已经放了一个 example 脚本：

- [examples/run_dataset_loading_example.py](/home/ubuntu/URSA-MATH/examples/run_dataset_loading_example.py)

它会做 4 件事：

1. 读取 `MMathCoT-1M` 和 `DualMath-1.1M` 的 raw 样本。
2. 生成 policy 兼容 manifest：`mmathcot_mathvista_compat.jsonl`
3. 生成 PRM 兼容 manifest：`mmathcot_prm_compat.jsonl`
4. 直接调用现有 `prepare_data()` 和 `prepare_input()`，把加载后的结果打印出来。

直接运行：

```bash
python examples/run_dataset_loading_example.py
```

如果你已经把真实图片放到了 `datasets/URSA-MATH/images`，它会优先使用真实图片；如果还没下图片，它会自动回退到仓库自带的 `figures/framework.png`，这样 example 也能立刻跑通。

指定真实图片根目录的运行方式：

```bash
python examples/run_dataset_loading_example.py \
  --image-root datasets/URSA-MATH/images
```

## 7. 一句话判断是否加载成功

你只需要看 example 输出里这几项：

- `policy_loader_demo.loaded_prompt_preview`
- `policy_loader_demo.loaded_image_size`
- `policy_loader_demo.origin_keys`
- `prm_loader_demo.prepared_input_preview`

只要这四项都正常打印出来，就说明：

- raw 数据读到了
- 兼容 manifest 生成对了
- 现有 inference loader 确实把图片和文本都吃进去了
- PRM 侧输入文本也已经按现有代码要求构造好了
