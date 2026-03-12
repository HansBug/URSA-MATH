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

### 3.4 完整的 datasets 路径配置（含软链）

跑完前面的下载、解压、软链和本文后面的 example/validation 命令后，推荐你把 `datasets/URSA-MATH` 维持成下面这个结构。这里把软链显式标出来了：

```text
datasets/URSA-MATH/
├── DualMath-1.1M/
│   ├── .cache/
│   │   └── huggingface/
│   ├── README.md
│   └── train.jsonl
├── MMathCoT-1M/
│   ├── .cache/
│   │   └── huggingface/
│   ├── DataEngine_Geometry.7z
│   ├── Geo170K.7z
│   ├── Mavis_Extra.7z
│   ├── README.md
│   ├── VarsityTutors.7z
│   └── train.jsonl
├── _example_manifests/
│   ├── mmathcot_mathvista_compat.jsonl
│   ├── mmathcot_prm_compat.jsonl
│   └── mmathcot_prm_pair_compat.jsonl
├── _loader_validation/                  # 跑 examples/validate_dataset_entrypoints.py 后生成
│   ├── mmathcot_mathvista_compat_sample0.jsonl
│   ├── mmathcot_prm_pair_sample0_abs_image.jsonl
│   ├── mmathcot_prm_pair_sample0_abs_image_avg_sampling8.pt
│   ├── mmathcot_prm_pair_sample0_abs_image_min_sampling8.pt
│   └── vllm_policy_stub.jsonl
└── images/
    ├── DataEngine_Geometry/
    │   └── rule_base_geo_vision_dom/
    ├── DataEngine_Geometry.extract.ok
    ├── Geo170K/
    │   ├── geo3k/
    │   └── geoqa_plus/
    ├── Geo170K.extract.ok
    ├── Mavis_Extra/
    │   ├── function_wo/
    │   ├── geo_cap_to_question/
    │   └── meta_gen/
    ├── Mavis_Extra.extract.ok
    ├── VarsityTutors/
    ├── VarsityTutors.extract.ok
    ├── data_images/                     # MathV360K 上游 zip 解压出的真实目录
    ├── MathV-360k -> data_images/       # 软链，给 raw image_url 前缀用
    ├── MathV-360k.extract.ok
    ├── RGB_images/                      # multimath-300k 上游 zip 解压出的真实目录
    ├── Multimath/
    │   └── RGB_images -> ../RGB_images/ # 软链，给 raw image_url 前缀用
    └── Multimath.extract.ok
```

两个最容易配错的地方：

- `MathV-360k` 必须是一个指向 `data_images/` 的软链。
- `Multimath/RGB_images` 必须是一个指向 `../RGB_images/` 的软链。

所以 policy 侧传给 loader 的 `--image_root` 必须是：

```text
datasets/URSA-MATH/images
```

而不是下面这些错误写法：

- `datasets/URSA-MATH/images/data_images`
- `datasets/URSA-MATH/images/MathV-360k`
- `datasets/URSA-MATH/images/RGB_images`
- `datasets/URSA-MATH/images/Multimath`

如果你想确认自己的目录是不是这个形状，直接跑：

```bash
tree -a -L 3 -F datasets/URSA-MATH
```

你至少应该看到这两行软链：

```text
datasets/URSA-MATH/images/MathV-360k -> data_images/
datasets/URSA-MATH/images/Multimath/RGB_images -> ../RGB_images/
```

如果你不想用软链，也可以改成实体目录搬运：

```text
mv datasets/URSA-MATH/images/data_images datasets/URSA-MATH/images/MathV-360k
mkdir -p datasets/URSA-MATH/images/Multimath
mv datasets/URSA-MATH/images/RGB_images datasets/URSA-MATH/images/Multimath/RGB_images
```

但这种做法会失去上游原始目录名，后面排查问题时不如软链清晰，所以默认仍然建议软链。

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

### 5.1 先生成兼容 manifest

仓库里已经放了一个 example 脚本：

- [examples/run_dataset_loading_example.py](/home/ubuntu/URSA-MATH/examples/run_dataset_loading_example.py)

它现在会做 5 件事：

1. 读取 `MMathCoT-1M` 和 `DualMath-1.1M` 的 raw 样本。
2. 生成 policy 兼容 manifest：`mmathcot_mathvista_compat.jsonl`
3. 生成 PRM 单样本 manifest：`mmathcot_prm_compat.jsonl`
4. 生成 PRM 正负样本对 manifest：`mmathcot_prm_pair_compat.jsonl`
5. 直接调用现有 `prepare_data()` 和 `prepare_input()`，把 loader 真正吃到的数据打印出来。

直接运行：

```bash
python examples/run_dataset_loading_example.py \
  --image-root datasets/URSA-MATH/images
```

如果你还没放真实图片，它会回退到仓库自带的 `figures/framework.png`；如果已经有真实图片，`policy_loader_demo.used_fallback_image` 应该是 `false`。

这里要特别注意：

- policy manifest 里的 `image` 仍然是相对 `image_root` 的 raw 路径，比如 `Mavis_Extra/meta_gen/...png`
- PRM manifest 里的 `image` 必须是能被 `Image.open()` 直接打开的完整路径，不能再只写 raw 相对路径

### 5.2 policy 侧：借用 `mathvista` loader

现有 [`inference/vllm_infer.py`](/home/ubuntu/URSA-MATH/inference/vllm_infer.py) 的 `mathvista` 分支只要求每条数据至少有：

- `question`
- `image`

兼容格式示例：

```json
{"question":"...","image":"MathV-360k/Geometry3K/images/454.png","ground_truth":"A","policy_instruction":"you are given a math problem image, please solve the problem step by step.\nQuestion:...","reference_output":"Step 1: ...\n†Answer: A"}
```

只验证 loader 时，可以直接走现有 `prepare_data()`：

```python
from inference.vllm_infer import prepare_data
input_data, origin_data = prepare_data(
    dataset="mathvista",
    data_path="datasets/URSA-MATH/_example_manifests/mmathcot_mathvista_compat.jsonl",
    image_root="datasets/URSA-MATH/images",
)
```

真实入口点命令本身不需要改：

```bash
CUDA_VISIBLE_DEVICES=0 python inference/vllm_infer.py \
  --model ./checkpoints/URSA-8B \
  --dataset mathvista \
  --data_path datasets/URSA-MATH/_example_manifests/mmathcot_mathvista_compat.jsonl \
  --image_root datasets/URSA-MATH/images \
  --output_file datasets/URSA-MATH/_loader_validation/vllm_policy_real.jsonl \
  --temperature 0 \
  --max_tokens 128 \
  --num_return_sequences 1
```

2026-03-12 在这台机器上的实测结论是：

- `prepare_data()` 能正确读到 `datasets/URSA-MATH/images/Mavis_Extra/meta_gen/textbook_collect_1220-100_170.png`
- 真正失败点不在数据，而在 vLLM 运行时环境：模型权重加载后报 `torch.ops._C.rms_norm` / `torch.ops._C_cache_ops.reshape_and_cache` 缺失
- 所以如果你看到这两个报错，含义是「数据已经吃到了，但本地 vLLM 自定义算子环境坏了」，不是 `datasets/` 目录配错了

这一条样本的正确语义结果应该是：

- `ground_truth == "A"`
- 如果真实模型能正常跑通，`extraction[0]` 也应该抽到 `"A"`
- `model_answer[0]` 的思维链不要求逐字匹配，但最后抽取出的选项应该是 `A`

### 5.3 PRM 侧：直接复用 `prm_infer_score.py` 的 `.jsonl` 输入

现有 [`inference/prm_infer_score.py`](/home/ubuntu/URSA-MATH/inference/prm_infer_score.py) 的 `.jsonl` 路径要求每条记录至少有：

- `input`
- `image`
- `label`

其中 `input` 的构造方式不要自己重写，直接调用现有的 `prepare_input(question, response)`。

这里有 3 个文档里最容易漏掉的细节：

1. 从仓库根目录直接跑 `python inference/prm_infer_score.py` 时，需要补 `PYTHONPATH=.`，否则会报 `ModuleNotFoundError: No module named 'models'`
2. `.jsonl` 分支不会使用 `--image_root` 去拼图片路径，它会直接 `Image.open(data[i]["image"])`
3. `--output_path` 只是文件名前缀，真正落盘的是 `*_avg_sampling8.pt`、`*_min_sampling8.pt` 这类派生文件

所以兼容格式示例应该写成这样，`image` 直接给完整路径：

```json
{"input":"You are given a problem and a step-by-step solution. You need to check the correctness of each step.\nQuestion:...\nStep 1: ... и\nStep 2: ... и\n†Answer: A","image":"/abs/path/to/datasets/URSA-MATH/images/MathV-360k/Geometry3K/images/454.png","label":1}
```

如果你要做「结果是否正确」的验证，最稳妥的做法不是只喂 1 条样本，而是给同一题喂 2 个候选解答：

- 第 1 条是原始正确解，`label=1`
- 第 2 条是手工改坏的错误解，`label=0`
- 然后设 `--best_of_n 2`

真实打分命令：

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python inference/prm_infer_score.py \
  --dataset_name custom \
  --data_path datasets/URSA-MATH/_loader_validation/mmathcot_prm_pair_sample0_abs_image.jsonl \
  --model_path ./checkpoints/URSA-RM-8B \
  --dtype torch.bfloat16 \
  --output_path datasets/URSA-MATH/_loader_validation/mmathcot_prm_pair_sample0_abs_image.pt \
  --image_root datasets/URSA-MATH/images \
  --cuda_device 0 \
  --cuda_sum 1 \
  --best_of_n 2
```

2026-03-12 在这台机器上的实测结果是：

- 命令可以正常跑完
- 会生成 8 个结果文件：`*_avg_sampling8.pt`、`*_avg_sampling16.pt`、`*_avg_sampling32.pt`、`*_avg_sampling64.pt`、`*_min_sampling8.pt`、`*_min_sampling16.pt`、`*_min_sampling32.pt`、`*_min_sampling64.pt`
- 这 8 个文件里的 tensor 全部都是 `[1]`

为什么 `[1]` 才是对的：

- 我们把正样本放在第 1 条，并把它的 `label` 设成了 `1`
- 把错误样本放在第 2 条，并把它的 `label` 设成了 `0`
- 所以只要 PRM 真正偏向正确解，它选出来的 label 就应该是 `1`

### 5.4 一键实测脚本

为了避免你手工拼命令，我把上面的入口点验证固化成了一个脚本：

- [examples/validate_dataset_entrypoints.py](/home/ubuntu/URSA-MATH/examples/validate_dataset_entrypoints.py)

直接运行：

```bash
python examples/validate_dataset_entrypoints.py --policy-mode auto
```

它会按下面的顺序做事：

1. 从 `MMathCoT-1M` 取第 0 条样本
2. 生成 policy manifest 和 PRM 正负样本对 manifest
3. 先真实调用 `python inference/vllm_infer.py`
4. 如果真实 vLLM 因自定义算子缺失失败，再用 stub 后端重新调用同一个入口点，只验证 loader 是否吃到了数据
5. 真实调用 `PYTHONPATH=. python inference/prm_infer_score.py`
6. 检查 policy 输出和 PRM `.pt` 的内容是否符合预期

正确结果应该怎么看：

- `policy.status == "passed_with_stub_backend"` 说明 loader 路径是通的，但本机 vLLM 环境坏了
- `policy.stub_backend.validated_output.extraction == ["A"]` 说明 policy 入口点已经把这条样本的 manifest、图片和原始字段都吃进去了
- `prm.status == "passed"` 且所有 `selected_labels` 都是 `[1]`，说明 PRM 入口点已经把正负样本对吃进去，并且打分方向是对的

如果你只想单独复核输出，也可以直接跑：

```bash
python - <<'PY'
import jsonlines
import torch
from pathlib import Path

policy_path = Path("datasets/URSA-MATH/_loader_validation/vllm_policy_stub.jsonl")
with policy_path.open("r", encoding="utf-8") as f:
    row = next(iter(jsonlines.Reader(f)))
print("policy.extraction =", row["extraction"])
print("policy.keys =", sorted(row.keys()))

for path in sorted(Path("datasets/URSA-MATH/_loader_validation").glob("*_sampling*.pt")):
    print(path.name, torch.load(path, weights_only=True).tolist())
PY
```

## 6. 一句话判断是否加载成功

最短的判断标准就是：

- example 输出里 `policy_loader_demo.loaded_image_size` 不是空
- example 输出里 `prm_loader_demo.resolved_image_path` 指向真实图片
- `python examples/validate_dataset_entrypoints.py --policy-mode auto` 输出里 `policy.stub_backend.validated_output.extraction == ["A"]`
- 同一个输出里 `prm.result.validated_output.selected_labels` 全部是 `[1]`

满足这 4 条，就可以认为：

- raw 数据读到了
- 兼容 manifest 生成对了
- 现有 policy loader 确实吃到了图片和文本
- 现有 PRM 入口点也确实吃到了图片和文本
- 并且 PRM 在这条正负样本对上的选择方向是正确的
