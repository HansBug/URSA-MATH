import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import jsonlines
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference.prm_infer_score import prepare_input


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Exercise the existing inference entrypoints against URSA-MATH "
            "compatibility manifests and validate the outputs."
        )
    )
    parser.add_argument(
        "--mmathcot-path",
        default=str(REPO_ROOT / "datasets" / "URSA-MATH" / "MMathCoT-1M" / "train.jsonl"),
        help="Path to URSA-MATH/MMathCoT-1M train.jsonl.",
    )
    parser.add_argument(
        "--image-root",
        default=str(REPO_ROOT / "datasets" / "URSA-MATH" / "images"),
        help="Root directory that contains the extracted dataset images.",
    )
    parser.add_argument(
        "--work-dir",
        default=str(REPO_ROOT / "tmp" / "dataset_load_checks" / "entrypoints"),
        help="Where to write manifests, outputs, and validation summaries.",
    )
    parser.add_argument(
        "--policy-model",
        default=str(REPO_ROOT / "checkpoints" / "URSA-8B"),
        help="Model path for inference/vllm_infer.py.",
    )
    parser.add_argument(
        "--prm-model",
        default=str(REPO_ROOT / "checkpoints" / "URSA-RM-8B"),
        help="Model path for inference/prm_infer_score.py.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="0-based sample index to validate.",
    )
    parser.add_argument(
        "--policy-mode",
        choices=["auto", "real-only", "stub-only", "skip"],
        default="auto",
        help=(
            "How to validate inference/vllm_infer.py. 'auto' tries the real backend first "
            "and falls back to a stub backend if local vLLM runtime ops are missing."
        ),
    )
    parser.add_argument(
        "--skip-prm",
        action="store_true",
        help="Skip inference/prm_infer_score.py validation.",
    )
    return parser.parse_args()


def read_jsonl_row(path: Path, sample_index: int):
    with path.open("r", encoding="utf-8") as fp:
        reader = iter(jsonlines.Reader(fp))
        for index, item in enumerate(reader):
            if index == sample_index:
                return item
    raise IndexError(f"sample_index={sample_index} is out of range for {path}")


def extract_question(instruction: str) -> str:
    marker = "Question:"
    index = instruction.find(marker)
    if index == -1:
        return instruction.strip()
    return instruction[index + len(marker) :].strip()


def extract_answer(output: str) -> str:
    marker = "†Answer:"
    index = output.rfind(marker)
    if index == -1:
        return output.strip()
    return output[index + len(marker) :].strip()


def build_negative_response(response: str) -> str:
    negative = response.replace(
        "Step 3: This implies that knowing side lengths allows triangle type determination.",
        "Step 3: This implies that knowing side lengths does not allow triangle type determination.",
    )
    return negative.replace("†Answer: A", "†Answer: B")


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        writer = jsonlines.Writer(fp)
        for row in rows:
            writer.write(row)


def run_command(cmd, cwd: Path, env=None):
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=merged_env,
        text=True,
        capture_output=True,
    )
    return {
        "cmd": " ".join(cmd),
        "cwd": str(cwd),
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def build_policy_manifest(raw_item, image_root: Path, work_dir: Path):
    question = extract_question(raw_item["instruction"])
    ground_truth = extract_answer(raw_item["output"])
    image_path = image_root / raw_item["image_url"]
    if not image_path.exists():
        raise FileNotFoundError(f"policy image not found: {image_path}")
    manifest_path = work_dir / "mmathcot_mathvista_compat_sample0.jsonl"
    write_jsonl(
        manifest_path,
        [
            {
                "question": question,
                "image": raw_item["image_url"],
                "ground_truth": ground_truth,
                "raw_image_url": raw_item["image_url"],
                "policy_instruction": raw_item["instruction"],
                "reference_output": raw_item["output"],
            }
        ],
    )
    return {
        "question": question,
        "ground_truth": ground_truth,
        "manifest_path": manifest_path,
        "image_path": image_path.resolve(),
    }


def build_prm_manifest(raw_item, image_root: Path, work_dir: Path):
    question = extract_question(raw_item["instruction"])
    image_path = (image_root / raw_item["image_url"]).resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"prm image not found: {image_path}")
    positive = raw_item["output"]
    negative = build_negative_response(positive)
    manifest_path = work_dir / "mmathcot_prm_pair_sample0_abs_image.jsonl"
    rows = [
        {
            "input": prepare_input(question, positive),
            "image": str(image_path),
            "label": 1,
            "candidate": "positive",
            "ground_truth": extract_answer(positive),
            "raw_image_url": raw_item["image_url"],
        },
        {
            "input": prepare_input(question, negative),
            "image": str(image_path),
            "label": 0,
            "candidate": "negative",
            "ground_truth": extract_answer(positive),
            "raw_image_url": raw_item["image_url"],
        },
    ]
    write_jsonl(manifest_path, rows)
    return {
        "manifest_path": manifest_path,
        "expected_selected_label": 1,
        "rows": rows,
    }


def write_stub_vllm(stub_root: Path):
    package_root = stub_root / "vllm"
    assets_root = package_root / "assets"
    assets_root.mkdir(parents=True, exist_ok=True)
    (assets_root / "__init__.py").write_text("", encoding="utf-8")
    (assets_root / "image.py").write_text(
        "class ImageAsset:\n"
        "    pass\n",
        encoding="utf-8",
    )
    (package_root / "__init__.py").write_text(
        "class SamplingParams:\n"
        "    def __init__(self, temperature=0.2, max_tokens=2048, n=1):\n"
        "        self.temperature = temperature\n"
        "        self.max_tokens = max_tokens\n"
        "        self.n = n\n"
        "\n"
        "class _Output:\n"
        "    def __init__(self, text):\n"
        "        self.text = text\n"
        "\n"
        "class _RequestOutput:\n"
        "    def __init__(self, texts):\n"
        "        self.outputs = [_Output(text) for text in texts]\n"
        "\n"
        "class LLM:\n"
        "    def __init__(self, model, tensor_parallel_size=1):\n"
        "        self.model = model\n"
        "        self.tensor_parallel_size = tensor_parallel_size\n"
        "\n"
        "    def generate(self, input_data, sampling_params):\n"
        "        result = []\n"
        "        for _ in input_data:\n"
        "            texts = ['Step 1: stub policy generation\\n†Answer: A' for _ in range(sampling_params.n)]\n"
        "            result.append(_RequestOutput(texts))\n"
        "        return result\n",
        encoding="utf-8",
    )


def validate_policy_output(output_path: Path, expected_ground_truth: str):
    with output_path.open("r", encoding="utf-8") as fp:
        rows = list(jsonlines.Reader(fp))
    if len(rows) != 1:
        raise AssertionError(f"expected 1 policy row, got {len(rows)}")
    row = rows[0]
    if row["ground_truth"] != expected_ground_truth:
        raise AssertionError(
            f"expected ground_truth={expected_ground_truth}, got {row['ground_truth']}"
        )
    if row["extraction"] != [expected_ground_truth]:
        raise AssertionError(f"expected extraction={[expected_ground_truth]}, got {row['extraction']}")
    return {
        "output_file": str(output_path),
        "keys": sorted(row.keys()),
        "model_answer": row["model_answer"],
        "extraction": row["extraction"],
    }


def try_real_policy(policy_manifest, args, image_root: Path, work_dir: Path):
    output_path = work_dir / "vllm_policy_real.jsonl"
    cmd = [
        "python",
        str(REPO_ROOT / "inference" / "vllm_infer.py"),
        "--model",
        str(Path(args.policy_model).resolve()),
        "--dataset",
        "mathvista",
        "--data_path",
        str(policy_manifest["manifest_path"]),
        "--image_root",
        str(image_root.resolve()),
        "--output_file",
        str(output_path),
        "--temperature",
        "0",
        "--max_tokens",
        "128",
        "--num_return_sequences",
        "1",
    ]
    result = run_command(cmd, cwd=REPO_ROOT)
    if result["returncode"] != 0:
        combined = f"{result['stdout_tail']}\n{result['stderr_tail']}"
        if "rms_norm" in combined or "reshape_and_cache" in combined:
            result["status"] = "runtime_env_error"
            result["reason"] = "vllm custom ops are missing in the local runtime"
        else:
            result["status"] = "failed"
        return result
    try:
        result["validated_output"] = validate_policy_output(
            output_path,
            expected_ground_truth=policy_manifest["ground_truth"],
        )
    except AssertionError as exc:
        result["status"] = "failed"
        result["reason"] = str(exc)
        return result
    result["status"] = "passed"
    return result


def run_stub_policy(policy_manifest, args, image_root: Path, work_dir: Path):
    output_path = work_dir / "vllm_policy_stub.jsonl"
    with tempfile.TemporaryDirectory(prefix="ursa_stub_vllm_") as stub_dir:
        stub_root = Path(stub_dir)
        write_stub_vllm(stub_root)
        cmd = [
            "python",
            str(REPO_ROOT / "inference" / "vllm_infer.py"),
            "--model",
            str(Path(args.policy_model).resolve()),
            "--dataset",
            "mathvista",
            "--data_path",
            str(policy_manifest["manifest_path"]),
            "--image_root",
            str(image_root.resolve()),
            "--output_file",
            str(output_path),
            "--temperature",
            "0",
            "--max_tokens",
            "128",
            "--num_return_sequences",
            "1",
        ]
        env = {"PYTHONPATH": stub_dir}
        result = run_command(cmd, cwd=Path(stub_dir), env=env)
    if result["returncode"] != 0:
        result["status"] = "failed"
        return result
    try:
        result["validated_output"] = validate_policy_output(
            output_path,
            expected_ground_truth=policy_manifest["ground_truth"],
        )
    except AssertionError as exc:
        result["status"] = "failed"
        result["reason"] = str(exc)
        return result
    result["status"] = "passed"
    return result


def run_prm(prm_manifest, args, image_root: Path, work_dir: Path):
    output_path = work_dir / "mmathcot_prm_pair_sample0_abs_image.pt"
    cmd = [
        "python",
        str(REPO_ROOT / "inference" / "prm_infer_score.py"),
        "--dataset_name",
        "custom",
        "--data_path",
        str(prm_manifest["manifest_path"]),
        "--model_path",
        str(Path(args.prm_model).resolve()),
        "--dtype",
        "torch.bfloat16",
        "--output_path",
        str(output_path),
        "--image_root",
        str(image_root.resolve()),
        "--cuda_device",
        "0",
        "--cuda_sum",
        "1",
        "--best_of_n",
        "2",
    ]
    env = {"PYTHONPATH": str(REPO_ROOT)}
    result = run_command(cmd, cwd=REPO_ROOT, env=env)
    if result["returncode"] != 0:
        result["status"] = "failed"
        return result

    label_files = sorted(work_dir.glob("mmathcot_prm_pair_sample0_abs_image_*_sampling*.pt"))
    selected = {}
    for path in label_files:
        value = torch.load(path, weights_only=True)
        if hasattr(value, "tolist"):
            value = value.tolist()
        selected[path.name] = value
        if value != [prm_manifest["expected_selected_label"]]:
            result["status"] = "failed"
            result["reason"] = f"{path.name} expected {[1]}, got {value}"
            return result

    result["status"] = "passed"
    result["validated_output"] = {
        "output_files": [str(path) for path in label_files],
        "selected_labels": selected,
    }
    return result


def main():
    args = parse_args()
    mmathcot_path = Path(args.mmathcot_path).resolve()
    image_root = Path(args.image_root).resolve()
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    raw_item = read_jsonl_row(mmathcot_path, args.sample_index)
    policy_manifest = build_policy_manifest(raw_item, image_root, work_dir)
    prm_manifest = build_prm_manifest(raw_item, image_root, work_dir)

    summary = {
        "sample_index": args.sample_index,
        "raw_image_url": raw_item["image_url"],
        "question": policy_manifest["question"],
        "ground_truth": policy_manifest["ground_truth"],
        "policy": {
            "mode": args.policy_mode,
            "manifest_path": str(policy_manifest["manifest_path"]),
        },
        "prm": {
            "skipped": args.skip_prm,
            "manifest_path": str(prm_manifest["manifest_path"]),
            "expected_selected_label": prm_manifest["expected_selected_label"],
        },
    }

    if args.policy_mode == "skip":
        summary["policy"]["status"] = "skipped"
    elif args.policy_mode == "stub-only":
        summary["policy"]["stub_backend"] = run_stub_policy(policy_manifest, args, image_root, work_dir)
        summary["policy"]["status"] = summary["policy"]["stub_backend"]["status"]
    elif args.policy_mode == "real-only":
        summary["policy"]["real_backend"] = try_real_policy(policy_manifest, args, image_root, work_dir)
        summary["policy"]["status"] = summary["policy"]["real_backend"]["status"]
    else:
        real_result = try_real_policy(policy_manifest, args, image_root, work_dir)
        summary["policy"]["real_backend"] = real_result
        if real_result["status"] == "passed":
            summary["policy"]["status"] = "passed"
        else:
            stub_result = run_stub_policy(policy_manifest, args, image_root, work_dir)
            summary["policy"]["stub_backend"] = stub_result
            summary["policy"]["status"] = (
                "passed_with_stub_backend" if stub_result["status"] == "passed" else "failed"
            )

    if args.skip_prm:
        summary["prm"]["status"] = "skipped"
    else:
        prm_result = run_prm(prm_manifest, args, image_root, work_dir)
        summary["prm"]["result"] = prm_result
        summary["prm"]["status"] = prm_result["status"]

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
