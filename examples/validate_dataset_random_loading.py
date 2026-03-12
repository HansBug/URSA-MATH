import argparse
import json
import os
import random
import sys
from pathlib import Path

import jsonlines
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference.vllm_infer import prepare_data


EMPTY_QUESTION_FALLBACK = (
    "Please solve the math problem shown in the image step by step and provide the final answer."
)
MISSING_GROUND_TRUTH_FALLBACK = "[missing-ground-truth]"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Validate MMathCoT-1M and DualMath-1.1M for field completeness, "
            "image existence, and loader compatibility."
        )
    )
    parser.add_argument(
        "--mmathcot-path",
        default=str(REPO_ROOT / "datasets" / "URSA-MATH" / "MMathCoT-1M" / "train.jsonl"),
        help="Path to URSA-MATH/MMathCoT-1M train.jsonl.",
    )
    parser.add_argument(
        "--dualmath-path",
        default=str(REPO_ROOT / "datasets" / "URSA-MATH" / "DualMath-1.1M" / "train.jsonl"),
        help="Path to URSA-MATH/DualMath-1.1M train.jsonl.",
    )
    parser.add_argument(
        "--image-root",
        default=str(REPO_ROOT / "datasets" / "URSA-MATH" / "images"),
        help="Root directory that contains the extracted image tree.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10000,
        help="Number of random rows to sample from each dataset when --mode=sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260312,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "tmp" / "dataset_load_checks" / "random_loading"),
        help="Where to write sampled manifests and the validation summary.",
    )
    parser.add_argument(
        "--mode",
        choices=["sample", "all"],
        default="sample",
        help=(
            "'sample' validates a random subset of size --sample-size. "
            "'all' validates every current row in both train.jsonl files."
        ),
    )
    return parser.parse_args()


def extract_question(instruction: str):
    marker = "Question:"
    index = instruction.find(marker)
    if index == -1:
        question = instruction.strip()
        return question, False
    question = instruction[index + len(marker) :].strip()
    if question:
        return question, False
    return EMPTY_QUESTION_FALLBACK, True


def extract_answer(output: str):
    marker = "†Answer:"
    index = output.rfind(marker)
    if index == -1:
        answer = output.strip()
        return answer, False
    answer = output[index + len(marker) :].strip()
    if answer:
        return answer, False
    return MISSING_GROUND_TRUTH_FALLBACK, True


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        writer = jsonlines.Writer(fp)
        for row in rows:
            writer.write(row)


def reservoir_sample_jsonl(path: Path, sample_size: int, rng: random.Random):
    sample = []
    total = 0
    with path.open("r", encoding="utf-8") as fp:
        for index, item in enumerate(jsonlines.Reader(fp)):
            wrapped = {"source_index": index, "row": item}
            total += 1
            if index < sample_size:
                sample.append(wrapped)
                continue
            replace_at = rng.randint(0, index)
            if replace_at < sample_size:
                sample[replace_at] = wrapped
    if total < sample_size:
        raise ValueError(f"{path} only has {total} rows, which is smaller than sample_size={sample_size}")
    sample.sort(key=lambda item: item["source_index"])
    return sample, total


def read_all_jsonl_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as fp:
        for index, item in enumerate(jsonlines.Reader(fp)):
            rows.append({"source_index": index, "row": item})
    return rows, len(rows)


def require_non_empty_str(value, field_name, dataset_name, source_index):
    if not isinstance(value, str) or not value.strip():
        raise AssertionError(
            f"{dataset_name}[{source_index}] field '{field_name}' must be a non-empty string"
        )


def require_exists_isfile(path: Path, dataset_name, source_index):
    if not os.path.exists(path):
        raise AssertionError(f"{dataset_name}[{source_index}] missing image: {path}")
    if not os.path.isfile(path):
        raise AssertionError(f"{dataset_name}[{source_index}] image is not a file: {path}")


def require_openable_image(path: Path, dataset_name, source_index):
    with Image.open(path) as image:
        width, height = image.size
    if width <= 0 or height <= 0:
        raise AssertionError(f"{dataset_name}[{source_index}] invalid image size: {path}")
    return [width, height]


def validate_mmathcot(rows_to_check, image_root: Path, output_dir: Path, manifest_tag: str):
    dataset_name = "MMathCoT-1M"
    compat_rows = []
    image_sizes = []
    empty_question_fallback_rows = []
    missing_ground_truth_rows = []
    for sampled in rows_to_check:
        source_index = sampled["source_index"]
        row = sampled["row"]
        for field_name in ["image_url", "instruction", "output"]:
            require_non_empty_str(row.get(field_name), field_name, dataset_name, source_index)
        image_path = image_root / row["image_url"]
        require_exists_isfile(image_path, dataset_name, source_index)
        image_sizes.append(require_openable_image(image_path, dataset_name, source_index))
        question, used_fallback = extract_question(row["instruction"])
        ground_truth, used_ground_truth_fallback = extract_answer(row["output"])
        require_non_empty_str(question, "question", dataset_name, source_index)
        require_non_empty_str(ground_truth, "ground_truth", dataset_name, source_index)
        if used_fallback:
            empty_question_fallback_rows.append(source_index)
        if used_ground_truth_fallback:
            missing_ground_truth_rows.append(source_index)
        compat_rows.append(
            {
                "question": question,
                "image": row["image_url"],
                "ground_truth": ground_truth,
                "raw_image_url": row["image_url"],
                "policy_instruction": row["instruction"],
                "reference_output": row["output"],
                "source_index": source_index,
            }
        )

    manifest_path = output_dir / f"mmathcot_mathvista_{manifest_tag}.jsonl"
    write_jsonl(manifest_path, compat_rows)
    input_data, origin_data = prepare_data(
        dataset="mathvista",
        data_path=str(manifest_path),
        image_root=str(image_root.resolve()),
    )
    if len(input_data) != len(compat_rows) or len(origin_data) != len(compat_rows):
        raise AssertionError(
            f"{dataset_name} prepare_data length mismatch: "
            f"{len(input_data)} / {len(origin_data)} / {len(compat_rows)}"
        )
    for compat_row, loaded_row, origin_row in zip(compat_rows, input_data, origin_data):
        if origin_row["raw_image_url"] != compat_row["raw_image_url"]:
            raise AssertionError(f"{dataset_name}[{compat_row['source_index']}] origin/raw_image_url mismatch")
        prompt = loaded_row["prompt"]
        require_non_empty_str(prompt, "prompt", dataset_name, compat_row["source_index"])
        if compat_row["question"] not in prompt:
            raise AssertionError(f"{dataset_name}[{compat_row['source_index']}] prompt does not contain question")
        loaded_image = loaded_row["multi_modal_data"]["image"]
        width, height = loaded_image.size
        if width <= 0 or height <= 0:
            raise AssertionError(f"{dataset_name}[{compat_row['source_index']}] loaded image size is invalid")

    return {
        "sampled_rows": len(compat_rows),
        "manifest_path": str(manifest_path),
        "required_fields_checked": [
            "image_url",
            "instruction",
            "output",
            "question",
            "image",
            "ground_truth",
            "raw_image_url",
            "policy_instruction",
            "reference_output",
        ],
        "loader_name": "prepare_data(dataset='mathvista')",
        "image_check": "os.path.exists + os.path.isfile + PIL.Image.open",
        "first_image_size": image_sizes[0],
        "first_source_index": compat_rows[0]["source_index"],
        "empty_question_fallback_count": len(empty_question_fallback_rows),
        "empty_question_fallback_examples": empty_question_fallback_rows[:20],
        "empty_question_fallback_prompt": EMPTY_QUESTION_FALLBACK,
        "missing_ground_truth_count": len(missing_ground_truth_rows),
        "missing_ground_truth_examples": missing_ground_truth_rows[:20],
        "missing_ground_truth_placeholder": MISSING_GROUND_TRUTH_FALLBACK,
    }


def derive_dualmath_label(output: str) -> int:
    pos_count = output.count("<pos>")
    neg_count = output.count("<neg>")
    if pos_count == 0 and neg_count == 0:
        raise AssertionError("DualMath output does not contain any <pos>/<neg> tag")
    return 1 if neg_count == 0 and pos_count > 0 else 0


def validate_dualmath(rows_to_check, image_root: Path, output_dir: Path, manifest_tag: str):
    dataset_name = "DualMath-1.1M"
    compat_rows = []
    image_sizes = []
    for sampled in rows_to_check:
        source_index = sampled["source_index"]
        row = sampled["row"]
        for field_name in ["image_url", "instruction", "output"]:
            require_non_empty_str(row.get(field_name), field_name, dataset_name, source_index)
        image_path = (image_root / row["image_url"]).resolve()
        require_exists_isfile(image_path, dataset_name, source_index)
        image_sizes.append(require_openable_image(image_path, dataset_name, source_index))
        label = derive_dualmath_label(row["output"])
        compat_row = {
            "input": row["instruction"].strip(),
            "image": str(image_path),
            "label": label,
            "raw_image_url": row["image_url"],
            "source_instruction": row["instruction"],
            "source_output": row["output"],
            "pos_count": row["output"].count("<pos>"),
            "neg_count": row["output"].count("<neg>"),
            "source_index": source_index,
        }
        require_non_empty_str(compat_row["input"], "input", dataset_name, source_index)
        require_non_empty_str(compat_row["image"], "image", dataset_name, source_index)
        if compat_row["label"] not in (0, 1):
            raise AssertionError(f"{dataset_name}[{source_index}] label must be 0 or 1")
        compat_rows.append(compat_row)

    manifest_path = output_dir / f"dualmath_prm_{manifest_tag}.jsonl"
    write_jsonl(manifest_path, compat_rows)

    loaded_rows = []
    with manifest_path.open("r", encoding="utf-8") as fp:
        for item in jsonlines.Reader(fp):
            loaded_rows.append(item)
    if len(loaded_rows) != len(compat_rows):
        raise AssertionError(
            f"{dataset_name} jsonl reload length mismatch: {len(loaded_rows)} / {len(compat_rows)}"
        )
    for row in loaded_rows:
        require_non_empty_str(row["input"], "input", dataset_name, row["source_index"])
        require_non_empty_str(row["image"], "image", dataset_name, row["source_index"])
        require_exists_isfile(Path(row["image"]), dataset_name, row["source_index"])

    return {
        "sampled_rows": len(compat_rows),
        "manifest_path": str(manifest_path),
        "required_fields_checked": [
            "image_url",
            "instruction",
            "output",
            "input",
            "image",
            "label",
        ],
        "loader_name": "prm_infer_score.py jsonl branch field contract",
        "image_check": "os.path.exists + os.path.isfile + PIL.Image.open",
        "first_image_size": image_sizes[0],
        "first_source_index": compat_rows[0]["source_index"],
    }


def attach_total_rows(result, total_rows):
    result["dataset_total_rows"] = total_rows
    return result


def main():
    args = parse_args()
    mmathcot_path = Path(args.mmathcot_path).resolve()
    dualmath_path = Path(args.dualmath_path).resolve()
    image_root = Path(args.image_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "sample":
        rng_mmath = random.Random(args.seed)
        rng_dualmath = random.Random(args.seed + 1)
        mmathcot_rows, mmathcot_total = reservoir_sample_jsonl(mmathcot_path, args.sample_size, rng_mmath)
        dualmath_rows, dualmath_total = reservoir_sample_jsonl(dualmath_path, args.sample_size, rng_dualmath)
        mmathcot_tag = f"random{args.sample_size}"
        dualmath_tag = f"random{args.sample_size}"
        pass_criteria = [
            "sampled_rows == sample_size_requested",
            "required fields are non-empty and fully present",
            "every referenced image satisfies os.path.exists(path) and os.path.isfile(path)",
            "every sampled image can be opened by PIL.Image.open",
            "MMathCoT compatibility manifest can be loaded by prepare_data(dataset='mathvista')",
            "DualMath compatibility manifest satisfies prm_infer_score.py jsonl field contract",
        ]
        summary_name = "dataset_random_loading_summary.json"
    else:
        mmathcot_rows, mmathcot_total = read_all_jsonl_rows(mmathcot_path)
        dualmath_rows, dualmath_total = read_all_jsonl_rows(dualmath_path)
        mmathcot_tag = "all"
        dualmath_tag = "all"
        pass_criteria = [
            "rows_checked == dataset_total_rows for both datasets",
            "required fields are non-empty and fully present",
            "every referenced image satisfies os.path.exists(path) and os.path.isfile(path)",
            "every referenced image can be opened by PIL.Image.open",
            "the full current MMathCoT subset can be loaded by prepare_data(dataset='mathvista')",
            "the full current DualMath subset satisfies prm_infer_score.py jsonl field contract",
        ]
        summary_name = "dataset_all_loading_summary.json"

    mmathcot_result = attach_total_rows(
        validate_mmathcot(mmathcot_rows, image_root, output_dir, mmathcot_tag),
        mmathcot_total,
    )
    dualmath_result = attach_total_rows(
        validate_dualmath(dualmath_rows, image_root, output_dir, dualmath_tag),
        dualmath_total,
    )
    mmathcot_result["rows_checked"] = len(mmathcot_rows)
    dualmath_result["rows_checked"] = len(dualmath_rows)
    if args.mode == "sample":
        mmathcot_result["sampled_rows"] = len(mmathcot_rows)
        dualmath_result["sampled_rows"] = len(dualmath_rows)

    summary = {
        "mode": args.mode,
        "seed": args.seed,
        "sample_size_requested": args.sample_size,
        "pass_criteria": pass_criteria,
        "mmathcot_policy_check": mmathcot_result,
        "dualmath_prm_check": dualmath_result,
        "status": "passed",
    }

    summary_path = output_dir / summary_name
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
