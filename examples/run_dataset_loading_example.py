import argparse
import json
import sys
from pathlib import Path

import jsonlines


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference.prm_infer_score import prepare_input
from inference.vllm_infer import prepare_data


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build URSA raw-dataset compatibility manifests without changing inference/"
            "model code, then load them through the existing inference helpers."
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
        help=(
            "Root directory that contains real image assets laid out by image_url prefixes. "
            "If the requested image is missing, the example falls back to --fallback-image."
        ),
    )
    parser.add_argument(
        "--fallback-image",
        default=str(REPO_ROOT / "figures" / "framework.png"),
        help="Fallback image used when the real dataset image is not present locally.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="0-based sample index to preview from the raw datasets.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "tmp" / "dataset_load_checks" / "example_manifests"),
        help="Where to write compatibility jsonl files.",
    )
    return parser.parse_args()


def read_jsonl_row(path: Path, sample_index: int):
    with path.open("r", encoding="utf-8") as fp:
        reader = jsonlines.Reader(fp)
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


def choose_image_field(
    raw_image_url: str,
    image_root: Path,
    fallback_image: Path,
):
    real_image = (image_root / raw_image_url).resolve()
    if real_image.exists():
        return {
            "effective_image_root": str(image_root),
            "image_field": raw_image_url,
            "resolved_image_path": str(real_image),
            "used_fallback": False,
        }
    return {
        "effective_image_root": str(fallback_image.parent),
        "image_field": fallback_image.name,
        "resolved_image_path": str(fallback_image.resolve()),
        "used_fallback": True,
    }


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        writer = jsonlines.Writer(fp)
        for row in rows:
            writer.write(row)


def preview(text: str, limit: int = 240) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[:limit] + "..."


def build_negative_response(response: str) -> str:
    negative = response.replace(
        "Step 3: This implies that knowing side lengths allows triangle type determination.",
        "Step 3: This implies that knowing side lengths does not allow triangle type determination.",
    )
    return negative.replace("†Answer: A", "†Answer: B")


def main():
    args = parse_args()

    mmathcot_path = Path(args.mmathcot_path).resolve()
    dualmath_path = Path(args.dualmath_path).resolve()
    image_root = Path(args.image_root).resolve()
    fallback_image = Path(args.fallback_image).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not mmathcot_path.exists():
        raise FileNotFoundError(f"MMathCoT file not found: {mmathcot_path}")
    if not dualmath_path.exists():
        raise FileNotFoundError(f"DualMath file not found: {dualmath_path}")
    if not fallback_image.exists():
        raise FileNotFoundError(f"Fallback image not found: {fallback_image}")

    mmathcot_raw = read_jsonl_row(mmathcot_path, args.sample_index)
    dualmath_raw = read_jsonl_row(dualmath_path, args.sample_index)

    question = extract_question(mmathcot_raw["instruction"])
    ground_truth = extract_answer(mmathcot_raw["output"])
    image_meta = choose_image_field(mmathcot_raw["image_url"], image_root, fallback_image)

    policy_manifest_path = output_dir / "mmathcot_mathvista_compat.jsonl"
    policy_manifest_row = {
        "question": question,
        "image": image_meta["image_field"],
        "ground_truth": ground_truth,
        "raw_image_url": mmathcot_raw["image_url"],
        "policy_instruction": mmathcot_raw["instruction"],
        "reference_output": mmathcot_raw["output"],
    }
    write_jsonl(policy_manifest_path, [policy_manifest_row])

    prm_manifest_path = output_dir / "mmathcot_prm_compat.jsonl"
    prm_manifest_row = {
        "input": prepare_input(question, mmathcot_raw["output"]),
        "image": image_meta["resolved_image_path"],
        "label": 1,
        "ground_truth": ground_truth,
        "raw_image_url": mmathcot_raw["image_url"],
    }
    write_jsonl(prm_manifest_path, [prm_manifest_row])

    prm_pair_manifest_path = output_dir / "mmathcot_prm_pair_compat.jsonl"
    prm_pair_rows = [
        {
            "input": prepare_input(question, mmathcot_raw["output"]),
            "image": image_meta["resolved_image_path"],
            "label": 1,
            "candidate": "positive",
            "ground_truth": ground_truth,
            "raw_image_url": mmathcot_raw["image_url"],
        },
        {
            "input": prepare_input(question, build_negative_response(mmathcot_raw["output"])),
            "image": image_meta["resolved_image_path"],
            "label": 0,
            "candidate": "negative",
            "ground_truth": ground_truth,
            "raw_image_url": mmathcot_raw["image_url"],
        },
    ]
    write_jsonl(prm_pair_manifest_path, prm_pair_rows)

    input_data, origin_data = prepare_data(
        dataset="mathvista",
        data_path=str(policy_manifest_path),
        image_root=image_meta["effective_image_root"],
    )
    loaded_item = input_data[0]
    loaded_origin = origin_data[0]
    loaded_image = loaded_item["multi_modal_data"]["image"]

    result = {
        "raw_mmathcot_sample": {
            "image_url": mmathcot_raw["image_url"],
            "question_preview": preview(question),
            "ground_truth": ground_truth,
            "reference_output_preview": preview(mmathcot_raw["output"]),
        },
        "raw_dualmath_sample": {
            "image_url": dualmath_raw["image_url"],
            "instruction_preview": preview(dualmath_raw["instruction"]),
            "output_preview": preview(dualmath_raw["output"]),
            "pos_label_count": dualmath_raw["output"].count("<pos>"),
            "neg_label_count": dualmath_raw["output"].count("<neg>"),
        },
        "policy_loader_demo": {
            "compat_manifest_path": str(policy_manifest_path),
            "effective_image_root": image_meta["effective_image_root"],
            "resolved_image_path": image_meta["resolved_image_path"],
            "used_fallback_image": image_meta["used_fallback"],
            "loaded_prompt_preview": preview(loaded_item["prompt"]),
            "loaded_image_size": list(loaded_image.size),
            "origin_keys": sorted(loaded_origin.keys()),
        },
        "prm_loader_demo": {
            "compat_manifest_path": str(prm_manifest_path),
            "pair_manifest_path": str(prm_pair_manifest_path),
            "resolved_image_path": prm_manifest_row["image"],
            "prepared_input_preview": preview(prm_manifest_row["input"]),
            "label": prm_manifest_row["label"],
        },
        "prm_pair_demo": {
            "pair_count": len(prm_pair_rows),
            "candidate_order": [row["candidate"] for row in prm_pair_rows],
            "expected_selected_label": 1,
            "negative_input_preview": preview(prm_pair_rows[1]["input"]),
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
