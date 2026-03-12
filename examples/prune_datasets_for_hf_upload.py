import argparse
import os
import random
import shutil
import sys
from pathlib import Path

import jsonlines


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Prune datasets/URSA-MATH down to the minimum subset required by the "
            "documented DATASET_LOAD.md checks."
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
        default=5000,
        help="Random sample size used by the documented validation script.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260312,
        help="Sampling seed used by the documented validation script.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete files. Without this flag the script only reports what would be kept.",
    )
    return parser.parse_args()


def reservoir_sample_image_urls(path: Path, sample_size: int, seed: int):
    rng = random.Random(seed)
    sample = []
    total = 0
    with path.open("r", encoding="utf-8") as fp:
        for index, item in enumerate(jsonlines.Reader(fp)):
            total += 1
            wrapped = {"source_index": index, "image_url": item["image_url"]}
            if index < sample_size:
                sample.append(wrapped)
                continue
            replace_at = rng.randint(0, index)
            if replace_at < sample_size:
                sample[replace_at] = wrapped
    sample.sort(key=lambda item: item["source_index"])
    return sample, total


def read_all_image_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as fp:
        for index, item in enumerate(jsonlines.Reader(fp)):
            rows.append({"source_index": index, "image_url": item["image_url"]})
    return rows


def collect_prefix_check_images(path: Path):
    want = {
        "MathV-360k",
        "Multimath",
        "DataEngine_Geometry",
        "Geo170K",
        "Mavis_Extra",
        "VarsityTutors",
    }
    found = {}
    with path.open("r", encoding="utf-8") as fp:
        for index, item in enumerate(jsonlines.Reader(fp)):
            prefix = item["image_url"].split("/")[0]
            if prefix in want and prefix not in found:
                found[prefix] = {"source_index": index, "image_url": item["image_url"]}
                if len(found) == len(want):
                    break
    return found


def logical_to_physical(image_url: str):
    if image_url.startswith("MathV-360k/"):
        return Path("data_images") / image_url.removeprefix("MathV-360k/")
    if image_url.startswith("Multimath/RGB_images/"):
        return Path("RGB_images") / image_url.removeprefix("Multimath/RGB_images/")
    return Path(image_url)


def keep_source_indices(rows):
    return {item["source_index"] for item in rows}


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def stage_kept_files(image_root: Path, keep_physical_paths, stage_root: Path):
    for relative_path in sorted(keep_physical_paths):
        source = image_root / relative_path
        destination = stage_root / relative_path
        ensure_parent(destination)
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)


def prune_and_restore(image_root: Path, keep_physical_paths, stage_root: Path):
    physical_roots = [
        "DataEngine_Geometry",
        "Geo170K",
        "Mavis_Extra",
        "VarsityTutors",
        "data_images",
        "RGB_images",
    ]
    stage_kept_files(image_root, keep_physical_paths, stage_root)
    for root_name in physical_roots:
        target = image_root / root_name
        if target.exists():
            shutil.rmtree(target)
    for root_name in physical_roots:
        staged = stage_root / root_name
        if staged.exists():
            shutil.move(str(staged), str(image_root / root_name))
        else:
            (image_root / root_name).mkdir(parents=True, exist_ok=True)
    shutil.rmtree(stage_root)


def prune_jsonl_rows(path: Path, keep_indices):
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with path.open("r", encoding="utf-8") as src, tmp_path.open("w", encoding="utf-8") as dst:
        writer = jsonlines.Writer(dst)
        for index, item in enumerate(jsonlines.Reader(src)):
            if index in keep_indices:
                writer.write(item)
    tmp_path.replace(path)


def main():
    args = parse_args()
    mmathcot_path = Path(args.mmathcot_path).resolve()
    dualmath_path = Path(args.dualmath_path).resolve()
    image_root = Path(args.image_root).resolve()

    mmathcot_sample, mmathcot_total = reservoir_sample_image_urls(
        mmathcot_path,
        args.sample_size,
        args.seed,
    )
    dualmath_sample, dualmath_total = reservoir_sample_image_urls(
        dualmath_path,
        args.sample_size,
        args.seed + 1,
    )
    prefix_images = collect_prefix_check_images(mmathcot_path)

    # Keep the current already-pruned subset stable on repeated runs.
    if mmathcot_total <= args.sample_size + len(prefix_images):
        mmathcot_rows_to_keep = read_all_image_rows(mmathcot_path)
    else:
        mmathcot_rows_to_keep = mmathcot_sample

    if dualmath_total <= args.sample_size + 1:
        dualmath_rows_to_keep = read_all_image_rows(dualmath_path)
    else:
        dualmath_rows_to_keep = dualmath_sample

    mmathcot_keep_indices = keep_source_indices(mmathcot_rows_to_keep)
    mmathcot_keep_indices.update(item["source_index"] for item in prefix_images.values())
    mmathcot_keep_indices.add(0)

    dualmath_keep_indices = keep_source_indices(dualmath_rows_to_keep)
    dualmath_keep_indices.add(0)

    keep_logical = set(item["image_url"] for item in mmathcot_rows_to_keep)
    keep_logical.update(item["image_url"] for item in dualmath_rows_to_keep)
    keep_logical.update(item["image_url"] for item in prefix_images.values())
    keep_logical.add("Mavis_Extra/meta_gen/textbook_collect_1220-100_170.png")
    keep_logical.add("MathV-360k/Geometry3K/images/454.png")

    keep_physical = {logical_to_physical(path) for path in keep_logical}
    missing = [str(path) for path in sorted(keep_physical) if not (image_root / path).is_file()]
    if missing:
        raise FileNotFoundError(
            "Refusing to prune because some required files are already missing. "
            f"Examples: {missing[:10]}"
        )

    archive_paths = sorted((mmathcot_path.parent).glob("*.7z"))
    cache_paths = [
        mmathcot_path.parent / ".cache",
        dualmath_path.parent / ".cache",
    ]
    generated_dirs = [
        REPO_ROOT / "datasets" / "URSA-MATH" / "_example_manifests",
        REPO_ROOT / "datasets" / "URSA-MATH" / "_loader_validation",
    ]

    print(f"keep logical images: {len(keep_logical)}")
    print(f"keep physical files: {len(keep_physical)}")
    print(f"keep MMathCoT rows: {len(mmathcot_keep_indices)}")
    print(f"keep DualMath rows: {len(dualmath_keep_indices)}")
    print("archives to remove:")
    for path in archive_paths:
        print(f"  {path}")
    print("cache directories to remove:")
    for path in cache_paths:
        print(f"  {path}")
    print("generated directories to clean:")
    for path in generated_dirs:
        print(f"  {path}")

    if not args.apply:
        print("dry-run only; re-run with --apply to actually delete files")
        return

    stage_root = image_root.parent / ".prune_stage"
    if stage_root.exists():
        shutil.rmtree(stage_root)
    stage_root.mkdir(parents=True, exist_ok=True)
    prune_and_restore(image_root, keep_physical, stage_root)
    prune_jsonl_rows(mmathcot_path, mmathcot_keep_indices)
    prune_jsonl_rows(dualmath_path, dualmath_keep_indices)

    for path in archive_paths:
        if path.exists():
            path.unlink()
    for path in cache_paths:
        if path.exists():
            shutil.rmtree(path)
    for directory in generated_dirs:
        if directory.exists():
            shutil.rmtree(directory)
            directory.mkdir(parents=True, exist_ok=True)

    print("prune completed")


if __name__ == "__main__":
    main()
