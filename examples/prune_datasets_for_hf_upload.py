import argparse
import os
import random
import shutil
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
        default=10000,
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


def read_image_url_at_index(path: Path, wanted_index: int):
    with path.open("r", encoding="utf-8") as fp:
        for index, item in enumerate(jsonlines.Reader(fp)):
            if index == wanted_index:
                return {"source_index": index, "image_url": item["image_url"]}
    raise IndexError(f"{path} does not contain source_index={wanted_index}")


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
    mmathcot_sample0 = read_image_url_at_index(mmathcot_path, 0)
    dualmath_sample0 = read_image_url_at_index(dualmath_path, 0)

    keep_logical = set(item["image_url"] for item in mmathcot_sample)
    keep_logical.update(item["image_url"] for item in dualmath_sample)
    keep_logical.update(item["image_url"] for item in prefix_images.values())
    keep_logical.add(mmathcot_sample0["image_url"])
    keep_logical.add(dualmath_sample0["image_url"])

    keep_physical = {logical_to_physical(path) for path in keep_logical}
    missing = [str(path) for path in sorted(keep_physical) if not (image_root / path).is_file()]
    if missing:
        raise FileNotFoundError(
            "Refusing to prune because some required files are already missing. "
            f"Examples: {missing[:10]}"
        )

    archive_paths = sorted((mmathcot_path.parent).glob("*.7z"))
    source_archives = [
        REPO_ROOT / "datasets" / "_sources" / "MathV360K" / "data_images.zip",
        REPO_ROOT / "datasets" / "_sources" / "multimath-300k" / "images.zip",
    ]
    cache_paths = [
        mmathcot_path.parent / ".cache",
        dualmath_path.parent / ".cache",
    ]
    generated_dirs = [
        REPO_ROOT / "datasets" / "URSA-MATH" / "_example_manifests",
        REPO_ROOT / "datasets" / "URSA-MATH" / "_loader_validation",
        REPO_ROOT / "tmp" / "dataset_load_checks",
    ]
    source_dirs = [
        REPO_ROOT / "datasets" / "_sources" / "MathV360K",
        REPO_ROOT / "datasets" / "_sources" / "multimath-300k",
    ]

    print(f"keep logical images: {len(keep_logical)}")
    print(f"keep physical files: {len(keep_physical)}")
    print(f"keep MMathCoT rows: full jsonl retained ({mmathcot_total})")
    print(f"keep DualMath rows: full jsonl retained ({dualmath_total})")
    print("archives to remove:")
    for path in archive_paths:
        print(f"  {path}")
    print("source archives to remove:")
    for path in source_archives:
        print(f"  {path}")
    print("cache directories to remove:")
    for path in cache_paths:
        print(f"  {path}")
    print("generated directories to clean:")
    for path in generated_dirs:
        print(f"  {path}")
    print("source directories to remove after archive deletion:")
    for path in source_dirs:
        print(f"  {path}")

    if not args.apply:
        print("dry-run only; re-run with --apply to actually delete files")
        return

    stage_root = image_root.parent / ".prune_stage"
    if stage_root.exists():
        shutil.rmtree(stage_root)
    stage_root.mkdir(parents=True, exist_ok=True)
    prune_and_restore(image_root, keep_physical, stage_root)

    for marker in image_root.glob("*.extract.ok"):
        marker.unlink()

    for path in archive_paths:
        if path.exists():
            path.unlink()
    for path in source_archives:
        if path.exists():
            path.unlink()
    for path in cache_paths:
        if path.exists():
            shutil.rmtree(path)
    for directory in generated_dirs:
        if directory.exists():
            shutil.rmtree(directory)
    for directory in source_dirs:
        if directory.exists():
            shutil.rmtree(directory)

    mathv_link = image_root / "MathV-360k"
    if mathv_link.exists() or mathv_link.is_symlink():
        mathv_link.unlink()
    mathv_link.symlink_to("data_images")

    multimath_dir = image_root / "Multimath"
    if multimath_dir.exists() and multimath_dir.is_symlink():
        multimath_dir.unlink()
    multimath_dir.mkdir(exist_ok=True)
    multimath_link = multimath_dir / "RGB_images"
    if multimath_link.exists() or multimath_link.is_symlink():
        multimath_link.unlink()
    multimath_link.symlink_to("../RGB_images")

    print("prune completed")


if __name__ == "__main__":
    main()
