import argparse
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Remove non-essential dataset artifacts while keeping the full raw jsonl files "
            "and the full extracted image tree required by ALL-row validation."
        )
    )
    parser.add_argument(
        "--dataset-root",
        default=str(REPO_ROOT / "datasets" / "URSA-MATH"),
        help="Root directory of the restored URSA-MATH dataset tree.",
    )
    parser.add_argument(
        "--sources-root",
        default=str(REPO_ROOT / "datasets" / "_sources"),
        help="Directory that holds downloaded archive sources.",
    )
    parser.add_argument(
        "--temp-root",
        default=str(REPO_ROOT / "tmp" / "dataset_load_checks"),
        help="Temporary validation output directory to remove.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete files. Without this flag the script only reports what would be removed.",
    )
    return parser.parse_args()


def remove_path(path: Path):
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def collect_unneeded_paths(dataset_root: Path, sources_root: Path, temp_root: Path):
    removable = []

    top_level_keep = {"MMathCoT-1M", "DualMath-1.1M", "images"}
    if dataset_root.exists():
        removable.extend(
            sorted(path for path in dataset_root.iterdir() if path.name not in top_level_keep)
        )

    mmathcot_dir = dataset_root / "MMathCoT-1M"
    dualmath_dir = dataset_root / "DualMath-1.1M"
    image_root = dataset_root / "images"

    if mmathcot_dir.exists():
        removable.extend(
            sorted(path for path in mmathcot_dir.iterdir() if path.name != "train.jsonl")
        )
    if dualmath_dir.exists():
        removable.extend(
            sorted(path for path in dualmath_dir.iterdir() if path.name != "train.jsonl")
        )

    keep_image_roots = {
        "DataEngine_Geometry",
        "Geo170K",
        "Mavis_Extra",
        "VarsityTutors",
        "data_images",
        "RGB_images",
    }
    if image_root.exists():
        removable.extend(
            sorted(path for path in image_root.iterdir() if path.name not in keep_image_roots)
        )
        removable.extend(sorted(image_root.glob("*.extract.ok")))

    multimath_dir = image_root / "Multimath"
    if multimath_dir.exists() and multimath_dir.is_dir():
        removable.extend(sorted(multimath_dir.iterdir()))

    if sources_root.exists():
        removable.append(sources_root)
    if temp_root.exists():
        removable.append(temp_root)

    deduped = []
    seen = set()
    for path in removable:
        resolved = path.resolve() if path.exists() else path
        key = str(resolved)
        if key not in seen:
            seen.add(key)
            deduped.append(path)
    return deduped


def ensure_required_layout(dataset_root: Path):
    image_root = dataset_root / "images"
    required_paths = [
        dataset_root / "MMathCoT-1M" / "train.jsonl",
        dataset_root / "DualMath-1.1M" / "train.jsonl",
        image_root / "DataEngine_Geometry",
        image_root / "Geo170K",
        image_root / "Mavis_Extra",
        image_root / "VarsityTutors",
        image_root / "data_images",
        image_root / "RGB_images",
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Refusing to clean because required full-dataset paths are missing: "
            + ", ".join(missing)
        )


def restore_symlinks(dataset_root: Path):
    image_root = dataset_root / "images"

    mathv_link = image_root / "MathV-360k"
    if mathv_link.exists() or mathv_link.is_symlink():
        remove_path(mathv_link)
    mathv_link.symlink_to("data_images")

    multimath_dir = image_root / "Multimath"
    if multimath_dir.exists() and multimath_dir.is_symlink():
        remove_path(multimath_dir)
    multimath_dir.mkdir(exist_ok=True)

    multimath_link = multimath_dir / "RGB_images"
    if multimath_link.exists() or multimath_link.is_symlink():
        remove_path(multimath_link)
    multimath_link.symlink_to("../RGB_images")


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    sources_root = Path(args.sources_root).resolve()
    temp_root = Path(args.temp_root).resolve()

    ensure_required_layout(dataset_root)
    removable = collect_unneeded_paths(dataset_root, sources_root, temp_root)

    print("paths to remove:")
    for path in removable:
        print(f"  {path}")

    if not args.apply:
        print("dry-run only; re-run with --apply to actually delete files")
        return

    for path in removable:
        if path.exists() or path.is_symlink():
            remove_path(path)

    restore_symlinks(dataset_root)
    print("cleanup completed")


if __name__ == "__main__":
    main()
