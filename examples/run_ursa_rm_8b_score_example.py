import argparse
import json
import sys
import time
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference.prm_infer_score import prepare_input, single_inference
from models.ursa_model import UrsaForTokenClassification, UrsaProcessor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load URSA-RM-8B on GPU and score a short step-by-step response."
    )
    parser.add_argument(
        "--model-path",
        default=str(REPO_ROOT / "checkpoints" / "URSA-RM-8B"),
        help="Local path to the URSA-RM-8B checkpoint directory.",
    )
    parser.add_argument(
        "--image-path",
        default=str(REPO_ROOT / "figures" / "framework.png"),
        help="Path to the example image.",
    )
    parser.add_argument(
        "--question",
        default="How many training stages are shown in this diagram?",
        help="Question text for the reward model input.",
    )
    parser.add_argument(
        "--response",
        default=(
            "Step 1: The figure is a training framework diagram. "
            "Step 2: I can identify four training stages in the diagram. "
            "†Answer: 4"
        ),
        help="Step-by-step response to score.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="CUDA device to load the reward model onto, for example cuda:0.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. URSA-RM-8B should be loaded onto GPU.")

    model_path = Path(args.model_path)
    image_path = Path(args.image_path)
    device = torch.device(args.device)

    if device.type != "cuda":
        raise ValueError("This example is intended to run on GPU. Please use a cuda device.")
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image path does not exist: {image_path}")

    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)

    start = time.time()
    processor = UrsaProcessor.from_pretrained(model_path)
    model = UrsaForTokenClassification.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    load_seconds = time.time() - start

    input_prompt = prepare_input(args.question, args.response)
    with torch.inference_mode():
        min_score, avg_score = single_inference(
            processor=processor,
            model=model,
            cuda_device=device.index or 0,
            input_prompt=input_prompt,
            image=str(image_path),
            do_sample=False,
        )

    result = {
        "model_class": type(model).__name__,
        "device": str(next(model.parameters()).device),
        "dtype": str(next(model.parameters()).dtype),
        "load_seconds": round(load_seconds, 2),
        "peak_mem_gb": round(torch.cuda.max_memory_allocated(device) / (1024**3), 2),
        "min_score": round(float(min_score.item()), 6),
        "avg_score": round(float(avg_score.item()), 6),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
