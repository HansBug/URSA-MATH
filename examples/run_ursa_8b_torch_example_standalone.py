import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image


EXAMPLES_DIR = Path(__file__).resolve().parent
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from standalone_ursa import UrsaForConditionalGeneration, UrsaProcessor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Standalone URSA-8B torch example without importing repo models/ code."
    )
    parser.add_argument(
        "--model-path",
        default=str(EXAMPLES_DIR.parent / "checkpoints" / "URSA-8B"),
        help="Local path to the URSA-8B checkpoint directory.",
    )
    parser.add_argument(
        "--image-path",
        default=str(EXAMPLES_DIR.parent / "figures" / "framework.png"),
        help="Path to the example image.",
    )
    parser.add_argument(
        "--question",
        default="How many training stages are shown in this diagram? Answer briefly.",
        help="Question text appended after the <|image|> token.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="CUDA device to load the model onto, for example cuda:0.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=48,
        help="Maximum number of new tokens to generate.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. URSA-8B should be loaded onto GPU.")

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
    model = UrsaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    load_seconds = time.time() - start

    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"<|image|>{args.question}"},
    ]
    prompt = processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
    )
    image = Image.open(image_path).convert("RGB")
    inputs = processor(
        text=[prompt],
        images=[image],
        return_tensors="pt",
        padding=True,
    ).to(device, torch.bfloat16)

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

    prompt_len = inputs["input_ids"].shape[1]
    new_tokens = generated[0, prompt_len:]
    generated_text = processor.decode(new_tokens, skip_special_tokens=True).strip()

    result = {
        "model_class": type(model).__name__,
        "device": str(next(model.parameters()).device),
        "dtype": str(next(model.parameters()).dtype),
        "load_seconds": round(load_seconds, 2),
        "peak_mem_gb": round(torch.cuda.max_memory_allocated(device) / (1024**3), 2),
        "generated_text": generated_text,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
