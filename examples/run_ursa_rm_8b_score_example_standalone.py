import argparse
import json
import sys
import time
from pathlib import Path

import regex as re
import torch
from PIL import Image


EXAMPLES_DIR = Path(__file__).resolve().parent
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from standalone_ursa import UrsaForTokenClassification, UrsaProcessor


PROMPT = (
    "You are given a problem and a step-by-step solution. "
    "You need to check the correctness of each step.\nQuestion:"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Standalone URSA-RM-8B scoring example without importing repo inference/ code."
    )
    parser.add_argument(
        "--model-path",
        default=str(EXAMPLES_DIR.parent / "checkpoints" / "URSA-RM-8B"),
        help="Local path to the URSA-RM-8B checkpoint directory.",
    )
    parser.add_argument(
        "--image-path",
        default=str(EXAMPLES_DIR.parent / "figures" / "framework.png"),
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


def return_score(scores: torch.Tensor, operation: str):
    if scores.numel() == 0:
        return torch.tensor([0.0], device=scores.device if scores.is_cuda else "cpu")
    scores = scores.view(-1)
    if operation == "min":
        return torch.min(scores)
    if operation == "avg":
        return torch.mean(scores)
    raise ValueError(f"Unsupported operation: {operation}")


def replace_specific_plus_minus_with_ki(text: str):
    pattern = r"Step \d+"
    matches = list(re.finditer(pattern, text))
    positions = [(match.start(), match.end()) for match in matches]

    text_list = list(text)
    insert_pos = []

    try:
        for i in range(1, len(positions)):
            for j in range(positions[i][0] - 1, positions[i - 1][1], -1):
                if text_list[j] not in {" ", "\n"}:
                    insert_pos.append(j + 1)
                    break

        answer_start = text.find("†Answer:")
        for j in range(answer_start - 1, positions[-1][1], -1):
            if text_list[j] not in {" ", "\n"}:
                insert_pos.append(j + 1)
                break

        for index in sorted(insert_pos, reverse=True):
            text = text[:index] + " и" + text[index:]
        return text
    except Exception:
        return text + " и"


def prepare_input(question: str, response: str):
    instruction = PROMPT + question + "\n" + response
    return replace_specific_plus_minus_with_ki(instruction)


def single_inference(
    processor: UrsaProcessor,
    model: UrsaForTokenClassification,
    device: torch.device,
    input_prompt: str,
    image_path: str,
):
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "<|image|>" + input_prompt},
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

    tag_id = processor.tokenizer.encode(" и", add_special_tokens=False)
    with torch.inference_mode():
        reward = model(**inputs).logits
        input_ids = inputs["input_ids"].view(-1)
        insert_values = torch.full((575,), -1, device=input_ids.device)
        input_ids = torch.cat((input_ids[:1], insert_values, input_ids[1:]))
        reward = reward.view(-1)[input_ids == tag_id[0]]
        reward = torch.sigmoid(reward).view(-1)
        min_score = return_score(reward, "min")
        avg_score = return_score(reward, "avg")
    return min_score, avg_score


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
    min_score, avg_score = single_inference(
        processor=processor,
        model=model,
        device=device,
        input_prompt=input_prompt,
        image_path=str(image_path),
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
