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


PROMPT = "You are given a problem and a step-by-step solution. You need to check the correctness of each step.\nQuestion:"
TAG_TOKEN = " и"
DEFAULT_GROUND_TRUTH_ANSWER = "3"
STEP_PATTERN = re.compile(r"(Step\s+\d+\s*:[\s\S]*?)(?=Step\s+\d+\s*:|†Answer:|$)")
DEFAULT_RESPONSE = (
    "Step 1: The figure explicitly labels Stage 1 as VL Alignment. "
    "Step 2: The top-right panel is labeled Stage 2 Math SFT. "
    "Step 3: The bottom-left panel is labeled Stage 3 PRM Training and Verifying. "
    "†Answer: 3"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load standalone URSA-RM-8B on GPU and print per-step PRM scores."
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
        default="How many numbered training stages are shown in this diagram?",
        help="Question text for the reward model input.",
    )
    parser.add_argument(
        "--response",
        action="append",
        dest="responses",
        help=(
            "Repeat this flag to score one or more responses. If omitted, the example "
            "uses a built-in response."
        ),
    )
    parser.add_argument(
        "--ground-truth-answer",
        default=DEFAULT_GROUND_TRUTH_ANSWER,
        help="Ground-truth final answer. This is reported in the output for reference.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="CUDA device to load the reward model onto, for example cuda:0.",
    )
    return parser.parse_args()


def return_score(scores: torch.Tensor, operation: str):
    if scores.numel() == 0:
        return torch.tensor([0.0])
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


def round_float(value, digits: int = 6):
    if value is None:
        return None
    return round(float(value), digits)


def parse_step_texts(response: str):
    return [match.group(1).strip() for match in STEP_PATTERN.finditer(response)]


def load_responses(args):
    return args.responses or [DEFAULT_RESPONSE]


def score_step_rewards(
    processor: UrsaProcessor,
    model: UrsaForTokenClassification,
    device: torch.device,
    prepared_input: str,
    image_path: Path,
):
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "<|image|>" + prepared_input},
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    with Image.open(image_path) as raw_image:
        image = raw_image.convert("RGB")
    inputs = processor(prompt, [image], return_tensors="pt").to(device, torch.bfloat16)

    tag_id = processor.tokenizer.encode(TAG_TOKEN, add_special_tokens=False)
    with torch.inference_mode():
        reward_logits = model(**inputs).logits
        input_ids = inputs["input_ids"].view(-1)
        insert_values = torch.full((575,), -1, device=input_ids.device)
        input_ids = torch.cat((input_ids[:1], insert_values, input_ids[1:]))

        # Keep the same token-position extraction logic as inference/prm_infer_score.py.
        step_rewards = reward_logits.view(-1)[input_ids == tag_id[0]]
        step_rewards = torch.sigmoid(step_rewards).view(-1).detach().float().cpu()

    min_score = return_score(step_rewards, "min")
    avg_score = return_score(step_rewards, "avg")
    return step_rewards.tolist(), float(min_score.item()), float(avg_score.item())


def build_response_result(
    response_id: int,
    response: str,
    prepared_input: str,
    step_rewards,
    min_score: float,
    avg_score: float,
):
    step_texts = parse_step_texts(response)
    paired_length = max(len(step_texts), len(step_rewards))
    steps = []
    for step_index in range(paired_length):
        steps.append(
            {
                "step_index": step_index + 1,
                "step_text": step_texts[step_index] if step_index < len(step_texts) else None,
                "prm_score": (
                    round_float(step_rewards[step_index])
                    if step_index < len(step_rewards)
                    else None
                ),
            }
        )

    return {
        "response_id": response_id,
        "response": response,
        "prepared_input": prepared_input,
        "step_count_in_text": len(step_texts),
        "scored_step_count": len(step_rewards),
        "step_reward_alignment_ok": len(step_texts) == len(step_rewards),
        "steps": steps,
        "min_prm_score": round_float(min_score),
        "avg_prm_score": round_float(avg_score),
    }


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

    ground_truth_answer = args.ground_truth_answer.strip() if args.ground_truth_answer else None
    responses = load_responses(args)

    start = time.time()
    processor = UrsaProcessor.from_pretrained(model_path)
    model = UrsaForTokenClassification.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    load_seconds = time.time() - start

    response_results = []
    for response_id, response in enumerate(responses):
        prepared_input = prepare_input(args.question, response)
        step_rewards, min_score, avg_score = score_step_rewards(
            processor=processor,
            model=model,
            device=device,
            prepared_input=prepared_input,
            image_path=image_path,
        )
        response_results.append(
            build_response_result(
                response_id=response_id,
                response=response,
                prepared_input=prepared_input,
                step_rewards=step_rewards,
                min_score=min_score,
                avg_score=avg_score,
            )
        )

    result = {
        "model_class": type(model).__name__,
        "device": str(next(model.parameters()).device),
        "dtype": str(next(model.parameters()).dtype),
        "load_seconds": round(load_seconds, 2),
        "peak_mem_gb": round(torch.cuda.max_memory_allocated(device) / (1024**3), 2),
        "image_path": str(image_path),
        "question": args.question,
        "ground_truth_answer": ground_truth_answer,
        "responses": response_results,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
