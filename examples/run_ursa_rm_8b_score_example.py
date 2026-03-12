import argparse
import json
import sys
import time
from pathlib import Path

import regex as re
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference.prm_infer_score import (
    extract_answer_try_all_methods,
    prepare_input,
)
from models.ursa_model import UrsaForTokenClassification, UrsaProcessor


TAG_TOKEN = " и"
DEFAULT_GROUND_TRUTH_ANSWER = "3"
DEFAULT_RHO = 0.3
DEFAULT_GAMMA = 0.5
STEP_PATTERN = re.compile(r"(Step\s+\d+\s*:[\s\S]*?)(?=Step\s+\d+\s*:|†Answer:|$)")
DEFAULT_RESPONSES = [
    (
        "Step 1: The figure explicitly labels Stage 1 as VL Alignment. "
        "Step 2: The top-right panel is labeled Stage 2 Math SFT. "
        "Step 3: The bottom-left panel is labeled Stage 3 PRM Training and Verifying. "
        "†Answer: 3"
    ),
    (
        "Step 1: The top row contains two numbered training stages. "
        "Step 2: The lower-left block is also labeled Stage 3, so the diagram contains three stages in total. "
        "†Answer: 3"
    ),
    (
        "Step 1: The diagram has four large regions, so there must be four training stages. "
        "Step 2: I will trust that first impression. "
        "†Answer: 4"
    ),
    (
        "Step 1: The figure first looks like it might have four sections. "
        "Step 2: However, only three of them carry numbered stage labels: Stage 1, Stage 2, and Stage 3. "
        "Step 3: The inference-time scaling panel is an application block rather than another training stage. "
        "†Answer: 3"
    ),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Load URSA-RM-8B on GPU, emit per-step PRM rewards, and summarize the "
            "BoN / GRPO / PS-GRPO statistics discussed in the paper."
        )
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
        default="How many numbered training stages are shown in this diagram?",
        help="Question text for the reward model input.",
    )
    parser.add_argument(
        "--response",
        action="append",
        dest="responses",
        help=(
            "Repeat this flag to score one or more rollout responses. If omitted, "
            "the example uses a built-in rollout group that exercises BoN and PS-GRPO."
        ),
    )
    parser.add_argument(
        "--responses-file",
        help=(
            "Optional JSON file containing a list of rollout strings or objects with "
            '{"response": "...", "outcome_reward": 1}.'
        ),
    )
    parser.add_argument(
        "--ground-truth-answer",
        default=DEFAULT_GROUND_TRUTH_ANSWER,
        help=(
            "Ground-truth final answer used to derive binary outcome rewards when "
            "--outcome-reward is not provided."
        ),
    )
    parser.add_argument(
        "--outcome-reward",
        action="append",
        type=float,
        dest="outcome_rewards",
        help=(
            "Optional rollout-level outcome reward. Repeat once per --response to "
            "override automatic answer matching."
        ),
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=DEFAULT_RHO,
        help="Drop-moment threshold rho from Equation 5 / Equation 6 in the paper.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=DEFAULT_GAMMA,
        help="Penalty gamma from Equation 6 in the paper.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="CUDA device to load the reward model onto, for example cuda:0.",
    )
    return parser.parse_args()


def round_float(value, digits: int = 6):
    if value is None:
        return None
    return round(float(value), digits)


def canonicalize_answer(text: str):
    cleaned = re.sub(r"^\s*(?:†\s*)?answer\s*:\s*", "", text.strip(), flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", cleaned).strip()


def normalize_answer(text: str):
    return canonicalize_answer(text).lower()


def population_std(values):
    if not values:
        return 0.0
    tensor = torch.tensor(values, dtype=torch.float32)
    return float(tensor.std(unbiased=False).item())


def safe_standardize(values):
    if not values:
        return []
    tensor = torch.tensor(values, dtype=torch.float32)
    std = tensor.std(unbiased=False)
    if std.item() == 0:
        return [0.0 for _ in values]
    return ((tensor - tensor.mean()) / std).tolist()


def parse_step_texts(response: str):
    return [match.group(1).strip() for match in STEP_PATTERN.finditer(response)]


def load_rollout_specs(args):
    rollout_specs = []

    if args.responses_file:
        responses_path = Path(args.responses_file)
        if not responses_path.exists():
            raise FileNotFoundError(f"Responses file does not exist: {responses_path}")
        with responses_path.open("r", encoding="utf-8") as file:
            raw_items = json.load(file)
        if not isinstance(raw_items, list):
            raise ValueError("--responses-file must contain a JSON list.")
        for item in raw_items:
            if isinstance(item, str):
                rollout_specs.append({"response": item})
            elif isinstance(item, dict) and "response" in item:
                rollout_specs.append(
                    {
                        "response": item["response"],
                        "outcome_reward": item.get("outcome_reward"),
                    }
                )
            else:
                raise ValueError(
                    "--responses-file items must be either strings or objects with a 'response' key."
                )

    if args.responses:
        rollout_specs.extend({"response": response} for response in args.responses)

    if not rollout_specs:
        rollout_specs = [{"response": response} for response in DEFAULT_RESPONSES]

    if args.outcome_rewards:
        if len(args.outcome_rewards) != len(rollout_specs):
            raise ValueError(
                "The number of --outcome-reward values must match the number of rollout responses."
            )
        for rollout_spec, outcome_reward in zip(rollout_specs, args.outcome_rewards):
            rollout_spec["outcome_reward"] = outcome_reward

    return rollout_specs


def resolve_outcome_reward(response: str, rollout_spec: dict, ground_truth_answer: str | None):
    if rollout_spec.get("outcome_reward") is not None:
        return float(rollout_spec["outcome_reward"])

    if not ground_truth_answer:
        return None

    predicted_answer = extract_answer_try_all_methods(response)
    return float(normalize_answer(predicted_answer) == normalize_answer(ground_truth_answer))


def score_process_rewards(
    processor: UrsaProcessor,
    model: UrsaForTokenClassification,
    device: torch.device,
    input_prompt: str,
    image_path: Path,
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
    with Image.open(image_path) as raw_image:
        image = raw_image.convert("RGB")
    inputs = processor(
        text=[prompt],
        images=[image],
        return_tensors="pt",
        padding=True,
    ).to(device, torch.bfloat16)

    tag_id = processor.tokenizer.encode(TAG_TOKEN, add_special_tokens=False)
    with torch.inference_mode():
        reward_logits = model(**inputs).logits
        input_ids = inputs["input_ids"].view(-1)
        insert_values = torch.full((575,), -1, device=input_ids.device)
        input_ids = torch.cat((input_ids[:1], insert_values, input_ids[1:]))
        reward = reward_logits.view(-1)[input_ids == tag_id[0]]
        reward = torch.sigmoid(reward).view(-1).detach().float().cpu()
    return reward.tolist()


def analyze_drop_moment(process_rewards, rho: float):
    relative_drops = []
    max_relative_drop = 0.0
    max_relative_drop_pair = None

    for index in range(len(process_rewards) - 1):
        current_score = process_rewards[index]
        next_score = process_rewards[index + 1]
        denominator = current_score if abs(current_score) > 1e-12 else 1e-12
        relative_drop = (current_score - next_score) / denominator
        relative_drops.append(
            {
                "from_step": index + 1,
                "to_step": index + 2,
                "relative_drop": round_float(relative_drop),
            }
        )
        if relative_drop > max_relative_drop:
            max_relative_drop = float(relative_drop)
            max_relative_drop_pair = {
                "from_step": index + 1,
                "to_step": index + 2,
            }

    return {
        "relative_drops": relative_drops,
        "max_relative_drop": round_float(max_relative_drop),
        "triggered": max_relative_drop > rho,
        "threshold_rho": round_float(rho),
        "max_relative_drop_pair": max_relative_drop_pair,
    }


def build_rollout_result(
    rollout_index: int,
    response: str,
    process_rewards,
    outcome_reward,
    rho: float,
    gamma: float,
):
    step_texts = parse_step_texts(response)
    predicted_answer = canonicalize_answer(extract_answer_try_all_methods(response))
    drop_moment = analyze_drop_moment(process_rewards, rho)
    avg_process_reward = sum(process_rewards) / len(process_rewards) if process_rewards else 0.0
    min_process_reward = min(process_rewards) if process_rewards else 0.0
    ps_grpo_reward = None
    variant1_rollout_reward = None

    if outcome_reward is not None:
        variant1_rollout_reward = float(outcome_reward) + float(avg_process_reward)
        if outcome_reward > 0:
            ps_grpo_reward = 1.0 - gamma if drop_moment["triggered"] else 1.0
        else:
            ps_grpo_reward = 0.0

    step_scores = []
    paired_length = max(len(step_texts), len(process_rewards))
    for step_index in range(paired_length):
        step_scores.append(
            {
                "step_index": step_index + 1,
                "step_text": step_texts[step_index] if step_index < len(step_texts) else None,
                "process_reward": (
                    round_float(process_rewards[step_index])
                    if step_index < len(process_rewards)
                    else None
                ),
            }
        )

    return {
        "rollout_id": rollout_index,
        "response": response,
        "parsed_answer": predicted_answer,
        "step_count_in_text": len(step_texts),
        "process_reward_count": len(process_rewards),
        "step_reward_alignment_ok": len(step_texts) == len(process_rewards),
        "step_scores": step_scores,
        "process_reward_sequence": [round_float(score) for score in process_rewards],
        "min_process_reward": round_float(min_process_reward),
        "avg_process_reward": round_float(avg_process_reward),
        "outcome_reward": round_float(outcome_reward),
        "drop_moment": drop_moment,
        "paper_metrics": {
            "bon_selection_score": round_float(avg_process_reward),
            "variant1_rollout_reward": round_float(variant1_rollout_reward),
            "variant2_step_rewards": [round_float(score) for score in process_rewards],
            "ps_grpo_reward": round_float(ps_grpo_reward),
        },
    }


def add_group_metrics(rollout_results, gamma: float, rho: float):
    avg_process_rewards = [rollout["avg_process_reward"] for rollout in rollout_results]
    outcome_rewards = [rollout["outcome_reward"] for rollout in rollout_results]
    has_complete_outcome_rewards = all(
        outcome_reward is not None for outcome_reward in outcome_rewards
    )

    bon_winner = max(rollout_results, key=lambda rollout: rollout["avg_process_reward"])
    process_advantages = safe_standardize(avg_process_rewards)
    if has_complete_outcome_rewards:
        outcome_advantages = safe_standardize(outcome_rewards)
        variant1_rewards = [
            rollout["paper_metrics"]["variant1_rollout_reward"] for rollout in rollout_results
        ]
        ps_grpo_rewards = [rollout["paper_metrics"]["ps_grpo_reward"] for rollout in rollout_results]
        variant1_advantages = safe_standardize(variant1_rewards)
        ps_grpo_advantages = safe_standardize(ps_grpo_rewards)
    else:
        outcome_advantages = [None] * len(rollout_results)
        variant1_rewards = []
        ps_grpo_rewards = []
        variant1_advantages = [None] * len(rollout_results)
        ps_grpo_advantages = [None] * len(rollout_results)

    for index, rollout in enumerate(rollout_results):
        outcome_advantage = outcome_advantages[index]
        process_advantage = process_advantages[index]
        rollout["paper_metrics"]["vanilla_grpo_advantage"] = round_float(outcome_advantage)
        rollout["paper_metrics"]["variant1_advantage"] = round_float(variant1_advantages[index])
        rollout["paper_metrics"]["variant2_process_advantage_term"] = round_float(
            process_advantage
        )
        rollout["paper_metrics"]["variant2_outcome_advantage_term"] = round_float(
            outcome_advantage
        )
        rollout["paper_metrics"]["variant2_step_advantages"] = (
            [
                round_float(score * process_advantage + outcome_advantage)
                for score in rollout["process_reward_sequence"]
            ]
            if outcome_advantage is not None
            else None
        )
        rollout["paper_metrics"]["ps_grpo_advantage"] = round_float(ps_grpo_advantages[index])

    return {
        "paper_hyperparameters": {
            "rho": round_float(rho),
            "gamma": round_float(gamma),
            "standardization": "population_std",
        },
        "group_size": len(rollout_results),
        "bon_by_mean_process_reward": {
            "selected_rollout_id": bon_winner["rollout_id"],
            "selected_avg_process_reward": bon_winner["avg_process_reward"],
            "selected_answer": bon_winner["parsed_answer"],
        },
        "group_statistics": {
            "avg_process_reward_mean": round_float(sum(avg_process_rewards) / len(avg_process_rewards)),
            "avg_process_reward_std": round_float(population_std(avg_process_rewards)),
            "outcome_reward_mean": (
                round_float(sum(outcome_rewards) / len(outcome_rewards))
                if has_complete_outcome_rewards
                else None
            ),
            "outcome_reward_std": (
                round_float(population_std(outcome_rewards))
                if has_complete_outcome_rewards
                else None
            ),
            "variant1_reward_mean": (
                round_float(sum(variant1_rewards) / len(variant1_rewards))
                if has_complete_outcome_rewards
                else None
            ),
            "variant1_reward_std": (
                round_float(population_std(variant1_rewards))
                if has_complete_outcome_rewards
                else None
            ),
            "ps_grpo_reward_mean": (
                round_float(sum(ps_grpo_rewards) / len(ps_grpo_rewards))
                if has_complete_outcome_rewards
                else None
            ),
            "ps_grpo_reward_std": (
                round_float(population_std(ps_grpo_rewards))
                if has_complete_outcome_rewards
                else None
            ),
        },
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

    rollout_specs = load_rollout_specs(args)
    ground_truth_answer = args.ground_truth_answer.strip() if args.ground_truth_answer else None

    start = time.time()
    processor = UrsaProcessor.from_pretrained(model_path)
    model = UrsaForTokenClassification.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    load_seconds = time.time() - start

    rollout_results = []
    for rollout_index, rollout_spec in enumerate(rollout_specs):
        response = rollout_spec["response"]
        input_prompt = prepare_input(args.question, response)
        process_rewards = score_process_rewards(
            processor=processor,
            model=model,
            device=device,
            input_prompt=input_prompt,
            image_path=image_path,
        )
        outcome_reward = resolve_outcome_reward(response, rollout_spec, ground_truth_answer)
        rollout_results.append(
            build_rollout_result(
                rollout_index=rollout_index,
                response=response,
                process_rewards=process_rewards,
                outcome_reward=outcome_reward,
                rho=args.rho,
                gamma=args.gamma,
            )
        )

    group_result = add_group_metrics(rollout_results, gamma=args.gamma, rho=args.rho)

    result = {
        "model_class": type(model).__name__,
        "device": str(next(model.parameters()).device),
        "dtype": str(next(model.parameters()).dtype),
        "load_seconds": round(load_seconds, 2),
        "peak_mem_gb": round(torch.cuda.max_memory_allocated(device) / (1024**3), 2),
        "image_path": str(image_path),
        "question": args.question,
        "ground_truth_answer": ground_truth_answer,
        "paper_references": {
            "bon_selection": "Appendix B.2 Equation 10",
            "drop_moment": "Section 4 Equation 5",
            "ps_grpo_reward": "Section 4 Equation 6",
            "grpo_advantage": "Appendix B.1 Equation 7",
            "prm_variant1_and_variant2": "Appendix B.1 Equation 9",
        },
        **group_result,
        "rollouts": rollout_results,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
