import argparse
import importlib.util
import json
import site
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
VLLM_REPO_PACKAGE = REPO_ROOT / "vllm" / "vllm"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load URSA-8B with the repo-local vLLM runtime and run a short multimodal generation."
    )
    parser.add_argument(
        "--model-path",
        default=str(REPO_ROOT / "checkpoints" / "URSA-8B"),
        help="Local path to the URSA-8B checkpoint directory.",
    )
    parser.add_argument(
        "--image-path",
        default=str(REPO_ROOT / "figures" / "framework.png"),
        help="Path to the example image.",
    )
    parser.add_argument(
        "--question",
        default="How many training stages are shown in this diagram? Answer briefly.",
        help="Question text appended after the <|image|> token.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size passed to vLLM.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        help="vLLM model dtype, for example bfloat16 or float16.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Maximum context length passed to vLLM.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=48,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. Use 0.0 for greedy decoding.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
        help="Fraction of visible GPU memory that vLLM may reserve.",
    )
    parser.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable CUDA graph capture on the first run to make the example more predictable.",
    )
    return parser.parse_args()


def find_vllm_prebuilt_package():
    candidate_roots = list(site.getsitepackages())
    user_site = site.getusersitepackages()
    if user_site:
        candidate_roots.append(user_site)

    for root in candidate_roots:
        candidate = Path(root) / "vllm_pre_built"
        if candidate.exists():
            return candidate
    return None


def repo_local_vllm_is_self_contained():
    has_core_extension = any(VLLM_REPO_PACKAGE.glob("_C*.so"))
    has_flash_attn_package = (VLLM_REPO_PACKAGE / "vllm_flash_attn").exists()
    return has_core_extension and has_flash_attn_package


def bootstrap_repo_local_vllm():
    if "vllm" in sys.modules:
        return

    if not VLLM_REPO_PACKAGE.exists():
        raise FileNotFoundError(f"Repo-local vLLM package does not exist: {VLLM_REPO_PACKAGE}")

    search_paths = [str(VLLM_REPO_PACKAGE)]
    prebuilt_package = find_vllm_prebuilt_package()
    if prebuilt_package is not None:
        search_paths.append(str(prebuilt_package))
    elif not repo_local_vllm_is_self_contained():
        raise RuntimeError(
            "Could not find the compiled vLLM runtime. Run `bash start.sh` in the repo "
            "or activate an environment that contains `vllm_pre_built`."
        )

    spec = importlib.util.spec_from_file_location(
        "vllm",
        VLLM_REPO_PACKAGE / "__init__.py",
        submodule_search_locations=search_paths,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to build an import spec for repo-local vLLM: {VLLM_REPO_PACKAGE}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["vllm"] = module
    spec.loader.exec_module(module)


def build_prompt(model_path: Path, question: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"<|image|>{question}"},
    ]
    return tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
    )


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. URSA-8B via vLLM should be loaded onto GPU.")

    model_path = Path(args.model_path)
    image_path = Path(args.image_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image path does not exist: {image_path}")

    bootstrap_repo_local_vllm()
    from vllm import LLM, SamplingParams
    from vllm.distributed import cleanup_dist_env_and_memory

    prompt = build_prompt(model_path, args.question)
    with Image.open(image_path) as image_file:
        image = image_file.convert("RGB")

    llm = None
    try:
        sampling_params = SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_new_tokens,
        )

        load_start = time.time()
        llm = LLM(
            model=str(model_path),
            tensor_parallel_size=args.tensor_parallel_size,
            dtype=args.dtype,
            max_model_len=args.max_model_len,
            trust_remote_code=False,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=args.enforce_eager,
        )
        load_seconds = time.time() - load_start

        generate_start = time.time()
        outputs = llm.generate(
            [
                {
                    "prompt": prompt,
                    "multi_modal_data": {"image": image},
                }
            ],
            sampling_params=sampling_params,
        )
        generate_seconds = time.time() - generate_start

        result = {
            "backend": "vllm",
            "model_path": str(model_path),
            "image_path": str(image_path),
            "tensor_parallel_size": args.tensor_parallel_size,
            "dtype": args.dtype,
            "enforce_eager": args.enforce_eager,
            "load_seconds": round(load_seconds, 2),
            "generate_seconds": round(generate_seconds, 2),
            "generated_text": outputs[0].outputs[0].text,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        del llm
        cleanup_dist_env_and_memory()


if __name__ == "__main__":
    main()
