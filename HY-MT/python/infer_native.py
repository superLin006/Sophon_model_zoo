#!/usr/bin/env python3
"""HY-MT1.5-1.8B deterministic PyTorch baseline and converter reference dump."""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


def tensor_summary(tensor: torch.Tensor) -> dict:
    value = tensor.detach().float().cpu()
    return {
        "shape": list(value.shape),
        "dtype": str(tensor.dtype),
        "min": float(value.min()),
        "max": float(value.max()),
        "mean": float(value.mean()),
        "std": float(value.std()),
        "l2": float(torch.linalg.vector_norm(value)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--cases", default=str(Path(__file__).with_name("test_cases.json")))
    parser.add_argument("--output_dir", default=str(Path(__file__).parents[1] / "outputs" / "baseline"))
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--lightweight",
        action="store_true",
        help="Generate text reports without dumping large hidden-state/KV reference tensors.",
    )
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    load_start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
        attn_implementation="eager",
    ).eval()
    load_seconds = time.perf_counter() - load_start

    config = model.config
    report = {
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": str(model.device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "load_seconds": load_seconds,
        },
        "architecture": {
            "class": type(model).__name__,
            "model_type": config.model_type,
            "hidden_size": config.hidden_size,
            "intermediate_size": config.intermediate_size,
            "num_hidden_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "head_dim": config.head_dim,
            "vocab_size": config.vocab_size,
            "tie_word_embeddings": config.tie_word_embeddings,
            "use_qk_norm": config.use_qk_norm,
            "rope_theta": config.rope_theta,
            "rope_scaling": config.rope_scaling,
        },
        "cases": [],
    }

    for index, case in enumerate(cases):
        messages = [{"role": "user", "content": case["prompt"]}]
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
        ).to(model.device)

        reference = None
        prefill_seconds = None
        if not args.lightweight:
            prefill_start = time.perf_counter()
            with torch.inference_mode():
                reference = model(
                    input_ids=input_ids,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True,
                )
            if model.device.type == "cuda":
                torch.cuda.synchronize()
            prefill_seconds = time.perf_counter() - prefill_start

        generate_start = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        if model.device.type == "cuda":
            torch.cuda.synchronize()
        generate_seconds = time.perf_counter() - generate_start

        new_ids = generated[0, input_ids.shape[1]:]
        decoded = tokenizer.decode(new_ids, skip_special_tokens=True)
        selected = {}
        if reference is not None:
            past = reference.past_key_values
            first_key = past.layers[0].keys if hasattr(past, "layers") else past[0][0]
            first_value = past.layers[0].values if hasattr(past, "layers") else past[0][1]
            selected = {
                "embedding": reference.hidden_states[0],
                "layer0": reference.hidden_states[1],
                "layer15": reference.hidden_states[16],
                "layer31": reference.hidden_states[32],
                "final_hidden": reference.hidden_states[-1],
                "last_logits": reference.logits[:, -1:, :],
                "layer0_key": first_key,
                "layer0_value": first_value,
            }
            np.savez(
                out_dir / f"{case['id']}_reference.npz",
                input_ids=input_ids.cpu().numpy(),
                output_ids=new_ids.cpu().numpy(),
                **{name: value.detach().float().cpu().numpy() for name, value in selected.items()},
            )
        report["cases"].append({
            "id": case["id"],
            "prompt": case["prompt"],
            "rendered_prompt": tokenizer.decode(input_ids[0]),
            "input_ids": input_ids[0].tolist(),
            "input_length": input_ids.shape[1],
            "output_ids": new_ids.tolist(),
            "output": decoded,
            "prefill_seconds": prefill_seconds,
            "generate_seconds": generate_seconds,
            "generated_tokens": len(new_ids),
            "summaries": {name: tensor_summary(value) for name, value in selected.items()},
        })
        print(f"[{index + 1}/{len(cases)}] {case['id']}: {decoded}")

    (out_dir / "baseline.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Baseline written to {out_dir}")


if __name__ == "__main__":
    main()
