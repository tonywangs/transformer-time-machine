#!/usr/bin/env python3
"""Replay one immutable checkpoint offline and compare fresh observations.

Loads weights and tokenizer directly through Transformers, without importing the
extractor or reading its trace shards. Requires an already populated HF cache.
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

# Set before importing either library. Both loaders also require local files.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

ATTENTION_TOLERANCE = 6e-7
PROBABILITY_TOLERANCE = 1e-7


def require(condition, message):
    if not condition:
        raise ValueError(message)


def commit_sha(value, label):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value),
            f"{label} must be an immutable 40-character Git commit SHA")
    return value


def replay(args):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.perf_counter()
    raw = args.traces.read_bytes()  # One snapshot if another process replaces it.
    data = json.loads(raw)
    require(data.get("schema_version") == 1, "Expected trace schema version 1")
    require(args.cache_dir.is_dir(), f"Cache directory does not exist: {args.cache_dir}")
    checkpoints = data["checkpoints"]
    require(bool(checkpoints), "Trace contains no checkpoints")
    step = args.step if args.step is not None else checkpoints[-1]["step"]
    selected = [checkpoint for checkpoint in checkpoints if checkpoint["step"] == step]
    require(len(selected) == 1, f"Expected exactly one recorded checkpoint for step {step}")
    checkpoint = selected[0]
    model_sha = commit_sha(checkpoint["commit_sha"], "Model revision")
    tokenizer_sha = commit_sha(data["provenance"]["tokenizer_commit_sha"], "Tokenizer revision")
    metadata = data["model"]
    model_id = metadata["id"]
    require(isinstance(model_id, str)
            and re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", model_id),
            "Expected a Hugging Face owner/repository model ID")
    require(len(data["scenes"]) == 3, "This release expects all three recorded scenes")
    require(len({scene["id"] for scene in data["scenes"]}) == 3, "Duplicate scene IDs")

    torch.set_num_threads(min(4, os.cpu_count() or 1))
    print(f"Replaying {model_id}, step {step}, commit {model_sha}; local cache only",
          file=sys.stderr, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=tokenizer_sha, cache_dir=args.cache_dir,
        local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=model_sha, cache_dir=args.cache_dir,
        local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.float32, attn_implementation="eager")
    model.to(device="cpu", dtype=torch.float32).eval()
    require(not model.training, "Model must be in evaluation mode")
    require(model.config._attn_implementation == "eager", "Expected eager attention")
    dimensions = {
        "layers": model.config.num_hidden_layers,
        "heads": model.config.num_attention_heads,
        "hidden_size": model.config.hidden_size,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    for key, observed in dimensions.items():
        require(metadata[key] == observed, f"Recorded {key} does not match loaded model")

    reports = []
    with torch.inference_mode():
        for scene in data["scenes"]:
            label = scene["id"]
            frames = [frame for frame in scene["frames"] if frame["step"] == step]
            require(len(frames) == 1, f"{label}: expected exactly one frame for step {step}")
            stored = frames[0]
            token_ids = tokenizer.encode(scene["prompt"], add_special_tokens=False)
            decoded = [tokenizer.decode([token]) for token in token_ids]
            target_ids = tokenizer.encode(scene["target_label"], add_special_tokens=False)
            require(token_ids == scene["token_ids"], f"{label}: prompt token IDs differ")
            require(decoded == scene["tokens"], f"{label}: decoded prompt tokens differ")
            require(target_ids == scene["target_token_ids"], f"{label}: target token IDs differ")
            require(len(target_ids) == 1, f"{label}: expected one target token")
            require(tokenizer.decode(target_ids) == scene["target_label"],
                    f"{label}: target token does not decode to its label")

            inputs = torch.tensor([token_ids], dtype=torch.long, device="cpu")
            output = model(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                           output_attentions=True, use_cache=False)
            probabilities = output.logits[0, -1].float().softmax(dim=-1)
            require(bool(torch.isfinite(probabilities).all()), f"{label}: non-finite probabilities")
            require(output.attentions is not None
                    and all(tensor is not None for tensor in output.attentions),
                    f"{label}: model did not return attention weights")
            observed_attention = torch.stack([layer[0] for layer in output.attentions]).float()
            require(bool(torch.isfinite(observed_attention).all()), f"{label}: non-finite attention")
            saved_attention = torch.tensor(stored["attention"], dtype=torch.float64)
            expected_shape = (dimensions["layers"], dimensions["heads"], len(token_ids), len(token_ids))
            require(tuple(observed_attention.shape) == expected_shape,
                    f"{label}: unexpected replay attention shape")
            require(tuple(saved_attention.shape) == expected_shape,
                    f"{label}: stored attention shape differs")
            require(bool(torch.isfinite(saved_attention).all()), f"{label}: non-finite stored attention")
            # Compare every entry, both before and after the documented rounding.
            raw_attention_error = float((observed_attention.double() - saved_attention).abs().max())
            rounded_attention = torch.round(observed_attention * 1_000_000) / 1_000_000
            rounded_attention_error = float((rounded_attention.double() - saved_attention).abs().max())
            require(raw_attention_error <= ATTENTION_TOLERANCE,
                    f"{label}: raw attention max error {raw_attention_error:.10g} exceeds "
                    f"{ATTENTION_TOLERANCE:g}")
            require(rounded_attention_error <= ATTENTION_TOLERANCE,
                    f"{label}: rounded attention max error {rounded_attention_error:.10g} exceeds "
                    f"{ATTENTION_TOLERANCE:g}")

            values, indices = torch.topk(probabilities, k=8, largest=True, sorted=True)
            require(len(stored["top_tokens"]) == 8, f"{label}: expected eight stored top tokens")
            require(indices.tolist() == [entry["id"] for entry in stored["top_tokens"]],
                    f"{label}: top-eight token IDs/order differ")
            require([tokenizer.decode([token]) for token in indices.tolist()]
                    == [entry["token"] for entry in stored["top_tokens"]],
                    f"{label}: top-eight decoded tokens differ")
            saved_probabilities = torch.tensor(
                [entry["probability"] for entry in stored["top_tokens"]], dtype=torch.float64)
            require(bool(torch.isfinite(saved_probabilities).all()),
                    f"{label}: non-finite stored top probabilities")
            top_probability_error = float((values.double() - saved_probabilities).abs().max())
            target_probability = float(probabilities[target_ids].sum())
            target_probability_error = abs(target_probability - stored["target_probability"])
            require(top_probability_error <= PROBABILITY_TOLERANCE,
                    f"{label}: top probability max error {top_probability_error:.10g} exceeds "
                    f"{PROBABILITY_TOLERANCE:g}")
            require(target_probability_error <= PROBABILITY_TOLERANCE,
                    f"{label}: target probability error {target_probability_error:.10g} exceeds "
                    f"{PROBABILITY_TOLERANCE:g}")
            reports.append({
                "scene": label, "prompt_tokens": len(token_ids),
                "tokenization_and_top_ids_match": True,
                "attention_entries_checked": saved_attention.numel(),
                "max_raw_attention_error": raw_attention_error,
                "max_rounded_attention_error": rounded_attention_error,
                "max_top_probability_error": top_probability_error,
                "target_probability_error": target_probability_error,
            })
            del output

    error_keys = ("max_raw_attention_error", "max_rounded_attention_error",
                  "max_top_probability_error", "target_probability_error")
    return {
        "valid": True, "model": model_id, "step": step,
        "model_commit_sha": model_sha, "tokenizer_commit_sha": tokenizer_sha,
        "trace_sha256": hashlib.sha256(raw).hexdigest(),
        "replay_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "device": "cpu", "dtype": "float32", "attention_implementation": "eager",
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "local_files_only": True,
        "attention_tolerance": ATTENTION_TOLERANCE,
        "probability_tolerance": PROBABILITY_TOLERANCE,
        "scenes": reports,
        "max_errors": {key: max(report[key] for report in reports) for key in error_keys},
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, default=root / "dist/data/traces.json")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--step", type=int, help="Recorded step to replay; defaults to the last checkpoint")
    args = parser.parse_args()
    try:
        result = replay(args)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, ImportError) as exc:
        print(f"REPLAY FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
