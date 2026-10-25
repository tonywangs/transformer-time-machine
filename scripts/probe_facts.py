#!/usr/bin/env python3
"""Screen a fixed list of factual completions using cached, immutable checkpoints.

This is a disclosed example-selection exercise, not a factual-accuracy benchmark.
The candidate list and selection rule are declared before inference. Every result
is retained, including failures and targets unsuitable for the single-token UI.
No downloads, training, in-context answers, or generated intermediate states.
"""
import argparse
import gc
import hashlib
import json
import os
import platform
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Set before either model-loading library is imported. Each loader also passes
# local_files_only=True, so a missing cache entry fails instead of downloading.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_ID = "EleutherAI/pythia-70m-deduped"
CANDIDATES = [
    dict(id="france_original", prompt="The capital of France is", target=" Paris"),
    dict(id="paris_country", prompt="Paris is the capital of", target=" France"),
    dict(id="france_possessive", prompt="France's capital is", target=" Paris"),
    dict(id="rome_country", prompt="Rome is the capital of", target=" Italy"),
    dict(id="tokyo_country", prompt="Tokyo is the capital of", target=" Japan"),
    dict(id="madrid_country", prompt="Madrid is the capital of", target=" Spain"),
    dict(id="ottawa_country", prompt="Ottawa is the capital of", target=" Canada"),
    dict(id="largest_planet", prompt="The largest planet in our solar system is", target=" Jupiter"),
    dict(id="earth_orbit", prompt="The Earth revolves around the", target=" sun"),
    dict(id="week_days", prompt="There are seven days in a", target=" week"),
    dict(id="year_months", prompt="There are twelve months in a", target=" year"),
    dict(id="largest_ocean", prompt="The largest ocean on Earth is the", target=" Pacific"),
    dict(id="water_freezing", prompt="Water freezes at zero degrees", target=" Celsius"),
    dict(id="gold_symbol", prompt="The chemical symbol for gold is", target=" Au"),
    dict(id="human_exhalation", prompt="Humans breathe oxygen and exhale carbon", target=" dioxide"),
    dict(id="triangle_sides", prompt="A triangle has three", target=" sides"),
    # Follow-up round: after the initial 16 failed to select a Paris answer,
    # declare these eight formats together before running any of them.
    dict(id="paris_city_of", prompt="The capital of France is the city of", target=" Paris"),
    dict(id="paris_capital_city", prompt="France's capital city is", target=" Paris"),
    dict(id="paris_city_colon", prompt="France's capital city:", target=" Paris"),
    dict(id="paris_colon", prompt="The capital of France:", target=" Paris"),
    dict(id="paris_country_field", prompt="Country: France\nCapital:", target=" Paris"),
    dict(id="paris_qa", prompt="Q: What is the capital of France?\nA:", target=" Paris"),
    dict(id="paris_answer", prompt="What is the capital of France?\nAnswer:", target=" Paris"),
    dict(id="paris_apposition", prompt="France's capital,", target=" Paris"),
]
SELECTION_RULE = (
    "Eligible only if the target is one tokenizer token, is not top-1 at step 0, "
    "and is top-1 at step 143000. Prefer france_original, then paris_country, "
    "then france_possessive, then the eight follow-up Paris formats in declared order if eligible; otherwise select the eligible candidate "
    "with highest final target probability, breaking ties by declared list order. "
    "If either endpoint is absent, make no recommendation. This selection is "
    "curated for an illustrative success and must not be presented as representative accuracy."
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(value, label):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value),
            f"{label} must be an immutable 40-character Git commit SHA")
    return value


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True,
                        help="Existing Hugging Face cache; this script is always offline")
    parser.add_argument("--traces", type=Path,
                        default=Path(__file__).resolve().parents[1] / "dist/data/traces.json",
                        help="Existing trace manifest supplies immutable checkpoint/tokenizer SHAs")
    parser.add_argument("--steps", default="0,143000",
                        help="Comma-separated recorded training steps, unique and increasing")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    steps = [int(value) for value in args.steps.split(",")]
    require(steps and steps == sorted(set(steps)), "Steps must be unique and increasing")
    require(len(steps) <= 38, "This bounded screen supports at most 38 checkpoints")
    require(args.cache_dir.is_dir(), f"Cache directory does not exist: {args.cache_dir}")
    raw_manifest = args.traces.read_bytes()
    manifest = json.loads(raw_manifest)
    require(manifest.get("schema_version") == 1, "Unsupported trace manifest")
    require(manifest["model"]["id"] == MODEL_ID, "The fixed protocol expects Pythia-70M-deduped")
    checkpoints = []
    for step in steps:
        matches = [item for item in manifest["checkpoints"] if item["step"] == step]
        require(len(matches) == 1, f"Step {step} is absent or duplicated in the manifest")
        checkpoint = matches[0]
        sha(checkpoint["commit_sha"], f"Checkpoint {step}")
        checkpoints.append(checkpoint)
    tokenizer_sha = sha(manifest["provenance"]["tokenizer_commit_sha"], "Tokenizer")

    import huggingface_hub
    import tokenizers
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.perf_counter()
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=tokenizer_sha, cache_dir=args.cache_dir,
        trust_remote_code=False, local_files_only=True)
    candidates = []
    for specification in CANDIDATES:
        candidate = dict(specification)
        candidate["prompt_token_ids"] = tokenizer.encode(candidate["prompt"], add_special_tokens=False)
        candidate["target_token_ids"] = tokenizer.encode(candidate["target"], add_special_tokens=False)
        candidate["single_token_target"] = len(candidate["target_token_ids"]) == 1
        candidate["observations"] = []
        if not candidate["single_token_target"]:
            candidate["not_evaluated_reason"] = "Target is not a single tokenizer token; incompatible with this UI."
        candidates.append(candidate)
    protocol = json.dumps(CANDIDATES, sort_keys=True, separators=(",", ":")).encode()
    report = dict(
        schema_version=1, status="running", model_id=MODEL_ID,
        purpose="Disclosed selection of an illustrative factual success, not an accuracy benchmark",
        screening_history="First 16 candidates were screened first; eight Paris-format variants were then declared together and tested. This report reruns all 24 under one script. Original screen: docs/history/fact-initial-screen.json.",
        created_at=datetime.now(timezone.utc).isoformat(),
        probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        candidate_list_sha256=hashlib.sha256(protocol).hexdigest(),
        source_trace_sha256=hashlib.sha256(raw_manifest).hexdigest(),
        tokenizer_commit_sha=tokenizer_sha, checkpoints=checkpoints,
        selection_rule=SELECTION_RULE,
        environment=dict(
            python_version=platform.python_version(), platform=platform.platform(),
            machine=platform.machine(), torch_version=str(torch.__version__),
            torch_build=torch.__config__.show(), transformers_version=transformers.__version__,
            huggingface_hub_version=huggingface_hub.__version__, tokenizers_version=tokenizers.__version__,
            device="cpu", dtype="float32", attention_implementation="eager", eval_mode=True,
            inference_mode=True, use_cache=False, local_files_only=True,
            num_threads=torch.get_num_threads(), seed=0, add_special_tokens=False),
        rank_definition="1 + number of vocabulary tokens with strictly greater probability; top1 separately compares argmax token ID",
        candidates=candidates, recommendation=None)
    write_report(args.output, report)
    for checkpoint in checkpoints:
        step = checkpoint["step"]
        print(f"Screening step {step}, {checkpoint['commit_sha']} (local cache only)",
              file=sys.stderr, flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, revision=checkpoint["commit_sha"], cache_dir=args.cache_dir,
            local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.float32, attn_implementation="eager")
        model.to(device="cpu", dtype=torch.float32).eval()
        require(model.config._attn_implementation == "eager", "Expected eager attention")
        with torch.inference_mode():
            for candidate in candidates:
                if not candidate["single_token_target"]:
                    continue
                inputs = torch.tensor([candidate["prompt_token_ids"]], dtype=torch.long)
                prediction = model(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                   output_attentions=False, use_cache=False)
                probabilities = prediction.logits[0, -1].float().softmax(dim=-1)
                require(bool(torch.isfinite(probabilities).all()), "Non-finite probabilities")
                target_id = candidate["target_token_ids"][0]
                target_probability = probabilities[target_id]
                values, indices = probabilities.topk(8)
                observation = dict(
                    step=step, commit_sha=checkpoint["commit_sha"],
                    target_probability=float(target_probability),
                    target_rank=1 + int((probabilities > target_probability).sum()),
                    target_is_top1=int(probabilities.argmax()) == target_id,
                    top_tokens=[dict(id=int(index), token=tokenizer.decode([int(index)]), probability=float(value))
                                for value, index in zip(values, indices)])
                candidate["observations"].append(observation)
                print(f"  {candidate['id']}: target rank {observation['target_rank']}, "
                      f"p={observation['target_probability']:.4%}; "
                      f"top={observation['top_tokens'][0]['token']!r}", file=sys.stderr, flush=True)
                del prediction
        write_report(args.output, report)
        del model
        gc.collect()

    eligible = []
    if 0 in steps and 143000 in steps:
        for candidate in candidates:
            by_step = {item["step"]: item for item in candidate["observations"]}
            candidate["eligible_for_selection"] = bool(
                candidate["single_token_target"] and not by_step[0]["target_is_top1"]
                and by_step[143000]["target_is_top1"])
            if candidate["eligible_for_selection"]:
                eligible.append(candidate)
        preferred = [item for label in ("france_original", "paris_country", "france_possessive", *[item["id"] for item in CANDIDATES[16:]])
                     for item in eligible if item["id"] == label]
        selected = preferred[0] if preferred else max(
            eligible, default=None,
            key=lambda item: next(obs["target_probability"] for obs in item["observations"] if obs["step"] == 143000))
        if selected is not None:
            report["recommendation"] = dict(id=selected["id"], prompt=selected["prompt"], target=selected["target"])
    report["status"] = "complete"
    report["wall_seconds"] = time.perf_counter() - started
    write_report(args.output, report)
    print(json.dumps(dict(output=str(args.output), recommendation=report["recommendation"],
                          eligible_candidates=[item["id"] for item in eligible]), indent=2))


if __name__ == "__main__":
    main()
