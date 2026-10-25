#!/usr/bin/env python3
"""Export observed Pythia checkpoints. No training, remote code, or synthetic frames."""
import argparse
import gc
import hashlib
import json
import math
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SCENES = [
    dict(id="echo", title="Follow the pattern", kicker="Repeated sequence",
         description="The same color sequence appears twice. Track the chance of continuing it with purple, and explore which earlier tokens each head attends to.",
         prompt=" red blue green yellow orange purple red blue green yellow orange", target_label=" purple"),
    dict(id="story", title="A familiar beginning", kicker="Language pattern",
         description="A familiar phrase, seen at different points in training. Watch the probability of time alongside the model's other next-token guesses.",
         prompt="Once upon a", target_label=" time"),
    dict(id="capital", title="A fact taking shape", kicker="Factual completion",
         description="An illustrative factual prompt. Follow Paris through training; its probability can rise or fall, and a small model may prefer another completion.",
         prompt="The capital of France is", target_label=" Paris"),
]


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
    tmp.replace(path)


def content_hash(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def validate_shard(shard, checkpoint, scene_specs, extractor_hash, environment,
                   environment_hash, model_id, expected_metadata=None):
    """Reject stale, incomplete, or mixed-source shards before appending any frames."""
    if not isinstance(shard, dict) or shard.get("schema_version") != 2:
        raise ValueError("unsupported extraction shard schema")
    expected = dict(checkpoint=checkpoint, scene_specs=scene_specs,
                    scene_ids=[s["id"] for s in scene_specs],
                    extractor_sha256=extractor_hash, environment=environment,
                    environment_sha256=environment_hash)
    for field, value in expected.items():
        if shard.get(field) != value:
            raise ValueError(f"cached shard {field} mismatch")
    frames = shard.get("frames")
    if not isinstance(frames, list) or len(frames) != len(scene_specs):
        raise ValueError("cached shard frame count mismatch")
    for scene, frame in zip(scene_specs, frames):
        if (not isinstance(frame, dict) or frame.get("scene_id") != scene["id"]
                or type(frame.get("step")) is not int or frame["step"] != checkpoint["step"]):
            raise ValueError("cached frame scene/checkpoint mismatch")
    metadata = shard.get("model")
    if not isinstance(metadata, dict) or metadata.get("id") != model_id:
        raise ValueError("cached shard model identity mismatch")
    if (metadata.get("label") != model_id.rsplit("/", 1)[-1]
            or metadata.get("source_url") != f"https://huggingface.co/{model_id}"):
        raise ValueError("cached shard model display metadata mismatch")
    for field in ("parameters", "layers", "heads", "hidden_size"):
        if type(metadata.get(field)) is not int or metadata[field] <= 0:
            raise ValueError(f"invalid cached model {field}")
    if expected_metadata is not None and metadata != expected_metadata:
        raise ValueError("model metadata differs across checkpoints")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-70m-deduped")
    ap.add_argument("--steps", default="0,64,512,2000,10000,30000,70000,143000")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = ap.parse_args()
    import torch
    import transformers
    import huggingface_hub
    import tokenizers
    from huggingface_hub import HfApi
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.manual_seed(0)
    steps = [int(s) for s in args.steps.split(",")]
    if len(steps) != len(set(steps)) or steps != sorted(steps):
        raise ValueError("Checkpoint steps must be unique and sorted")
    out = Path(args.output).resolve()
    cache = Path(args.cache_dir).resolve()
    shards = cache / "trace_shards_v2"
    shards.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    final_sha = api.model_info(args.model, revision="step143000").sha
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=final_sha, cache_dir=cache,
                                             trust_remote_code=False)
    scenes = []
    for spec in SCENES:
        ids = tokenizer.encode(spec["prompt"], add_special_tokens=False)
        targets = tokenizer.encode(spec["target_label"], add_special_tokens=False)
        if len(targets) != 1:
            raise ValueError(f"Expected single-token illustrative completion: {spec['target_label']!r}")
        scenes.append(dict(**spec, token_ids=ids,
                           tokens=[tokenizer.decode([i]) for i in ids],
                           target_token_ids=targets, frames=[]))
    extractor_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    tokenizer_state = (tokenizer.backend_tokenizer.to_str() if tokenizer.is_fast else
                       json.dumps(tokenizer.get_vocab(), sort_keys=True))
    device_info = dict(type=args.device, machine=platform.machine(), processor=platform.processor())
    if args.device == "cuda":
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        device_info.update(index=torch.cuda.current_device(), name=properties.name,
                           capability=list(torch.cuda.get_device_capability()),
                           total_memory=properties.total_memory,
                           multiprocessor_count=properties.multi_processor_count)
    environment = dict(
        python_version=platform.python_version(), python_implementation=platform.python_implementation(),
        platform=platform.platform(), torch_version=str(torch.__version__),
        torch_build=torch.__config__.show(), transformers_version=transformers.__version__,
        huggingface_hub_version=huggingface_hub.__version__, tokenizers_version=tokenizers.__version__,
        cuda_version=torch.version.cuda, cudnn_version=torch.backends.cudnn.version(),
        device=device_info, dtype="float32", num_threads=torch.get_num_threads(),
        num_interop_threads=torch.get_num_interop_threads(), seed=0,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        matmul_precision=torch.get_float32_matmul_precision(),
        cuda_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        attention_implementation="eager", inference_mode=True, eval_mode=True, use_cache=False,
        tokenizer=dict(model_id=args.model, commit_sha=final_sha,
                       implementation=type(tokenizer).__name__, is_fast=tokenizer.is_fast,
                       vocabulary_size=len(tokenizer), state_sha256=hashlib.sha256(tokenizer_state.encode()).hexdigest(),
                       special_tokens_map={k: str(v) for k, v in tokenizer.special_tokens_map.items()},
                       add_special_tokens=False))
    environment_hash = content_hash(environment)
    scene_specs = [{k: v for k, v in s.items() if k != "frames"} for s in scenes]
    result = dict(schema_version=1, model=None,
                  provenance=dict(extracted_at=datetime.now(timezone.utc).isoformat(),
                      device=args.device, platform=platform.platform(), dtype="float32",
                      torch_version=torch.__version__, transformers_version=transformers.__version__,
                      tokenizer_commit_sha=final_sha, extractor_sha256=extractor_hash,
                      environment=environment, environment_sha256=environment_hash,
                      notes=["Actual eval-mode checkpoint forward passes; no synthetic intermediate frames.",
                             "Targets are illustrative next tokens, not an accuracy benchmark.",
                             "Attention weights are rounded to six decimal places.",
                             "Concentration measures last-query focus, not head importance or causality.",
                             "No implicit beginning-of-sequence token; pretrained tokenizer from final checkpoint.",
                             "Training steps are sampled at unequal intervals; playback visits recorded checkpoints."]),
                  checkpoints=[], scenes=scenes)
    started = time.perf_counter()
    for step in steps:
        revision = f"step{step}"
        sha = api.model_info(args.model, revision=revision).sha
        checkpoint = dict(step=step, revision=revision, commit_sha=sha,
                          source_url=f"https://huggingface.co/{args.model}/tree/{sha}")
        shard_path = shards / f"{args.model.replace('/', '--')}-{sha}-{extractor_hash[:12]}-{environment_hash[:12]}.json"
        shard = None
        if shard_path.exists():
            try:
                candidate = json.loads(shard_path.read_text())
                validate_shard(candidate, checkpoint, scene_specs, extractor_hash, environment,
                               environment_hash, args.model, result["model"])
                shard = candidate
                print(f"Reusing verified extraction shard {revision}", flush=True)
            except (ValueError, UnicodeError) as error:
                print(f"Ignoring invalid extraction shard {revision}: {error}", flush=True)
        if shard is None:
            print(f"Loading {args.model} {revision} ({sha})", flush=True)
            model = AutoModelForCausalLM.from_pretrained(args.model, revision=sha, cache_dir=cache,
                       torch_dtype=torch.float32, attn_implementation="eager", trust_remote_code=False)
            model = model.to(args.device).eval()
            metadata = dict(id=args.model, label=args.model.rsplit("/", 1)[-1],
                parameters=sum(p.numel() for p in model.parameters()),
                layers=model.config.num_hidden_layers, heads=model.config.num_attention_heads,
                hidden_size=model.config.hidden_size, source_url=f"https://huggingface.co/{args.model}")
            if result["model"] is not None and metadata != result["model"]:
                raise ValueError("Model metadata differs across checkpoints")
            frames = []
            with torch.inference_mode():
                for scene in scenes:
                    tokens = torch.tensor([scene["token_ids"]], device=args.device)
                    prediction = model(input_ids=tokens, attention_mask=torch.ones_like(tokens),
                                       output_attentions=True, use_cache=False)
                    if not prediction.attentions or any(a is None for a in prediction.attentions):
                        raise RuntimeError("Model did not return actual attention matrices")
                    probs = torch.softmax(prediction.logits[0, -1].float(), dim=-1)
                    values, indices = probs.topk(8)
                    top = [dict(id=int(i), token=tokenizer.decode([int(i)]), probability=float(v))
                           for v, i in zip(values.cpu(), indices.cpu())]
                    attention = torch.stack([a[0].float().cpu() for a in prediction.attentions])
                    if not torch.isfinite(attention).all() or not torch.isfinite(probs).all():
                        raise RuntimeError("Non-finite model observation")
                    n = tokens.shape[-1]
                    if not torch.allclose(attention.sum(-1), torch.ones_like(attention.sum(-1)), atol=1e-5):
                        raise RuntimeError("Attention rows do not sum to one")
                    if torch.triu(attention, diagonal=1).abs().max() > 1e-7:
                        raise RuntimeError("Causal attention mask violated")
                    entropy = -(attention[:, :, -1, :] * attention[:, :, -1, :].clamp_min(1e-30).log()).sum(-1)
                    focus = (1 - entropy / math.log(n)).clamp(0, 1)
                    rounded = (attention * 1_000_000).round().div(1_000_000).tolist()
                    frame = dict(scene_id=scene["id"], step=step, top_tokens=top,
                                 target_probability=float(probs[scene["target_token_ids"]].sum()),
                                 attention=rounded, concentration=focus.tolist())
                    frames.append(frame)
                    print(f"  {scene['id']}: top={top[0]['token']!r} ({top[0]['probability']:.3%}), "
                          f"target={frame['target_probability']:.3%}", flush=True)
                    del prediction
            shard = dict(schema_version=2, model=metadata, checkpoint=checkpoint, frames=frames,
                         scene_ids=[s["id"] for s in scenes], scene_specs=scene_specs,
                         extractor_sha256=extractor_hash, environment=environment,
                         environment_sha256=environment_hash,
                         extracted_at=datetime.now(timezone.utc).isoformat())
            validate_shard(shard, checkpoint, scene_specs, extractor_hash, environment,
                           environment_hash, args.model, result["model"])
            atomic_json(shard_path, shard)
            del model
            gc.collect()
        result["model"] = shard["model"]
        result["checkpoints"].append(checkpoint)
        for scene, frame in zip(result["scenes"], shard["frames"]):
            scene["frames"].append(frame)
        # A partial dataset is valid: checkpoint count is read from the file, never fabricated.
        atomic_json(out, result)
    result["provenance"]["extraction_wall_seconds"] = round(time.perf_counter() - started, 3)
    atomic_json(out, result)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    out.with_suffix(".sha256").write_text(f"{digest}  {out.name}\n")
    print(f"Saved {len(steps)} checkpoints × {len(scenes)} prompts; {out.stat().st_size:,} bytes; sha256={digest}", flush=True)


if __name__ == "__main__":
    main()
