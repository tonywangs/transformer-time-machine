#!/usr/bin/env python3
"""Independently validate recorded traces, using only the Python standard library.

This checks internal consistency and file provenance, not whether a model was run.
It never generates observations, imports the extractor, or downloads model files.
"""
import argparse
import ast
import hashlib
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path


class InvalidTrace(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise InvalidTrace(message)


def mapping(value, location):
    require(isinstance(value, dict), f"{location}: expected object")
    return value


def sequence(value, location, length=None):
    require(isinstance(value, list), f"{location}: expected array")
    if length is not None:
        require(len(value) == length,
                f"{location}: expected {length} entries, got {len(value)}")
    return value


def integer(value, location, minimum=0):
    require(type(value) is int and value >= minimum,
            f"{location}: expected integer >= {minimum}")
    return value


def number(value, location, minimum=0.0, maximum=None):
    require(type(value) in (int, float) and math.isfinite(value),
            f"{location}: expected finite number")
    require(value >= minimum and (maximum is None or value <= maximum),
            f"{location}: number outside permitted range")
    return float(value)


def string(value, location, nonempty=True):
    require(isinstance(value, str) and (bool(value) or not nonempty),
            f"{location}: expected {'nonempty ' if nonempty else ''}string")
    return value


def no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def invalid_constant(value):
    raise InvalidTrace(f"JSON contains non-finite constant {value}")


def read_scene_specs(extractor):
    """Read the literal SCENES declaration without executing any source code."""
    module = ast.parse(extractor.read_text(encoding="utf-8"))
    for node in module.body:
        if (isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "SCENES" for t in node.targets)):
            require(isinstance(node.value, ast.List), "extractor SCENES must be a literal list")
            specs = []
            for entry in node.value.elts:
                if isinstance(entry, ast.Dict):
                    specs.append(ast.literal_eval(entry))
                else:
                    require(isinstance(entry, ast.Call)
                            and isinstance(entry.func, ast.Name) and entry.func.id == "dict"
                            and not entry.args and all(k.arg is not None for k in entry.keywords),
                            "extractor SCENES contains unsupported nonliteral construction")
                    specs.append({k.arg: ast.literal_eval(k.value) for k in entry.keywords})
            require(bool(specs), "extractor SCENES is empty")
            return specs
    raise InvalidTrace("extractor has no literal SCENES declaration")


def entropy_term(p):
    return -p * math.log(p) if p > 0 else 0.0


def concentration_bounds(row):
    # Rounded probabilities were computed before concentration. Bound their
    # possible entropy instead of requiring exact equality after rounding.
    radius = 0.0000006  # half a decimal unit plus float32 serialization allowance
    low_entropy = high_entropy = 0.0
    for p in row:
        lo, hi = max(0.0, p - radius), min(1.0, p + radius)
        terms = [entropy_term(lo), entropy_term(hi)]
        low_entropy += min(terms)
        if lo <= 1.0 / math.e <= hi:
            terms.append(1.0 / math.e)
        high_entropy += max(terms)
    denominator = math.log(len(row))
    return (max(0.0, 1.0 - high_entropy / denominator) - 0.000002,
            min(1.0, 1.0 - low_entropy / denominator) + 0.000002)


def validate(trace_path, extractor_path, sidecar_path):
    raw = trace_path.read_bytes()
    actual_digest = hashlib.sha256(raw).hexdigest()
    sidecar = sidecar_path.read_text(encoding="utf-8").strip()
    match = re.fullmatch(r"([0-9a-f]{64})[ \t]+\*?(.+)", sidecar)
    require(match is not None, "SHA-256 sidecar must contain one digest and filename")
    require(match.group(2) == trace_path.name, "SHA-256 sidecar names a different file")
    require(match.group(1) == actual_digest, "SHA-256 sidecar does not match trace bytes")
    data = mapping(json.loads(raw, object_pairs_hook=no_duplicate_keys,
                              parse_constant=invalid_constant), "root")
    require(type(data.get("schema_version")) is int and data["schema_version"] == 1,
            "schema_version must be integer 1")
    metadata = mapping(data.get("model"), "model")
    require(metadata.get("id") == "EleutherAI/pythia-70m-deduped",
            "unexpected model: this release validator requires EleutherAI/pythia-70m-deduped")
    for field, expected in (("layers", 6), ("heads", 8), ("hidden_size", 512),
                            ("parameters", 70_426_624)):
        require(integer(metadata.get(field), f"model.{field}", 1) == expected,
                f"model.{field}: expected {expected} for Pythia 70M")
    string(metadata.get("label"), "model.label")
    model_url = "https://huggingface.co/" + metadata["id"]
    require(metadata.get("source_url") == model_url, "incorrect model source_url")
    layers, heads = metadata["layers"], metadata["heads"]
    provenance = mapping(data.get("provenance"), "provenance")
    source_digest = hashlib.sha256(extractor_path.read_bytes()).hexdigest()
    require(provenance.get("extractor_sha256") == source_digest,
            "extractor_sha256 does not match the current extractor script")
    require(provenance.get("dtype") == "float32", "dtype must be float32")
    require(provenance.get("device") in ("cpu", "cuda"), "unknown extraction device")
    for key in ("torch_version", "transformers_version"):
        string(provenance.get(key), "provenance." + key)
    timestamp = datetime.fromisoformat(string(provenance.get("extracted_at"),
                                             "provenance.extracted_at").replace("Z", "+00:00"))
    require(timestamp.tzinfo is not None, "extracted_at must include a timezone")
    number(provenance.get("extraction_wall_seconds"), "provenance.extraction_wall_seconds")
    require(re.fullmatch(r"[0-9a-f]{40}", str(provenance.get("tokenizer_commit_sha"))),
            "tokenizer_commit_sha must be a 40-character lowercase Git SHA")
    notes = sequence(provenance.get("notes"), "provenance.notes")
    require(bool(notes), "provenance.notes must not be empty")
    for note in notes:
        string(note, "provenance.notes[]")

    checkpoints = sequence(data.get("checkpoints"), "checkpoints")
    require(bool(checkpoints), "no recorded checkpoints")
    steps, commits = [], set()
    for i, checkpoint in enumerate(checkpoints):
        checkpoint = mapping(checkpoint, f"checkpoints[{i}]")
        step = integer(checkpoint.get("step"), f"checkpoints[{i}].step")
        require(not steps or step > steps[-1], "checkpoint steps must be unique and increasing")
        require(checkpoint.get("revision") == f"step{step}", "checkpoint revision/step mismatch")
        sha = string(checkpoint.get("commit_sha"), "checkpoint.commit_sha")
        require(re.fullmatch(r"[0-9a-f]{40}", sha), "invalid checkpoint Git SHA")
        require(sha not in commits, "two checkpoints resolve to the same commit")
        require(checkpoint.get("source_url") == f"{model_url}/tree/{sha}",
                "checkpoint source_url does not pin its recorded commit")
        steps.append(step)
        commits.add(sha)

    specs = read_scene_specs(extractor_path)
    scenes = sequence(data.get("scenes"), "scenes", len(specs))
    scene_ids = set()
    token_spellings = {}
    rows_checked = 0
    max_row_error = 0.0
    for index, (scene, spec) in enumerate(zip(scenes, specs)):
        where = f"scenes[{index}]"
        scene = mapping(scene, where)
        for key, value in spec.items():
            require(scene.get(key) == value, f"{where}.{key}: does not match extractor scene")
        require(scene["id"] not in scene_ids, "duplicate scene ID")
        scene_ids.add(scene["id"])
        ids = sequence(scene.get("token_ids"), where + ".token_ids")
        require(len(ids) >= 2, f"{where}: concentration requires at least two prompt tokens")
        tokens = sequence(scene.get("tokens"), where + ".tokens", len(ids))
        for token_id, token in zip(ids, tokens):
            integer(token_id, where + ".token_ids[]")
            string(token, where + ".tokens[]", nonempty=False)
            require(token_id not in token_spellings or token_spellings[token_id] == token,
                    "inconsistent display text for the same token ID")
            token_spellings[token_id] = token
        require("".join(tokens) == scene["prompt"], f"{where}: decoded tokens do not reconstruct prompt")
        targets = sequence(scene.get("target_token_ids"), where + ".target_token_ids", 1)
        integer(targets[0], where + ".target_token_ids[0]")
        frames = sequence(scene.get("frames"), where + ".frames", len(steps))
        for frame_index, frame in enumerate(frames):
            location = f"{where}.frames[{frame_index}]"
            frame = mapping(frame, location)
            require(type(frame.get("step")) is int and frame["step"] == steps[frame_index],
                    location + ": frame step/order does not match checkpoints")
            target_probability = number(frame.get("target_probability"),
                                        location + ".target_probability", maximum=1.0)
            top = sequence(frame.get("top_tokens"), location + ".top_tokens", 8)
            previous = 1.0
            top_ids, total = set(), 0.0
            found_target = False
            for rank, candidate in enumerate(top):
                candidate = mapping(candidate, location + f".top_tokens[{rank}]")
                token_id = integer(candidate.get("id"), "top token ID")
                require(token_id not in top_ids, location + ": duplicate top token ID")
                top_ids.add(token_id)
                token = string(candidate.get("token"), "top token text", nonempty=False)
                require(token_id not in token_spellings or token_spellings[token_id] == token,
                        "inconsistent display text for the same token ID")
                token_spellings[token_id] = token
                probability = number(candidate.get("probability"), "top token probability", maximum=1.0)
                require(probability <= previous + 1e-12, location + ": top probabilities not descending")
                previous = probability
                total += probability
                if token_id == targets[0]:
                    found_target = True
                    require(abs(probability - target_probability) <= 1e-7,
                            location + ": target probability disagrees with top-token probability")
                    require(token == scene["target_label"], location + ": target token text mismatch")
            require(total <= 1.0 + 1e-6, location + ": top-eight probabilities sum above one")
            require(found_target or target_probability <= previous + 1e-7,
                    location + ": omitted target outranks an alleged top-eight token")
            attention = sequence(frame.get("attention"), location + ".attention", layers)
            concentration = sequence(frame.get("concentration"), location + ".concentration", layers)
            n = len(ids)
            for layer_index in range(layers):
                layer = sequence(attention[layer_index], location + ".attention.layer", heads)
                focus_layer = sequence(concentration[layer_index], location + ".concentration.layer", heads)
                for head_index in range(heads):
                    head_location = location + f".attention[{layer_index}][{head_index}]"
                    head = sequence(layer[head_index], head_location, n)
                    for query, row in enumerate(head):
                        row = sequence(row, head_location + f"[{query}]", n)
                        for key, value in enumerate(row):
                            value = number(value, head_location + f"[{query}][{key}]", maximum=1.0)
                            require(abs(value - round(value, 6)) <= 8e-8,
                                    head_location + ": weights not rounded to six decimal places")
                            require(key <= query or value == 0.0,
                                    head_location + ": future-token attention is nonzero")
                        error = abs(math.fsum(row) - 1.0)
                        require(error <= n * 0.0000006 + 0.000002,
                                head_location + f": row {query} sum error {error:g} exceeds rounding bound")
                        max_row_error = max(max_row_error, error)
                        rows_checked += 1
                    focus = number(focus_layer[head_index], "concentration", maximum=1.0)
                    lower, upper = concentration_bounds(head[-1])
                    require(lower <= focus <= upper,
                            head_location + ": concentration disagrees with last-query entropy")
    return {"valid": True, "model": metadata["id"], "checkpoints": len(steps),
            "scenes": len(scenes), "frames": len(steps) * len(scenes),
            "attention_rows_checked": rows_checked, "max_row_sum_error": max_row_error,
            "sha256": actual_digest, "extractor_sha256": source_digest,
            "limits": "Internal consistency and hash verification; not an independent model rerun."}


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, default=root / "dist/data/traces.json")
    parser.add_argument("--extractor", type=Path, default=root / "scripts/extract_traces.py")
    parser.add_argument("--sha256", type=Path, help="Defaults to traces path with .sha256 suffix")
    args = parser.parse_args()
    try:
        summary = validate(args.traces, args.extractor, args.sha256 or args.traces.with_suffix(".sha256"))
    except (OSError, ValueError, TypeError, KeyError, SyntaxError) as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
