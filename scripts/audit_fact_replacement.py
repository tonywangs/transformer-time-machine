#!/usr/bin/env python3
"""Audit a factual-example replacement independently of the extraction code."""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def max_difference(before, after):
    if isinstance(before, list):
        require(isinstance(after, list) and len(before) == len(after), "Numeric array shapes differ")
        return max((max_difference(a, b) for a, b in zip(before, after)), default=0.0)
    return abs(before - after)


def trajectory(scene):
    targets = scene["target_token_ids"]
    records = []
    for frame in scene["frames"]:
        rank = next((i + 1 for i, token in enumerate(frame["top_tokens"]) if token["id"] in targets), None)
        records.append(dict(step=frame["step"], target_rank_in_top8=rank,
                            target_probability=frame["target_probability"],
                            top_token=frame["top_tokens"][0]["token"]))
    wins = [record["step"] for record in records if record["target_rank_in_top8"] == 1]
    losses = [dict(from_step=a["step"], to_step=b["step"])
              for a, b in zip(records, records[1:])
              if a["target_rank_in_top8"] == 1 and b["target_rank_in_top8"] != 1]
    return dict(prompt=scene["prompt"], target=scene["target_label"],
                top1_count=len(wins), first_top1_step=wins[0] if wins else None,
                all_top1_steps=wins, top1_reversals=losses,
                peak=max(records, key=lambda record: record["target_probability"]),
                final=records[-1], checkpoints=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("before", "after", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--screen", type=Path,
                        default=Path(__file__).resolve().parents[1] / "docs/fact-candidate-screen.json")
    args = parser.parse_args()
    before, before_hash = read(args.before)
    after, after_hash = read(args.after)
    screen, screen_hash = read(args.screen)
    require(before["model"] == after["model"], "Model identity or dimensions changed")
    require(before["checkpoints"] == after["checkpoints"], "Checkpoint manifest changed")
    require(before["provenance"]["tokenizer_commit_sha"] == after["provenance"]["tokenizer_commit_sha"],
            "Tokenizer identity changed")
    old = {scene["id"]: scene for scene in before["scenes"]}
    new = {scene["id"]: scene for scene in after["scenes"]}
    require(set(old) == set(new) == {"story", "echo", "capital"}, "Unexpected scene IDs")
    recommendation = screen["recommendation"]
    require(screen["status"] == "complete" and recommendation is not None, "No completed screening recommendation")
    require(new["capital"]["prompt"] == recommendation["prompt"]
            and new["capital"]["target_label"] == recommendation["target"], "Replacement differs from screened recommendation")
    unchanged = {}
    for identity in ("story", "echo"):
        a, b = old[identity], new[identity]
        for field in ("prompt", "target_label", "token_ids", "tokens", "target_token_ids"):
            require(a[field] == b[field], f"{identity}: {field} changed")
        require([f["step"] for f in a["frames"]] == [f["step"] for f in b["frames"]], "Frame steps changed")
        differences = {}
        for field in ("target_probability", "attention", "concentration"):
            differences[field] = max(max_difference(x[field], y[field]) for x, y in zip(a["frames"], b["frames"]))
        top_identity_equal = all([(t["id"], t["token"]) for t in x["top_tokens"]]
                                 == [(t["id"], t["token"]) for t in y["top_tokens"]]
                                 for x, y in zip(a["frames"], b["frames"]))
        differences["top_probabilities_by_rank"] = max(max_difference(
            [t["probability"] for t in x["top_tokens"]], [t["probability"] for t in y["top_tokens"]])
            for x, y in zip(a["frames"], b["frames"]))
        unchanged[identity] = dict(scene_exact=a == b, top_token_identities_equal=top_identity_equal,
                                   maximum_absolute_differences=differences)
    report = dict(schema_version=1, model_identity_unchanged=True, checkpoint_manifest_unchanged=True,
                  tokenizer_identity_unchanged=True, checkpoint_count=len(after["checkpoints"]),
                  hashes=dict(before_trace=before_hash, after_trace=after_hash, screening_report=screen_hash,
                              audit_script=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                              before_extractor=before["provenance"]["extractor_sha256"],
                              after_extractor=after["provenance"]["extractor_sha256"]),
                  unchanged_scenes=unchanged, old_fact=trajectory(old["capital"]), new_fact=trajectory(new["capital"]),
                  total_attention_entries=sum(len(row) for s in after["scenes"] for f in s["frames"]
                                              for layer in f["attention"] for head in layer for row in head))
    require(report["old_fact"]["top1_count"] == 0, "Before dataset does not reproduce the reported Paris failure")
    require(report["new_fact"]["checkpoints"][0]["target_rank_in_top8"] != 1, "Replacement already wins at initialization")
    require(report["new_fact"]["final"]["target_rank_in_top8"] == 1, "Replacement does not finish with the expected answer first")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in ("old_fact", "new_fact")}, indent=2))
    print(json.dumps({key: value for key, value in report["new_fact"].items() if key != "checkpoints"}, indent=2))


if __name__ == "__main__":
    main()
