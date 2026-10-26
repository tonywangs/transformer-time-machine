"""Bake only pinned Pythia weights/tokenizer into the custom worker image."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import hf_hub_download


def main():
    manifest = json.loads(Path("/service/model-manifest.json").read_text())

    def fetch(checkpoint):
        for filename in ("config.json", "model.safetensors"):
            hf_hub_download(manifest["model"]["id"], filename,
                            revision=checkpoint["commit_sha"], cache_dir="/models")
        print(f"Cached immutable checkpoint {checkpoint['step']}", flush=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch, manifest["checkpoints"]))
    for filename in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        hf_hub_download(manifest["model"]["id"], filename,
                        revision=manifest["tokenizer_commit_sha"], cache_dir="/models")


if __name__ == "__main__":
    main()
