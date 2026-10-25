#!/usr/bin/env python3
"""Run the recorded CPU experiment on Modal, with bounded resources and runtime.

Optional runner: requires Python 3.11, the Modal client, and an authenticated account locally.
Only the four named pipeline files are uploaded; weights stay in an ephemeral
container. This creates a temporary run, not a persistent deployment or schedule.
"""
from pathlib import Path

import modal

SOURCE = Path(__file__).resolve().parent
app = modal.App("transformer-time-machine-extraction")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.8.0", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("transformers==4.56.2", "huggingface_hub==0.35.3")
    .env({"HF_HUB_DISABLE_XET": "0", "HF_XET_CHUNK_CACHE_SIZE_BYTES": "0",
          "HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false"})
)
for filename in ("extract_traces.py", "validate_traces.py", "replay_checkpoint.py",
                 "checkpoint-steps.txt"):
    image = image.add_local_file(SOURCE / filename, f"/pipeline/{filename}", copy=True)


@app.function(image=image, cpu=(4, 4), memory=(8192, 8192), timeout=900,
              max_containers=1, retries=0, scaledown_window=2,
              serialized=True, include_source=False)
def extract():
    import concurrent.futures
    import gzip
    import json
    import subprocess
    import sys
    import time
    from pathlib import Path
    from huggingface_hub import HfApi, hf_hub_download

    started = time.perf_counter()
    cache = "/tmp/hf-cache"
    out = Path("/tmp/observations")
    out.mkdir()
    steps = Path("/pipeline/checkpoint-steps.txt").read_text().strip()
    model_id = "EleutherAI/pythia-70m-deduped"

    def prefetch(step):
        revision = HfApi().model_info(model_id, revision=f"step{step}").sha
        for filename in ("config.json", "model.safetensors"):
            hf_hub_download(model_id, filename, revision=revision, cache_dir=cache)
        return step

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [pool.submit(prefetch, step) for step in steps.split(",")]
        for count, job in enumerate(concurrent.futures.as_completed(jobs), 1):
            print(f"Cached checkpoint {job.result()}; {count}/{len(jobs)}", flush=True)
    prefetch_seconds = time.perf_counter() - started
    subprocess.run([sys.executable, "/pipeline/extract_traces.py", "--steps", steps,
                    "--cache-dir", cache, "--output", str(out / "traces.json")], check=True)
    validation = subprocess.check_output([
        sys.executable, "/pipeline/validate_traces.py", "--traces", str(out / "traces.json"),
        "--extractor", "/pipeline/extract_traces.py"], text=True)
    reports = {"validation-data.json": json.loads(validation)}
    for step, name in ((0, "step0"), (5000, "new-checkpoint"), (143000, "final")):
        replay = subprocess.check_output([
            sys.executable, "/pipeline/replay_checkpoint.py", "--traces", str(out / "traces.json"),
            "--cache-dir", cache, "--step", str(step)], text=True)
        reports[f"replay-{name}.json"] = json.loads(replay)
    reports["modal-run.json"] = {
        "cpu_limit": 4, "memory_limit_mib": 8192, "timeout_seconds": 900,
        "gpu": None, "prefetch_seconds": prefetch_seconds,
        "function_wall_seconds": time.perf_counter() - started,
        "runtime_packages": subprocess.check_output([
            sys.executable, "-m", "pip", "freeze"], text=True).splitlines(),
    }
    bundle = {"traces": (out / "traces.json").read_text(),
              "sha256": (out / "traces.sha256").read_text(), "reports": reports}
    return gzip.compress(json.dumps(bundle).encode(), compresslevel=6)


@app.local_entrypoint()
def main(output: str = "modal-observations.json.gz"):
    result = extract.remote()
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(result)
    print(f"Verified experiment bundle saved to {path} ({len(result):,} bytes)")
