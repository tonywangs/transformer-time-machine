# Transformer Time Machine

[Open the live demo](https://transformer-time-machine.vercel.app/)

An interactive, recorded-data view of Pythia's training. Scrub checkpoints while the same prompt stays fixed; inspect actual attention and next-token probabilities.

The first dataset targets **Pythia-70M-deduped**, eight checkpoints, and three short curated prompts. It is a visual prototype, not a representative capability benchmark. No Modal training or GPU job is needed to reproduce this small dataset.

The first dataset is extracted and verified. See the [verification record](docs/verification.md), including independent replay results for initialization and the final checkpoint.

## Preview

From this directory:

```sh
python3 -m http.server 8765 --bind 127.0.0.1 --directory dist
```

Open `http://127.0.0.1:8765`. The site loads `dist/data/traces.json`; missing data produces an error, not synthetic values. No JavaScript build or npm install is required.

## Reproduce the observations

Use Python 3.12 and install the pinned main dependencies into a virtual environment:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/extract_traces.py \
  --cache-dir /tmp/transformer-time-machine-cache \
  --output dist/data/traces.json
.venv/bin/python scripts/validate_traces.py --traces dist/data/traces.json
.venv/bin/python scripts/replay_checkpoint.py \
  --cache-dir /tmp/transformer-time-machine-cache
node --check dist/app.js
```

Extraction downloads public model checkpoints from Hugging Face, one at a time. Cache weights outside this project's source/deployment tree. It executes actual float32, evaluation-mode, eager-attention forward passes on CPU; `--device cuda` is optional. It never trains weights, uses remote custom code, or calls a paid model API.

The default steps are `0,64,512,2000,10000,30000,70000,143000`. Each is resolved to its immutable commit SHA. `--steps` can specify an ordered subset. Resume uses cached extraction shards keyed by checkpoint and script hash. The output includes the exact tokenizer revision, token IDs, source revisions, library versions, dtype/device, extraction-script hash, and a `.sha256` sidecar. Changing source code invalidates previous extraction shards.

`requirements-lock.txt` records the complete package set used for the first extraction (Python 3.12, Apple silicon CPU). Install it instead of `requirements.txt` to reproduce that package set. Cached extraction shards also require matching tokenizer, device, and runtime metadata; incompatible shards are recomputed. The replay utility makes no network calls and reloads the final saved checkpoint from the local weight cache, comparing fresh outputs against the exported observations.

## Reading the display

- Predictions always refer to the **next token after the full prompt**. Selecting an earlier query changes only the attention view.
- Attention is the recorded probability matrix: query rows, key columns, with future positions masked.
- Concentration is normalized attention entropy. It describes focus, not causal importance.
- A target such as ` Paris` is an illustrative completion. Its probability is not an accuracy score, and need not increase during training.
- Playback visits discrete, unevenly spaced recorded steps. No intermediate model states are synthesized.
- The model is small and pretrained; it is not a chat assistant. These three handpicked prompts cannot support general conclusions about learning.

See [the trace contract](docs/data-contract.md) for tensor axes and semantics. 

## Deploy to Vercel

This is a static site. `vercel.json` selects the Other framework preset, skips installation and builds, and publishes `dist/`. Import this GitHub repository into Vercel, or run `vercel --prod` from its root. No environment variables or secrets are required. The connected Vercel project deploys changes pushed to `main`.

Only files under `dist/` are served. Model extraction is a separate Python workflow and never runs during deployment.

## Source and attribution

`dist/` contains only static assets and recorded traces; `scripts/` contains extraction and validation code. Keep model weights, caches, credentials, and virtual environments out of source and deployment archives.

The model and checkpoint trajectory come from [EleutherAI Pythia](https://github.com/EleutherAI/pythia) and its [Hugging Face model repository](https://huggingface.co/EleutherAI/pythia-70m-deduped). [AttentionViz](https://arxiv.org/abs/2305.03210) is related visualization work. This project presents measured snapshots; it does not claim a new interpretability method.
