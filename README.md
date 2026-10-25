# Transformer Time Machine

[Open the live demo](https://transformer-time-machine.vercel.app/)

An interactive, recorded-data view of Pythia's training. Scrub checkpoints while the same prompt stays fixed; inspect actual attention and next-token probabilities.

The dataset covers **Pythia-70M-deduped**, 38 checkpoints, and three short curated prompts. It is a visual prototype, not a representative capability benchmark. No Modal training or GPU job is needed to reproduce this small dataset.

The factual example completes “The capital of France is the city of” with “ Paris”. It was selected for a visible success through a [documented candidate screen](docs/fact-candidate-screen.json): 16 initial prompts, then eight Paris-format variants. The original “The capital of France is” did not put Paris first at any sampled checkpoint. This wording change illustrates prompt sensitivity; it is not a general factual-accuracy result.

The observations are extracted and verified. See the [verification record](docs/verification.md), including independent checkpoint replays.

## Preview

From this directory:

```sh
python3 -m http.server 8765 --bind 127.0.0.1 --directory dist
```

Run `node scripts/test-playback.cjs` for deterministic playback regression checks.

Open `http://127.0.0.1:8765`. The site loads `dist/data/traces.json`; missing data produces an error, not synthetic values. No JavaScript build or npm install is required.

## Reproduce the observations

Use Python 3.12 and install the pinned main dependencies into a virtual environment:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/extract_traces.py \
  --cache-dir /tmp/transformer-time-machine-cache \
  --steps "$(cat scripts/checkpoint-steps.txt)" \
  --output dist/data/traces.json
.venv/bin/python scripts/validate_traces.py --traces dist/data/traces.json
.venv/bin/python scripts/replay_checkpoint.py \
  --cache-dir /tmp/transformer-time-machine-cache
node --check dist/app.js
```

Extraction downloads public model checkpoints from Hugging Face, one at a time. Cache weights outside this project's source/deployment tree. It executes actual float32, evaluation-mode, eager-attention forward passes on CPU; `--device cuda` is optional. It never trains weights, uses remote custom code, or calls a paid model API.

The published 38-step selection is in [scripts/checkpoint-steps.txt](scripts/checkpoint-steps.txt). It retains the original eight training steps, adds every available early checkpoint through step 1,000, samples each 1,000-step checkpoint through 10,000, and then uses wider intervals through step 143,000. These steps were chosen to resolve early changes and the reversals visible in the first release, before inspecting their new outputs. They are a subset of the model repository's 154 available training checkpoints.

Without `--steps`, the extractor still uses the original eight-step pilot: `0,64,512,2000,10000,30000,70000,143000`. Every requested revision is resolved to its immutable commit SHA. Resume uses cached extraction shards keyed by checkpoint, script hash, and runtime environment. The output includes the exact tokenizer revision, token IDs, source revisions, library versions, dtype/device, extraction-script hash, and a `.sha256` sidecar. Changing source code invalidates previous extraction shards. Downloading all 38 float32 weight files requires roughly 11 GB of cache space; the website serves only the compact observations.

`requirements-lock.txt` records the complete package set used for the first extraction (Python 3.12, Apple silicon CPU). Install it instead of `requirements.txt` to reproduce that package set. Cached extraction shards also require matching tokenizer, device, and runtime metadata; incompatible shards are recomputed. The replay utility makes no network calls and reloads the final saved checkpoint from the local weight cache, comparing fresh outputs against the exported observations.

### Optional Modal runner

The expanded release uses a Modal CPU container to avoid slow local weight downloads. With the Modal client installed under Python 3.11 and your account authenticated (the local and remote Python versions must match for this serialized function):

```sh
modal run scripts/extract_on_modal.py --output /tmp/modal-observations.json.gz
```

This temporary job has one container, a four-core CPU limit, an 8 GiB memory limit, and a 15-minute function timeout, with no automatic retries or GPU allocation. It uploads only the extractor, validator, replay script, and checkpoint list. All 38 checkpoints are evaluated in the same container; independent replays check steps 0, 5,000, the factual example’s first observed top-one checkpoint, and 143,000. The gzip file contains a JSON bundle with `traces` and `sha256` strings and a `reports` object. It does not deploy or replace website files automatically. Runtime packages and function duration are recorded in [docs/modal-run.json](docs/modal-run.json); the local client used for this release was Modal 1.1.1. Model weights stay in ephemeral container storage.

## Reading the display

Desktop windows at least 1,100 × 620 CSS pixels use a single-screen dashboard: prompt, attention, and predictions sit side by side above a reserved playback dock. Charts resize to their panels; crowded token labels stagger rather than scrolling out of view. Smaller or zoomed windows use a readable scrolling layout.

- Predictions always refer to the **next token after the full prompt**. Selecting an earlier query changes only the attention view.
- Attention is the recorded probability matrix: query rows, key columns, with future positions masked.
- Concentration is normalized attention entropy. It describes focus, not causal importance.
- A target such as ` Paris` is an illustrative completion. Its probability is not an accuracy score, and need not increase during training.
- Playback visits discrete, unevenly spaced recorded steps. The playhead travels smoothly between them; printed probabilities and training steps change only at recorded observations. Bar lengths and attention colors briefly transition for readability. No intermediate model states are synthesized.
- Pause freezes the playhead in place; resume and speed changes preserve its progress. Reduced-motion preferences disable interpolated movement and transitions.
- The exact-step menu jumps to any recorded checkpoint. Playback speed changes how long each observation stays on screen; it does not represent training time.
- The history chart is clickable. Its shortcut jumps to the tracked completion's highest probability among the recorded checkpoints, which is not necessarily the highest point between them.
- The model is small and pretrained; it is not a chat assistant. These three handpicked prompts cannot support general conclusions about learning.

See [the trace contract](docs/data-contract.md) for tensor axes and semantics. 

## Deploy to Vercel

This is a static site. `vercel.json` selects the Other framework preset, skips installation and builds, and publishes `dist/`. Import this GitHub repository into Vercel, or run `vercel --prod` from its root. No environment variables or secrets are required. The connected Vercel project deploys changes pushed to `main`.

Only files under `dist/` are served. Model extraction is a separate Python workflow and never runs during deployment.

## Source and attribution

`dist/` contains only static assets and recorded traces; `scripts/` contains extraction, validation, prompt-screening, and playback-test code. Keep model weights, caches, credentials, and virtual environments out of source and deployment archives.

The model and checkpoint trajectory come from [EleutherAI Pythia](https://github.com/EleutherAI/pythia) and its [Hugging Face model repository](https://huggingface.co/EleutherAI/pythia-70m-deduped). [AttentionViz](https://arxiv.org/abs/2305.03210) is related visualization work. This project presents measured snapshots; it does not claim a new interpretability method.
