# Transformer Time Machine

[Open the live demo](https://transformer-time-machine.vercel.app/)

An interactive view of Pythia's training. Choose an example or type your own short prompt, then scrub checkpoints to inspect actual attention and next-token probabilities.

The bundled datasets cover **Pythia 70M, 160M, 410M, and 1B** (deduped), 38 checkpoints each, and three short curated prompts. It is a visual prototype, not a representative capability benchmark. Exploring the recorded examples requires no inference job.

The factual example completes “The capital of France is the city of” with “ Paris”. It was selected for a visible success through a [documented candidate screen](docs/fact-candidate-screen.json): 16 initial prompts, then eight Paris-format variants. The original 70M prompt “The capital of France is” did not put Paris first at any sampled checkpoint. This wording change illustrates prompt sensitivity; it is not a general factual-accuracy result.

The observations are extracted and verified. See the [verification record](docs/verification.md), including independent checkpoint replays.

## Your own prompt

Choose **Your prompt**, type an unfinished phrase, and press **Analyze**. A bounded Modal L4 GPU worker evaluates the exact input at all 38 immutable checkpoints. Once the full run arrives, the same timeline, attention heads, Flow/Matrix views, and playback work on your observations. Choose Pythia 70M, 160M, 410M, or 1B from the model dropdown. Preset tabs load that model’s recorded examples; Your prompt runs live inference. Preset navigation never overwrites the custom draft. All four use the deduplicated Pythia training series and the same 38 training steps. Nothing is generated while typing, and an incomplete run never replaces the previous result.

Custom prompts support up to 32 tokenizer tokens and 400 characters; overlong prompts are rejected rather than truncated. The history tracks the **final checkpoint's most likely next token** across training. This is not a correct-answer label. Pythia is a small pretrained completion model, not a chat assistant.

A first run can take several minutes, especially for larger models; it evaluates 38 separate weight snapshots. Completed model/prompt combinations are cached in browser memory for quick comparisons. Cancel stops the job; failures leave existing examples usable. About the data downloads the currently displayed custom experiment, including its prompt, tokens, probabilities, attention, runtime metadata, and immutable checkpoint sources. Custom observations stay in browser memory for this visit. Server access expires after an hour; expired records are cleaned on later requests and otherwise follow Modal Dict retention. See [the service contract and limits](docs/custom-api.md) and [the Vercel proxy setup](docs/custom-prompt-proxy.md).

## Preview

From this directory:

```sh
python3 -m http.server 8765 --bind 127.0.0.1 --directory dist
```

Run `node scripts/test-playback.cjs`, `node scripts/test-custom-runner.cjs`, and `node scripts/test-proxy.cjs` for playback, experiment lifecycle, and server boundary checks. `python scripts/test_custom_service.py` tests the backend without account access (the tensor checks also need PyTorch).

Open `http://127.0.0.1:8765`. This static preview loads the three recorded examples; custom inference needs the configured `/api/analyze` Vercel function. Missing data or an unavailable API produces an error, not synthetic values. No JavaScript build or npm install is required.

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

The paper light theme is the default. The header’s Light/Dark switch remembers your choice locally and preserves the selected checkpoint, head, token, and playback state. Both SVG and canvas visualizations use matching palettes; higher attention is darker blue in light mode and brighter green in dark mode.

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

`vercel.json` selects the Other framework preset, skips installation and builds, publishes `dist/`, and enables the Node function in `api/analyze.js`. Import this GitHub repository into Vercel, or run `vercel --prod` from its root. The connected Vercel project deploys changes pushed to `main`.

The recorded examples need no credentials. Custom prompts require the separately deployed `scripts/serve_custom.py` Modal service and server-only `TTM_MODAL_URL` and `TTM_API_KEY` variables. The same key must be in Modal's `transformer-time-machine-api` secret. Never expose the key to browser code. The service scales to zero, permits one worker, limits job duration and admission, and has a fixed expiry. Model weights stay in its private Modal Volume, never Vercel assets. See the setup documents above.

## Source and attribution

`dist/` contains only static assets and recorded traces; `scripts/` contains extraction, validation, prompt-screening, and playback-test code. Keep model weights, caches, credentials, and virtual environments out of source and deployment archives.

The model and checkpoint trajectory come from [EleutherAI Pythia](https://github.com/EleutherAI/pythia) and its [Hugging Face model repository](https://huggingface.co/EleutherAI/pythia-70m-deduped). [AttentionViz](https://arxiv.org/abs/2305.03210) is related visualization work. This project presents measured snapshots; it does not claim a new interpretability method.


### Rebuild the larger-model examples

Run `modal run scripts/build_model_examples.py` with the existing model Volume prepared.
It evaluates all three fixed-reference prompts at each of 38 pinned checkpoints,
reusing each loaded model for the three prompts. The resulting
`dist/data/examples-{160m,410m,1b}.json` files are real observations, served as
static data. Run `python scripts/validate_model_examples.py` to check all model
manifests, prompt/target identities, causal attention, normalization, and focus.
`node --test scripts/test-preset-navigation.cjs` tests the navigation regression.
