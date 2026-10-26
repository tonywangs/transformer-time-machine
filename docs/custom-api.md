# Custom prompt inference

`scripts/serve_custom.py` defines the Modal app `transformer-time-machine-custom`.
Its `api` web function accepts POSTs at `/`. It requires the Modal secret
`transformer-time-machine-api` containing `TTM_API_KEY`. The Vercel proxy keeps
this key server-side and forwards `Authorization: Bearer …` plus `X-TTM-Client`,
a 64-character lowercase HMAC fingerprint of the trusted client IP.

The gateway serializes admission with one container and a local lock. The
separate inference function has one L4 GPU, 4 CPU cores, 16 GiB RAM, a
900-second execution timeout, no retries, and no minimum warm containers.
`dist/data/models.json` pins 38 checkpoints each for 70M, 160M, 410M, and 1B
(all deduped). `modal run scripts/serve_custom.py::prepare` explicitly populates
the `transformer-time-machine-models` Volume before deployment; public requests
cannot download arbitrary weights. Inference loads local files only and retains
at most one checkpoint in GPU memory. The worker scales to zero after two minutes.
The Volume persists between runs; HTTP status polls never load model weights.

## Requests and responses

Start: `{"action":"start","prompt":"Once upon a","model":"1b"}` returns HTTP 202 with
`{job_id,status:"queued",completed:0,total:38,expires_at}`. `job_id` is a UUIDv4.
The model is allowlisted (`70m`, `160m`, `410m`, `1b`); omission defaults to `70m` for older clients. Model identity is saved with the job and checked in the browser along with every checkpoint SHA.
The exact prompt is preserved, including leading/trailing whitespace. Empty
prompts and more than 400 Unicode characters fail before admission. The worker
rejects prompts outside 1–32 tokenizer tokens before loading a model.

Poll: `{"action":"poll","job_id":"…","cursor":0}` returns HTTP 200 with
`{job_id,status,completed,total,expires_at}` and, when ready,
`{scene,model,checkpoints,provenance,frame,index}`. The response contains at most
one frame, exactly the requested chronological index 0–37. If that frame is
not ready, `frame` and `index` are absent. Advance the cursor only after receiving
the requested frame. A complete worker can still have frames left to download.
Scene metadata is repeated to make polling retries safe. No poll consumes start
quota. Polling every 1–2 seconds while waiting is sufficient; download already
available frames consecutively.

Cancel: `{"action":"cancel","job_id":"…"}` marks a live job cancelled and
cancels its Modal call. Only the initiating client fingerprint may poll or cancel
a job; all other clients receive 404.

Statuses are `queued`, `running`, `complete`, `cancelled`, and `error`. Failed
jobs include a safe `error` and short `code`. Admission errors use a 4xx status
and `{error,code,retry_after?}`. Failed startup uses HTTP 503. The UI should keep
the previous complete observation visible and assemble all 38 frames before
switching to the new scene. Missing frames must never be interpolated or filled
with another checkpoint's values.

## Measurements and target

The final checkpoint runs first. Its actual top token becomes the fixed target
for the whole timeline; `scene.target_selection` is
`final_checkpoint_argmax`. Label this “Final checkpoint’s top guess.” It is not
a correctness label, and the final top guess can be punctuation or a special
token. The final frame still occupies chronological index 37. The remaining
checkpoints run in order from index 0.

New live frames use CUDA float32 with TF32 disabled, eager attention, eval/inference mode, no KV cache,
and the recorded final tokenizer with no implicit BOS. Attention is rounded to
six decimal places as Python scalars. Even the largest supported attention layout
(410M: 24 layers × 16 heads × 32 × 32 tokens) fits the 4 MiB proxy response cap.
The original bundled 70M examples retain their recorded CPU observations. Focus is one minus normalized entropy of the final query's
attention; a one-token prompt has focus 1. The manifest, extractor, runtime,
tokenizer hash, and original immutable checkpoint URLs accompany each result.
The selected manifest hash is SHA-256 of Python `json.dumps(manifest, sort_keys=True).encode()`.

## Public compute and retention limits

- 8 accepted starts per client per rolling hour; 200 accepted starts per UTC day.
- At most 2 outstanding jobs globally: one running and one waiting.
- Requests are accepted until 2026-12-25 00:21:15 UTC. Redeploying this source
  does not extend that fixed date.
- Job access expires after one hour. Expired metadata and frames are deleted on
  the next start/expired poll that encounters them. With no subsequent traffic,
  physical records can remain until Modal Dict's own expiry (normally seven days
  after last read/write); do not promise immediate physical deletion.
- Prompts are not printed in application logs or published as static assets.
  The backend receives user-entered text and stores it temporarily to run jobs.

The gateway/worker bounds are separate: lowering worker concurrency limits peak
resource use but is not a total-spend quota. Daily admission limits and the fixed
sunset provide the latter. Accepted invalid-token runs also count toward quota.

## Verification

Run `python scripts/test_custom_service.py` in a Python environment with Modal
installed. The offline tests exercise ownership, quota boundaries, concurrency,
expiry, cancellation, hard-timeout recovery, manifest identity, payload bounds,
and exact prompt preservation without making account calls.

Before production, run one authorized real request and compare its checkpoint
0/final values against an independent forward pass. Check cold and warm timing,
32-token and single-token prompts, cancellation, and browser assembly through
the Vercel proxy. Runtime failures never fall back to synthetic observations.

Official references: [Modal HTTP timeouts](https://modal.com/docs/guide/webhook-timeouts),
[Modal scaling](https://modal.com/docs/guide/scale),
[Vercel payload limits](https://vercel.com/docs/functions/limitations).

## Model-switching UI

The dropdown preserves the active tab and selected training step. Each model has
three recorded preset scenes in `dist/data/traces.json` (70M) or
`dist/data/examples-{size}.json`. Preset clicks never start inference or change
the custom textarea. Their fixed reference tokens are time, purple, and Paris;
these remain reference targets even when a model predicts something else.

On Your prompt, changing models reruns the custom draft (or restores its cached
result). Completed custom results and the draft stay separate from preset
navigation. The client retains at most four complete model/prompt results plus
the latest custom result per model; refreshing clears these caches. A completed
run installs model metadata, checkpoint manifest, and frames together. Pending
model downloads cannot overwrite a more recent model or tab selection. Attention
heads remain accessible in groups of six layers.

The first run can take several minutes to load and evaluate 38 separate weight
snapshots. It is not a chat endpoint. The browser allows up to 32 minutes for
one queued job plus a 15-minute worker run; HTTP requests remain individually
bounded to 30 seconds at the proxy. Cancel stops outstanding inference.
