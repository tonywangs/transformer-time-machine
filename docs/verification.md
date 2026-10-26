# Verification record

Updated October 25, 2026. Current dataset SHA-256: `21e806364dace6d96c2b51349ea480e92031d58b77f7af86012e0896423390a3`.

## Current dataset and extraction

Pythia-70M-deduped has 70,426,624 parameters, six layers, and eight heads. This release contains the same [38 selected checkpoints](../scripts/checkpoint-steps.txt), three prompts, 114 recorded forward passes, and 353,856 stored attention values. The factual prompt changed to `The capital of France is the city of`, with the single-token target ` Paris`.

All three scenes were recomputed together in a Modal Linux CPU container with float32 and eager attention. The [run record](modal-run.json) reports 29.12 seconds for prefetching and 87.58 seconds for the complete function, including extraction and independent replays. Resources were bounded to one container, four CPU cores, 8 GiB memory, a 15-minute timeout, and no automatic retries. No GPU training or paid model API was used. These are measured runtimes, not billing amounts.

- [Validation](validation-data.json) passed for all 114 frames and 40,128 attention rows, covering dimensions, checkpoint identity, probabilities, causality, concentration, source hash, and dataset checksum. Maximum rounded row-sum error was about `3.01e-6`.
- Independent replays at [step 0](replay-step0.json), [step 5,000](replay-new-checkpoint.json), [step 12,000](replay-fact-first-success.json), and [step 143,000](replay-final.json) loaded immutable cached weights without importing the extractor or reading its shards. All three scenes matched exactly in tokenization, top-eight identities and probabilities, target probabilities, and rounded attention. Maximum raw attention difference from stored rounded values was `5.3644e-7`, within the `6e-7` tolerance.
- The [replacement audit](fact-replacement-audit.json) confirms unchanged model, checkpoint manifest, and tokenizer identity. Recomputed story and color scenes are exactly equal to the previous release. The factual scene is newly measured; no local probe values were spliced into the published Linux dataset.

## Why the factual prompt changed

The original `The capital of France is` never ranked ` Paris` first in the 38 observations. Its highest recorded Paris probability was 0.85%, at step 60,000; the final checkpoint assigned 0.19% and preferred ` the`. It remains documented in the replacement audit.

The replacement was deliberately selected to illustrate a factual completion that becomes the top guess. An [initial screen of 16 candidates](history/fact-initial-screen.json) was followed by eight Paris-format variants. The [complete 24-candidate screen](fact-candidate-screen.json) retains every tested prompt, endpoint result, selection rule, and local runtime record. Selection required a single-token target, not top-one at initialization, and top-one at the final checkpoint. This is disclosed curation for a visual example, not an accuracy benchmark.

For the new wording, Paris first becomes the top guess at the **recorded step 12,000**, with 14.45% probability. It wins at seven sampled checkpoints: 12,000, 15,000, 20,000, 100,000, 120,000, 130,000, and 143,000. Its probability peaks at 19.48% at step 15,000 and ends at 11.22%. It loses first place between the sampled pairs 20,000→25,000, 100,000→110,000, and 130,000→140,000. The first observed success is independently replayed; the true transition could lie between sampled checkpoints.

This demonstrates sensitivity to prompt wording and a non-monotonic next-token trajectory. It does not show that the original prompt improved, that a fact was acquired permanently at one exact step, or that the model has general factual competence. The unchanged color example still never selects purple as its top next token. Attention concentration describes focus, not causal importance.

## Current factual interface verification

Browser checks with the replacement dataset confirmed that 1,100 × 620 CSS pixels still fit the full page with zero overflow in all three panels. The final view showed Paris at 11.2%; advancing from step 10,000 to 12,000 changed the top guess from ` the` to ` Paris` at 14.5%; the peak shortcut selected step 15,000 at 19.5%. The eight-token Matrix view worked. At 390 × 844, the scrolling page measured 390 × 2,149 with no horizontal overflow. No console errors or warnings were observed, and all 12 playback regression tests passed against the current JavaScript.

## Earlier interface verification

The following browser checks were completed during the dashboard and motion revisions against the previous dataset, SHA-256 `874e2b2786d982cf0a8e22122df28bca5551e2f0ba3dd38aad28284579160266`:

- At 1,100 × 620, 1,280 × 650, 1,366 × 768, and 1,440 × 900 CSS pixels, the desktop page fit the viewport without panel overflow. All 11 color-token labels, 48 heads, predictions, history, and playback controls remained visible. No page scaling or overflow clipping was used. Phone and short-window layouts retained readable scrolling without horizontal overflow.
- Flow/Matrix switching, query/head selection, exact-step selection, keyboard scrubbing, peak shortcuts, resizing, playback speed changes, and modal pause were checked without console errors. During 0.5× playback the visual playhead moved from 40.77% to 41.75% over approximately 0.88 seconds while recorded step 5,000 and leading probability 4.48% stayed fixed.
- Pause/resume preserves fractional progress. Numeric readouts remain discrete; only presentation properties animate. Exact-zero attention paths stay invisible. Reduced-motion preferences disable interpolated presentation. Deterministic transport regression checks are available with `node scripts/test-playback.cjs`.

## Historical extraction change

The [archived expansion audit](history/expansion-audit.json) covers the earlier move from eight Mac CPU checkpoints to 38 Linux CPU checkpoints, with its original hashes and numerical comparisons intact. It is not a validation report for the current factual replacement. The current validation and replay reports above all identify the new dataset checksum.


## Paper and dark themes

The first visit opens in a warm paper light theme, with a persisted header switch for the original dark palette. Theme selection runs before stylesheet paint. Source-level bootstrap checks passed for no saved preference, either valid preference, invalid values, and blocked storage. Browser checks confirmed persistence after reload and immediate SVG/canvas repaint without resetting checkpoint, selected head/query, view, or active playback.

All three light-mode prompts fit at 1,100 × 620 CSS pixels with zero document or panel overflow. The 390 × 844 layout retained zero horizontal overflow and a visible theme control. Flow, Matrix, the About dialog, and both palette legends were inspected. Token-label ink is selected by actual luminance; a 1,001-shade sweep measured minimum contrast above 4.58:1 in both palettes. No application console errors were observed; an unrelated browser extension emitted an error during testing. The 12 playback regression checks passed. Theme changes do not alter the recorded dataset or extraction code; the current dataset checksum above is unchanged.

## Custom prompts — October 25, 2026

The custom-prompt path was exercised through the public Vercel API, not a local mock. `The opposite of hot is` produced all 38 frames and five exact prompt tokens. The first cold run took 94.886 seconds end to end, including 82.768 seconds in the inference worker. The final top token was ` the` with probability 0.19162645936012268; an unintuitive continuation is retained as measured.

An independent Linux replay loaded the pinned first and final checkpoints directly through Transformers in the same runtime image, without importing the serving observation helpers. Top-eight IDs and probabilities, target probabilities, and all 48 head concentrations matched exactly. All 2,400 attention entries matched within the documented six-decimal rounding (maximum error below 5e-7). See [the Linux replay report](custom-linux-replay.json).

A separate Apple-silicon replay had identical tokenization, checkpoint hashes, and top-eight rankings but did **not** pass the strict numerical tolerances. Its largest final attention difference was 0.0013674 and top-token probability difference 0.00004379. Those cross-platform differences are retained in [the comparison report](custom-cross-platform-verification.json); they were not hidden by loosening the replay threshold.

The automated checks cover complete ordered assembly, immutable checkpoint/tokenizer identity, cancellation and late responses, transient failures, request limits, client ownership, quotas, expiry, one-token focus, and the 32-token payload bound. The 12 existing playback tests, 14 experiment-runner tests, 13 proxy tests, and 11 backend tests passed. Bundled observations retain SHA-256 `21e806364dace6d96c2b51349ea480e92031d58b77f7af86012e0896423390a3`.

Browser checks confirmed completed 1-token and 32-token live experiments (including all 48 single-token heads at 100% focus), a rejected 33-token prompt without truncation, cancellation preserving the last result, all 38 selectable steps, Flow/Matrix inspection, exact prompt preservation, a read-only editor during analysis, and a visible distinction between the pending draft and the previous displayed result. The 1100 × 620 desktop view has no page overflow; the 390-pixel mobile view has no horizontal overflow. The backend rejects unauthenticated direct requests with HTTP 401.
