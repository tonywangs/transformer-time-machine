# First prototype verification

Completed October 25, 2026. Dataset SHA-256: `38136584c548324f0655d36a2f151aae8679cd410d7e868fa882dc32dedaf2ad`.

## Actual scope

Pythia-70M-deduped: 70,426,624 parameters, six layers, eight heads. Eight training checkpoints from step 0 through 143,000, three fixed prompts, 24 recorded forward passes, and 59,520 stored attention values. Extraction ran on local CPU in float32. No Modal jobs or paid model API calls were launched.

## Checks completed

- [Data validation](validation-data.json): all 24 frames passed dimension, checkpoint identity, probability, causality, concentration, source-hash, and checksum checks. All 7,296 attention rows were checked; maximum rounded row-sum error was about `3.01e-6`.
- [Independent initialization replay](replay-step0.json) and [independent final-checkpoint replay](replay-final.json): reloaded immutable local weights without importing the extractor or using extraction shards. All three prompts' tokenization, top-eight token IDs/text, top probabilities, target probabilities, and rounded attention entries matched exactly. Maximum difference between unrounded attention and stored values was `5.3644e-7`, within the declared `6e-7` rounding tolerance.
- JavaScript syntax check passed. Browser tests covered scene changes, flow/matrix switching, head and query selection, playback from the final step, keyboard timeline endpoints, and experiment details. The browser reported no JavaScript errors during the final pass.
- Desktop and narrow layouts were inspected. At a 390 CSS-pixel viewport, page width and scroll width both measured 390 after correcting head-grid overflow. The attention flow has its own horizontal scroll region for long token sequences.

## What the examples show

For `Once upon a`, the illustrative next-token probability of ` time` is approximately 2.96% at step 2,000, 0.43% at step 10,000, and 7.69% at the final checkpoint. For the color sequence, ` purple` is approximately 7.80% at step 2,000 and 0.14% at the final checkpoint. These reversals are retained in the display.

These are curated prompt observations from one small model trajectory, not a discovery about general capability acquisition. Attention concentration does not establish head importance or causal explanation. This release has neither custom prompt inference nor interpolated training states.

## Next useful iteration

Inspect additional real prompts and checkpoints to find visually legible stories, record the curation choices, and produce a short demo recording. Broader models or Modal extraction should follow a bounded pilot; they are not needed to use this static prototype.
