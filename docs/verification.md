# Expanded timeline verification

Completed October 25, 2026. Dataset SHA-256: `874e2b2786d982cf0a8e22122df28bca5551e2f0ba3dd38aad28284579160266`.

## Actual scope

Pythia-70M-deduped: 70,426,624 parameters, six layers, eight heads. The timeline now contains 38 training checkpoints from step 0 through 143,000, three fixed prompts, 114 recorded forward passes, and 282,720 stored attention values. The selected steps are listed in [the checkpoint manifest](../scripts/checkpoint-steps.txt). They include the original eight training steps and 30 additional steps, selected before inspecting the new outputs.

All observations were recomputed together on a Modal Linux CPU container in float32 with eager attention. [The run record](modal-run.json) captures packages, resource limits, and timing: 34.64 seconds to prefetch weights and 92.86 seconds for the complete function, including extraction and independent replays. No GPU training or paid model API was used. The function had a 15-minute timeout, four-core CPU limit, 8 GiB memory limit, one container, and no automatic retries. These durations are measured runtime, not a billing statement.

## Checks completed

- [Data validation](validation-data.json): all 114 frames passed dimension, checkpoint identity, probability, causality, concentration, source-hash, and checksum checks. All 34,656 attention rows were checked; maximum rounded row-sum error was about `3.01e-6`. Local validation of the returned bundle matched the remote report.
- Independent replays at [initialization](replay-step0.json), [step 5,000](replay-new-checkpoint.json), and [the final checkpoint](replay-final.json) reloaded immutable cached weights without importing the extractor or using its shards. All three prompts' tokenization, top-eight token IDs/text, probabilities, and rounded attention entries matched exactly. Maximum difference between unrounded attention and stored values was `5.3644e-7`, within the declared `6e-7` rounding tolerance.
- [Expansion audit](expansion-audit.json): all 38 checkpoint SHAs matched the independently enumerated Hugging Face revisions. Exact prompts, token IDs, target IDs, tokenizer revision, and extractor source hash were preserved.
- JavaScript syntax and whitespace checks passed. Browser checks covered exact-step selection, changing playback speed while playing, stopping at the endpoint, per-scene peak jumps, slider/select synchronization, and chart-dot clicks at tablet width. Chart clicks now account for SVG aspect-ratio margins.
- At a 390 CSS-pixel viewport, page width and scroll width both measured 390; the sampling note and new controls remain visible. No JavaScript errors were observed in the test browser.

## Environment change from the first release

The first eight-step release ran on an Apple silicon CPU. This release reruns those steps on Linux alongside all new steps, so its floats are not byte-identical to the old release. Across the 24 overlapping frames, the largest absolute probability difference is about `6.52e-5` (0.00652 percentage points), and the largest stored attention difference is about `0.005962` (0.5962 percentage points). The expansion audit retains the measured differences. The current release's independent replays are exact within its own recorded environment; no Mac and Linux frames are mixed in the published dataset.

## What the examples show

For `Once upon a`, ` time` is the top next-token guess at step 2,000, loses that position at 3,000, returns at 5,000, and loses it again at 6,000. Its highest probability among these 38 snapshots is about 11.70% at step 100,000, compared with 7.69% at the final checkpoint. The denser early timeline exposes changes hidden between the original observations.

For the color sequence, ` purple` peaks at about 7.80% at step 2,000 and ends at 0.14%. For the factual prompt, ` Paris` peaks at about 0.85% at step 60,000. Neither is the top next token at any recorded checkpoint. A model preferring ` the` after the factual prompt could still begin a longer answer; immediate token rank is not a full-answer evaluation.

These are curated prompt observations from one small model trajectory, not a discovery about general capability acquisition. Attention concentration does not establish head importance or causal explanation. Peaks are maxima over the recorded samples, not a claim about unobserved training steps. This release has neither custom prompt inference nor interpolated model states.

## Single-screen dashboard revision

The desktop UI reserves layout rows for the header, prompt tabs, three simultaneous panels, and playback dock. Browser checks at 1,100 × 620, 1,280 × 650, 1,366 × 768, and 1,440 × 900 CSS pixels confirmed document dimensions equal the viewport and no vertical overflow inside the prompt, attention, or prediction panels. At the smallest size, all 11 color-token labels, all 48 head controls, the prediction list, and history fit above the dock. The color cycle remains two distinct lines. No page scaling or overflow clipping is used to obtain the fit.

The attention SVG now uses its panel's actual dimensions and staggers crowded labels. A ResizeObserver redraws SVG/canvas geometry when available space changes. Matrix switching, token/head selection, step selection, play/pause, and resizing passed browser checks with no console errors. The 390-pixel phone layout and a wide window shorter than 620 pixels retain readable scrolling without horizontal page overflow.

This is a presentation change. The 38-checkpoint observations, their checksum, and extraction/replay code are unchanged.


## Motion and microinteractions

Playback now uses one elapsed-time animation-frame loop with a continuous visual playhead and a separate integer observation index. Browser samples during 0.5× playback showed the playhead progressing from 40.77% to 41.75% across approximately 0.88 seconds while the displayed step remained 5,000 and the leading probability remained 4.48%. Pausing freezes progress; resume and speed changes preserve the fractional interval. Seeking stops playback and settles onto the selected checkpoint. The endpoint stops exactly at the final recorded observation.

Prediction rows retain token identity as their ranks change, using short positional transitions. Head buttons, query controls, and SVG nodes are retained across updates. Numeric readouts always use recorded values; only their presentation animates. Exact-zero attention paths remain invisible. Reduced-motion settings disable animated interpolation, including JavaScript transitions, while playback continues to visit recorded observations.

Browser checks covered playback, changing speed, final-stop behavior, keyboard scrubbing and its accessible value, switching prompts/views/heads/query tokens, and opening About while playing. The 1,100 × 620 dashboard retained zero document/panel overflow, and the 390-pixel mobile fallback retained zero horizontal overflow. No console errors or warnings were observed. Deterministic transport regression checks can be run with `node scripts/test-playback.cjs`.

The recorded dataset and extraction/replay scripts are unchanged; dataset SHA-256 remains `874e2b2786d982cf0a8e22122df28bca5551e2f0ba3dd38aad28284579160266`.
