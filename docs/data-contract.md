# Recorded trace format — version 1

The app reads `dist/data/traces.json`. It must never synthesize model observations when this file is missing. Show a useful loading/error state instead.

```text
schema_version: 1
model: {id, label, parameters, layers, heads, hidden_size, source_url}
provenance: {extracted_at, device, dtype, torch_version, transformers_version, extractor_sha256, notes: string[]}
checkpoints: [{step, revision, commit_sha, source_url}]
scenes: [{
  id, title, kicker, description, prompt,
  target_label, target_token_ids: number[],
  tokens: string[], token_ids: number[],
  frames: [{
    step,
    top_tokens: [{id, token, probability}],
    target_probability,
    attention: number[layer][head][query_token][key_token],
    concentration: number[layer][head]
  }]
}]
```

Arrays follow checkpoint order. All observations come from actual checkpoint forward passes with eval mode, float32, and eager attention. The prompt has no implicit BOS token. Probabilities use the final prompt position. `target_probability` sums the listed target token IDs; targets are illustrative completions, not a task accuracy score. For this first dataset each target is one token. Each attention entry is rounded to six decimal places. Concentration is `1 - entropy(last_query_attention)/log(prompt_token_count)`; it measures focus, not head importance. Query is row and key is column; masked future positions are zero.

The final prompt token can attend to every prompt token. A selected-token view must use the selected query row and make the causal mask clear. Model layers and heads are indexed from zero in the data; the UI may show one-based labels if consistently stated. Use stable token positions, not a newly projected embedding at each frame. Timeline playback changes discrete observed checkpoints; tween only presentation properties, never invent intermediate model measurements.

The front end must support any number of checkpoints/scenes that respects this schema. Public assets contain only small recorded traces, never model weights, credentials, or caches.
