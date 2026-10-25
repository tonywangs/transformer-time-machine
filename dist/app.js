/* Recorded observations only. No model outputs are generated in this client. */
'use strict';

const $ = (id) => document.getElementById(id);
const state = { data: null, scene: 0, frame: 0, layer: 0, head: 0, query: 0, view: 'flow', playing: false, timer: null };
const exampleOrder = ['story', 'echo', 'capital'];
const exampleCopy = {
  story: { tab: 'Finish a phrase', question: 'Once upon a… what?', explanation: '“Time” is the familiar continuation. Rewind training to see when it appears among the model’s guesses.', reference: 'Familiar completion' },
  echo: { tab: 'Repeat a pattern', question: 'Can it repeat six colors?', explanation: 'The first line gives a six-color cycle. The second repeats its first five colors. If the same cycle continues, “purple” comes next.', reference: 'If the cycle repeats' },
  capital: { tab: 'Complete a fact', question: 'Does it predict Paris?', explanation: '“Paris” is the direct factual completion. Track its probability alongside the model’s other next-token guesses.', reference: 'Direct completion' }
};
const number = new Intl.NumberFormat('en-US');
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const clamp = (n, a, b) => Math.max(a, Math.min(b, n));
const tokenText = (s) => String(s).replace(/ /g, '·').replace(/\n/g, '↵').replace(/\t/g, '⇥');
const displayToken = (s) => tokenText(s) || '∅';
const percent = (p) => p === 0 ? '0%' : p < 0.0001 ? '<0.01%' : `${(p * 100).toFixed(p >= .1 ? 1 : 2)}%`;
const compactStep = (s) => s >= 1000 ? `${Number((s / 1000).toFixed(1))}k` : String(s);
const scene = () => state.data.scenes[state.scene];
const frame = () => scene().frames[state.frame];
const weightColor = (p, minimum = 0) => {
  const t = Math.max(minimum, clamp(p, 0, 1));
  return `rgb(${Math.round(37 + 173 * t)},${Math.round(58 + 180 * t)},${Math.round(54 + 100 * t)})`;
};
const safeUrl = (value) => {
  try { const u = new URL(value); return ['https:', 'http:'].includes(u.protocol) ? u.href : '#'; } catch { return '#'; }
};

function validateData(data) {
  if (data.schema_version !== 1 || !data.model || !Array.isArray(data.checkpoints) || !data.checkpoints.length || !Array.isArray(data.scenes) || !data.scenes.length) throw new Error('The recorded observations have an unsupported format.');
  const { layers, heads } = data.model;
  if (!Number.isInteger(layers) || !Number.isInteger(heads) || layers < 1 || heads < 1) throw new Error('Model dimensions are missing from the trace.');
  data.scenes.forEach((s) => {
    if (!Array.isArray(s.tokens) || s.tokens.length < 1 || !Array.isArray(s.frames) || s.frames.length !== data.checkpoints.length) throw new Error('A prompt is missing its recorded tokens or checkpoints.');
    s.frames.forEach((f, i) => {
      if (f.step !== data.checkpoints[i].step || !Array.isArray(f.top_tokens) || !f.top_tokens.length || !Number.isFinite(f.target_probability)) throw new Error('A checkpoint observation is incomplete.');
      if (!Array.isArray(f.attention) || f.attention.length !== layers || !Array.isArray(f.concentration) || f.concentration.length !== layers) throw new Error('Attention observations do not match the model.');
      f.attention.forEach((layer, l) => {
        if (layer.length !== heads || f.concentration[l].length !== heads) throw new Error('An attention head is missing.');
        layer.forEach((head) => {
          if (head.length !== s.tokens.length || head.some((row) => row.length !== s.tokens.length)) throw new Error('Attention dimensions do not match the recorded tokens.');
        });
      });
    });
  });
}

async function loadData() {
  pause();
  $('loadingState').hidden = false;
  $('errorState').hidden = true;
  $('instrument').hidden = true;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 30000);
  try {
    const response = await fetch('data/traces.json', { signal: controller.signal, cache: 'no-cache' });
    if (!response.ok) throw new Error(`The observation file returned HTTP ${response.status}. Please try again in a moment.`);
    const data = await response.json();
    validateData(data);
    // Presentation order only: keep the downloaded observations unchanged.
    const order = (id) => exampleOrder.includes(id) ? exampleOrder.indexOf(id) : exampleOrder.length;
    state.data = { ...data, scenes: [...data.scenes].sort((a, b) => order(a.id) - order(b.id)) };
    state.scene = 0;
    state.frame = data.checkpoints.length - 1;
    state.layer = Math.min(2, data.model.layers - 1);
    state.head = 0;
    state.query = scene().tokens.length - 1;
    // Choose the most focused recorded head in the middle layer as a useful first view.
    const concentrations = frame().concentration[state.layer];
    state.head = concentrations.indexOf(Math.max(...concentrations));
    initDataUI();
    $('loadingState').hidden = true;
    $('instrument').hidden = false;
    render();
  } catch (error) {
    $('loadingState').hidden = true;
    $('errorState').hidden = false;
    $('errorMessage').textContent = error.name === 'AbortError' ? 'The observation file is taking too long to load. Please try again.' : error.message;
  } finally { clearTimeout(timeout); }
}

function initDataUI() {
  const m = state.data.model;
  $('modelSpec').innerHTML = `<span class="spec-name">${esc(m.label || m.id)}</span><span class="spec-line">${esc(typeof m.parameters === 'number' ? `${Math.round(m.parameters / 1e6)}M parameters` : m.parameters)} · ${m.layers} layers · ${m.heads} heads</span>`;
  $('sceneTabs').innerHTML = state.data.scenes.map((s, i) => `<button class="scene-tab" id="scene-${i}" role="tab" aria-selected="${i === state.scene}" aria-controls="instrument" data-scene="${i}"><span class="tab-number">0${i + 1}</span>${esc(exampleCopy[s.id]?.tab || s.title)}</button>`).join('');
  $('checkpointSlider').max = state.data.checkpoints.length - 1;
  $('timelineTicks').innerHTML = state.data.checkpoints.map(() => '<i></i>').join('');
  $('timelineStart').textContent = `STEP ${number.format(state.data.checkpoints[0].step)}`;
  $('timelineEnd').textContent = `STEP ${number.format(state.data.checkpoints.at(-1).step)}`;
  $('headColumnLabels').style.gridTemplateColumns = `repeat(${m.heads},1fr)`;
  $('headColumnLabels').innerHTML = Array.from({ length: m.heads }, (_, i) => `<span>${i + 1}</span>`).join('');
  $('headGrid').style.setProperty('--head-count', m.heads);
  buildProvenance();
}

function render() {
  if (!state.data) return;
  const activeId = document.activeElement?.id;
  const s = scene();
  const f = frame();
  renderExample(s, f);
  $('headLabel').textContent = `Layer ${state.layer + 1} / Head ${state.head + 1}`;
  $('focusValue').textContent = `${(f.concentration[state.layer][state.head] * 100).toFixed(1)}%`;
  document.querySelectorAll('.scene-tab').forEach((b) => { b.setAttribute('aria-selected', String(Number(b.dataset.scene) === state.scene)); b.tabIndex = Number(b.dataset.scene) === state.scene ? 0 : -1; });
  $('queryTokens').innerHTML = s.tokens.map((t, i) => `<button id="query-${i}" class="query-token" data-query="${i}" aria-pressed="${state.query === i}" aria-label="Inspect token ${i + 1}: ${esc(t)}"><span class="token-index" aria-hidden="true">${String(i + 1).padStart(2, '0')}</span>${esc(displayToken(t))}</button>`).join('');
  renderHeads();
  renderVisualization();
  renderPredictions();
  renderHistory();
  renderTimeline();
  if (activeId && (activeId.startsWith('head-') || activeId.startsWith('query-'))) $(activeId)?.focus({ preventScroll: true });
}

function renderExample(s, f) {
  const copy = exampleCopy[s.id];
  $('sceneTitle').textContent = copy?.question || s.title;
  $('sceneDescription').textContent = copy?.explanation || s.description;
  // Line breaks group the existing color tokens; no tokens are added to the input.
  if (s.id === 'echo') {
    $('promptText').innerHTML = `<span class="sequence-line">${esc(s.tokens.slice(0, 6).join(''))}</span><span class="sequence-line">${esc(s.tokens.slice(6).join(''))}</span>`;
  } else $('promptText').textContent = s.prompt;
  $('referenceLabel').textContent = copy?.reference || 'Completion to track';
  $('referenceCompletion').textContent = s.target_label.trim();
  const top = f.top_tokens[0];
  $('exampleGuess').textContent = `“${top.token.trim() || displayToken(top.token)}” · ${percent(top.probability)}`;
  const rank = f.top_tokens.findIndex((t) => s.target_token_ids.includes(t.id));
  const target = s.target_label.trim();
  const snapshot = rank === 0
    ? `At step ${number.format(f.step)}, “${target}” is the top guess (${percent(f.target_probability)}).`
    : `At step ${number.format(f.step)}, “${target}” gets ${percent(f.target_probability)}${rank < 0 ? ' and is outside the top eight guesses' : ` and ranks #${rank + 1}`}.`;
  const wins = s.frames.filter((item) => s.target_token_ids.includes(item.top_tokens[0].id)).length;
  const note = s.id === 'echo' && wins === 0
    ? `None of these ${s.frames.length} checkpoints chooses “${target}” as its top next token. This example shows an unreliable continuation.`
    : s.id === 'capital'
      ? 'This tests the immediate next token. Another word could begin a longer answer.'
      : 'Press Play training to watch the guesses change. Progress can reverse between checkpoints.';
  $('exampleObservation').textContent = `${snapshot} ${note}`;
}

function renderHeads() {
  $('headGrid').innerHTML = frame().concentration.map((row, l) => `<div class="head-row"><span class="head-row-label" aria-hidden="true">${l + 1}</span>${row.map((c, h) => `<button id="head-${l}-${h}" class="head-cell" data-layer="${l}" data-head="${h}" style="background:${weightColor(c)}" aria-label="Layer ${l + 1}, head ${h + 1}, focus ${(c * 100).toFixed(1)} percent" aria-pressed="${state.layer === l && state.head === h}" title="Layer ${l + 1} · Head ${h + 1} · ${(c * 100).toFixed(1)}% focus"></button>`).join('')}</div>`).join('');
}

function renderVisualization() {
  $('flowGraph').toggleAttribute('hidden', state.view !== 'flow');
  $('matrixWrap').hidden = state.view !== 'matrix';
  $('flowButton').classList.toggle('active', state.view === 'flow');
  $('matrixButton').classList.toggle('active', state.view === 'matrix');
  $('flowButton').setAttribute('aria-pressed', String(state.view === 'flow'));
  $('matrixButton').setAttribute('aria-pressed', String(state.view === 'matrix'));
  if (state.view === 'flow') renderFlow(); else renderMatrix();
  const row = frame().attention[state.layer][state.head][state.query];
  const maxIndex = row.indexOf(Math.max(...row));
  $('attentionDetail').textContent = `${displayToken(scene().tokens[state.query])} → ${displayToken(scene().tokens[maxIndex])}  ·  ${percent(row[maxIndex])} attention`;
}

function renderFlow() {
  const s = scene();
  const row = frame().attention[state.layer][state.head][state.query];
  const svg = $('flowGraph');
  const available = $('visualization').clientWidth;
  const widths = s.tokens.map((t) => Math.max(34, Math.min(110, displayToken(t).length * 6.2 + 12)));
  const natural = widths.reduce((a, b) => a + b, 0) + (s.tokens.length - 1) * 9 + 26;
  const width = Math.max(available, natural);
  const height = 226;
  const surplus = Math.max(0, width - natural);
  const gap = 9 + (s.tokens.length > 1 ? surplus / (s.tokens.length - 1) : 0);
  let cursor = 13 + (s.tokens.length === 1 ? surplus / 2 : 0);
  const positions = widths.map((w) => { const x = cursor + w / 2; cursor += w + gap; return x; });
  const source = positions[state.query];
  const baseline = 171;
  const strongest = Math.max(...row);
  let content = `<defs><pattern id="flowDots" width="20" height="20" patternUnits="userSpaceOnUse"><circle cx="1" cy="1" r=".6" fill="#29403d"/></pattern></defs><rect x="0" y="16" width="${width}" height="154" fill="url(#flowDots)" opacity=".6"/><line x1="13" x2="${width - 13}" y1="${baseline}" y2="${baseline}" stroke="#304542" stroke-dasharray="2 5"/>`;
  // Each curve is the actual selected query-to-key observation; no interpolated frames.
  row.forEach((weight, k) => {
    if (k > state.query || weight <= 0 || k === state.query) return;
    const end = positions[k];
    const distance = Math.abs(end - source);
    const lift = Math.min(155, 25 + distance * .31);
    const opacity = .15 + .85 * Math.sqrt(weight / Math.max(strongest, .000001));
    const stroke = 1 + 5 * Math.sqrt(weight);
    content += `<path d="M ${source} ${baseline} C ${source} ${baseline - lift}, ${end} ${baseline - lift}, ${end} ${baseline}" fill="none" stroke="#cdeba1" stroke-opacity="${opacity}" stroke-width="${stroke}" data-weight-key="${k}"><title>${esc(displayToken(s.tokens[state.query]))} to ${esc(displayToken(s.tokens[k]))}: ${percent(weight)}</title></path>`;
  });
  if (row[state.query] > 0) content += `<path d="M ${source - 5} ${baseline} C ${source - 25} ${baseline - 44}, ${source + 25} ${baseline - 44}, ${source + 5} ${baseline}" fill="none" stroke="#cdeba1" stroke-opacity="${.15 + .85 * Math.sqrt(row[state.query] / Math.max(strongest, .000001))}" stroke-width="${1 + 5 * Math.sqrt(row[state.query])}"/>`;
  positions.forEach((x, i) => {
    const selected = i === state.query;
    const future = i > state.query;
    const w = widths[i];
    const color = future ? '#213131' : weightColor(Math.sqrt(row[i] || 0), .04);
    content += `<g class="token-node" data-query="${i}" tabindex="0" role="button" aria-label="Query token ${i + 1}: ${esc(s.tokens[i])}" aria-pressed="${selected}"><circle cx="${x}" cy="${baseline}" r="${future ? 2 : 2 + 5 * Math.sqrt(row[i] || 0)}" fill="${future ? '#364646' : '#cdeba1'}"/><rect x="${x - w / 2}" y="184" width="${w}" height="25" rx="3" fill="${color}" stroke="${selected ? '#d2ee9a' : future ? '#2d3b3b' : '#4b6253'}" stroke-width="${selected ? 1.5 : 1}"/><text x="${x}" y="200" text-anchor="middle" fill="${future ? '#667a76' : row[i] > .25 ? '#102019' : selected ? '#f1f7df' : '#d3dfcc'}" font-size="10">${esc(displayToken(s.tokens[i]))}</text><text x="${x}" y="222" text-anchor="middle" font-size="7" fill="${selected ? '#d2ee9a' : '#66827a'}">${selected ? 'QUERY' : future ? 'FUTURE' : String(i + 1).padStart(2, '0')}</text></g>`;
  });
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  svg.style.minWidth = `${width}px`;
  svg.innerHTML = content;
  svg.setAttribute('aria-label', `Attention from token ${state.query + 1}, ${s.tokens[state.query]}, at layer ${state.layer + 1}, head ${state.head + 1}. Select a token to inspect its attention.`);
}

let matrixLayout = null;
function renderMatrix() {
  const canvas = $('attentionMatrix');
  const wrapper = $('matrixWrap');
  const size = Math.min(226, wrapper.clientHeight || 226);
  const n = scene().tokens.length;
  const left = 40;
  const top = 18;
  const gridSize = size - top - 12;
  const cell = gridSize / n;
  const width = gridSize + left + 50;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = width * dpr;
  canvas.height = size * dpr;
  canvas.style.width = `${width}px`;
  canvas.style.height = `${size}px`;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, width, size);
  const matrix = frame().attention[state.layer][state.head];
  matrix.forEach((row, q) => row.forEach((value, k) => {
    ctx.fillStyle = k > q ? '#172527' : weightColor(Math.sqrt(value));
    ctx.fillRect(left + k * cell, top + q * cell, Math.max(1, cell - .7), Math.max(1, cell - .7));
  }));
  ctx.strokeStyle = '#d2ee9a';
  ctx.lineWidth = 1;
  ctx.strokeRect(left - 2, top + state.query * cell - 1, gridSize + 3, cell + 1);
  ctx.font = '8px ui-monospace, monospace';
  ctx.fillStyle = '#96a9a6';
  ctx.textAlign = 'center';
  ctx.fillText('KEY TOKEN →', left + gridSize / 2, 9);
  ctx.save(); ctx.translate(11, top + gridSize / 2); ctx.rotate(-Math.PI / 2); ctx.fillText('QUERY TOKEN →', 0, 0); ctx.restore();
  ctx.textAlign = 'right';
  const interval = n > 15 ? Math.ceil(n / 10) : 1;
  for (let q = 0; q < n; q += interval) ctx.fillText(String(q + 1), left - 7, top + (q + .65) * cell);
  ctx.fillStyle = '#66827a'; ctx.textAlign = 'left';
  ctx.fillText('FUTURE', left + gridSize - 28, top + 14);
  ctx.fillText('MASKED', left + gridSize - 28, top + 25);
  matrixLayout = { left, top, cell, n, width, height: size };
  $('matrixTip').hidden = true;
}

function renderPredictions() {
  const tokens = frame().top_tokens.slice(0, 8);
  const leading = tokens[0];
  $('leadingToken').textContent = displayToken(leading.token);
  $('leadingToken').title = `Token ID ${leading.id}: ${JSON.stringify(leading.token)}`;
  $('leadingProbability').textContent = percent(leading.probability);
  const maximum = Math.max(...tokens.map((t) => t.probability));
  $('probabilityBars').innerHTML = tokens.map((t) => `<div class="probability-row" title="Token ${t.id}: ${esc(JSON.stringify(t.token))}; probability ${percent(t.probability)}"><span class="probability-token">${esc(displayToken(t.token))}</span><div class="probability-bar-track" aria-hidden="true"><div class="probability-bar" style="width:${maximum ? (t.probability / maximum * 100) : 0}%"></div></div><span class="probability-value">${percent(t.probability)}</span></div>`).join('');
  $('targetLabel').textContent = `“${scene().target_label}”`;
  $('targetProbability').textContent = percent(frame().target_probability);
}

function renderHistory() {
  const values = scene().frames.map((f) => f.target_probability);
  const max = Math.max(...values, .000001);
  const height = 86;
  const x = (i) => 9 + i / Math.max(1, values.length - 1) * 302;
  const y = (p) => 105 - p / max * height;
  const points = values.map((p, i) => `${x(i)},${y(p)}`).join(' ');
  const currentX = x(state.frame);
  const currentY = y(values[state.frame]);
  $('probabilityHistory').innerHTML = `<defs><linearGradient id="historyFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#e8bd7b" stop-opacity=".16"/><stop offset="1" stop-color="#e8bd7b" stop-opacity="0"/></linearGradient></defs><line x1="9" y1="105" x2="311" y2="105" stroke="#304144"/><line x1="9" y1="62" x2="311" y2="62" stroke="#304144" stroke-dasharray="2 5"/><text x="9" y="10" fill="#78938b" font-family="ui-monospace,monospace" font-size="8">${percent(max)}</text><polygon points="9,105 ${points} 311,105" fill="url(#historyFill)"/><polyline points="${points}" fill="none" stroke="#e8bd7b" stroke-width="1.8"/>${values.map((p, i) => `<circle cx="${x(i)}" cy="${y(p)}" r="${i === state.frame ? 0 : 2}" fill="#e8bd7b" opacity=".55"><title>Step ${number.format(state.data.checkpoints[i].step)}: ${percent(p)}</title></circle>`).join('')}<line x1="${currentX}" y1="18" x2="${currentX}" y2="105" stroke="#e8bd7b" stroke-opacity=".4" stroke-dasharray="2 3"/><circle cx="${currentX}" cy="${currentY}" r="4" fill="#e8bd7b" stroke="#101a1c" stroke-width="2"/>`;
  $('probabilityHistory').setAttribute('aria-label', `Probability of ${scene().target_label} across ${values.length} checkpoints. Current step ${frame().step}: ${percent(frame().target_probability)}. Chart maximum ${percent(max)}.`);
  $('historyStart').textContent = `STEP ${compactStep(state.data.checkpoints[0].step)}`;
  $('historyEnd').textContent = `STEP ${compactStep(state.data.checkpoints.at(-1).step)}`;
}

function renderTimeline() {
  $('stepNumber').textContent = number.format(frame().step);
  $('checkpointCount').textContent = `${state.frame + 1} / ${state.data.checkpoints.length} observations`;
  $('checkpointSlider').value = state.frame;
  $('checkpointSlider').setAttribute('aria-valuetext', `Training step ${number.format(frame().step)}, checkpoint ${state.frame + 1} of ${state.data.checkpoints.length}`);
  $('checkpointSlider').style.setProperty('--progress', `${state.frame / Math.max(1, state.data.checkpoints.length - 1) * 100}%`);
  $('previousButton').disabled = state.frame === 0;
  $('nextButton').disabled = state.frame === state.data.checkpoints.length - 1;
  $('playButton').disabled = state.data.checkpoints.length < 2;
}

function setFrame(index, stopPlayback = true) {
  if (!state.data) return;
  if (stopPlayback) pause();
  state.frame = clamp(index, 0, state.data.checkpoints.length - 1);
  render();
  if (!state.playing) $('screenReaderStatus').textContent = `Training step ${number.format(frame().step)}. Most likely next token: ${frame().top_tokens[0].token}, ${percent(frame().top_tokens[0].probability)}.`;
}
function pause() {
  clearInterval(state.timer);
  state.timer = null;
  state.playing = false;
  $('playIcon').textContent = '▶';
  $('playLabel').textContent = 'Play training';
  $('playButton').setAttribute('aria-label', 'Play checkpoints');
}
function play() {
  if (!state.data || state.data.checkpoints.length < 2) return;
  if (state.playing) { pause(); return; }
  if (state.frame >= state.data.checkpoints.length - 1) setFrame(0);
  state.playing = true;
  $('playIcon').textContent = 'Ⅱ';
  $('playLabel').textContent = 'Pause';
  $('playButton').setAttribute('aria-label', 'Pause checkpoints');
  state.timer = setInterval(() => {
    if (state.frame >= state.data.checkpoints.length - 1) { pause(); return; }
    setFrame(state.frame + 1, false);
    if (state.frame === state.data.checkpoints.length - 1) pause();
  }, 1200);
}

function buildProvenance() {
  const { model: m, provenance: p, checkpoints } = state.data;
  const notes = (p.notes || []).map((note) => `<li>${esc(note)}</li>`).join('');
  $('provenanceContent').innerHTML = `<section class="provenance-section"><h3>MODEL</h3><dl><dt>Repository</dt><dd><a href="${esc(safeUrl(m.source_url))}" target="_blank" rel="noopener noreferrer">${esc(m.id)} ↗</a></dd><dt>Architecture</dt><dd>${m.layers} layers · ${m.heads} heads · ${esc(m.hidden_size)} hidden dimensions</dd><dt>Parameters</dt><dd>${esc(typeof m.parameters === 'number' ? number.format(m.parameters) : m.parameters)}</dd></dl></section><section class="provenance-section"><h3>HOW TO READ THIS</h3><p>Flow curves and matrix cells display attention weights from a recorded forward pass. The head grid summarizes those weights as concentration. A query can attend to itself and earlier tokens; future tokens are masked. Layer, head, and token labels start at 1.</p><p>The head grid always measures focus for the final prompt token, even when another query is selected. Focus is one minus normalized attention entropy—not a measure of head importance. Probability bars are scaled relative to the top token; the printed percentages are absolute probabilities. History and timeline positions are evenly spaced observations, not equal amounts of training.</p><p>“·” marks a space, “↵” a newline, and “⇥” a tab. Tokens can be word fragments. Attention alone does not establish a causal explanation of an output.</p></section><section class="provenance-section"><h3>EXTRACTION RECORD</h3><dl><dt>Extracted</dt><dd>${esc(p.extracted_at)}</dd><dt>Device / precision</dt><dd>${esc(p.device)} / ${esc(p.dtype)}</dd><dt>PyTorch</dt><dd>${esc(p.torch_version)}</dd><dt>Transformers</dt><dd>${esc(p.transformers_version)}</dd><dt>Extractor SHA-256</dt><dd>${esc(p.extractor_sha256)}</dd></dl>${notes ? `<ul>${notes}</ul>` : ''}</section><section class="provenance-section"><h3>IMMUTABLE CHECKPOINT SOURCES</h3><div class="checkpoint-list">${checkpoints.map((c) => `<div class="checkpoint-source"><span>${number.format(c.step)}</span><a href="${esc(safeUrl(c.source_url))}" target="_blank" rel="noopener noreferrer" title="${esc(c.commit_sha)}">${esc(c.revision)} · ${esc(c.commit_sha.slice(0, 12))} ↗</a></div>`).join('')}</div></section>`;
}

function openProvenance() {
  if (!state.data) {
    $('provenanceContent').innerHTML = '<p class="quiet">The model’s provenance will appear here once its recorded observations have loaded.</p>';
  }
  pause();
  $('provenanceDialog').showModal();
}

$('sceneTabs').addEventListener('click', (event) => {
  const button = event.target.closest('[data-scene]');
  if (!button) return;
  pause();
  state.scene = Number(button.dataset.scene);
  state.query = scene().tokens.length - 1;
  render();
});
$('sceneTabs').addEventListener('keydown', (event) => {
  if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key) || !state.data) return;
  event.preventDefault(); event.stopPropagation();
  let index = event.key === 'Home' ? 0 : event.key === 'End' ? state.data.scenes.length - 1 : (state.scene + (event.key === 'ArrowRight' ? 1 : -1) + state.data.scenes.length) % state.data.scenes.length;
  $(`scene-${index}`).click(); $(`scene-${index}`).focus();
});
$('headGrid').addEventListener('click', (event) => {
  const button = event.target.closest('[data-layer]');
  if (!button) return;
  state.layer = Number(button.dataset.layer); state.head = Number(button.dataset.head); render();
});
function selectQuery(event) {
  const element = event.target.closest('[data-query]');
  if (!element) return;
  state.query = Number(element.dataset.query); render();
}
$('queryTokens').addEventListener('click', selectQuery);
$('flowGraph').addEventListener('click', selectQuery);
$('flowGraph').addEventListener('keydown', (event) => {
  if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); selectQuery(event); }
});
$('flowGraph').addEventListener('pointerover', (event) => {
  const node = event.target.closest('[data-query]');
  if (!node || !state.data) return;
  const key = Number(node.dataset.query);
  $('attentionDetail').textContent = key > state.query ? 'Future token · hidden by the causal mask' : `${displayToken(scene().tokens[state.query])} → ${displayToken(scene().tokens[key])}  ·  ${percent(frame().attention[state.layer][state.head][state.query][key])} attention`;
});
$('flowButton').addEventListener('click', () => { state.view = 'flow'; renderVisualization(); });
$('matrixButton').addEventListener('click', () => { state.view = 'matrix'; renderVisualization(); });
$('attentionMatrix').addEventListener('pointermove', (event) => {
  if (!matrixLayout) return;
  const r = event.currentTarget.getBoundingClientRect();
  const x = (event.clientX - r.left) / r.width * matrixLayout.width;
  const y = (event.clientY - r.top) / r.height * matrixLayout.height;
  const k = Math.floor((x - matrixLayout.left) / matrixLayout.cell);
  const q = Math.floor((y - matrixLayout.top) / matrixLayout.cell);
  if (k < 0 || q < 0 || k >= matrixLayout.n || q >= matrixLayout.n) { $('matrixTip').hidden = true; return; }
  const tip = $('matrixTip');
  tip.hidden = false;
  tip.textContent = k > q ? `${q + 1} → ${k + 1}: future token masked` : `${displayToken(scene().tokens[q])} → ${displayToken(scene().tokens[k])}: ${percent(frame().attention[state.layer][state.head][q][k])}`;
  const parent = $('matrixWrap').getBoundingClientRect();
  tip.style.left = `${clamp(event.clientX - parent.left + 12, 0, parent.width - tip.offsetWidth)}px`;
  tip.style.top = `${clamp(event.clientY - parent.top - tip.offsetHeight - 8, 0, parent.height - tip.offsetHeight)}px`;
});
$('attentionMatrix').addEventListener('pointerleave', () => { $('matrixTip').hidden = true; });
$('attentionMatrix').addEventListener('click', (event) => {
  if (!matrixLayout) return;
  const r = event.currentTarget.getBoundingClientRect();
  const y = (event.clientY - r.top) / r.height * matrixLayout.height;
  const q = Math.floor((y - matrixLayout.top) / matrixLayout.cell);
  if (q >= 0 && q < matrixLayout.n) { state.query = q; render(); }
});
$('probabilityHistory').addEventListener('click', (event) => {
  if (!state.data) return;
  const r = event.currentTarget.getBoundingClientRect();
  const fraction = ((event.clientX - r.left) / r.width * 320 - 9) / 302;
  setFrame(Math.round(fraction * (state.data.checkpoints.length - 1)));
});
$('checkpointSlider').addEventListener('input', (event) => setFrame(Number(event.target.value)));
$('previousButton').addEventListener('click', () => setFrame(state.frame - 1));
$('nextButton').addEventListener('click', () => setFrame(state.frame + 1));
$('playButton').addEventListener('click', play);
$('retryButton').addEventListener('click', loadData);
$('aboutButton').addEventListener('click', openProvenance);
$('provenanceButton').addEventListener('click', openProvenance);
$('closeDialog').addEventListener('click', () => $('provenanceDialog').close());
$('provenanceDialog').addEventListener('click', (event) => { if (event.target === event.currentTarget) { const r = event.currentTarget.getBoundingClientRect(); if (event.clientX < r.left || event.clientX > r.right || event.clientY < r.top || event.clientY > r.bottom) event.currentTarget.close(); } });
document.addEventListener('keydown', (event) => {
  if (!state.data || $('provenanceDialog').open || ['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement.tagName)) return;
  if (event.key === 'ArrowLeft') { event.preventDefault(); setFrame(state.frame - 1); }
  if (event.key === 'ArrowRight') { event.preventDefault(); setFrame(state.frame + 1); }
  if (event.code === 'Space' && !['BUTTON', 'A'].includes(document.activeElement.tagName)) { event.preventDefault(); play(); }
});
document.addEventListener('visibilitychange', () => { if (document.hidden) pause(); });
let resizeTimer;
window.addEventListener('resize', () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => { if (state.data) renderVisualization(); }, 120); });
loadData();
