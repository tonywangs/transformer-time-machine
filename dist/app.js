/* Recorded observations only. No model outputs are generated in this client. */
'use strict';

const $ = (id) => document.getElementById(id);
const state = { data: null, scene: 0, frame: 0, layer: 0, head: 0, query: 0, view: 'flow', playing: false, playbackSpeed: 1 };
const motionPreference = window.matchMedia('(prefers-reduced-motion: reduce)');
const transport = { position: 0, raf: null, lastTime: null, seek: null };
const motionAnimations = new WeakMap();
function settle(element, keyframes = [{ opacity: .45, transform: 'translateY(3px)' }, { opacity: 1, transform: 'translateY(0)' }], duration = 220) {
  motionAnimations.get(element)?.cancel();
  if (motionPreference.matches || !element.animate) return;
  const animation = element.animate(keyframes, { duration, easing: 'cubic-bezier(.22,1,.36,1)' });
  motionAnimations.set(element, animation);
}
function updateReadout(id, text) {
  const element = $(id);
  if (element.textContent === text) return;
  const populated = element.textContent.length > 0;
  element.textContent = text;
  if (populated) settle(element);
}
// Preserve SVG nodes so attention styles can transition and keyboard focus survives updates.
function reconcileSvg(parent, markup) {
  const template = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  template.innerHTML = markup;
  const key = (node) => node.getAttribute('data-motion-key') || (node.hasAttribute('data-query') ? `token-${node.getAttribute('data-query')}` : node.id);
  function sync(target, source) {
    const old = [...target.children];
    [...source.children].forEach((next, index) => {
      const id = key(next);
      let current = id ? old.find((node) => key(node) === id) : old[index];
      if (!current || current.tagName !== next.tagName || (!id && key(current))) current = next.cloneNode(true);
      else {
        [...current.attributes].forEach((a) => { if (!next.hasAttribute(a.name)) current.removeAttribute(a.name); });
        [...next.attributes].forEach((a) => { if (current.getAttribute(a.name) !== a.value) current.setAttribute(a.name, a.value); });
        if (next.children.length) sync(current, next);
        else if (current.textContent !== next.textContent) current.textContent = next.textContent;
      }
      if (target.children[index] !== current) target.insertBefore(current, target.children[index] || null);
    });
    while (target.children.length > source.children.length) target.lastElementChild.remove();
  }
  sync(parent, template);
}
function updateSelectionIndicators() {
  const tabs = $('sceneTabs');
  const tab = tabs.querySelector('[aria-selected="true"]');
  const segment = $('flowButton').parentElement;
  const selected = state.view === 'flow' ? $('flowButton') : $('matrixButton');
  if (tab) {
    tabs.style.setProperty('--tab-left', `${tab.offsetLeft}px`);
    tabs.style.setProperty('--tab-top', `${tab.offsetTop}px`);
    tabs.style.setProperty('--tab-width', `${tab.offsetWidth}px`);
    tabs.style.setProperty('--tab-height', `${tab.offsetHeight}px`);
    tabs.classList.add('has-selection-indicator');
  }
  segment.style.setProperty('--segment-x', `${selected.offsetLeft}px`);
  segment.style.setProperty('--segment-width', `${selected.offsetWidth}px`);
  segment.classList.add('has-selection-indicator');
}
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
    transport.position = state.frame;
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
  const tickStride = Math.max(1, Math.ceil((state.data.checkpoints.length - 1) / 16));
  $('timelineTicks').innerHTML = state.data.checkpoints.map((_, i, checkpoints) => `<i class="${i % tickStride === 0 || i === checkpoints.length - 1 ? 'major-tick' : 'minor-tick'}"></i>`).join('');
  $('checkpointSelect').innerHTML = state.data.checkpoints.map((checkpoint, i) => `<option value="${i}">Step ${number.format(checkpoint.step)}</option>`).join('');
  $('playbackSpeed').value = String(state.playbackSpeed);
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
  if ($('queryTokens').dataset.scene !== s.id) {
    $('queryTokens').innerHTML = s.tokens.map((t, i) => `<button id="query-${i}" class="query-token" data-query="${i}" title="Token ${i + 1}: ${esc(t)}" aria-label="Inspect token ${i + 1}: ${esc(t)}"><span class="token-index" aria-hidden="true">${String(i + 1).padStart(2, '0')}</span>${esc(displayToken(t))}</button>`).join('');
    $('queryTokens').dataset.scene = s.id;
  }
  $('queryTokens').querySelectorAll('button').forEach((button, i) => button.setAttribute('aria-pressed', String(state.query === i)));
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
  updateReadout('exampleGuess', `“${top.token.trim() || displayToken(top.token)}” · ${percent(top.probability)}`);
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
  if (!$('headGrid').children.length) {
    $('headGrid').innerHTML = frame().concentration.map((row, l) => `<div class="head-row"><span class="head-row-label" aria-hidden="true">${l + 1}</span>${row.map((_, h) => `<button id="head-${l}-${h}" class="head-cell" data-layer="${l}" data-head="${h}"></button>`).join('')}</div>`).join('');
  }
  frame().concentration.forEach((row, l) => row.forEach((c, h) => {
    const cell = $(`head-${l}-${h}`);
    cell.style.background = weightColor(c);
    cell.setAttribute('aria-label', `Layer ${l + 1}, head ${h + 1}, focus ${(c * 100).toFixed(1)} percent`);
    cell.setAttribute('aria-pressed', String(state.layer === l && state.head === h));
    cell.title = `Layer ${l + 1} · Head ${h + 1} · ${(c * 100).toFixed(1)}% focus`;
  }));
}

function renderVisualization() {
  $('flowGraph').toggleAttribute('hidden', state.view !== 'flow');
  $('matrixWrap').hidden = state.view !== 'matrix';
  $('flowButton').classList.toggle('active', state.view === 'flow');
  $('matrixButton').classList.toggle('active', state.view === 'matrix');
  $('flowButton').setAttribute('aria-pressed', String(state.view === 'flow'));
  $('matrixButton').setAttribute('aria-pressed', String(state.view === 'matrix'));
  if (state.view === 'flow') renderFlow(); else renderMatrix();
  updateSelectionIndicators();
  const row = frame().attention[state.layer][state.head][state.query];
  const maxIndex = row.indexOf(Math.max(...row));
  $('attentionDetail').textContent = `${displayToken(scene().tokens[state.query])} → ${displayToken(scene().tokens[maxIndex])}  ·  ${percent(row[maxIndex])} attention`;
}

function renderFlow() {
  const s = scene();
  const row = frame().attention[state.layer][state.head][state.query];
  const svg = $('flowGraph');
  const available = Math.max(1, $('visualization').clientWidth);
  const widths = s.tokens.map((t) => Math.max(34, Math.min(110, displayToken(t).length * 6.2 + 12)));
  const natural = widths.reduce((a, b) => a + b, 0) + (s.tokens.length - 1) * 9 + 26;
  const width = available;
  const height = Math.max(100, $('visualization').clientHeight || 226);
  const compact = natural > width;
  const padding = Math.min(Math.max(...widths) / 2 + 8, width / 4);
  const spacing = (width - padding * 2) / Math.max(1, s.tokens.length - 1);
  const bands = compact ? Math.max(1, Math.ceil((Math.max(...widths) + 4) / spacing)) : 1;
  const labelHeight = compact ? 20 : 25;
  const labelStride = labelHeight + 5;
  const labelTop = height - 14 - labelHeight - (bands - 1) * labelStride;
  const surplus = Math.max(0, width - natural);
  const gap = 9 + (s.tokens.length > 1 ? surplus / (s.tokens.length - 1) : 0);
  let cursor = 13 + (s.tokens.length === 1 ? surplus / 2 : 0);
  // Stagger crowded labels instead of pushing real tokens beyond the viewport.
  const positions = widths.map((w, i) => {
    if (compact) return padding + i * spacing;
    const x = cursor + w / 2; cursor += w + gap; return x;
  });
  const source = positions[state.query];
  const baseline = labelTop - 13;
  const strongest = Math.max(...row);
  let content = `<defs><pattern id="flowDots" width="20" height="20" patternUnits="userSpaceOnUse"><circle cx="1" cy="1" r=".6" fill="#29403d"/></pattern></defs><rect x="0" y="16" width="${width}" height="${Math.max(0, baseline - 16)}" fill="url(#flowDots)" opacity=".6"/><line x1="13" x2="${width - 13}" y1="${baseline}" y2="${baseline}" stroke="#304542" stroke-dasharray="2 5"/>`;
  // Each curve is the actual selected query-to-key observation; no interpolated frames.
  row.forEach((weight, k) => {
    if (k > state.query || k === state.query) return;
    const end = positions[k];
    const distance = Math.abs(end - source);
    const lift = Math.min(baseline - 8, 25 + distance * .31);
    const opacity = weight > 0 ? .15 + .85 * Math.sqrt(weight / Math.max(strongest, .000001)) : 0;
    const stroke = 1 + 5 * Math.sqrt(weight);
    content += `<path class="attention-link" data-motion-key="edge-${k}" d="M ${source} ${baseline} C ${source} ${baseline - lift}, ${end} ${baseline - lift}, ${end} ${baseline}" fill="none" stroke="#cdeba1" stroke-opacity="${opacity}" stroke-width="${stroke}" data-weight-key="${k}"><title>${esc(displayToken(s.tokens[state.query]))} to ${esc(displayToken(s.tokens[k]))}: ${percent(weight)}</title></path>`;
  });
  content += `<path class="attention-link" data-motion-key="self" d="M ${source - 5} ${baseline} C ${source - 25} ${baseline - Math.min(44, baseline - 8)}, ${source + 25} ${baseline - Math.min(44, baseline - 8)}, ${source + 5} ${baseline}" fill="none" stroke="#cdeba1" stroke-opacity="${row[state.query] > 0 ? .15 + .85 * Math.sqrt(row[state.query] / Math.max(strongest, .000001)) : 0}" stroke-width="${1 + 5 * Math.sqrt(row[state.query])}"/>`;
  positions.forEach((x, i) => {
    const selected = i === state.query;
    const future = i > state.query;
    const w = widths[i];
    const labelY = labelTop + (i % bands) * labelStride;
    const color = future ? '#213131' : weightColor(Math.sqrt(row[i] || 0), .04);
    content += `<g class="token-node" data-query="${i}" tabindex="0" role="button" aria-label="Query token ${i + 1}: ${esc(s.tokens[i])}" aria-pressed="${selected}"><circle cx="${x}" cy="${baseline}" r="${future ? 2 : 2 + 5 * Math.sqrt(row[i] || 0)}" fill="${future ? '#364646' : '#cdeba1'}"/><line x1="${x}" y1="${baseline + 6}" x2="${x}" y2="${labelY - 3}" stroke="#4b6253" stroke-opacity=".5"/><rect x="${x - w / 2}" y="${labelY}" width="${w}" height="${labelHeight}" rx="3" fill="${color}" stroke="${selected ? '#d2ee9a' : future ? '#2d3b3b' : '#4b6253'}" stroke-width="${selected ? 1.5 : 1}"/><text x="${x}" y="${labelY + labelHeight / 2 + 3}" text-anchor="middle" fill="${future ? '#667a76' : row[i] > .25 ? '#102019' : selected ? '#f1f7df' : '#d3dfcc'}" font-size="10">${esc(displayToken(s.tokens[i]))}</text><text x="${x}" y="${height - 2}" text-anchor="middle" font-size="7" fill="${selected ? '#d2ee9a' : '#66827a'}">${selected ? 'QUERY' : future ? 'FUTURE' : String(i + 1).padStart(2, '0')}</text></g>`;
  });
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  svg.style.minWidth = '0';
  svg.style.width = '100%';
  svg.style.height = '100%';
  reconcileSvg(svg, content);
  svg.setAttribute('aria-label', `Attention from token ${state.query + 1}, ${s.tokens[state.query]}, at layer ${state.layer + 1}, head ${state.head + 1}. Select a token to inspect its attention.`);
}

let matrixLayout = null;
function renderMatrix() {
  const canvas = $('attentionMatrix');
  const snapshot = `${state.scene}:${state.frame}:${state.layer}:${state.head}:${state.query}`;
  if (canvas.dataset.snapshot && canvas.dataset.snapshot !== snapshot) settle(canvas, [{ opacity: .6 }, { opacity: 1 }], 180);
  canvas.dataset.snapshot = snapshot;
  const wrapper = $('matrixWrap');
  const size = Math.min(wrapper.clientHeight || 226, Math.max(80, wrapper.clientWidth - 60));
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
  updateReadout('leadingToken', displayToken(leading.token));
  $('leadingToken').title = `Token ID ${leading.id}: ${JSON.stringify(leading.token)}`;
  updateReadout('leadingProbability', percent(leading.probability));
  const maximum = Math.max(...tokens.map((t) => t.probability));
  const list = $('probabilityBars');
  const oldRows = new Map([...list.children].map((row) => [Number(row.dataset.token), { row, top: row.getBoundingClientRect().top }]));
  for (const [id, { row }] of oldRows) if (!tokens.some((token) => token.id === id)) row.remove();
  const entering = [];
  tokens.forEach((token, index) => {
    let row = oldRows.get(token.id)?.row;
    if (!row) {
      row = document.createElement('div');
      row.className = 'probability-row';
      row.dataset.token = token.id;
      row.innerHTML = `<span class="probability-token"></span><div class="probability-bar-track" aria-hidden="true"><div class="probability-bar" style="width:0%"></div></div><span class="probability-value"></span>`;
      entering.push(row);
    }
    row.title = `Token ${token.id}: ${JSON.stringify(token.token)}; probability ${percent(token.probability)}`;
    row.querySelector('.probability-token').textContent = displayToken(token.token);
    row.querySelector('.probability-value').textContent = percent(token.probability);
    if (list.children[index] !== row) list.insertBefore(row, list.children[index] || null);
  });
  // FLIP keeps a token's identity attached to its bar as rankings change.
  const nextTops = new Map([...list.children].map((row) => [row, row.getBoundingClientRect().top]));
  tokens.forEach((token, index) => {
    const row = list.children[index];
    row.querySelector('.probability-bar').style.width = `${maximum ? token.probability / maximum * 100 : 0}%`;
    if (entering.includes(row)) settle(row, [{ opacity: 0, transform: 'translateX(6px)' }, { opacity: 1, transform: 'translateX(0)' }]);
    else {
      const dy = oldRows.get(token.id).top - nextTops.get(row);
      if (Math.abs(dy) > .5) settle(row, [{ transform: `translateY(${dy}px)` }, { transform: 'translateY(0)' }], 240);
    }
  });
  $('targetLabel').textContent = `“${scene().target_label}”`;
  updateReadout('targetProbability', percent(frame().target_probability));
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
  reconcileSvg($('probabilityHistory'), `<defs><linearGradient id="historyFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#e8bd7b" stop-opacity=".16"/><stop offset="1" stop-color="#e8bd7b" stop-opacity="0"/></linearGradient></defs><line x1="9" y1="105" x2="311" y2="105" stroke="#304144"/><line x1="9" y1="62" x2="311" y2="62" stroke="#304144" stroke-dasharray="2 5"/><text x="9" y="10" fill="#78938b" font-family="ui-monospace,monospace" font-size="8">${percent(max)}</text><polygon points="9,105 ${points} 311,105" fill="url(#historyFill)"/><polyline points="${points}" fill="none" stroke="#e8bd7b" stroke-width="1.8"/>${values.map((p, i) => `<circle cx="${x(i)}" cy="${y(p)}" r="${i === state.frame ? 0 : 2}" fill="#e8bd7b" opacity=".55"><title>Step ${number.format(state.data.checkpoints[i].step)}: ${percent(p)}</title></circle>`).join('')}<line x1="${currentX}" y1="18" x2="${currentX}" y2="105" stroke="#e8bd7b" stroke-opacity=".4" stroke-dasharray="2 3"/><circle class="history-current" data-motion-key="history-current" cx="${currentX}" cy="${currentY}" r="4" fill="#e8bd7b" stroke="#101a1c" stroke-width="2"/>`);
  $('probabilityHistory').setAttribute('aria-label', `Probability of ${scene().target_label} across ${values.length} checkpoints. Current step ${frame().step}: ${percent(frame().target_probability)}. Chart maximum ${percent(max)}.`);
  $('historyStart').textContent = `STEP ${compactStep(state.data.checkpoints[0].step)}`;
  $('historyEnd').textContent = `STEP ${compactStep(state.data.checkpoints.at(-1).step)}`;
  const peakIndex = values.indexOf(Math.max(...values));
  const peak = scene().frames[peakIndex];
  $('peakProbability').dataset.frame = peakIndex;
  $('peakProbability').innerHTML = `<span>Highest recorded chance <span aria-hidden="true">↗</span></span><strong>“${esc(scene().target_label.trim())}” · ${percent(peak.target_probability)} · Step ${number.format(peak.step)}</strong>`;
  $('peakProbability').setAttribute('aria-label', `Jump to highest recorded chance of ${scene().target_label.trim()}: ${percent(peak.target_probability)}, step ${number.format(peak.step)}`);
}

function renderTimeline() {
  updateReadout('stepNumber', number.format(frame().step));
  $('checkpointCount').textContent = `${state.frame + 1} / ${state.data.checkpoints.length} observations`;
  $('checkpointSlider').value = state.frame;
  $('checkpointSelect').value = String(state.frame);
  $('checkpointSlider').setAttribute('aria-valuetext', `Training step ${number.format(frame().step)}, checkpoint ${state.frame + 1} of ${state.data.checkpoints.length}`);
  paintPlayhead();
  [...$('timelineTicks').children].forEach((tick, i) => { tick.classList.toggle('is-past', i < state.frame); tick.classList.toggle('is-current', i === state.frame); });
  $('previousButton').disabled = state.frame === 0;
  $('nextButton').disabled = state.frame === state.data.checkpoints.length - 1;
  $('playButton').disabled = state.data.checkpoints.length < 2;
}

function paintPlayhead() {
  const position = motionPreference.matches ? state.frame : transport.position;
  const progress = position / Math.max(1, (state.data?.checkpoints.length || 1) - 1);
  $('timelineRail').style.setProperty('--playhead', `${progress * 100}%`);
  $('timelineRail').style.setProperty('--playhead-scale', progress);
}
function cancelTransport() {
  cancelAnimationFrame(transport.raf);
  transport.raf = null;
  transport.lastTime = null;
  transport.seek = null;
}
function seekPlayhead(index, immediate = false) {
  cancelTransport();
  if (immediate || motionPreference.matches || Math.abs(index - transport.position) < .001) {
    transport.position = index;
    paintPlayhead();
    return;
  }
  transport.seek = { from: transport.position, to: index, start: null, duration: 360 };
  transport.raf = requestAnimationFrame(tickTransport);
}
function tickTransport(now) {
  transport.raf = null;
  if (transport.seek) {
    const seek = transport.seek;
    if (seek.start === null) seek.start = now;
    const t = clamp((now - seek.start) / seek.duration, 0, 1);
    transport.position = seek.from + (seek.to - seek.from) * (1 - Math.pow(1 - t, 3));
    paintPlayhead();
    if (t < 1) { transport.raf = requestAnimationFrame(tickTransport); return; }
    transport.seek = null;
    transport.lastTime = now;
  }
  if (!state.playing) return;
  if (transport.lastTime === null) transport.lastTime = now;
  const elapsed = now - transport.lastTime;
  transport.lastTime = now;
  const last = state.data.checkpoints.length - 1;
  transport.position = Math.min(last, transport.position + elapsed / (1200 / state.playbackSpeed));
  const observedFrame = Math.floor(transport.position + 1e-9);
  if (observedFrame !== state.frame) {
    state.frame = observedFrame;
    render();
  }
  paintPlayhead();
  if (transport.position >= last) { pause(); return; }
  transport.raf = requestAnimationFrame(tickTransport);
}
function setFrame(index, immediate = false) {
  if (!state.data) return;
  pause();
  state.frame = clamp(Math.round(index), 0, state.data.checkpoints.length - 1);
  seekPlayhead(state.frame, immediate);
  render();
  $('screenReaderStatus').textContent = `Training step ${number.format(frame().step)}. Most likely next token: ${frame().top_tokens[0].token}, ${percent(frame().top_tokens[0].probability)}.`;
}
function pause() {
  cancelTransport();
  state.playing = false;
  document.querySelector('.timeline-panel').classList.remove('is-playing');
  $('playIcon').textContent = '▶';
  $('playLabel').textContent = 'Play training';
  $('playButton').setAttribute('aria-label', 'Play checkpoints');
}
function play() {
  if (!state.data || state.data.checkpoints.length < 2) return;
  if (state.playing) { pause(); return; }
  // Finish an interrupted seek before continuing, or smoothly rewind at the endpoint.
  if (state.frame >= state.data.checkpoints.length - 1) setFrame(0);
  else if (transport.position < state.frame || transport.position >= state.frame + 1) seekPlayhead(state.frame);
  state.playing = true;
  document.querySelector('.timeline-panel').classList.add('is-playing');
  $('playIcon').textContent = 'Ⅱ';
  $('playLabel').textContent = 'Pause';
  $('playButton').setAttribute('aria-label', 'Pause checkpoints');
  if (transport.raf === null) transport.raf = requestAnimationFrame(tickTransport);
}

motionPreference.addEventListener('change', () => {
  if (motionPreference.matches) {
    document.getAnimations().forEach((animation) => animation.cancel());
    if (transport.seek) {
      transport.position = state.frame;
      transport.seek = null;
    }
  }
  paintPlayhead();
});

function buildProvenance() {
  const { model: m, provenance: p, checkpoints } = state.data;
  const notes = (p.notes || []).map((note) => `<li>${esc(note)}</li>`).join('');
  $('provenanceContent').innerHTML = `<section class="provenance-section"><h3>MODEL</h3><dl><dt>Repository</dt><dd><a href="${esc(safeUrl(m.source_url))}" target="_blank" rel="noopener noreferrer">${esc(m.id)} ↗</a></dd><dt>Architecture</dt><dd>${m.layers} layers · ${m.heads} heads · ${esc(m.hidden_size)} hidden dimensions</dd><dt>Parameters</dt><dd>${esc(typeof m.parameters === 'number' ? number.format(m.parameters) : m.parameters)}</dd></dl></section><section class="provenance-section"><h3>HOW TO READ THIS</h3><p>Flow curves and matrix cells display attention weights from a recorded forward pass. The head grid summarizes those weights as concentration. A query can attend to itself and earlier tokens; future tokens are masked. Layer, head, and token labels start at 1.</p><p>The head grid always measures focus for the final prompt token, even when another query is selected. Focus is one minus normalized attention entropy—not a measure of head importance. Probability bars are scaled relative to the top token; the printed percentages are absolute probabilities. History and timeline positions are evenly spaced observations, not equal amounts of training. The playhead glides between checkpoints; printed values update only when a recorded checkpoint is reached. Bar lengths and attention colors briefly transition for readability.</p><p>“·” marks a space, “↵” a newline, and “⇥” a tab. Tokens can be word fragments. Attention alone does not establish a causal explanation of an output.</p></section><section class="provenance-section"><h3>EXTRACTION RECORD</h3><dl><dt>Extracted</dt><dd>${esc(p.extracted_at)}</dd><dt>Device / precision</dt><dd>${esc(p.device)} / ${esc(p.dtype)}</dd><dt>PyTorch</dt><dd>${esc(p.torch_version)}</dd><dt>Transformers</dt><dd>${esc(p.transformers_version)}</dd><dt>Extractor SHA-256</dt><dd>${esc(p.extractor_sha256)}</dd></dl>${notes ? `<ul>${notes}</ul>` : ''}</section><section class="provenance-section"><h3>IMMUTABLE CHECKPOINT SOURCES</h3><div class="checkpoint-list">${checkpoints.map((c) => `<div class="checkpoint-source"><span>${number.format(c.step)}</span><a href="${esc(safeUrl(c.source_url))}" target="_blank" rel="noopener noreferrer" title="${esc(c.commit_sha)}">${esc(c.revision)} · ${esc(c.commit_sha.slice(0, 12))} ↗</a></div>`).join('')}</div></section>`;
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
  if (state.scene === Number(button.dataset.scene)) return;
  state.scene = Number(button.dataset.scene);
  state.query = scene().tokens.length - 1;
  setFrame(state.frame);
  document.querySelectorAll('.prompt-panel,.prediction-panel,.visualization').forEach((panel) => settle(panel, undefined, 240));
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
function setView(view) {
  if (state.view === view) return;
  state.view = view;
  renderVisualization();
  settle(view === 'flow' ? $('flowGraph') : $('matrixWrap'));
}
$('flowButton').addEventListener('click', () => setView('flow'));
$('matrixButton').addEventListener('click', () => setView('matrix'));
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
  const svg = event.currentTarget;
  const transform = svg.getScreenCTM();
  if (!transform) return;
  const point = svg.createSVGPoint();
  point.x = event.clientX;
  point.y = event.clientY;
  const fraction = (point.matrixTransform(transform.inverse()).x - 9) / 302;
  setFrame(Math.round(fraction * (state.data.checkpoints.length - 1)));
});
$('peakProbability').addEventListener('click', (event) => setFrame(Number(event.currentTarget.dataset.frame)));
$('checkpointSlider').addEventListener('input', (event) => setFrame(Number(event.target.value), true));
$('checkpointSlider').addEventListener('pointermove', (event) => {
  if (!state.data || event.buttons) { $('timelinePreview').hidden = true; return; }
  const rect = event.currentTarget.getBoundingClientRect();
  const fraction = clamp((event.clientX - rect.left - 6) / (rect.width - 12), 0, 1);
  const index = Math.round(fraction * (state.data.checkpoints.length - 1));
  const preview = $('timelinePreview');
  preview.textContent = `Step ${number.format(state.data.checkpoints[index].step)}`;
  preview.hidden = false;
  const left = 6 + index / (state.data.checkpoints.length - 1) * (rect.width - 12);
  preview.style.left = `${clamp(left, preview.offsetWidth / 2, rect.width - preview.offsetWidth / 2)}px`;
});
$('checkpointSlider').addEventListener('pointerleave', () => { $('timelinePreview').hidden = true; });
$('checkpointSlider').addEventListener('pointerdown', () => { pause(); $('timelinePreview').hidden = true; });
$('checkpointSelect').addEventListener('change', (event) => setFrame(Number(event.target.value)));
$('playbackSpeed').addEventListener('change', (event) => {
  const speed = Number(event.target.value);
  if (![0.5, 1, 2, 4].includes(speed)) return;
  state.playbackSpeed = speed;
  // The next animation frame uses this speed without resetting its fractional position.
});
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
let visualizationResizeFrame;
function resizeVisualization() {
  cancelAnimationFrame(visualizationResizeFrame);
  visualizationResizeFrame = requestAnimationFrame(() => {
    if (state.data && !$('instrument').hidden) renderVisualization();
  });
}
// Layout changes from scene text and viewport height both resize the drawing area.
if (typeof ResizeObserver !== 'undefined') {
  new ResizeObserver(resizeVisualization).observe($('visualization'));
} else {
  window.addEventListener('resize', resizeVisualization);
}
loadData();
