/* Real observations only: bundled examples or completed server-side forward passes. */
'use strict';

const $ = (id) => document.getElementById(id);
const palettes = {
  dark: {
    background: '#101a1c', low: [37, 58, 54], high: [210, 238, 154],
    dots: '#29403d', baseline: '#304542', flow: '#cdeba1',
    masked: '#213131', maskedDot: '#364646', maskedBorder: '#2d3b3b', maskedText: '#96a9a6',
    stem: '#4b6253', selected: '#d2ee9a', ink: '#000000', inverse: '#ffffff',
    axis: '#66827a', label: '#96a9a6', matrixMask: '#172527',
    history: '#e8bd7b', historyGrid: '#304144', historyAxis: '#78938b',
    lineWeight: 5, nodeRadius: 5, nodeBase: 2
  },
  light: {
    background: '#f7f7f2', low: [232, 236, 231], high: [65, 101, 124],
    dots: '#b7c2bd', baseline: '#bec8c2', flow: '#41657c',
    masked: '#eef0eb', maskedDot: '#aeb9b4', maskedBorder: '#cbd2cc', maskedText: '#626e70',
    stem: '#a0aea6', selected: '#41657c', ink: '#000000', inverse: '#ffffff',
    axis: '#626e70', label: '#626e70', matrixMask: '#edf0ea',
    history: '#886a42', historyGrid: '#d5dad4', historyAxis: '#626e70',
    lineWeight: 2.4, nodeRadius: 2.8, nodeBase: 1.7
  }
};
let palette = palettes[document.documentElement.dataset.theme] || palettes.light;
const state = { data: null, scene: 0, frame: 0, layer: 0, head: 0, query: 0, view: 'flow', playing: false, playbackSpeed: 1 };
let modelCatalog, bundledData;
let selectedModel = '1b';
const completedModels = new Map();
const presetModels = new Map();
const customResults = new Map();
let activeTab = 'story';
let modelLoading = false, modelSwitchGeneration = 0, requestIntent = 0;
let pendingCancellation = Promise.resolve();
const custom = { selected: false, sceneIndex: null, busy: false, status: '', statusKind: '' };
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
  capital: { tab: 'Complete a fact', question: 'When does Paris become #1?', explanation: 'Finish the sentence with a city. Rewind to see when “Paris” becomes the model’s top guess.', reference: 'Direct completion' }
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
  return `rgb(${palette.low.map((channel, i) => Math.round(channel + (palette.high[i] - channel) * t)).join(",")})`;
};
// Choose readable ink from the actual shade, including intermediate heatmap values.
function tokenInk(rgb) {
  const luminance = (channels) => channels.map((v) => {
    const value = v / 255;
    return value <= .04045 ? value / 12.92 : Math.pow((value + .055) / 1.055, 2.4);
  }).reduce((sum, value, i) => sum + value * [.2126, .7152, .0722][i], 0);
  const hex = (value) => [1, 3, 5].map((i) => parseInt(value.slice(i, i + 2), 16));
  const background = luminance(rgb.match(/\d+/g).map(Number));
  const contrast = (color) => {
    const foreground = luminance(hex(color));
    return (Math.max(background, foreground) + .05) / (Math.min(background, foreground) + .05);
  };
  return contrast(palette.ink) >= contrast(palette.inverse) ? palette.ink : palette.inverse;
}
function applyTheme(theme, remember = false) {
  const dark = theme === 'dark';
  const selected = dark ? 'dark' : 'light';
  document.documentElement.dataset.theme = selected;
  document.documentElement.style.colorScheme = selected;
  palette = palettes[selected];
  document.querySelector('meta[name="theme-color"]').content = palette.background;
  $('themeLabel').textContent = dark ? 'Dark' : 'Light';
  $('themeToggle').setAttribute('aria-pressed', String(dark));
  $('themeToggle').setAttribute('aria-label', `Switch to ${dark ? 'light' : 'dark'} mode`);
  $('themeToggle').title = `Switch to ${dark ? 'light' : 'dark'} mode`;
  $('headFocusDescription').textContent = `${dark ? 'Brighter' : 'Darker'} squares focus on fewer tokens.`;
  if (remember) { try { localStorage.setItem('ttm-theme', selected); } catch (_) {} }
  // Redraw canvas as well as SVG; theme changes preserve playback and every selection.
  if (state.data) { renderHeads(); renderVisualization(); renderHistory(); }
}
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

function renderCustomUI() {
  const panel = document.querySelector('.prompt-panel');
  $('customPromptForm').hidden = !custom.selected;
  if (custom.selected) panel.dataset.customMode = scene().custom ? 'result' : 'draft';
  else delete panel.dataset.customMode;
  const selector = $('modelSelect');
  if (selector) selector.value = selectedModel;
  $('customPromptForm').setAttribute('aria-busy', String(custom.busy));
  $('customPromptInput').readOnly = custom.busy;
  $('analyzePromptButton').disabled = modelLoading || custom.busy || !$('customPromptInput').value.trim();
  $('analyzePromptButton').textContent = custom.busy ? 'Analyzing…' : 'Analyze';
  $('cancelPromptButton').hidden = !custom.busy;
  $('customPromptStatus').textContent = custom.status;
  $('customPromptStatus').dataset.state = custom.statusKind;
  $('presetStatus').hidden = custom.selected || !custom.status;
  $('presetStatus').textContent = custom.status;
  $('presetStatus').dataset.state = custom.statusKind;
  const modelDiffers = state.data.model.id !== modelCatalog?.[selectedModel]?.model.id;
  if ($('displayedModel')) $('displayedModel').textContent = modelDiffers ? `Viewing ${state.data.model.label.replace('-deduped', '')}` : `${state.data.model.layers} layers · ${state.data.model.heads} heads`;
  const draftDiffers = custom.selected && (modelDiffers || !scene().custom || $('customPromptInput').value !== scene().prompt);
  $('customPendingNote').hidden = !modelLoading && !modelDiffers && (!custom.selected || (!custom.busy && !draftDiffers));
  $('customPendingNote').textContent = modelDiffers ? `Still showing ${state.data.model.label.replace('-deduped', '')} · ${modelLoading || custom.busy ? 'loading selected model' : 'Selected model unavailable'}` : scene().custom
    ? custom.busy ? 'Previous result · analysis in progress' : 'Draft changed · Analyze to update'
    : `Showing example: ${exampleCopy[scene().id]?.tab || scene().title}`;
  $('customPendingNote').title = `Currently displayed prompt: ${scene().prompt}`;
  $('customPendingNote').setAttribute('aria-label', `${$('customPendingNote').textContent}. Currently displayed prompt: ${scene().prompt}`);
}

async function experimentRequest(body) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 35000);
  try {
    const response = await fetch('/api/analyze', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body), signal: controller.signal, cache: 'no-store' });
    let result;
    try { result = await response.json(); } catch (_) { throw new Error('The experiment server is unavailable. Please try again.'); }
    if (!response.ok) {
      const error = new Error(typeof result.error === 'string' ? result.error : 'The experiment could not be started. Please try again.');
      error.permanent = response.status >= 400 && response.status < 500;
      throw error;
    }
    return result;
  } catch (error) {
    if (error.name === 'AbortError') throw new Error('The experiment server is taking too long to respond.');
    throw error;
  } finally { clearTimeout(timeout); }
}

const experiment = new PromptExperiment({
  request: experimentRequest,
  onProgress(result) {
    custom.busy = true;
    custom.statusKind = '';
    const completed = Math.min(result.total, Math.max(0, result.completed || 0));
    custom.status = result.status === 'starting' ? 'Starting your experiment…'
      : result.status === 'queued' ? 'Waking the model… larger models can take several minutes.'
      : completed === result.total ? `Loading views · ${Math.min(result.total, (result.index ?? 0) + 1)} / ${result.total}`
      : `${completed} / ${result.total} checkpoints measured`;
    renderCustomUI();
  },
  onComplete(result) {
    // Install model, checkpoints, and observations as one validated record.
    const next = { schema_version: 1, model: result.model, checkpoints: result.checkpoints,
      provenance: result.provenance, scenes: [result] };
    validateData(next);
    customResults.set(selectedModel, result);
    activeTab = 'custom';
    completedModels.set(`${selectedModel}:${result.prompt}`, result);
    if (completedModels.size > 4) completedModels.delete(completedModels.keys().next().value);
    pause();
    state.data = next;
    state.scene = 0;
    custom.sceneIndex = 0;
    custom.selected = true;
    custom.busy = false;
    custom.statusKind = 'success';
    custom.status = '';
    state.layer = Math.min(state.layer, result.model.layers - 1);
    state.frame = custom.resumeFrame ?? result.frames.length - 1;
    custom.resumeFrame = null;
    transport.position = state.frame;
    state.query = result.tokens.length - 1;
    const focus = result.frames[state.frame].concentration[state.layer];
    state.head = focus.indexOf(Math.max(...focus));
    $('headGrid').replaceChildren();
    initDataUI();
    render();
    settle($('visualization'));
    $('scene-custom').title = 'Your experiment is ready';
    $('screenReaderStatus').textContent = 'Your prompt is ready across all 38 checkpoints.';
  },
  onError(message) {
    custom.resumeFrame = null;
    custom.busy = false;
    custom.statusKind = 'error';
    custom.status = message;
    renderCustomUI();
  }
});

$('customPromptForm').addEventListener('submit', (event) => {
  event.preventDefault();
  if (!state.data || custom.busy) return;
  pause();
  runSelectedModel($('customPromptInput').value);
});
$('customPromptInput').addEventListener('input', () => {
  if (!custom.busy) { custom.status = ''; custom.statusKind = ''; }
  renderCustomUI();
});
$('cancelPromptButton').addEventListener('click', () => {
  cancelAnalysis();
  custom.busy = false;
  custom.resumeFrame = null;
  custom.status = 'Cancelled. Your previous observations are still here.';
  custom.statusKind = '';
  renderCustomUI();
});
window.addEventListener('pagehide', () => {
  if (experiment.job) navigator.sendBeacon('/api/analyze', new Blob([JSON.stringify({ action: 'cancel', job_id: experiment.job })], { type: 'application/json' }));
});

async function loadData() {
  pause();
  $('loadingState').hidden = false;
  $('errorState').hidden = true;
  $('instrument').hidden = true;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 30000);
  try {
    const response = await fetch(selectedModel === '70m' ? 'data/traces.json' : `data/examples-${selectedModel}.json`, { signal: controller.signal, cache: 'no-cache' });
    if (!response.ok) throw new Error(`The observation file returned HTTP ${response.status}. Please try again in a moment.`);
    const data = await response.json();
    const catalogResponse = await fetch('data/models.json', { signal: controller.signal });
    if (!catalogResponse.ok) throw new Error('The model catalog could not load.');
    modelCatalog = await catalogResponse.json();
    validateData(data);
    // Presentation order only: keep the downloaded observations unchanged.
    const order = (id) => exampleOrder.includes(id) ? exampleOrder.indexOf(id) : exampleOrder.length;
    state.data = { ...data, scenes: [...data.scenes].sort((a, b) => order(a.id) - order(b.id)) };
    bundledData = { ...state.data, scenes: [...state.data.scenes] };
    presetModels.set(selectedModel, bundledData);
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
  if (!$('modelSelect')) $('modelSpec').innerHTML = `<label class="model-picker"><select id="modelSelect" aria-label="Pythia model size">${Object.entries(modelCatalog).map(([key, manifest]) => `<option value="${key}" ${key === selectedModel ? 'selected' : ''}>Pythia ${key.toUpperCase()}</option>`).join('')}</select></label><span class="spec-line" id="displayedModel">Viewing ${esc(m.label.replace('-deduped', ''))} · ${m.layers} layers · ${m.heads} heads</span>`;
  $('displayedModel').textContent = `${m.layers} layers · ${m.heads} heads`;
  if (!$('sceneTabs').children.length) $('sceneTabs').innerHTML = bundledData.scenes.map((s, i) => `<button class="scene-tab" id="scene-${i}" role="tab" aria-selected="${!custom.selected && i === state.scene}" aria-controls="instrument" data-scene="${i}">${esc(exampleCopy[s.id]?.tab || s.title)}</button>`).join('') + '<button class="scene-tab" id="scene-custom" role="tab" aria-selected="false" aria-controls="instrument" data-custom="true">Your prompt</button>';
  $('checkpointSlider').max = state.data.checkpoints.length - 1;
  const tickStride = Math.max(1, Math.ceil((state.data.checkpoints.length - 1) / 16));
  $('timelineTicks').innerHTML = state.data.checkpoints.map((_, i, checkpoints) => `<i class="${i % tickStride === 0 || i === checkpoints.length - 1 ? 'major-tick' : 'minor-tick'}"></i>`).join('');
  $('checkpointSelect').innerHTML = state.data.checkpoints.map((checkpoint, i) => `<option value="${i}">Step ${number.format(checkpoint.step)}</option>`).join('');
  $('playbackSpeed').value = String(state.playbackSpeed);
  $('timelineStart').textContent = `${number.format(state.data.checkpoints[0].step)}`;
  $('timelineEnd').textContent = `${number.format(state.data.checkpoints.at(-1).step)}`;
  $('headColumnLabels').style.gridTemplateColumns = `repeat(${m.heads},1fr)`;
  $('headColumnLabels').innerHTML = Array.from({ length: m.heads }, (_, i) => `<span>${i + 1}</span>`).join('');
  $('headGrid').style.setProperty('--head-count', m.heads);
  $('layerPage').innerHTML = Array.from({ length: Math.ceil(m.layers / 6) }, (_, page) => `<option value="${page}">Layers ${page * 6 + 1}–${Math.min(m.layers, page * 6 + 6)}</option>`).join('');
  $('layerPage').hidden = m.layers <= 6;
  buildProvenance();
}

function render() {
  if (!state.data) return;
  const activeId = document.activeElement?.id;
  const s = scene();
  const f = frame();
  renderExample(s, f);
  $('headLabel').textContent = `Layer ${state.layer + 1} · Head ${state.head + 1}`;
  $('focusValue').textContent = `${(f.concentration[state.layer][state.head] * 100).toFixed(1)}%`;
  document.querySelectorAll('.scene-tab').forEach((b) => {
    const selected = b.dataset.custom ? custom.selected : !custom.selected && bundledData.scenes[Number(b.dataset.scene)]?.id === activeTab;
    b.setAttribute('aria-selected', String(selected)); b.tabIndex = selected ? 0 : -1;
  });
  renderCustomUI();
  if ($('queryTokens').dataset.scene !== s.id) {
    $('queryTokens').innerHTML = s.tokens.map((t, i) => `<button id="query-${i}" class="query-token" data-query="${i}" title="Token ${i + 1}: ${esc(t)}" aria-label="Inspect token ${i + 1}: ${esc(t)}"><span class="token-index" aria-hidden="true">${String(i + 1).padStart(2, '0')}</span>${esc(displayToken(t))}</button>`).join('');
    $('queryTokens').dataset.scene = s.id;
    $('queryTokens').classList.toggle('has-many-tokens', s.tokens.length > 12);
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
  // Line breaks group the existing color tokens; no tokens are added to the input.
  if (s.id === 'echo') {
    $('promptText').innerHTML = `<span class="sequence-line">${esc(s.tokens.slice(0, 6).join(''))}</span><span class="sequence-line">${esc(s.tokens.slice(6).join(''))}</span>`;
  } else $('promptText').textContent = s.prompt;
  $('referenceLabel').textContent = s.custom ? 'Tracking' : 'Reference';
  $('referenceLabel').title = s.custom ? 'The final checkpoint’s top guess, tracked across training; not a correct-answer label.' : copy?.reference || 'Expected continuation';
  $('referenceCompletion').textContent = s.target_label.trim() || displayToken(s.target_label);

}

function renderHeads() {
  if (!$('headGrid').children.length) {
    $('headGrid').innerHTML = frame().concentration.map((row, l) => `<div class="head-row"><span class="head-row-label" aria-hidden="true">${l + 1}</span>${row.map((_, h) => `<button id="head-${l}-${h}" class="head-cell" data-layer="${l}" data-head="${h}"></button>`).join('')}</div>`).join('');
  }
  const page = Math.floor(state.layer / 6);
  [...$('headGrid').children].forEach((row, index) => { row.hidden = Math.floor(index / 6) !== page; });
  $('layerPage').value = String(page);
  frame().concentration.forEach((row, l) => row.forEach((c, h) => {
    const cell = $(`head-${l}-${h}`);
    cell.style.background = weightColor(c);
    cell.setAttribute('aria-label', `Layer ${l + 1}, head ${h + 1}, focus ${(c * 100).toFixed(1)} percent`);
    cell.setAttribute('aria-pressed', String(state.layer === l && state.head === h));
    cell.title = `Layer ${l + 1} · Head ${h + 1} · ${(c * 100).toFixed(1)}% focus`;
  }));
}

function renderVisualization() {
  const combined = window.innerWidth >= 1100 && $('visualization').clientHeight >= 300;
  $('visualization').classList.toggle('combined-attention', combined);
  $('flowButton').parentElement.hidden = combined;
  $('flowGraph').toggleAttribute('hidden', !combined && state.view !== 'flow');
  $('matrixWrap').hidden = !combined && state.view !== 'matrix';
  $('flowButton').classList.toggle('active', state.view === 'flow');
  $('matrixButton').classList.toggle('active', state.view === 'matrix');
  $('flowButton').setAttribute('aria-pressed', String(state.view === 'flow'));
  $('matrixButton').setAttribute('aria-pressed', String(state.view === 'matrix'));
  if (combined || state.view === 'flow') renderFlow();
  if (combined || state.view === 'matrix') renderMatrix();
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
  const height = Math.max(100, svg.clientHeight || 226);
  // Dense prompts use numbered nodes; the query strip and tooltips retain every exact token.
  const widest = Math.max(...s.tokens.map((t) => Math.max(34, Math.min(110, displayToken(t).length * 6.2 + 12))));
  const potentialSpacing = (available - Math.min(widest + 16, available / 2)) / Math.max(1, s.tokens.length - 1);
  const textBands = Math.ceil((widest + 4) / Math.max(1, potentialSpacing));
  const numbered = s.tokens.length > 12 || textBands > Math.max(1, Math.floor((height - 65) / 25));
  const labels = s.tokens.map((t, i) => numbered ? String(i + 1) : displayToken(t).length > 15 ? `${displayToken(t).slice(0, 14)}…` : displayToken(t));
  const widths = labels.map((label) => Math.max(numbered ? 21 : 34, Math.min(110, label.length * 6.2 + 12)));
  const natural = widths.reduce((a, b) => a + b, 0) + (s.tokens.length - 1) * 9 + 26;
  const width = available;
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
  let content = `<defs><pattern id="flowDots" width="20" height="20" patternUnits="userSpaceOnUse"><circle cx="1" cy="1" r=".6" fill="${palette.dots}"/></pattern></defs><rect x="0" y="16" width="${width}" height="${Math.max(0, baseline - 16)}" fill="url(#flowDots)" opacity=".6"/><line x1="13" x2="${width - 13}" y1="${baseline}" y2="${baseline}" stroke="${palette.baseline}" stroke-dasharray="2 5"/>`;
  // Each curve is the actual selected query-to-key observation; no interpolated frames.
  row.forEach((weight, k) => {
    if (k > state.query || k === state.query) return;
    const end = positions[k];
    const distance = Math.abs(end - source);
    const lift = Math.min(baseline - 8, 25 + distance * .31);
    const opacity = weight > 0 ? .15 + .85 * Math.sqrt(weight / Math.max(strongest, .000001)) : 0;
    const stroke = 1 + palette.lineWeight * Math.sqrt(weight);
    content += `<path class="attention-link" data-motion-key="edge-${k}" d="M ${source} ${baseline} C ${source} ${baseline - lift}, ${end} ${baseline - lift}, ${end} ${baseline}" fill="none" stroke="${palette.flow}" stroke-opacity="${opacity}" stroke-width="${stroke}" data-weight-key="${k}"><title>${esc(displayToken(s.tokens[state.query]))} to ${esc(displayToken(s.tokens[k]))}: ${percent(weight)}</title></path>`;
  });
  content += `<path class="attention-link" data-motion-key="self" d="M ${source - 5} ${baseline} C ${source - 25} ${baseline - Math.min(44, baseline - 8)}, ${source + 25} ${baseline - Math.min(44, baseline - 8)}, ${source + 5} ${baseline}" fill="none" stroke="${palette.flow}" stroke-opacity="${row[state.query] > 0 ? .15 + .85 * Math.sqrt(row[state.query] / Math.max(strongest, .000001)) : 0}" stroke-width="${1 + palette.lineWeight * Math.sqrt(row[state.query])}"/>`;
  positions.forEach((x, i) => {
    const selected = i === state.query;
    const future = i > state.query;
    const w = widths[i];
    const labelY = labelTop + (i % bands) * labelStride;
    const color = future ? palette.masked : weightColor(Math.sqrt(row[i] || 0), .04);
    content += `<g class="token-node" data-query="${i}" tabindex="0" role="button" aria-label="Query token ${i + 1}: ${esc(s.tokens[i])}" aria-pressed="${selected}"><title>${i + 1}: ${esc(displayToken(s.tokens[i]))}</title><circle cx="${x}" cy="${baseline}" r="${future ? 2 : palette.nodeBase + palette.nodeRadius * Math.sqrt(row[i] || 0)}" fill="${future ? palette.maskedDot : palette.flow}"/><line x1="${x}" y1="${baseline + 6}" x2="${x}" y2="${labelY - 3}" stroke="${palette.stem}" stroke-opacity=".5"/><rect x="${x - w / 2}" y="${labelY}" width="${w}" height="${labelHeight}" rx="3" fill="${color}" stroke="${selected ? palette.selected : future ? palette.maskedBorder : palette.stem}" stroke-width="${selected ? 1.5 : 1}"/><text x="${x}" y="${labelY + labelHeight / 2 + 3}" text-anchor="middle" fill="${future ? palette.maskedText : tokenInk(color)}" font-size="10">${esc(labels[i])}</text><text x="${x}" y="${height - 2}" text-anchor="middle" font-size="7" fill="${selected ? palette.selected : palette.axis}">${selected ? (numbered ? '↑' : 'QUERY') : numbered ? '' : future ? 'FUTURE' : String(i + 1).padStart(2, '0')}</text></g>`;
  });
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  svg.style.minWidth = '0';
  svg.style.width = '100%';
  svg.style.height = '100%';
  reconcileSvg(svg, content);
  svg.setAttribute('aria-label', `Attention from token ${state.query + 1}, ${s.tokens[state.query]}, at layer ${state.layer + 1}, head ${state.head + 1}. Select a token to inspect its attention.`);
}

let matrixLayout = null;
let matrixAnimation = null;
let matrixDisplay = null;
function renderMatrix() {
  const canvas = $('attentionMatrix');
  cancelAnimationFrame(matrixAnimation);
  matrixAnimation = null;
  const wrapper = $('matrixWrap');
  const size = Math.min(wrapper.clientHeight || 226, Math.max(80, wrapper.clientWidth - 60));
  const n = scene().tokens.length;
  const left = 40;
  const top = 18;
  const gridSize = size - top - 12;
  const cell = gridSize / n;
  const width = gridSize + left + 50;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  if (canvas.width !== Math.floor(width * dpr)) canvas.width = width * dpr;
  if (canvas.height !== Math.floor(size * dpr)) canvas.height = size * dpr;
  canvas.style.width = `${width}px`;
  canvas.style.height = `${size}px`;
  const ctx = canvas.getContext('2d');
  const target = frame().attention[state.layer][state.head].map(row => row.map(value => Math.sqrt(value)));
  const from = matrixDisplay?.length === n ? matrixDisplay : target;
  const changing = from.some((row, q) => row.some((value, k) => Math.abs(value - target[q][k]) > .00001));
  const start = performance.now();
  function paint(now) {
    const progress = motionPreference.matches || !changing ? 1 : Math.min(1, (now - start) / 220);
    const eased = 1 - Math.pow(1 - progress, 3);
    matrixDisplay = target.map((row, q) => row.map((value, k) => from[q][k] + (value - from[q][k]) * eased));
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, size);
    matrixDisplay.forEach((row, q) => row.forEach((value, k) => {
    ctx.fillStyle = k > q ? palette.matrixMask : weightColor(value);
    ctx.fillRect(left + k * cell, top + q * cell, Math.max(1, cell - .7), Math.max(1, cell - .7));
  }));
  ctx.strokeStyle = palette.selected;
  ctx.lineWidth = 1;
  ctx.strokeRect(left - 2, top + state.query * cell - 1, gridSize + 3, cell + 1);
  ctx.font = '8px ui-monospace, monospace';
  ctx.fillStyle = palette.label;
  ctx.textAlign = 'center';
  ctx.fillText('KEY TOKEN →', left + gridSize / 2, 9);
  ctx.save(); ctx.translate(11, top + gridSize / 2); ctx.rotate(-Math.PI / 2); ctx.fillText('QUERY TOKEN →', 0, 0); ctx.restore();
  ctx.textAlign = 'right';
  const interval = n > 15 ? Math.ceil(n / 10) : 1;
  for (let q = 0; q < n; q += interval) ctx.fillText(String(q + 1), left - 7, top + (q + .65) * cell);
  ctx.fillStyle = palette.axis; ctx.textAlign = 'left';
  ctx.fillText('FUTURE', left + gridSize - 28, top + 14);
  ctx.fillText('MASKED', left + gridSize - 28, top + 25);
    matrixAnimation = progress < 1 ? requestAnimationFrame(paint) : null;
  }
  paint(start);
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
  $('targetLabel').textContent = `P(“${scene().target_label.trim() || displayToken(scene().target_label)}”)`;
  updateReadout('targetProbability', percent(frame().target_probability));
}

function renderHistory() {
  const values = scene().frames.map((f) => f.target_probability);
  const max = Math.max(...values, .000001);
  const chart = $('probabilityHistory');
  const bounds = chart.getBoundingClientRect();
  const width = Math.max(100, bounds.width || 320);
  const chartHeight = Math.max(50, bounds.height || 120);
  chart.setAttribute('viewBox', `0 0 ${width} ${chartHeight}`);
  const bottom = chartHeight - 10;
  const middle = (bottom + 19) / 2;
  const right = width - 9;
  const height = bottom - 19;
  const x = (i) => 9 + i / Math.max(1, values.length - 1) * (width - 18);
  const y = (p) => bottom - p / max * height;
  const points = values.map((p, i) => `${x(i)},${y(p)}`).join(' ');
  const currentX = x(state.frame);
  const currentY = y(values[state.frame]);
  reconcileSvg($('probabilityHistory'), `<defs><linearGradient id="historyFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="${palette.history}" stop-opacity=".16"/><stop offset="1" stop-color="${palette.history}" stop-opacity="0"/></linearGradient></defs><line x1="9" y1="${bottom}" x2="${right}" y2="${bottom}" stroke="${palette.historyGrid}"/><line x1="9" y1="${middle}" x2="${right}" y2="${middle}" stroke="${palette.historyGrid}" stroke-dasharray="2 5"/><text x="9" y="10" fill="${palette.historyAxis}" font-family="ui-monospace,monospace" font-size="8">${percent(max)}</text><polygon points="9,${bottom} ${points} ${right},${bottom}" fill="url(#historyFill)"/><polyline points="${points}" fill="none" stroke="${palette.history}" stroke-width="1.1" vector-effect="non-scaling-stroke"/>${values.map((p, i) => `<circle cx="${x(i)}" cy="${y(p)}" r="${i === state.frame ? 0 : 1.35}" fill="${palette.history}" opacity=".55"><title>Step ${number.format(state.data.checkpoints[i].step)}: ${percent(p)}</title></circle>`).join('')}<line id="historyPlayhead" x1="${currentX}" y1="18" x2="${currentX}" y2="${bottom}" stroke="${palette.history}" stroke-opacity=".4" stroke-dasharray="2 3"/><circle id="historyDot" class="history-current" data-motion-key="history-current" cx="${currentX}" cy="${currentY}" r="3" fill="${palette.history}" stroke="${palette.background}" stroke-width="1.3"/>`);
  $('probabilityHistory').setAttribute('aria-label', `Probability of ${scene().target_label} across ${values.length} checkpoints. Current step ${frame().step}: ${percent(frame().target_probability)}. Chart maximum ${percent(max)}.`);
  paintPlayhead();
  $('historyStart').textContent = `${compactStep(state.data.checkpoints[0].step)}`;
  $('historyEnd').textContent = `${compactStep(state.data.checkpoints.at(-1).step)}`;
  const peakIndex = values.indexOf(Math.max(...values));
  const peak = scene().frames[peakIndex];
  $('peakProbability').dataset.frame = peakIndex;
  $('peakProbability').innerHTML = `<span>Peak</span><strong>${percent(peak.target_probability)} · ${compactStep(peak.step)} <span aria-hidden="true">↗</span></strong>`;
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
  const historyPlayhead = $('historyPlayhead');
  if (historyPlayhead) {
    const chart = $('probabilityHistory');
    const bounds = chart.getBoundingClientRect();
    const x = 9 + progress * (Math.max(100, bounds.width || 320) - 18);
    historyPlayhead.setAttribute('x1', x);
    historyPlayhead.setAttribute('x2', x);
    const dot = $('historyDot');
    if (dot && state.data) {
      const values = scene().frames.map(f => f.target_probability);
      const lower = Math.min(Math.floor(position), values.length - 1);
      const upper = Math.min(lower + 1, values.length - 1);
      const probability = values[lower] + (values[upper] - values[lower]) * (position - lower);
      const bottom = Math.max(50, bounds.height || 120) - 10;
      dot.setAttribute('cx', x);
      dot.setAttribute('cy', bottom - probability / Math.max(...values, .000001) * (bottom - 19));
    }
  }
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

let observationDownload = null;
function buildProvenance() {
  const { model: m, checkpoints } = state.data;
  const p = scene().custom ? scene().provenance : state.data.provenance;
  const notes = (p.notes || []).map((note) => `<li>${esc(note)}</li>`).join('');
  const promptNote = exampleCopy[scene().id]?.explanation || 'The tracked token is the final checkpoint’s top guess, not a correct-answer label. Your exact prompt, including spaces, is used at every checkpoint.';
  $('provenanceContent').innerHTML = `<section class="provenance-section"><h3>PROMPT</h3><p>${esc(promptNote)}</p></section><section class="provenance-section"><h3>MODEL</h3><dl><dt>Repository</dt><dd><a href="${esc(safeUrl(m.source_url))}" target="_blank" rel="noopener noreferrer">${esc(m.id)} ↗</a></dd><dt>Architecture</dt><dd>${m.layers} layers · ${m.heads} heads · ${esc(m.hidden_size)} hidden dimensions</dd><dt>Parameters</dt><dd>${esc(typeof m.parameters === 'number' ? number.format(m.parameters) : m.parameters)}</dd></dl></section><section class="provenance-section"><h3>HOW TO READ THIS</h3><p>Flow curves and matrix cells display attention weights from a recorded forward pass. The head grid summarizes those weights as concentration. A query can attend to itself and earlier tokens; future tokens are masked. Layer, head, and token labels start at 1.</p><p>The head grid always measures focus for the final prompt token, even when another query is selected. Focus is one minus normalized attention entropy—not a measure of head importance. Probability bars are scaled relative to the top token; the printed percentages are absolute probabilities. History and timeline positions are evenly spaced observations, not equal amounts of training. The playhead and chart marker glide between checkpoints along the plotted line; intermediate marker positions are visual interpolation, not additional measurements. Printed values update only when a recorded checkpoint is reached. Bar lengths and attention colors briefly transition for readability.</p><p>The bundled 70M factual sentence was selected from a <a href="https://github.com/tonywangs/transformer-time-machine/blob/main/docs/fact-candidate-screen.json" target="_blank" rel="noopener noreferrer">documented prompt screen ↗</a>. The original 70M prompt “The capital of France is” never put Paris first in those checkpoints; the city-name wording gives different predictions. This curated example is not a general knowledge benchmark.</p><p>“·” marks a space, “↵” a newline, and “⇥” a tab. Tokens can be word fragments. Attention alone does not establish a causal explanation of an output.</p></section><section class="provenance-section"><h3>EXTRACTION RECORD</h3><dl><dt>Extracted</dt><dd>${esc(p.extracted_at)}</dd><dt>Device / precision</dt><dd>${esc(p.device)} / ${esc(p.dtype)}</dd><dt>PyTorch</dt><dd>${esc(p.torch_version)}</dd><dt>Transformers</dt><dd>${esc(p.transformers_version)}</dd><dt>Extractor SHA-256</dt><dd>${esc(p.extractor_sha256)}</dd></dl>${notes ? `<ul>${notes}</ul>` : ''}</section><section class="provenance-section"><h3>IMMUTABLE CHECKPOINT SOURCES</h3><div class="checkpoint-list">${checkpoints.map((c) => `<div class="checkpoint-source"><span>${number.format(c.step)}</span><a href="${esc(safeUrl(c.source_url))}" target="_blank" rel="noopener noreferrer" title="${esc(c.commit_sha)}">${esc(c.revision)} · ${esc(c.commit_sha.slice(0, 12))} ↗</a></div>`).join('')}</div></section>`;
}

function openProvenance() {
  if (!state.data) {
    $('provenanceContent').innerHTML = '<p class="quiet">The model’s provenance will appear here once its recorded observations have loaded.</p>';
  }
  if (state.data) {
    buildProvenance();
    const link = document.querySelector('.dialog-actions a');
    if (observationDownload) URL.revokeObjectURL(observationDownload);
    observationDownload = null;
    if (scene().custom) {
      const record = { schema_version: 1, model: state.data.model, checkpoints: state.data.checkpoints, provenance: scene().provenance, scenes: [scene()] };
      observationDownload = URL.createObjectURL(new Blob([JSON.stringify(record)], { type: 'application/json' }));
    }
    const observedSize = Object.keys(modelCatalog).find((size) => modelCatalog[size].model.id === state.data.model.id);
    link.href = observationDownload || (observedSize === '70m' ? 'data/traces.json' : `data/examples-${observedSize}.json`);
    link.download = scene().custom ? 'my-prompt-observations.json' : 'transformer-time-machine-traces.json';
  }
  pause();
  $('provenanceDialog').showModal();
}


function cancelAnalysis() {
  requestIntent += 1;
  pendingCancellation = Promise.allSettled([pendingCancellation, experiment.cancel()]);
  custom.busy = false;
  custom.resumeFrame = null;
}

function installPreset(id = activeTab) {
  const index = bundledData.scenes.findIndex((s) => s.id === id);
  pause();
  state.data = bundledData;
  state.scene = Math.max(0, index);
  state.layer = Math.min(state.layer, state.data.model.layers - 1);
  state.head = Math.min(state.head, state.data.model.heads - 1);
  state.query = scene().tokens.length - 1;
  state.frame = Math.min(state.frame, state.data.checkpoints.length - 1);
  transport.position = state.frame;
  $('headGrid').replaceChildren();
  initDataUI();
  render();
}

function selectTab(tab) {
  cancelAnalysis();
  activeTab = tab;
  custom.selected = tab === 'custom';
  custom.status = modelLoading ? 'Loading recorded examples…' : '';
  custom.statusKind = '';
  if (custom.selected) {
    const previous = customResults.get(selectedModel);
    if (previous && !modelLoading) { custom.resumeFrame = state.frame; experiment.onComplete(previous); }
    else { installPreset('story'); renderCustomUI(); }
  } else installPreset(tab);
}

async function runSelectedModel(prompt, { preserveFrame = false } = {}) {
  const intent = ++requestIntent;
  const size = selectedModel;
  custom.resumeFrame = preserveFrame ? state.frame : null;
  pause();
  activeTab = 'custom';
  custom.selected = true;
  const cached = completedModels.get(`${size}:${prompt}`);
  render();
  await pendingCancellation;
  if (intent !== requestIntent || size !== selectedModel || modelLoading) return;
  if (cached) { await experiment.cancel(); if (intent === requestIntent) experiment.onComplete(cached); }
  else experiment.start(prompt, modelCatalog[size].checkpoints, size);
}

async function changeModel(size) {
  const generation = ++modelSwitchGeneration;
  cancelAnalysis();
  selectedModel = size;
  modelLoading = true;
  custom.status = 'Loading recorded examples…';
  custom.statusKind = '';
  render();
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 60000);
  try {
    let data = presetModels.get(size);
    if (!data) {
      const response = await fetch(size === '70m' ? 'data/traces.json' : `data/examples-${size}.json`, { signal: controller.signal });
      if (!response.ok) throw new Error('The recorded examples could not load. Choose the model again to retry.');
      data = await response.json();
      validateData(data);
      const manifest = modelCatalog[size];
      if (data.model.id !== manifest.model.id || data.checkpoints.length !== manifest.checkpoints.length || data.checkpoints.some((c, i) => c.commit_sha !== manifest.checkpoints[i]?.commit_sha)) throw new Error('These examples do not match the selected model. Please refresh.');
      data.scenes.sort((a, b) => exampleOrder.indexOf(a.id) - exampleOrder.indexOf(b.id));
      presetModels.set(size, data);
    }
    await pendingCancellation;
    if (generation !== modelSwitchGeneration) return;
    bundledData = data;
    modelLoading = false;
    custom.status = '';
    if (activeTab === 'custom') {
      const previous = customResults.get(size);
      if (previous && previous.prompt === $('customPromptInput').value) { custom.resumeFrame = state.frame; experiment.onComplete(previous); }
      else if ($('customPromptInput').value.trim()) runSelectedModel($('customPromptInput').value, { preserveFrame: true });
      else selectTab('custom');
    } else installPreset(activeTab);
  } catch (error) {
    if (generation !== modelSwitchGeneration) return;
    modelLoading = false;
    custom.statusKind = 'error';
    custom.status = error.name === 'AbortError' ? 'The examples took too long to load. Choose another model and retry.' : error.message;
    renderCustomUI();
  } finally { clearTimeout(timeout); }
}
$('modelSpec').addEventListener('change', (event) => {
  if (event.target.id === 'modelSelect' && modelCatalog[event.target.value]) changeModel(event.target.value);
});
$('layerPage').addEventListener('change', () => {
  state.layer = Number($('layerPage').value) * 6;
  render();
});

$('sceneTabs').addEventListener('click', (event) => {
  const button = event.target.closest('[data-scene],[data-custom]');
  if (!button) return;
  const tab = button.dataset.custom ? 'custom' : exampleOrder[Number(button.dataset.scene)];
  if (tab === activeTab) return;
  selectTab(tab);
});
$('sceneTabs').addEventListener('keydown', (event) => {
  if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key) || !state.data) return;
  event.preventDefault(); event.stopPropagation();
  const tabs = [...$('sceneTabs').querySelectorAll('button')];
  const current = tabs.indexOf(event.target.closest('button'));
  const index = event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 : (current + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length;
  tabs[index].click(); tabs[index].focus();
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
  const fraction = (point.matrixTransform(transform.inverse()).x - 9) / (svg.viewBox.baseVal.width - 18);
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
$('themeToggle').addEventListener('click', () => applyTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark', true));
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
    if (state.data && !$('instrument').hidden) { renderVisualization(); renderHistory(); }
  });
}
// Layout changes from scene text and viewport height both resize the drawing area.
if (typeof ResizeObserver !== 'undefined') {
  const observer = new ResizeObserver(resizeVisualization);
  observer.observe($('visualization'));
  observer.observe($('probabilityHistory'));
} else {
  window.addEventListener('resize', resizeVisualization);
}
applyTheme(document.documentElement.dataset.theme);
loadData();
