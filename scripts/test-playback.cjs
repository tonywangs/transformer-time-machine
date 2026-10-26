#!/usr/bin/env node
'use strict';

// Run from the repository root: node scripts/test-playback.cjs
// Execute the production transport in a VM with a controlled clock and minimal
// DOM boundary. No browser, dependencies, model loading, or real timers needed.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '..', 'dist', 'app.js'), 'utf8');
function section(start, end) {
  const from = source.indexOf(start);
  const to = source.indexOf(end, from);
  assert(from >= 0 && to > from, `Production section not found: ${start}`);
  return source.slice(from, to);
}
const production = [
  section('const $ =', 'const motionAnimations ='),
  section('const number =', 'const safeUrl ='),
  section('function paintPlayhead()', 'function buildProvenance()'),
  section('function renderFlow()', 'let matrixLayout =')
].join('\n');

function harness() {
  const callbacks = new Map();
  const elements = new Map();
  const renderedFrames = [];
  let requestId = 0;
  let now = 0;
  let api;
  let cancelledAnimations = 0;
  const preference = {
    matches: false,
    addEventListener(type, callback) {
      assert.equal(type, 'change');
      this.change = callback;
    }
  };
  function element(id) {
    if (!elements.has(id)) elements.set(id, {
      textContent: '',
      innerHTML: '',
      clientWidth: 640,
      clientHeight: 240,
      attributes: {},
      setAttribute(name, value) { this.attributes[name] = value; },
      classList: { add() {}, remove() {} },
      style: { properties: {}, setProperty(name, value) { this.properties[name] = value; } }
    });
    return elements.get(id);
  }
  const context = vm.createContext({
    window: { matchMedia: () => preference },
    document: {
      documentElement: { dataset: { theme: 'dark' } },
      getElementById: element,
      querySelector: element,
      getAnimations: () => [{ cancel: () => { cancelledAnimations += 1; } }]
    },
    requestAnimationFrame(callback) { callbacks.set(++requestId, callback); return requestId; },
    cancelAnimationFrame(id) { callbacks.delete(id); },
    render() {
      assert(Number.isInteger(api.state.frame), 'The model frame must remain discrete');
      assert(api.state.frame >= 0 && api.state.frame < 38, 'Frame outside the recorded set');
      renderedFrames.push(api.state.frame);
    },
    reconcileSvg(parent, markup) { parent.innerHTML = markup; }
  });
  vm.runInContext(production, context, { filename: 'app.js (production transport and flow)' });
  api = vm.runInContext('({state, transport, play, pause, setFrame, renderFlow})', context);
  api.state.data = {
    checkpoints: Array.from({ length: 38 }, (_, i) => ({ step: i * 1000 })),
    scenes: [{
      tokens: [' a', ' b', ' c'],
      frames: Array.from({ length: 38 }, (_, i) => ({
        step: i * 1000,
        top_tokens: [{ token: ' example', probability: .1 }],
        attention: [[[[1, 0, 0], [0, 1, 0], [0, .25, .75]]]]
      }))
    }]
  };
  function tick(milliseconds = 0) {
    now += milliseconds;
    const pending = [...callbacks.values()];
    callbacks.clear();
    pending.forEach(callback => callback(now));
    assert(callbacks.size <= 1, 'More than one transport animation loop is pending');
  }
  function reduced(value) {
    preference.matches = value;
    preference.change();
  }
  return { ...api, tick, reduced, element, renderedFrames,
    pending: () => callbacks.size, cancelledAnimations: () => cancelledAnimations };
}

function near(actual, expected) {
  assert(Math.abs(actual - expected) < 1e-8, `Expected ${expected}, received ${actual}`);
}
function test(name, run) {
  run(harness());
  console.log(`PASS ${name}`);
}

test('playhead moves continuously while model observations remain discrete', h => {
  h.play(); h.tick();
  for (let i = 1; i <= 5; i += 1) {
    h.tick(200);
    near(h.transport.position, i / 6);
    const chartX = 9 + (i / 6) / 37 * (640 - 18);
    near(h.element('historyPlayhead').attributes.x1, chartX);
    near(h.element('historyPlayhead').attributes.x2, chartX);
    assert.equal(h.state.frame, 0);
    assert.equal(h.renderedFrames.length, 0);
  }
  h.tick(200);
  near(h.transport.position, 1);
  assert.equal(h.state.frame, 1);
  assert.deepEqual(h.renderedFrames, [1]);
});

test('pause freezes position and resume preserves fractional progress', h => {
  h.play(); h.tick(); h.tick(600); h.pause();
  assert.equal(h.pending(), 0);
  h.tick(5000);
  near(h.transport.position, .5);
  h.play(); h.tick();
  near(h.transport.position, .5);
  h.tick(600);
  assert.equal(h.state.frame, 1);
});

test('speed changes preserve the current position', h => {
  h.state.playbackSpeed = .5;
  h.play(); h.tick(); h.tick(1200);
  near(h.transport.position, .5);
  h.state.playbackSpeed = 4;
  h.tick(75);
  near(h.transport.position, .75);
  h.tick(75);
  assert.equal(h.state.frame, 1);
});

test('smooth seek stops playback and settles on the exact requested checkpoint', h => {
  h.play(); h.tick(); h.tick(600);
  h.setFrame(10);
  assert.equal(h.state.playing, false);
  assert.equal(h.state.frame, 10);
  assert.equal(h.element('screenReaderStatus').textContent.includes('10,000'), true);
  h.tick(); h.tick(180);
  assert(h.transport.position > .5 && h.transport.position < 10);
  h.tick(180);
  near(h.transport.position, 10);
  assert.equal(h.pending(), 0);
});

test('direct scrubbing cancels an in-flight seek without leftover callbacks', h => {
  h.setFrame(25); h.tick(); h.tick(100);
  h.setFrame(4, true);
  near(h.transport.position, 4);
  assert.equal(h.state.frame, 4);
  assert.equal(h.pending(), 0);
  h.tick(10000);
  near(h.transport.position, 4);
});

test('play resumes an interrupted seek before advancing checkpoints', h => {
  h.setFrame(10); h.tick(); h.tick(180); h.pause();
  const paused = h.transport.position;
  h.play(); h.tick();
  near(h.transport.position, paused);
  h.tick(360);
  near(h.transport.position, 10);
  h.tick(1200);
  assert.equal(h.state.frame, 11);
});

test('the final checkpoint stops playback exactly once', h => {
  h.setFrame(36, true);
  h.state.playbackSpeed = 4;
  h.play(); h.tick(); h.tick(300);
  assert.equal(h.state.frame, 37);
  assert.equal(h.state.playing, false);
  assert.equal(h.pending(), 0);
  near(h.transport.position, 37);
  const count = h.renderedFrames.length;
  h.tick(10000);
  assert.equal(h.renderedFrames.length, count);
});

test('play from the final checkpoint rewinds smoothly and restarts', h => {
  h.setFrame(37, true); h.play(); h.tick(); h.tick(180);
  assert.equal(h.state.frame, 0);
  assert(h.transport.position > 0 && h.transport.position < 37);
  h.tick(180);
  near(h.transport.position, 0);
  h.tick(1200);
  assert.equal(h.state.frame, 1);
});

test('reduced motion paints discrete positions while playback continues', h => {
  h.reduced(true);
  h.play(); h.tick(); h.tick(600);
  near(h.transport.position, .5);
  assert.equal(h.element('timelineRail').style.properties['--playhead-scale'], 0);
  h.tick(600);
  assert.equal(h.state.frame, 1);
  assert.equal(h.element('timelineRail').style.properties['--playhead-scale'], 1 / 37);
  h.setFrame(20);
  near(h.transport.position, 20);
  assert.equal(h.pending(), 0);
});

test('enabling reduced motion during a running seek preserves a valid playback state', h => {
  h.setFrame(10); h.play(); h.tick(); h.tick(180);
  h.reduced(true);
  near(h.transport.position, 10);
  assert.equal(h.transport.seek, null);
  assert.equal(h.cancelledAnimations(), 1);
  h.tick(); h.tick(1200);
  assert.equal(h.state.frame, 11);
});

test('rapid play/pause cycles never accumulate animation loops', h => {
  for (let i = 0; i < 20; i += 1) {
    h.play(); h.tick(); h.tick(25); h.pause();
    assert.equal(h.pending(), 0);
  }
  near(h.transport.position, 500 / 1200);
  assert.equal(h.state.frame, 0);
});

test('exact-zero attention paths are retained but invisible', h => {
  h.state.query = 2;
  h.renderFlow();
  function opacity(key) {
    const markup = h.element('flowGraph').innerHTML;
    const tag = markup.match(new RegExp(`<path\\b[^>]*data-motion-key="${key}"[^>]*>`));
    assert(tag, `Missing retained path: ${key}`);
    return Number(tag[0].match(/stroke-opacity="([^"]+)"/)[1]);
  }
  assert.equal(opacity('edge-0'), 0);
  assert(opacity('edge-1') > 0);
  assert(opacity('self') > 0);
  h.state.data.scenes[0].frames[0].attention[0][0][2] = [0, 1, 0];
  h.renderFlow();
  assert.equal(opacity('self'), 0);
});
