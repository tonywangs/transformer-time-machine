'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { PromptExperiment } = require('../dist/custom-runner.js');

const manifest = require('./model-manifest.json');
const checkpoints = manifest.checkpoints;
const steps = checkpoints.map((checkpoint) => checkpoint.step);
const prompt = '  The moon\n';
const JOB = '12345678-1234-4123-8123-123456789abc';
const OTHER = '12345678-1234-4123-8123-123456789def';
const flush = () => new Promise(setImmediate);
const deferred = () => {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
};
const admitted = (job = JOB) => ({ job_id: job, status: 'queued', completed: 0, total: 38 });
const observation = (cursor, job = JOB, text = prompt) => ({
  job_id: job, status: cursor === 37 ? 'complete' : 'running', completed: cursor + 1, total: 38,
  index: cursor, frame: { step: steps[cursor], observed: `${job}:${cursor}` },
  scene: { id: `custom-${job}`, prompt: text, tokens: ['example'] },
  provenance: { tokenizer_commit_sha: manifest.tokenizer_commit_sha },
  model: manifest.model, checkpoints: structuredClone(checkpoints)
});

function harness(implementation) {
  const calls = [], progress = [], completed = [], errors = [], waits = [];
  const runner = new PromptExperiment({
    request: async (payload) => {
      calls.push(payload);
      if (implementation) return implementation(payload);
      if (payload.action === 'start') return admitted();
      if (payload.action === 'poll') return observation(payload.cursor);
      return { status: 'cancelled' };
    },
    onProgress: (result) => progress.push(result),
    onComplete: (result) => completed.push(result),
    onError: (message) => errors.push(message),
    wait: async (ms) => { waits.push(ms); }
  });
  return { runner, calls, progress, completed, errors, waits };
}

test('assembles exactly the 38 observed chronological frames and preserves the exact prompt', async () => {
  const h = harness();
  await h.runner.start(prompt, checkpoints);
  assert.deepEqual(h.calls[0], { action: 'start', prompt, model: '70m' });
  assert.deepEqual(h.calls.filter((call) => call.action === 'poll').map((call) => call.cursor), steps.map((_, i) => i));
  assert.equal(h.completed.length, 1);
  assert.equal(h.completed[0].prompt, prompt);
  assert.equal(h.completed[0].custom, true);
  assert.deepEqual(h.completed[0].frames, steps.map((_, i) => observation(i).frame));
  assert.deepEqual(h.errors, []);
  assert.equal(h.runner.busy, false);
  assert.equal(h.runner.job, null);
});

test('37 observed frames never install a partial scene', async () => {
  const final = deferred();
  const h = harness((call) => call.action === 'start' ? admitted() : call.cursor === 37 ? final.promise : observation(call.cursor));
  const run = h.runner.start(prompt, checkpoints);
  await flush();
  assert.equal(h.completed.length, 0);
  assert.equal(h.runner.busy, true);
  final.resolve(observation(37));
  await run;
  assert.equal(h.completed.length, 1);
  assert.equal(h.completed[0].frames.length, 38);
});

test('cancelling immediately after start cannot resurrect admission', async () => {
  const h = harness();
  const run = h.runner.start(prompt, checkpoints);
  await h.runner.cancel();
  await run;
  assert.deepEqual(h.calls, []);
  assert.deepEqual(h.completed, []);
  assert.deepEqual(h.errors, []);
  assert.equal(h.runner.busy, false);
});

test('cancel before the start response cancels the admitted job as soon as its ID arrives', async () => {
  const start = deferred();
  const h = harness((call) => call.action === 'start' ? start.promise : { status: 'cancelled' });
  const run = h.runner.start(prompt, checkpoints);
  await flush();
  await h.runner.cancel();
  start.resolve(admitted());
  await run;
  assert.deepEqual(h.calls.map((call) => call.action), ['start', 'cancel']);
  assert.equal(h.calls[1].job_id, JOB);
  assert.deepEqual(h.completed, []);
  assert.deepEqual(h.errors, []);
});

test('late start response from a superseded run never installs or clears the newer run', async () => {
  const oldStart = deferred();
  const newPoll = deferred();
  let starts = 0;
  const newPrompt = 'A different prompt';
  const h = harness((call) => {
    if (call.action === 'start') return ++starts === 1 ? oldStart.promise : admitted(OTHER);
    if (call.action === 'cancel') return { status: 'cancelled' };
    if (call.cursor === 0) return newPoll.promise;
    return observation(call.cursor, OTHER, newPrompt);
  });
  const oldRun = h.runner.start(prompt, checkpoints);
  await flush();
  const newRun = h.runner.start(newPrompt, checkpoints);
  await flush();
  oldStart.resolve(admitted());
  await oldRun;
  assert.equal(h.runner.busy, true);
  assert.equal(h.runner.job, OTHER);
  assert.deepEqual(h.completed, []);
  assert.ok(h.calls.some((call) => call.action === 'cancel' && call.job_id === JOB));
  newPoll.resolve(observation(0, OTHER, newPrompt));
  await newRun;
  assert.equal(h.completed.length, 1);
  assert.equal(h.completed[0].prompt, newPrompt);
  assert.equal(h.runner.busy, false);
});

test('a late poll after cancellation is ignored and cannot render progress or completion', async () => {
  const poll = deferred();
  const h = harness((call) => call.action === 'start' ? admitted() : call.action === 'poll' ? poll.promise : { status: 'cancelled' });
  const run = h.runner.start(prompt, checkpoints);
  await flush();
  const progressCount = h.progress.length;
  await h.runner.cancel();
  poll.resolve(observation(0));
  await run;
  assert.equal(h.progress.length, progressCount);
  assert.deepEqual(h.completed, []);
  assert.deepEqual(h.errors, []);
});

test('transient polling errors retry the same cursor with bounded backoff', async () => {
  let failures = 0;
  const h = harness((call) => {
    if (call.action === 'start') return admitted();
    if (call.action === 'poll' && call.cursor === 0 && failures++ < 3) throw new Error('Network interrupted');
    return observation(call.cursor);
  });
  await h.runner.start(prompt, checkpoints);
  assert.deepEqual(h.waits, [1000, 2000, 3000]);
  assert.deepEqual(h.calls.slice(1, 5).map((call) => call.cursor), [0, 0, 0, 0]);
  assert.equal(h.completed.length, 1);
  assert.deepEqual(h.errors, []);
});

test('persistent errors stop after three retries, report the error, and cancel the job', async () => {
  const h = harness((call) => {
    if (call.action === 'start') return admitted();
    if (call.action === 'poll') throw new Error('Connection unavailable');
    return { status: 'cancelled' };
  });
  await h.runner.start(prompt, checkpoints);
  assert.equal(h.calls.filter((call) => call.action === 'poll').length, 4);
  assert.deepEqual(h.errors, ['Connection unavailable']);
  assert.equal(h.calls.at(-1).action, 'cancel');
  assert.deepEqual(h.completed, []);
  assert.equal(h.runner.busy, false);
});

test('permanent errors do not retry', async () => {
  const h = harness((call) => {
    if (call.action === 'start') return admitted();
    if (call.action === 'poll') throw Object.assign(new Error('Expired analysis'), { permanent: true });
    return { status: 'cancelled' };
  });
  await h.runner.start(prompt, checkpoints);
  assert.deepEqual(h.waits, []);
  assert.deepEqual(h.errors, ['Expired analysis']);
  assert.equal(h.calls.filter((call) => call.action === 'poll').length, 1);
});

test('queued responses wait without manufacturing frames', async () => {
  let queued = true;
  const h = harness((call) => {
    if (call.action === 'start') return admitted();
    if (queued) { queued = false; return admitted(); }
    return observation(call.cursor);
  });
  await h.runner.start(prompt, checkpoints);
  assert.deepEqual(h.waits, [750]);
  assert.equal(h.completed[0].frames.length, 38);
  assert.equal(h.completed[0].frames[0].observed, `${JOB}:0`);
});

test('out-of-order, mismatched, or incomplete responses never install', async () => {
  const invalid = [
    { ...observation(0), index: 1 },
    { ...observation(0), frame: { step: steps[1] } },
    { ...observation(0), job_id: OTHER },
    { ...observation(0), total: 37 },
    { ...admitted(), status: 'complete' },
    { ...admitted(), status: 'invented' },
    { ...admitted(), status: 'error', error: 'Prompt exceeds 32 tokens.' }
  ];
  for (const response of invalid) {
    const h = harness((call) => call.action === 'start' ? admitted() : call.action === 'poll' ? response : { status: 'cancelled' });
    await h.runner.start(prompt, checkpoints);
    assert.equal(h.completed.length, 0);
    assert.equal(h.errors.length, 1);
    assert.equal(h.calls.at(-1).action, 'cancel');
  }
});

test('missing provenance or a changed prompt cannot install observations', async () => {
  for (const kind of ['provenance', 'prompt']) {
    const h = harness((call) => {
      if (call.action === 'start') return admitted();
      if (call.action === 'cancel') return { status: 'cancelled' };
      const result = observation(call.cursor, JOB, kind === 'prompt' ? prompt.trim() : prompt);
      if (kind === 'provenance') delete result.provenance;
      return result;
    });
    await h.runner.start(prompt, checkpoints);
    assert.equal(h.completed.length, 0);
    assert.equal(h.errors.length, 1);
  }
});

test('every frame must carry the exact immutable checkpoint and tokenizer manifest', async () => {
  const mutations = [
    (result) => { delete result.checkpoints; },
    (result) => { result.checkpoints.pop(); },
    (result) => { result.checkpoints[5].commit_sha = '0'.repeat(40); },
    (result) => { result.checkpoints[5].step += 1; },
    (result) => { delete result.provenance.tokenizer_commit_sha; },
    (result) => { result.provenance.tokenizer_commit_sha = '0'.repeat(40); }
  ];
  for (const mutate of mutations) {
    const h = harness((call) => {
      if (call.action === 'start') return admitted();
      if (call.action === 'cancel') return { status: 'cancelled' };
      const result = observation(call.cursor);
      // Earlier metadata cannot license a different/missing final manifest.
      if (call.cursor === 37) mutate(result);
      return result;
    });
    await h.runner.start(prompt, checkpoints);
    assert.deepEqual(h.completed, []);
    assert.equal(h.errors.length, 1);
    assert.match(h.errors[0], /different checkpoint or tokenizer/);
    assert.equal(h.calls.at(-1).action, 'cancel');
  }
});

test('invalid prompt input reports an error without touching the current run or server', async () => {
  for (const text of ['', '  \n ', 'x'.repeat(401), null]) {
    const h = harness();
    await h.runner.start(text, checkpoints);
    assert.deepEqual(h.calls, []);
    assert.equal(h.errors.length, 1);
    assert.deepEqual(h.completed, []);
  }
});


test('rejects a different model even when checkpoint metadata matches', async () => {
  const h = harness((call) => {
    if (call.action === 'start') return admitted();
    if (call.action === 'cancel') return {};
    return { ...observation(call.cursor), model: { ...manifest.model, id: 'EleutherAI/pythia-1b-deduped' } };
  });
  await h.runner.start(prompt, checkpoints, '70m');
  assert.equal(h.completed.length, 0);
  assert.equal(h.errors.length, 1);
});

test('each larger model carries its own immutable manifest to completion', async () => {
  const catalog = require('../dist/data/models.json');
  for (const size of ['160m', '410m', '1b']) {
    const m = catalog[size];
    const h = harness((call) => call.action === 'start' ? admitted() : {
      ...observation(call.cursor), model: m.model, checkpoints: m.checkpoints,
      provenance: { tokenizer_commit_sha: m.tokenizer_commit_sha }
    });
    await h.runner.start(prompt, m.checkpoints, size);
    assert.equal(h.calls[0].model, size);
    assert.deepEqual(h.errors, []);
    assert.equal(h.completed[0].model.id, m.model.id);
  }
});
