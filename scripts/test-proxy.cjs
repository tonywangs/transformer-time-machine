'use strict';

const { test, beforeEach, afterEach } = require('node:test');
const assert = require('node:assert/strict');
const { EventEmitter } = require('node:events');
const { Readable } = require('node:stream');
const { createHmac } = require('node:crypto');
const analyze = require('../api/analyze.js');
const originalFetch = global.fetch;
const originalEnvironment = { ...process.env };
const KEY = 'a'.repeat(64);
const JOB = '12345678-1234-4123-8123-123456789abc';
let calls;

beforeEach(() => {
  process.env.VERCEL = '1';
  process.env.NODE_ENV = 'production';
  process.env.TTM_API_KEY = KEY;
  process.env.TTM_MODAL_URL = 'https://time-machine--gateway.modal.run/';
  delete process.env.VERCEL_URL;
  delete process.env.VERCEL_PROJECT_PRODUCTION_URL;
  calls = [];
  global.fetch = async (...args) => {
    calls.push(args);
    return Response.json({ job_id: JOB, status: 'queued', completed: 0, total: 38 }, { status: 202 });
  };
});
afterEach(() => {
  global.fetch = originalFetch;
  for (const name of ['VERCEL', 'NODE_ENV', 'TTM_API_KEY', 'TTM_MODAL_URL', 'VERCEL_URL', 'VERCEL_PROJECT_PRODUCTION_URL']) {
    if (originalEnvironment[name] === undefined) delete process.env[name];
    else process.env[name] = originalEnvironment[name];
  }
});

function request(body = { action: 'start', prompt: 'Once upon a' }, overrides = {}) {
  const req = Readable.from([typeof body === 'string' ? body : JSON.stringify(body)]);
  req.method = overrides.method || 'POST';
  req.headers = {
    host: 'transformer-time-machine.vercel.app',
    origin: 'https://transformer-time-machine.vercel.app',
    'content-type': 'application/json',
    'x-vercel-forwarded-for': '203.0.113.5',
    ...overrides.headers
  };
  req.socket = { remoteAddress: '127.0.0.1' };
  if (Object.hasOwn(overrides, 'parsedBody')) req.body = overrides.parsedBody;
  const res = new EventEmitter();
  res.headers = {};
  res.setHeader = (name, value) => { res.headers[name.toLowerCase()] = value; };
  res.end = (text) => { res.writableEnded = true; res.data = JSON.parse(text); };
  return { req, res };
}
async function run(body, overrides) {
  const pair = request(body, overrides);
  await analyze(pair.req, pair.res);
  return pair.res;
}

test('forwards only the validated prompt, keeps exact whitespace, and binds the trusted IP', async () => {
  const prompt = '  Once upon a\n';
  const result = await run({ action: 'start', prompt, client_id: 'attacker', url: 'https://evil.example' }, {
    headers: { 'x-forwarded-for': '1.2.3.4', 'x-ttm-client': 'attacker' }
  });
  assert.equal(result.statusCode, 202);
  assert.equal(calls.length, 1);
  assert.equal(calls[0][0], process.env.TTM_MODAL_URL);
  const options = calls[0][1];
  assert.deepEqual(JSON.parse(options.body), { action: 'start', prompt });
  assert.equal(options.headers.Authorization, `Bearer ${KEY}`);
  assert.equal(options.headers['X-TTM-Client'], createHmac('sha256', KEY).update('ttm-client-v1\0' + '203.0.113.5').digest('hex'));
  assert.equal(options.redirect, 'error');
  assert.equal(result.headers['cache-control'], 'private, no-store');
  assert.ok(!JSON.stringify(result.data).includes(KEY));
});

test('supports Vercel parsed JSON bodies and poll/cancel contract', async () => {
  const poll = { action: 'poll', job_id: JOB, cursor: 37, client_id: 'ignored' };
  assert.equal((await run({}, { parsedBody: poll })).statusCode, 202);
  assert.deepEqual(JSON.parse(calls[0][1].body), { action: 'poll', job_id: JOB, cursor: 37 });
  await run({ action: 'cancel', job_id: JOB, cursor: 10 });
  assert.deepEqual(JSON.parse(calls[1][1].body), { action: 'cancel', job_id: JOB });
});

test('invalid input never reaches the model service', async () => {
  const cases = [
    [{ action: 'start', prompt: '  \n ' }, 400],
    [{ action: 'start', prompt: 'a'.repeat(401) }, 400],
    [{ action: 'poll', job_id: JOB, cursor: 38 }, 400],
    [{ action: 'poll', job_id: JOB, cursor: -1 }, 400],
    [{ action: 'poll', job_id: JOB, cursor: 1.5 }, 400],
    [{ action: 'cancel', job_id: '../secret' }, 400],
    [{ action: 'deploy', prompt: 'x' }, 400],
    [[], 400],
    ['{invalid', 400],
    [{ action: 'start', prompt: 'ok', padding: 'x'.repeat(4096) }, 413]
  ];
  for (const [body, status] of cases) assert.equal((await run(body)).statusCode, status);
  assert.equal((await run({}, { headers: { 'content-type': 'text/plain' } })).statusCode, 415);
  assert.equal((await run({}, { headers: { 'content-length': '999999' } })).statusCode, 413);
  assert.equal(calls.length, 0);
});

test('Unicode character limits count code points without corrupting input', async () => {
  const prompt = '🧠'.repeat(400);
  assert.equal((await run({ action: 'start', prompt })).statusCode, 202);
  assert.equal(JSON.parse(calls[0][1].body).prompt, prompt);
});

test('rejects cross-site browser requests and unsupported methods', async () => {
  assert.equal((await run(undefined, { headers: { origin: 'https://evil.example' } })).statusCode, 403);
  assert.equal((await run(undefined, { headers: { 'sec-fetch-site': 'cross-site' } })).statusCode, 403);
  assert.equal((await run(undefined, { headers: { origin: 'https://transformer-time-machine.vercel.app/path' } })).statusCode, 403);
  const result = await run(undefined, { method: 'GET' });
  assert.equal(result.statusCode, 405);
  assert.equal(result.headers.allow, 'POST');
  assert.equal(calls.length, 0);
});

test('allows configured Vercel preview origins but not arbitrary vercel.app projects', async () => {
  process.env.VERCEL_URL = 'time-machine-preview-owner.vercel.app';
  assert.equal((await run(undefined, { headers: { origin: 'https://' + process.env.VERCEL_URL } })).statusCode, 202);
  assert.equal((await run(undefined, { headers: { origin: 'https://attacker.vercel.app' } })).statusCode, 403);
});

test('fails closed when the trusted Vercel IP is absent or ambiguous', async () => {
  for (const ip of ['', '1.2.3.4, 5.6.7.8', 'not-an-ip']) {
    assert.equal((await run(undefined, { headers: { 'x-vercel-forwarded-for': ip, 'x-forwarded-for': '1.2.3.4' } })).statusCode, 503);
  }
  assert.equal(calls.length, 0);
});

test('local development uses the connection IP and allows only its matching localhost origin', async () => {
  delete process.env.VERCEL;
  process.env.NODE_ENV = 'development';
  assert.equal((await run(undefined, { headers: { host: 'localhost:3000', origin: 'http://localhost:3000' } })).statusCode, 202);
  assert.equal(calls[0][1].headers['X-TTM-Client'], createHmac('sha256', KEY).update('ttm-client-v1\0' + '127.0.0.1').digest('hex'));
  assert.equal((await run(undefined, { headers: { host: 'localhost:3000', origin: 'http://localhost:4000' } })).statusCode, 403);
});

test('missing secrets and invalid upstream URLs disable inference', async () => {
  for (const url of ['https://evil.example/', 'http://time-machine.modal.run/', 'https://key@time-machine.modal.run/', 'https://time-machine.modal.run/?target=x']) {
    process.env.TTM_MODAL_URL = url;
    assert.equal((await run()).statusCode, 503);
  }
  process.env.TTM_MODAL_URL = 'https://time-machine.modal.run/';
  delete process.env.TTM_API_KEY;
  assert.equal((await run()).statusCode, 503);
  assert.equal(calls.length, 0);
});

test('preserves safe quota errors and retry timing without forwarding extra fields', async () => {
  global.fetch = async () => Response.json({ error: 'Daily limit reached.', code: 'daily_limit', retry_after: 120, internal: 'hidden' }, { status: 429 });
  const result = await run();
  assert.equal(result.statusCode, 429);
  assert.deepEqual(result.data, { error: 'Daily limit reached.', code: 'daily_limit', retry_after: 120 });
  assert.equal(result.headers['retry-after'], '120');
});

test('hides backend failures and rejects malformed or oversized output', async () => {
  global.fetch = async () => Response.json({ error: 'secret traceback' }, { status: 500 });
  let result = await run();
  assert.equal(result.statusCode, 503);
  assert.ok(!JSON.stringify(result.data).includes('secret'));
  global.fetch = async () => new Response('<html>oops</html>', { headers: { 'content-type': 'text/html' } });
  assert.equal((await run()).statusCode, 502);
  global.fetch = async () => Response.json({ data: 'x'.repeat(4 * 1024 * 1024) });
  assert.equal((await run()).statusCode, 502);
  global.fetch = async () => { throw new Error('private service URL and credentials'); };
  result = await run();
  assert.equal(result.statusCode, 502);
  assert.ok(!JSON.stringify(result.data).includes('credentials'));
});

test('times out the upstream request and cleans up listeners', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  global.fetch = async (_url, options) => new Promise((_resolve, reject) => {
    options.signal.addEventListener('abort', () => reject(new Error('aborted')), { once: true });
  });
  const { req, res } = request();
  const pending = analyze(req, res);
  await new Promise(setImmediate);
  t.mock.timers.tick(30000);
  await pending;
  assert.equal(res.statusCode, 504);
  assert.equal(req.listenerCount('aborted'), 0);
  assert.equal(res.listenerCount('close'), 0);
});

test('a disconnected client aborts the proxy request without writing a response', async () => {
  let signal;
  global.fetch = async (_url, options) => {
    signal = options.signal;
    return new Promise((_resolve, reject) => {
      signal.addEventListener('abort', () => reject(new Error('aborted')), { once: true });
    });
  };
  const { req, res } = request();
  const pending = analyze(req, res);
  await new Promise(setImmediate);
  res.destroyed = true;
  res.emit('close');
  await pending;
  assert.equal(signal.aborted, true);
  assert.equal(res.data, undefined);
  assert.equal(req.listenerCount('aborted'), 0);
  assert.equal(res.listenerCount('close'), 0);
});

test('forwards allowlisted model identity and rejects arbitrary model requests', async () => {
  for (const model of ['70m', '160m', '410m', '1b']) {
    assert.equal((await run({ action: 'start', prompt: 'Hello', model })).statusCode, 202);
    assert.equal(JSON.parse(calls.at(-1)[1].body).model, model);
  }
  for (const model of ['12b', '../model', [], {}, null]) {
    const count = calls.length;
    assert.equal((await run({ action: 'start', prompt: 'Hello', model })).statusCode, 400);
    assert.equal(calls.length, count);
  }
});
