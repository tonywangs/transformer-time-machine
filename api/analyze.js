'use strict';

// This function only proxies bounded requests. Inference, ownership checks, and
// admission limits live in the Modal service; the browser never receives its key.
const { createHmac } = require('node:crypto');
const { isIP } = require('node:net');

const MAX_REQUEST_BYTES = 4096;
const MAX_RESPONSE_BYTES = 4 * 1024 * 1024;
const TIMEOUT_MS = 30000;
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

function header(req, name) {
  const value = req.headers[name];
  return typeof value === 'string' ? value : '';
}

function respond(res, status, payload) {
  if (res.destroyed || res.writableEnded) return;
  res.statusCode = status;
  res.setHeader('Content-Type', 'application/json; charset=utf-8');
  res.setHeader('Cache-Control', 'private, no-store');
  res.setHeader('X-Content-Type-Options', 'nosniff');
  res.end(JSON.stringify(payload));
}

function error(status, message) {
  return Object.assign(new Error(message), { status });
}

function isAllowedOrigin(req) {
  if (header(req, 'sec-fetch-site') === 'cross-site') return false;
  const value = header(req, 'origin');
  if (!value) return true; // Non-browser clients still face backend admission limits.
  let origin;
  try { origin = new URL(value); } catch { return false; }
  if (origin.origin !== value) return false;
  const hosts = new Set([
    'transformer-time-machine.vercel.app',
    process.env.VERCEL_URL,
    process.env.VERCEL_PROJECT_PRODUCTION_URL
  ].filter(Boolean));
  if (origin.protocol === 'https:' && hosts.has(origin.host)) return true;
  return process.env.VERCEL !== '1' && process.env.NODE_ENV !== 'production'
    && origin.protocol === 'http:'
    && ['localhost', '127.0.0.1', '[::1]'].includes(origin.hostname)
    && origin.host === header(req, 'host');
}

async function readPayload(req) {
  const length = header(req, 'content-length');
  if (length && (!/^\d+$/.test(length) || Number(length) > MAX_REQUEST_BYTES)) {
    throw error(413, 'This request is too large.');
  }
  if (!/^application\/json(?:\s*;|$)/i.test(header(req, 'content-type'))) {
    throw error(415, 'Send a JSON request.');
  }
  let body = req.body;
  if (body === undefined) {
    const chunks = [];
    let bytes = 0;
    for await (const chunk of req) {
      const buffer = Buffer.from(chunk);
      bytes += buffer.length;
      if (bytes > MAX_REQUEST_BYTES) throw error(413, 'This request is too large.');
      chunks.push(buffer);
    }
    body = Buffer.concat(chunks).toString('utf8');
  }
  try {
    if (Buffer.isBuffer(body)) body = body.toString('utf8');
    if (typeof body === 'string') {
      if (Buffer.byteLength(body) > MAX_REQUEST_BYTES) throw error(413, 'This request is too large.');
      body = JSON.parse(body);
    } else if (Buffer.byteLength(JSON.stringify(body)) > MAX_REQUEST_BYTES) {
      throw error(413, 'This request is too large.');
    }
  } catch (failure) {
    if (failure.status) throw failure;
    throw error(400, 'This request is not valid JSON.');
  }
  if (!body || Array.isArray(body) || typeof body !== 'object') throw error(400, 'Send a JSON object.');
  if (body.action === 'start') {
    if (typeof body.prompt !== 'string' || !body.prompt.trim()) throw error(400, 'Enter a prompt first.');
    if ([...body.prompt].length > 400) throw error(400, 'Keep your prompt to 400 characters or fewer.');
    // Whitespace is part of the model input. Validate emptiness without trimming.
    const model = body.model === undefined ? '70m' : body.model;
    if (!['70m', '160m', '410m', '1b'].includes(model)) throw error(400, 'Choose a supported Pythia model.');
    return { action: 'start', prompt: body.prompt, ...(body.model !== undefined ? { model } : {}) };
  }
  if (body.action !== 'poll' && body.action !== 'cancel') throw error(400, 'Unknown analysis action.');
  if (typeof body.job_id !== 'string' || !UUID.test(body.job_id)) throw error(400, 'The analysis ID is invalid.');
  if (body.action === 'cancel') return { action: 'cancel', job_id: body.job_id };
  if (!Number.isInteger(body.cursor) || body.cursor < 0 || body.cursor > 37) throw error(400, 'The checkpoint cursor is invalid.');
  return { action: 'poll', job_id: body.job_id, cursor: body.cursor };
}

function clientFingerprint(req, key) {
  // Vercel replaces this header at its edge. Do not trust forwarded headers on a
  // directly exposed local Node server. Missing/ambiguous production IPs fail closed.
  const ip = process.env.VERCEL === '1'
    ? header(req, 'x-vercel-forwarded-for').trim()
    : req.socket?.remoteAddress;
  if (typeof ip !== 'string' || !isIP(ip)) throw error(503, 'Custom prompts are temporarily unavailable.');
  return createHmac('sha256', key).update(`ttm-client-v1\0${ip}`).digest('hex');
}

async function readResponse(response) {
  const length = response.headers.get('content-length');
  if (length && Number(length) > MAX_RESPONSE_BYTES) throw new Error('Oversized backend response');
  if (!/^application\/json(?:\s*;|$)/i.test(response.headers.get('content-type') || '')) throw new Error('Unexpected backend response');
  const reader = response.body?.getReader();
  if (!reader) throw new Error('Empty backend response');
  let bytes = 0;
  const chunks = [];
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      bytes += value.byteLength;
      if (bytes > MAX_RESPONSE_BYTES) throw new Error('Oversized backend response');
      chunks.push(Buffer.from(value));
    }
  } catch (failure) {
    await reader.cancel().catch(() => {});
    throw failure;
  } finally { reader.releaseLock(); }
  const body = JSON.parse(Buffer.concat(chunks).toString('utf8'));
  if (!body || Array.isArray(body) || typeof body !== 'object') throw new Error('Invalid backend response');
  return body;
}

module.exports = async function analyze(req, res) {
  if (req.method !== 'POST') {
    res.setHeader('Allow', 'POST');
    return respond(res, 405, { error: 'Use POST for custom prompts.' });
  }
  if (!isAllowedOrigin(req)) return respond(res, 403, { error: 'Open the time machine to run a prompt.' });
  let payload;
  let endpoint;
  let fingerprint;
  const key = process.env.TTM_API_KEY;
  try {
    payload = await readPayload(req);
    try { endpoint = new URL(process.env.TTM_MODAL_URL); } catch { /* handled below */ }
    if (!key || key.length < 32 || !endpoint || endpoint.protocol !== 'https:' || endpoint.username || endpoint.password
      || !endpoint.hostname.endsWith('.modal.run') || endpoint.port || endpoint.hash || endpoint.search) {
      throw error(503, 'Custom prompts are temporarily unavailable.');
    }
    fingerprint = clientFingerprint(req, key);
  } catch (failure) {
    return respond(res, failure.status || 400, { error: failure.status ? failure.message : 'This request is invalid.' });
  }

  const controller = new AbortController();
  let timedOut = false;
  const timeout = setTimeout(() => { timedOut = true; controller.abort(); }, TIMEOUT_MS);
  const disconnected = () => { if (!res.writableEnded) controller.abort(); };
  req.once('aborted', disconnected);
  res.once('close', disconnected);
  try {
    const upstream = await fetch(endpoint.href, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${key}`, 'X-TTM-Client': fingerprint },
      body: JSON.stringify(payload),
      signal: controller.signal,
      redirect: 'error'
    });
    if (upstream.status >= 500) {
      await upstream.body?.cancel();
      return respond(res, 503, { error: 'The model service is warming up or unavailable. Please try again shortly.' });
    }
    const data = await readResponse(upstream);
    if (!upstream.ok) {
      const safe = {
        error: typeof data.error === 'string' && data.error.length <= 300 ? data.error : 'This analysis request could not be completed.'
      };
      if (typeof data.code === 'string' && /^[a-z_]{1,40}$/.test(data.code)) safe.code = data.code;
      if (Number.isInteger(data.retry_after) && data.retry_after > 0 && data.retry_after <= 86400) {
        safe.retry_after = data.retry_after;
        res.setHeader('Retry-After', String(data.retry_after));
      }
      return respond(res, upstream.status >= 400 && upstream.status < 500 ? upstream.status : 502, safe);
    }
    return respond(res, upstream.status, data);
  } catch (_) {
    return respond(res, timedOut ? 504 : 502, {
      error: timedOut ? 'The model service took too long to respond. Please try again.' : 'The model service could not be reached. Please try again.'
    });
  } finally {
    clearTimeout(timeout);
    controller.abort();
    req.removeListener('aborted', disconnected);
    res.removeListener('close', disconnected);
  }
};
