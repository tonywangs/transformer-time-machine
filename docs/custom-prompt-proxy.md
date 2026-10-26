# Custom-prompt proxy

`api/analyze.js` is a dependency-free CommonJS Vercel Function. The recorded examples remain static files in `dist/`. Vercel detects root `api/` functions even when `outputDirectory` points to the static files; `vercel.json` gives this function a 60-second limit. Each upstream request has a shorter 30-second timeout. There is no long-running inference inside Vercel.

Configure these **server-side** environment variables for the deployment:

- `TTM_MODAL_URL`: the exact HTTPS Modal gateway endpoint, ending in `.modal.run` (an endpoint path is allowed). No credentials, query string, fragment, or alternate host is accepted.
- `TTM_API_KEY`: the same random secret used by the Modal gateway, at least 32 characters. Never put it in `dist/`, frontend code, a screenshot, or a committed environment file.

The proxy sends `Authorization: Bearer <key>` and `X-TTM-Client: <HMAC-SHA256 fingerprint>`. The fingerprint derives from the Vercel-provided `x-vercel-forwarded-for` IP and the secret; raw IP addresses are not sent to Modal. It is an admission-limit identifier, not a user account. People sharing a public IP share a limit. Key rotation changes fingerprints and invalidates ownership of existing jobs. The gateway must enforce job ownership, quotas, global compute limits, and expiry independently.

## Public request contract

All calls are `POST /api/analyze` with `Content-Type: application/json`:

```json
{"action":"start","prompt":"Once upon a"}
```

```json
{"action":"poll","job_id":"12345678-1234-4123-8123-123456789abc","cursor":0}
```

```json
{"action":"cancel","job_id":"12345678-1234-4123-8123-123456789abc"}
```

Only these fields are forwarded. Prompt whitespace is preserved exactly. The proxy allows at most 400 Unicode code points and a 4 KiB JSON body; the model service enforces its separate 32-token limit. Poll cursors are chronological checkpoint indices from 0 through 37. Each response contains at most one real checkpoint observation and must fit within 1 MiB. Responses are never HTTP-cached.

Same-site browser requests from the canonical production URL and Vercel's configured deployment URLs are accepted. Cross-site browser requests are rejected. Non-browser clients can call the endpoint; origin checks are not bot prevention. In production, a missing or ambiguous Vercel client-IP header disables custom requests instead of accepting a user-supplied identity. The direct local development path uses the socket IP and accepts a matching localhost origin only outside production.

Backend 4xx errors preserve a short error message, simple error code, and bounded `retry_after` value. Upstream network errors, malformed responses, oversized responses, and 5xx details are replaced with generic messages. No request bodies or secrets are logged by this function.

## Verification

Run:

```sh
node --test scripts/test-proxy.cjs
node --check api/analyze.js
```

The tests mock the model service; they verify the proxy boundary and do not perform inference. A deployment still needs one end-to-end run to verify environment variables, Vercel routing, platform headers, Modal authentication, and the real model response.

Static-only preview with `python3 -m http.server --directory dist` does not serve `/api/analyze`. Use Vercel's local development server with separately configured local environment variables when testing custom inference, or test a deployed preview. Recorded examples continue to work without backend configuration.

Official deployment references: [Node.js functions](https://vercel.com/docs/functions/runtimes/node-js), [project configuration](https://vercel.com/docs/project-configuration/vercel-json), and [Vercel request headers](https://vercel.com/docs/headers/request-headers).
