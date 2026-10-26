#!/usr/bin/env python3
"""Bounded, authenticated Pythia timeline service. Deploy explicitly with Modal.

Only immutable model snapshots are baked into the worker image. Prompts are
never logged. The small gateway admits jobs serially; a separate worker computes
real observations. No HTTP response contains more than one checkpoint.
"""
import hashlib
import hmac
import json
import math
import os
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

SOURCE = Path(__file__).resolve().parent
MANIFEST_PATH = SOURCE / "model-manifest.json"
MANIFEST_BYTES = MANIFEST_PATH.read_bytes()
MANIFEST = json.loads(MANIFEST_BYTES)
MANIFEST_SHA = hashlib.sha256(MANIFEST_BYTES).hexdigest()
EXTRACTOR_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
MODEL_ID = MANIFEST["model"]["id"]
CHECKPOINTS = MANIFEST["checkpoints"]
TOTAL = len(CHECKPOINTS)
MAX_TOKENS = 32
MAX_CHARS = 400
JOB_TIMEOUT = 180
ACCESS_TTL = 3600
MAX_OUTSTANDING = 2  # One running and at most one waiting.
GLOBAL_DAILY = 200
CLIENT_HOURLY = 8
# Fixed expiry: redeploying this source does not renew public compute access.
SUNSET_AT = 1798158075  # 2026-12-25 00:21:15 UTC.
TERMINAL = {"complete", "cancelled", "error"}
_MODEL_CACHE = {}
_TOKENIZER = None


def iso(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


class RequestError(Exception):
    def __init__(self, message, status=400, code="invalid_request", retry_after=None):
        super().__init__(message)
        self.status, self.code, self.retry_after = status, code, retry_after


def validate_request(payload):
    if not isinstance(payload, dict):
        raise RequestError("Send a JSON object.")
    action = payload.get("action")
    allowed = {"start": {"action", "prompt"}, "poll": {"action", "job_id", "cursor"},
               "cancel": {"action", "job_id"}}
    if not isinstance(action, str) or action not in allowed or set(payload) - allowed[action]:
        raise RequestError("Unknown request action or fields.")
    if action == "start":
        prompt = payload.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise RequestError("Enter a short prompt first.")
        if len(prompt) > MAX_CHARS:
            raise RequestError(f"Keep your prompt under {MAX_CHARS + 1} characters.", 422, "prompt_too_long")
        try:
            prompt.encode("utf-8", errors="strict")
        except UnicodeError:
            raise RequestError("The prompt contains an invalid character.") from None
    else:
        value = payload.get("job_id")
        try:
            parsed = uuid.UUID(value) if isinstance(value, str) else None
        except (ValueError, AttributeError):
            parsed = None
        if parsed is None or parsed.version != 4 or str(parsed) != value:
            raise RequestError("Invalid analysis identifier.")
        if action == "poll" and (type(payload.get("cursor")) is not int
                                  or not 0 <= payload["cursor"] < TOTAL):
            raise RequestError("Invalid checkpoint cursor.")
    return action


def _pop(store, key):
    try:
        return store.pop(key)
    except KeyError:
        return None


def public_progress(job, progress):
    result = {"job_id": job["id"], "status": progress.get("status", "queued"),
              "completed": progress.get("completed", 0), "total": TOTAL,
              "expires_at": iso(job["expires_at"])}
    for key in ("scene", "provenance", "error", "code"):
        if key in progress:
            result[key] = progress[key]
    if "scene" in progress:
        result.update(model=MANIFEST["model"], checkpoints=CHECKPOINTS)
    return result


class Gateway:
    """Admission and ownership checks, independent of Modal for offline tests.

    Calls must be serialized. In deployment the gateway has max_containers=1
    and the HTTP handler holds a local lock. Worker updates separate progress keys.
    """
    def __init__(self, store, spawn, cancel, clock=time.time):
        self.store, self.spawn, self.cancel, self.clock = store, spawn, cancel, clock

    def _ledger(self, now):
        ledger = self.store.get("ledger", {"jobs": {}, "starts": [], "active": []})
        ledger.setdefault("active", [])
        ledger["starts"] = [entry for entry in ledger["starts"] if entry[0] > now - 86400]
        for job_id, expiry in list(ledger["jobs"].items()):
            if expiry <= now:
                self._delete_job(job_id)
                del ledger["jobs"][job_id]
                # Deleting a frame is a remote store operation. Bound each
                # gateway request rather than sweeping hundreds of old jobs.
                break
        self.store["ledger"] = ledger
        return ledger

    def _delete_job(self, job_id):
        for key in [f"job:{job_id}", f"progress:{job_id}", f"cancel:{job_id}"]:
            _pop(self.store, key)
        for index in range(TOTAL):
            _pop(self.store, f"frame:{job_id}:{index}")

    def _progress(self, job, now):
        progress = self.store.get(f"progress:{job['id']}", {"status": "queued", "completed": 0})
        if self.store.get(f"cancel:{job['id']}", False):
            progress = {**progress, "status": "cancelled"}
        # Modal's hard worker timeout can terminate execution before its except
        # block runs. Do not leave such jobs appearing active forever.
        start = progress.get("started_at")
        overdue = ((start is not None and now > start + JOB_TIMEOUT + 15)
                   or (start is None and now > job["created_at"] + JOB_TIMEOUT * MAX_OUTSTANDING + 30))
        if progress.get("status") not in TERMINAL and overdue:
            progress = {**progress, "status": "error", "error": "This analysis took too long. Please try again.",
                        "code": "analysis_timeout"}
            self.store[f"progress:{job['id']}"] = progress
        return progress

    def handle(self, payload, client):
        action = validate_request(payload)
        if not isinstance(client, str) or re.fullmatch(r"[a-f0-9]{64}", client) is None:
            raise RequestError("The request could not be verified.", 403, "invalid_client")
        now = self.clock()
        if action == "start":
            if now >= SUNSET_AT:
                raise RequestError("Custom prompt analysis is paused. The recorded examples are still available.", 410, "service_paused")
            ledger = self._ledger(now)
            day = datetime.fromtimestamp(now, timezone.utc).date()
            daily = sum(datetime.fromtimestamp(t, timezone.utc).date() == day for t, _ in ledger["starts"])
            hourly = sum(who == client and t > now - 3600 for t, who in ledger["starts"])
            if daily >= GLOBAL_DAILY:
                raise RequestError("Today's custom prompt limit has been reached. Try again tomorrow.", 429, "daily_limit", 3600)
            if hourly >= CLIENT_HOURLY:
                raise RequestError("You've run eight prompts this hour. Try again a little later.", 429, "hourly_limit", 3600)
            active = []
            for job_id in ledger["active"]:
                job = self.store.get(f"job:{job_id}")
                if job and job["expires_at"] > now and self._progress(job, now)["status"] not in TERMINAL:
                    active.append(job)
            ledger["active"] = [job["id"] for job in active]
            self.store["ledger"] = ledger
            if any(job["client"] == client for job in active):
                raise RequestError("Your previous prompt is still running.", 409, "already_running")
            if len(active) >= MAX_OUTSTANDING:
                raise RequestError("The model is busy. Try again in a minute.", 429, "busy", 60)
            job_id = str(uuid.uuid4())
            job = {"id": job_id, "client": client, "prompt": payload["prompt"],
                   "created_at": now, "expires_at": now + ACCESS_TTL}
            self.store[f"job:{job_id}"] = job
            self.store[f"progress:{job_id}"] = {"status": "queued", "completed": 0}
            ledger["jobs"][job_id] = job["expires_at"]
            ledger["active"].append(job_id)
            ledger["starts"].append([now, client])
            self.store["ledger"] = ledger
            try:
                job["call_id"] = self.spawn(job_id)
                self.store[f"job:{job_id}"] = job
            except Exception:
                self.store[f"progress:{job_id}"] = {"status": "error", "completed": 0,
                    "error": "The model could not start. Please try again.", "code": "start_failed"}
                raise RequestError("The model could not start. Please try again.", 503, "start_failed") from None
            return public_progress(job, {"status": "queued", "completed": 0}), 202

        job_id = payload["job_id"]
        job = self.store.get(f"job:{job_id}")
        if not job or not hmac.compare_digest(job["client"], client):
            raise RequestError("This analysis was not found.", 404, "not_found")
        if now >= job["expires_at"]:
            self._delete_job(job_id)
            raise RequestError("This analysis has expired. Run your prompt again.", 410, "expired")
        progress = self._progress(job, now)
        if action == "cancel":
            if progress["status"] not in TERMINAL:
                self.store[f"cancel:{job_id}"] = True
                if job.get("call_id"):
                    self.cancel(job["call_id"])
                progress = {**progress, "status": "cancelled"}
            return public_progress(job, progress), 200
        result = public_progress(job, progress)
        if progress["status"] not in {"error", "cancelled"}:
            frame = self.store.get(f"frame:{job_id}:{payload['cursor']}")
            if frame is not None:
                result.update(frame=frame, index=payload["cursor"])
        return result, 200


def get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer
        _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_ID, revision=MANIFEST["tokenizer_commit_sha"],
            cache_dir="/models", local_files_only=True, trust_remote_code=False)
    return _TOKENIZER


def get_model(index):
    if index not in _MODEL_CACHE:
        import torch
        from transformers import AutoModelForCausalLM
        _MODEL_CACHE[index] = AutoModelForCausalLM.from_pretrained(MODEL_ID,
            revision=CHECKPOINTS[index]["commit_sha"], cache_dir="/models", local_files_only=True,
            torch_dtype=torch.float32, attn_implementation="eager", trust_remote_code=False).eval()
    return _MODEL_CACHE[index]


def observe(model, tokenizer, token_ids, index, target_id=None):
    import torch
    with torch.inference_mode():
        tokens = torch.tensor([token_ids], dtype=torch.long)
        result = model(input_ids=tokens, attention_mask=torch.ones_like(tokens), output_attentions=True, use_cache=False)
        probabilities = torch.softmax(result.logits[0, -1].float(), dim=-1)
        if not result.attentions or any(a is None for a in result.attentions):
            raise RuntimeError("Missing model attention")
        attention = torch.stack([a[0].float() for a in result.attentions])
        if not torch.isfinite(probabilities).all() or not torch.isfinite(attention).all():
            raise RuntimeError("Non-finite model output")
        if not torch.allclose(attention.sum(-1), torch.ones_like(attention.sum(-1)), atol=1e-5):
            raise RuntimeError("Invalid attention normalization")
        if torch.triu(attention, diagonal=1).abs().max() > 1e-7:
            raise RuntimeError("Invalid causal attention")
        values, indices = probabilities.topk(8)
        if target_id is None:
            target_id = int(indices[0])
        top = [{"id": int(i), "token": tokenizer.decode([int(i)]), "probability": float(v)}
               for v, i in zip(values, indices)]
        n = len(token_ids)
        entropy = -(attention[:, :, -1, :] * attention[:, :, -1, :].clamp_min(1e-30).log()).sum(-1)
        # A single token's only legal attention distribution has full focus.
        focus = torch.ones_like(entropy) if n == 1 else (1 - entropy / math.log(n)).clamp(0, 1)
        rounded = [[[[round(value, 6) for value in row] for row in head] for head in layer]
                   for layer in attention.tolist()]
        frame = {"scene_id": "custom", "step": CHECKPOINTS[index]["step"], "top_tokens": top,
                 "target_probability": float(probabilities[target_id]), "attention": rounded,
                 "concentration": focus.tolist()}
        return frame, target_id


def run_job(job_id, store):
    import platform
    import torch
    import transformers
    job = store.get(f"job:{job_id}")
    if not job or store.get(f"cancel:{job_id}", False):
        return
    started = time.time()
    progress = {"status": "running", "completed": 0, "started_at": started}
    store[f"progress:{job_id}"] = progress
    try:
        torch.set_num_threads(4)
        torch.manual_seed(0)
        tokenizer = get_tokenizer()
        token_ids = tokenizer.encode(job["prompt"], add_special_tokens=False)
        if not 1 <= len(token_ids) <= MAX_TOKENS:
            raise RequestError(f"This prompt uses {len(token_ids)} tokens. Keep it to {MAX_TOKENS} tokens or fewer.",
                               422, "too_many_tokens")
        final, target_id = observe(get_model(TOTAL - 1), tokenizer, token_ids, TOTAL - 1)
        scene = {"id": f"custom-{job_id}", "title": "Your prompt", "kicker": "Custom prompt", "description": "",
                 "prompt": job["prompt"], "tokens": [tokenizer.decode([i]) for i in token_ids],
                 "token_ids": token_ids, "target_label": tokenizer.decode([target_id]), "target_token_ids": [target_id],
                 "target_selection": "final_checkpoint_argmax"}
        provenance = {"extracted_at": iso(started), "device": "cpu", "dtype": "float32",
                      "platform": platform.platform(), "torch_version": str(torch.__version__),
                      "transformers_version": transformers.__version__,
                      "extractor_sha256": EXTRACTOR_SHA,
                      "manifest_sha256": MANIFEST_SHA, "tokenizer_commit_sha": MANIFEST["tokenizer_commit_sha"],
                      "notes": ["Actual eval-mode forward passes at all 38 immutable checkpoints.",
                                "Tracked token is the final checkpoint's top guess, not an accuracy label.",
                                "CPU float32, eager attention, inference mode, no KV cache, no implicit BOS.",
                                "Attention weights rounded to six decimals; no interpolated model observations.",
                                "Concentration is final-query focus; a one-token prompt has concentration 1."],
                      "environment": {"attention_implementation": "eager", "eval_mode": True,
                                      "inference_mode": True, "use_cache": False, "num_threads": 4,
                                      "seed": 0, "add_special_tokens": False}}
        progress.update(scene=scene, provenance=provenance)
        for index in [TOTAL - 1, *range(TOTAL - 1)]:
            if store.get(f"cancel:{job_id}", False):
                progress["status"] = "cancelled"
                break
            if time.time() - started >= JOB_TIMEOUT - 5:
                raise RequestError("This analysis took too long. Please try again.", 503, "analysis_timeout")
            frame = final if index == TOTAL - 1 else observe(get_model(index), tokenizer, token_ids, index, target_id)[0]
            frame["scene_id"] = scene["id"]
            store[f"frame:{job_id}:{index}"] = frame
            progress["completed"] += 1
            if progress["completed"] == TOTAL:
                progress["status"] = "complete"
                provenance["extraction_wall_seconds"] = round(time.time() - started, 3)
            store[f"progress:{job_id}"] = progress
    except RequestError as error:
        progress.update(status="error", error=str(error), code=error.code)
    except Exception as error:
        # Exception messages can contain prompt text. Log only the class name.
        print(f"Custom analysis failed: {type(error).__name__}", flush=True)
        progress.update(status="error", error="The model could not finish this prompt. Please try again.", code="analysis_failed")
    finally:
        store[f"progress:{job_id}"] = progress


# Keeping the Modal declarations below the pure service logic permits focused
# offline tests without invoking account APIs or downloading model snapshots.
import modal

app = modal.App("transformer-time-machine-custom")


def get_store():
    # Resolve at execution time: Modal 1.1.1 cannot pickle an unhydrated Dict
    # hidden inside a serialized function's nested helper dependencies.
    return modal.Dict.from_name("transformer-time-machine-custom-v1", create_if_missing=True)


base_image = (modal.Image.debian_slim(python_version="3.11")
              .pip_install("fastapi==0.116.1")
              .add_local_file(MANIFEST_PATH, "/service/model-manifest.json", copy=True)
              .env({"PYTHONPATH": "/service"}))
gateway_image = base_image.add_local_file(__file__, "/service/serve_custom.py", copy=True)
worker_image = (base_image
    .pip_install("torch==2.8.0", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("transformers==4.56.2", "huggingface_hub==0.35.3")
    .env({"HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false",
          "HF_XET_CHUNK_CACHE_SIZE_BYTES": "0"})
    .add_local_file(SOURCE / "cache_custom_models.py", "/service/cache_custom_models.py", copy=True)
    .run_commands("python /service/cache_custom_models.py")
    .add_local_file(__file__, "/service/serve_custom.py", copy=True))


@app.function(image=worker_image, cpu=(4, 4), memory=(16384, 16384), timeout=JOB_TIMEOUT,
              max_containers=1, min_containers=0, retries=0, scaledown_window=20,
              serialized=True, include_source=False)
def analyze(job_id):
    run_job(job_id, get_store())


@app.function(image=gateway_image, cpu=(0.25, 0.25), memory=(512, 512), timeout=30,
              max_containers=1, min_containers=0, scaledown_window=10,
              secrets=[modal.Secret.from_name("transformer-time-machine-api")],
              serialized=True, include_source=False)
@modal.asgi_app()
def api():
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    web = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    gate_lock = threading.Lock()
    worker = modal.Function.from_name("transformer-time-machine-custom", "analyze")
    gateway = Gateway(get_store(), lambda job_id: worker.spawn(job_id).object_id,
                      lambda call_id: modal.FunctionCall.from_id(call_id).cancel())

    @web.post("/")
    async def handle(request: Request):
        key = os.environ.get("TTM_API_KEY", "")
        provided = request.headers.get("authorization", "")
        if not key or not hmac.compare_digest(provided, f"Bearer {key}"):
            return JSONResponse({"error": "Unauthorized.", "code": "unauthorized"}, status_code=401)
        if request.headers.get("content-length", "0").isdigit() and int(request.headers.get("content-length", "0")) > 4096:
            return JSONResponse({"error": "The request is too large.", "code": "request_too_large"}, status_code=413)
        body = await request.body()
        if len(body) > 4096:
            return JSONResponse({"error": "The request is too large.", "code": "request_too_large"}, status_code=413)
        try:
            payload = json.loads(body)
            with gate_lock:
                result, status = gateway.handle(payload, request.headers.get("x-ttm-client", ""))
            return JSONResponse(result, status_code=status, headers={"Cache-Control": "no-store"})
        except (ValueError, UnicodeError):
            return JSONResponse({"error": "Send valid JSON.", "code": "invalid_json"}, status_code=400)
        except RequestError as error:
            result = {"error": str(error), "code": error.code}
            if error.retry_after is not None:
                result["retry_after"] = error.retry_after
            return JSONResponse(result, status_code=error.status, headers={"Cache-Control": "no-store"})

    return web
