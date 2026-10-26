"""Offline tests: no Modal account access, model download, or billed calls."""
import copy
import json
import unittest
from types import SimpleNamespace

import serve_custom as service


class Store(dict):
    # Modal Dict returns deserialized copies, not shared mutable references.
    def get(self, key, default=None):
        return copy.deepcopy(super().get(key, default))

    def __setitem__(self, key, value):
        super().__setitem__(key, copy.deepcopy(value))


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.store = Store()
        self.now = 1792972800.0
        self.cancelled = []
        self.gateway = service.Gateway(self.store, lambda job: "call-" + job,
                                       self.cancelled.append, lambda: self.now)
        self.client = "a" * 64

    def start(self, client=None, prompt="Once upon a"):
        return self.gateway.handle({"action": "start", "prompt": prompt}, client or self.client)[0]

    def finish(self, job):
        self.store[f"progress:{job['job_id']}"] = {"status": "complete", "completed": service.TOTAL}

    def assert_error(self, code, payload, client=None):
        with self.assertRaises(service.RequestError) as caught:
            self.gateway.handle(payload, client or self.client)
        self.assertEqual(caught.exception.code, code)

    def test_input_validation_and_exact_whitespace(self):
        for prompt in ("", " \n\t", None):
            self.assert_error("invalid_request", {"action": "start", "prompt": prompt})
        self.assert_error("prompt_too_long", {"action": "start", "prompt": "x" * 401})
        self.assert_error("invalid_request", {"action": [], "prompt": "a"})
        job = self.start(prompt="  hi\n")
        self.assertEqual(self.store[f"job:{job['job_id']}"]["prompt"], "  hi\n")

    def test_ownership_and_one_frame_per_poll(self):
        job = self.start()
        payload = {"action": "poll", "job_id": job["job_id"], "cursor": 0}
        self.assert_error("not_found", payload, "b" * 64)
        self.store[f"frame:{job['job_id']}:0"] = {"step": 0}
        self.store[f"frame:{job['job_id']}:37"] = {"step": 143000}
        response, status = self.gateway.handle(payload, self.client)
        self.assertEqual(status, 200)
        self.assertEqual(response["frame"], {"step": 0})
        self.assertEqual(response["index"], 0)
        self.assertNotIn("frames", response)
        self.assertEqual(len(self.store["ledger"]["starts"]), 1)

    def test_final_first_does_not_fill_missing_cursor(self):
        job = self.start()
        self.store[f"frame:{job['job_id']}:37"] = {"step": 143000}
        response, _ = self.gateway.handle({"action": "poll", "job_id": job["job_id"], "cursor": 0}, self.client)
        self.assertNotIn("frame", response)

    def test_concurrency_and_hourly_admission(self):
        first = self.start()
        self.assert_error("already_running", {"action": "start", "prompt": "hello"})
        second = self.start("b" * 64)
        self.assert_error("busy", {"action": "start", "prompt": "hello"}, "c" * 64)
        self.finish(first)
        self.finish(second)
        for _ in range(service.CLIENT_HOURLY - 1):
            job = self.start()
            self.finish(job)
        self.assert_error("hourly_limit", {"action": "start", "prompt": "hello"})
        self.now += 3601
        self.assertEqual(self.start()["status"], "queued")

    def test_global_quota_and_sunset(self):
        self.store["ledger"] = {"jobs": {}, "starts": [[self.now, "b" * 64]] * service.GLOBAL_DAILY}
        self.assert_error("daily_limit", {"action": "start", "prompt": "hello"})
        self.now = service.SUNSET_AT
        self.assert_error("service_paused", {"action": "start", "prompt": "hello"})

    def test_cancel_and_expiry(self):
        job = self.start()
        self.store[f"frame:{job['job_id']}:0"] = {"step": 0}
        response, _ = self.gateway.handle({"action": "cancel", "job_id": job["job_id"]}, self.client)
        self.assertEqual(response["status"], "cancelled")
        self.assertEqual(len(self.cancelled), 1)
        response, _ = self.gateway.handle({"action": "poll", "job_id": job["job_id"], "cursor": 0}, self.client)
        self.assertNotIn("frame", response)
        self.now += service.ACCESS_TTL
        self.assert_error("expired", {"action": "poll", "job_id": job["job_id"], "cursor": 0})
        self.assertNotIn(f"frame:{job['job_id']}:0", self.store)

    def test_hard_timeout_is_terminal(self):
        job = self.start()
        self.store[f"progress:{job['job_id']}"] = {"status": "running", "completed": 5, "started_at": self.now}
        self.now += service.JOB_TIMEOUT + 16
        response, _ = self.gateway.handle({"action": "poll", "job_id": job["job_id"], "cursor": 0}, self.client)
        self.assertEqual(response["status"], "error")
        self.assertEqual(response["code"], "analysis_timeout")

    def test_immutable_manifest_matches_recorded_dataset(self):
        traces = json.loads((service.SOURCE.parent / "dist/data/traces.json").read_text())
        self.assertEqual(service.CHECKPOINTS, traces["checkpoints"])
        self.assertEqual(service.MANIFEST["tokenizer_commit_sha"], traces["provenance"]["tokenizer_commit_sha"])
        self.assertEqual(service.TOTAL, 38)

    def test_maximum_rounded_frame_payload_is_bounded(self):
        frame = {"attention": [[[[0.123456] * 32 for _ in range(32)] for _ in range(8)] for _ in range(6)]}
        self.assertLess(len(json.dumps(frame).encode()), 500_000)


class ObservationTests(unittest.TestCase):
    def setUp(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch not installed; gateway tests do not require it")
        self.torch = torch

    def model(self, **inputs):
        torch = self.torch
        n = inputs["input_ids"].shape[-1]
        causal = torch.tril(torch.ones(n, n))
        causal /= causal.sum(-1, keepdim=True)
        attention = causal.repeat(1, 8, 1, 1)
        return SimpleNamespace(logits=torch.arange(10, dtype=torch.float32).repeat(1, n, 1),
                               attentions=[attention] * 6)

    def test_single_token_focus_and_target_are_actual_finite_values(self):
        tokenizer = SimpleNamespace(decode=lambda ids: f"token-{ids[0]}")
        frame, target = service.observe(self.model, tokenizer, [3], 37)
        self.assertEqual(target, 9)
        self.assertEqual(frame["target_probability"], frame["top_tokens"][0]["probability"])
        self.assertEqual(frame["concentration"], [[1.0] * 8] * 6)
        json.dumps(frame, allow_nan=False)

    def test_fixed_target_is_preserved_and_real_frame_stays_small(self):
        tokenizer = SimpleNamespace(decode=lambda ids: f"token-{ids[0]}")
        frame, target = service.observe(self.model, tokenizer, list(range(32)), 0, target_id=2)
        self.assertEqual(target, 2)
        self.assertNotEqual(frame["target_probability"], frame["top_tokens"][0]["probability"])
        self.assertEqual(frame["attention"][0][0][0][1], 0.0)
        self.assertEqual(frame["attention"][0][0][-1][-1], 0.03125)
        self.assertLess(len(json.dumps(frame, allow_nan=False).encode()), 500_000)


if __name__ == "__main__":
    unittest.main()
