#!/usr/bin/env python3
"""Run one real custom experiment through the public Vercel API and save it.

This consumes one accepted start from the service quota. It never substitutes
bundled observations and does not require a server credential.
"""
import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-url', default='https://transformer-time-machine.vercel.app')
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', choices=['70m', '160m', '410m', '1b'], default='70m')
    args = parser.parse_args()
    expected = json.loads((Path(__file__).parent.parent / 'dist/data/models.json').read_text())[args.model]
    def request(payload):
        req = urllib.request.Request(args.base_url.rstrip('/') + '/api/analyze',
            data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=40) as response:
                data = response.read(4 * 1024 * 1024 + 1)
                if len(data) > 4 * 1024 * 1024:
                    raise RuntimeError('Response exceeded one-frame limit')
                return json.loads(data)
        except urllib.error.HTTPError as error:
            payload = json.loads(error.read(4096))
            raise RuntimeError(f"HTTP {error.code}: {payload.get('error', 'Request failed')}") from None
    started = time.monotonic()
    admission = request({'action': 'start', 'prompt': args.prompt, 'model': args.model})
    job = admission['job_id']
    frames = []
    last_progress = None
    try:
        while len(frames) < len(expected['checkpoints']):
            if time.monotonic() - started > 1920:
                raise RuntimeError('Experiment exceeded smoke-test deadline')
            result = request({'action': 'poll', 'job_id': job, 'cursor': len(frames)})
            if result['status'] in {'error', 'cancelled'}:
                raise RuntimeError(result.get('error', result['status']))
            progress = result['completed']
            if progress != last_progress:
                print(f'Measured {progress}/{result["total"]}', flush=True)
                last_progress = progress
            if 'frame' not in result:
                time.sleep(.75)
                continue
            assert result['checkpoints'] == expected['checkpoints']
            assert result['model'] == expected['model']
            assert result['index'] == len(frames)
            assert result['frame']['step'] == expected['checkpoints'][len(frames)]['step']
            assert result['scene']['prompt'] == args.prompt
            frames.append(result['frame'])
        record = {'schema_version': 1, 'model': result['model'], 'checkpoints': result['checkpoints'],
                  'provenance': result['provenance'], 'scenes': [{**result['scene'], 'frames': frames}]}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, separators=(',', ':')))
        print(json.dumps({'saved': str(args.output), 'frames': len(frames),
            'tokens': len(result['scene']['tokens']), 'wall_seconds': round(time.monotonic() - started, 3),
            'inference_seconds': result['provenance'].get('extraction_wall_seconds'),
            'final_top_token': frames[-1]['top_tokens'][0]}, indent=2))
    except BaseException:
        try:
            request({'action': 'cancel', 'job_id': job})
        except Exception:
            pass
        raise


if __name__ == '__main__':
    main()
