"""Operator-only real inference check through the authenticated Modal SDK.

Run after deployment. Uses public test prompts, not browser/user records.
Saves full observations outside the deployed assets and a compact report.
"""
import argparse
import json
import time
import uuid
from pathlib import Path
import modal

parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=['70m','160m','410m','1b'], required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--prompt', default='The quick brown fox jumps over the')
args = parser.parse_args()
manifest = json.loads((Path(__file__).resolve().parents[1] / 'dist/data/models.json').read_text())[args.model]
store = modal.Dict.from_name('transformer-time-machine-custom-v1')
worker = modal.Function.from_name('transformer-time-machine-custom', 'analyze')
job_id = str(uuid.uuid4())
now = time.time()
store['job:'+job_id] = {'id':job_id, 'client':'0'*64, 'prompt':args.prompt,
    'model':args.model, 'created_at':now, 'expires_at':now+3600}
try:
    call = worker.spawn(job_id)
    last_completed = None
    while True:
        progress = store.get('progress:' + job_id, {'status': 'queued', 'completed': 0})
        if progress['completed'] != last_completed:
            print(f"{args.model}: {progress['completed']}/38", flush=True)
            last_completed = progress['completed']
        if progress['status'] in {'complete', 'error', 'cancelled'}:
            break
        if time.time() - now > 1920:
            call.cancel()
            raise TimeoutError('Operator verification exceeded deadline')
        time.sleep(5)
    call.get()
    progress = store['progress:'+job_id]
    if progress['status'] != 'complete':
        raise RuntimeError(str(progress))
    frames = [store[f'frame:{job_id}:{i}'] for i in range(38)]
    scene = {**progress['scene'], 'frames': frames}
    assert scene['prompt'] == args.prompt
    assert len(scene['tokens']) <= 32
    m = manifest['model']
    for i, f in enumerate(frames):
        assert f['step'] == manifest['checkpoints'][i]['step']
        assert len(f['attention']) == m['layers']
        assert all(len(layer) == m['heads'] for layer in f['attention'])
        assert all(abs(sum(row)-1) < 0.00004 for layer in f['attention'] for head in layer for row in head)
        assert len(json.dumps(f).encode()) + 50000 < 4*1024*1024
    record = {'schema_version':1, 'model':m, 'checkpoints':manifest['checkpoints'],
        'provenance':progress['provenance'], 'scenes':[scene]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, separators=(',', ':')))
    print(json.dumps({'model':args.model, 'tokens':len(scene['tokens']), 'frames':len(frames),
        'inference_seconds':progress['provenance']['extraction_wall_seconds'],
        'final_top':frames[-1]['top_tokens'][0], 'output':str(args.output)}, indent=2))
finally:
    for key in [f'job:{job_id}',f'progress:{job_id}',f'cancel:{job_id}',*[f'frame:{job_id}:{i}' for i in range(38)]]:
        try: store.pop(key)
        except KeyError: pass
