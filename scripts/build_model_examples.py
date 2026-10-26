"""Extract the three preset prompts for every supported larger model on Modal."""
import gzip
import json
from pathlib import Path
import modal
import serve_custom as service

ROOT = Path(__file__).resolve().parents[1]
app = modal.App('transformer-time-machine-examples')

@app.function(image=service.worker_image, gpu='L4', cpu=4, memory=16384,
              volumes={'/models': service.model_volume}, timeout=1800,
              max_containers=1, retries=0, serialized=True, include_source=False)
def extract(size, examples):
    import time, hashlib, platform, torch, transformers
    import serve_custom as s
    s.MANIFEST = s.CATALOG[size]
    s.MODEL_ID = s.MANIFEST['model']['id']
    s.CHECKPOINTS = s.MANIFEST['checkpoints']
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = s.get_tokenizer()
    scenes = []
    for original in examples:
        item = {k:v for k,v in original.items() if k != 'frames'}
        ids = tokenizer.encode(item['prompt'], add_special_tokens=False)
        targets = tokenizer.encode(item['target_label'], add_special_tokens=False)
        assert len(targets) == 1
        item.update(tokens=[tokenizer.decode([i]) for i in ids], token_ids=ids,
                    target_token_ids=targets, target_selection='preset_reference', frames=[])
        scenes.append(item)
    started = time.time()
    for i, checkpoint in enumerate(s.CHECKPOINTS):
        model = s.get_model(i)
        for scene in scenes:
            frame, _ = s.observe(model, tokenizer, scene['token_ids'], i, scene['target_token_ids'][0])
            frame['scene_id'] = scene['id']
            scene['frames'].append(frame)
        print(f'{size}: {i+1}/38', flush=True)
    record = {'schema_version':1, 'model':s.MANIFEST['model'], 'checkpoints':s.CHECKPOINTS,
        'scenes':scenes, 'provenance':{'extracted_at':s.iso(started),'device':'cuda',
        'gpu':torch.cuda.get_device_name(),'dtype':'float32','platform':platform.platform(),
        'torch_version':str(torch.__version__),'transformers_version':transformers.__version__,
        'extractor_sha256':s.EXTRACTOR_SHA,'tokenizer_commit_sha':s.MANIFEST['tokenizer_commit_sha'],
        'manifest_sha256':hashlib.sha256(json.dumps(s.MANIFEST,sort_keys=True).encode()).hexdigest(),
        'extraction_wall_seconds':round(time.time()-started,3),
        'notes':['Real forward passes at all 38 immutable checkpoints.',
                 'Preset reference tokens are fixed across models: time, purple, Paris; they are not forced predictions.',
                 'CUDA float32 with TF32 disabled; eager attention; evaluation/inference mode; no KV cache or implicit BOS.',
                 'Attention rounded to six decimal places.']}}
    return gzip.compress(json.dumps(record,separators=(',',':'),allow_nan=False).encode())

@app.local_entrypoint()
def main():
    original=json.loads((ROOT/'dist/data/traces.json').read_text())
    for size in ['160m','410m','1b']:
        path=ROOT/f'dist/data/examples-{size}.json'
        if path.exists():
            print('Already extracted',size)
            continue
        payload=extract.remote(size,original['scenes'])
        path.write_bytes(gzip.decompress(payload))
        print('Saved',path,flush=True)
