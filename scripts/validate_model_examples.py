"""Validate published larger-model presets independently of extraction code."""
import hashlib
import json
import math
from pathlib import Path
from validate_traces import concentration_bounds

ROOT=Path(__file__).resolve().parents[1]
catalog=json.loads((ROOT/'dist/data/models.json').read_text())
base=json.loads((ROOT/'dist/data/traces.json').read_text())
references={s['id']:s for s in base['scenes']}
reports=[]
for size in ['160m','410m','1b']:
    path=ROOT/f'dist/data/examples-{size}.json'
    raw=path.read_bytes();data=json.loads(raw);manifest=catalog[size]
    assert data['model']==manifest['model']
    assert data['checkpoints']==manifest['checkpoints']
    assert data['provenance']['tokenizer_commit_sha']==manifest['tokenizer_commit_sha']
    assert data['provenance']['extractor_sha256']==hashlib.sha256((ROOT/'scripts/serve_custom.py').read_bytes()).hexdigest()
    assert {s['id'] for s in data['scenes']}==set(references)
    rows=0;max_error=0
    for scene in data['scenes']:
        ref=references[scene['id']]
        for key in ['prompt','target_label','target_token_ids','tokens','token_ids']:
            assert scene[key]==ref[key],(size,scene['id'],key)
        assert scene['target_selection']=='preset_reference'
        assert len(scene['frames'])==38
        n=len(scene['tokens']);m=data['model']
        for i,f in enumerate(scene['frames']):
            assert f['step']==manifest['checkpoints'][i]['step'] and f['scene_id']==scene['id']
            assert math.isfinite(f['target_probability']) and 0<=f['target_probability']<=1
            assert len(f['top_tokens'])==8
            probs=[t['probability'] for t in f['top_tokens']]
            assert all(math.isfinite(p) and 0<=p<=1 for p in probs) and probs==sorted(probs,reverse=True)
            for t in f['top_tokens']:
                if t['id'] in scene['target_token_ids']: assert t['probability']==f['target_probability']
            assert len(f['attention'])==len(f['concentration'])==m['layers']
            for l,layer in enumerate(f['attention']):
                assert len(layer)==len(f['concentration'][l])==m['heads']
                for h,head in enumerate(layer):
                    assert len(head)==n
                    for q,row in enumerate(head):
                        assert len(row)==n and all(math.isfinite(v) and 0<=v<=1 for v in row)
                        assert all(v==0 for v in row[q+1:])
                        error=abs(sum(row)-1);max_error=max(max_error,error)
                        assert error<=n*0.0000005+0.000002
                        rows+=1
                    lo,hi=concentration_bounds(head[-1])
                    assert lo<=f['concentration'][l][h]<=hi
    reports.append({'model':data['model']['id'],'frames':114,'attention_rows_checked':rows,
      'maximum_row_sum_error':max_error,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),
      'extraction_seconds':data['provenance']['extraction_wall_seconds'],
      'final_top_tokens':{s['id']:s['frames'][-1]['top_tokens'][0] for s in data['scenes']}})
report={'method':'Real GPU forward passes; independent checks of pinned manifests, preset targets, probability ordering, causal attention, normalization, and entropy-derived focus.',
 'preset_extractor_sha256':hashlib.sha256((ROOT/'scripts/build_model_examples.py').read_bytes()).hexdigest(), 'models':reports}
(ROOT/'docs/preset-model-verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
