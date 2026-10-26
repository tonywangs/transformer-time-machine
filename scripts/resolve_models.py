"""Resolve public Pythia branch names once; runtime uses only these immutable SHAs."""
import json
import urllib.request
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
base = json.loads((ROOT / 'scripts/model-manifest.json').read_text())
catalog = {'70m': base}
def fetch(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)
for size in ['160m', '410m', '1b']:
    model_id = 'EleutherAI/pythia-' + size + '-deduped'
    url = 'https://huggingface.co/' + model_id
    refs = {r['name']: r['targetCommit'] for r in fetch('https://huggingface.co/api/models/' + model_id + '/refs')['branches']}
    sha = refs['step143000']
    cfg = fetch(url + '/resolve/' + sha + '/config.json')
    # Exact parameter count for GPTNeoX with untied input/output embeddings.
    h, layers, vocab = cfg['hidden_size'], cfg['num_hidden_layers'], cfg['vocab_size']
    params = 2*vocab*h + layers*(12*h*h + 13*h) + 2*h
    catalog[size] = {'schema_version': 1, 'model': {'id':model_id,'label':'pythia-'+size+'-deduped','parameters':params,
        'layers':layers,'heads':cfg['num_attention_heads'],'hidden_size':h,'source_url':url},
        'tokenizer_commit_sha':sha, 'checkpoints':[{'step':c['step'],'revision':c['revision'],'commit_sha':refs[c['revision']],
        'source_url':url+'/tree/'+refs[c['revision']]} for c in base['checkpoints']]}
    print(size, params, layers, cfg['num_attention_heads'])
(ROOT / 'dist/data/models.json').write_text(json.dumps(catalog, indent=2)+'\n')
