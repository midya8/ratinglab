"""Small browser bundle of checked saved values, no refit."""
import json
from pathlib import Path
from .experiment import saved
from .controls import controls
from .core import block_witness
ROOT=Path(__file__).resolve().parents[1]
def build(root=ROOT):
    r=saved();train=[r['edges'][j] for j in r['split']['train']];candidates=[]
    for c in r['candidates']+r['bias_candidates']:
        candidates.append({k:c[k] for k in ['id','model','status','train','validation','test','sweeps']}|{'history':[{k:h[k] for k in ('sweep','stage','objective','max_gradient','max_block_normal_residual')} for h in c['history']]})
    selected=next(c for c in r['candidates'] if c['id']==r['selected_id'])['model']
    data={k:r[k] for k in ['shape','edges','split','selected_id','selected_bias_id','global_test_rmse','cold','protocol','scope']};data.update(candidates=candidates,controls=controls(),witness=block_witness(selected,train,'user',0))
    (root/'docs/data.js').write_text('globalThis.RATING_DATA = '+json.dumps(data,separators=(',',':'),allow_nan=False)+';\n')
    refs=[]
    for c in candidates:
        m=c['model'];records=[]
        for row in r['edges']:
            u,i,_=row;terms=[a*b for a,b in zip(m['users'][u],m['items'][i])];records.append(m['mean']+m['user_bias'][u]+m['item_bias'][i]+sum(terms))
        refs.append(dict(id=c['id'],scores=records))
    (ROOT/'ratinglab/artifacts/browser-reference.json').write_text(json.dumps(refs,separators=(',',':'))+'\n')
    return data
if __name__=='__main__':build()
