"""Complete finite batch response references for two frozen selected models."""
import json
from pathlib import Path
from .batch import block_response,response
from .experiment import saved


def export():
    data=saved();cases=[]
    for fit in data['candidates']+data['bias_candidates']:
        if fit['id'] not in [data['selected_id'],data['selected_bias_id']]:continue
        m=fit['model']
        for anchor in data['split']['train'][:8]:
            u,i,_=data['edges'][anchor]
            for side,index in [('user',u),('item',i)]:
                full=[j for j in data['split']['train'] if data['edges'][j][0 if side=='user' else 1]==index]
                groups=[[],full[:1],full[:2],full[:3]]
                if len(full)<=64:groups.append(full)
                for rows in groups:
                    for delete in [False,True]:
                        deltas=None if delete else [1-data['edges'][j][2] for j in rows]
                        w=block_response(data,m,side,index,rows,0,deltas,delete)
                        cases.append({'model_id':fit['id'],'anchor':anchor,'witness':w})
    controls=[response([[1,1],[1,1],[1,0]],[1,2,0],1.,rows,[1,1],delete=True) for rows in [[],[0],[0,1],[0,1,2]]]
    return {'schema':1,'cases':cases,'controls':controls}


def write(path):
    Path(path).write_text(json.dumps(export(),separators=(',',':'),allow_nan=False)+'\n')
