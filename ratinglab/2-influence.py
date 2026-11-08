"""Finite rank-one response of ONE conditional ridge block, not retrained ALS."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from .core import block_witness
from .experiment import saved

SCOPE='One re-solved ridge block with training mean, penalty and all opposite-side parameters fixed. Not full ALS retraining, causal influence, uncertainty or unlearning.'

def response(A,target,regularization,row,query,delta=0.,delete=False):
    """Unique ridge baseline and finite edit/delete response, numerical arithmetic."""
    A=np.asarray(A,dtype=float);t=np.asarray(target,dtype=float);q=np.asarray(query,dtype=float)
    if A.ndim!=2 or not len(A) or not A.shape[1] or t.shape!=(len(A),) or q.shape!=(A.shape[1],) or not all(np.isfinite(v).all() for v in (A,t,q)):raise ValueError('finite nonempty design, matching target/query required')
    if type(row) is not int or not 0<=row<len(A) or isinstance(regularization,bool) or not isinstance(regularization,(int,float)) or not math.isfinite(regularization) or regularization<=0 or type(delete) is not bool or isinstance(delta,bool) or not isinstance(delta,(int,float)) or not math.isfinite(delta) or (delete and delta!=0):raise ValueError('valid row, positive penalty, finite delta; deletion cannot edit')
    d=A.shape[1];lam=float(regularization);H=A.T@A+lam*np.eye(d);rhs=A.T@t;a=A[row];base=np.linalg.solve(H,rhs);z=np.linalg.solve(H,a);h=float(a@z);den=1-h
    if den<=0 or h< -1e-12:raise ArithmeticError('numerical leverage outside admissible interval')
    residual=float(t[row]-a@base);coefficient=(-residual/den if delete else float(delta));point=base+coefficient*z
    changed_H=H-np.outer(a,a) if delete else H
    changed_rhs=rhs-a*t[row] if delete else rhs+a*delta
    base_score=float(q@base);new_score=float(q@point)
    return dict(row=row,delete=delete,delta=float(delta),normal=H.tolist(),rhs=rhs.tolist(),baseline=base.tolist(),direction=z.tolist(),leverage=h,deletion_denominator=den,denominator_lower_bound=lam/(lam+float(a@a)),training_residual=residual,coefficient=coefficient,point=point.tolist(),query=q.tolist(),query_derivative=float(q@z),baseline_score=base_score,score=new_score,score_change=new_score-base_score,deleted_row_residual=(residual/den if delete else None),max_normal_residual=float(np.max(np.abs(changed_H@point-changed_rhs))),baseline_normal_residual=float(np.max(np.abs(H@base-rhs))),scope=SCOPE)

def block_response(data,model,side,index,source_row,query_other,delta=0.,delete=False):
    """Source-row IDs are full dataset rows, never local block offsets."""
    limit=len(model['users'] if side=='user' else model['items']) if side in ('user','item') else 0
    other_limit=len(model['items'] if side=='user' else model['users']) if side in ('user','item') else 0
    if type(index) is not int or not 0<=index<limit or type(query_other) is not int or not 0<=query_other<other_limit or type(source_row) is not int or source_row not in data['split']['train']:raise ValueError('known side/index/query and training source row required')
    source=[j for j in data['split']['train'] if data['edges'][j][0 if side=='user' else 1]==index]
    if source_row not in source:raise ValueError('source row does not belong to selected block')
    opposite='items' if side=='user' else 'users';bias='item_bias' if side=='user' else 'user_bias';own='user_bias' if side=='user' else 'item_bias';ownfactor='users' if side=='user' else 'items'
    others=[data['edges'][j][1 if side=='user' else 0] for j in source]
    A=[[1,*model[opposite][j]] for j in others];target=[data['edges'][j][2]-model['mean']-model[bias][o] for j,o in zip(source,others)]
    if not delete and not 0<=data['edges'][source_row][2]+delta<=2:raise ValueError('edited rating must stay in [0,2]')
    q=[1,*model[opposite][query_other]];out=response(A,target,model['regularization'],source.index(source_row),q,delta,delete);offset=model['mean']+model[bias][query_other]
    saved_point=np.array([model[own][index],*model[ownfactor][index]])
    out.update(side=side,index=index,source_row=source_row,source_rows=source,query_other=query_other,original_rating=data['edges'][source_row][2],offset=offset,saved_score=float(offset+np.dot(q,saved_point)),baseline_score=out['baseline_score']+offset,score=out['score']+offset,saved_to_baseline_max_change=float(np.max(np.abs(saved_point-np.array(out['baseline'])))))
    return out

def references(root=None):
    data=saved();ids=[data['selected_id'],data['selected_bias_id']];out=[]
    for fit in data['candidates']+data['bias_candidates']:
        if fit['id'] not in ids:continue
        m=fit['model']
        for source in data['split']['train'][:24]:
            u,i,y=data['edges'][source]
            for side,index in [('user',u),('item',i)]:
                for query in [0,65 if side=='user' else 70,129 if side=='user' else 137]:
                    for delete,delta in [(False,1-y),(True,0)]:
                        r=block_response(data,m,side,index,source,query,delta,delete)
                        out.append(dict(model_id=fit['id'],side=side,index=index,source_row=source,query_other=query,delete=delete,delta=delta,**{k:r[k] for k in ('baseline','point','leverage','deletion_denominator','query_derivative','baseline_score','score','score_change','saved_score','saved_to_baseline_max_change')}))
    path=Path(root) if root else Path(__file__).parent/'artifacts/influence-reference.json';path.write_text(json.dumps(out,separators=(',',':'),allow_nan=False)+'\n');return out

def main():
    p=argparse.ArgumentParser(description=SCOPE);p.add_argument('command',choices=['demo','references']);p.add_argument('--output');a=p.parse_args()
    if a.command=='references':r=references(a.output);print(json.dumps(dict(references=len(r),scope=SCOPE)));return
    d=saved();c=next(c for c in d['candidates'] if c['id']==d['selected_id']);source=d['split']['train'][0];u,i,_=d['edges'][source];r=block_response(d,c['model'],'user',u,source,i,delete=True);print(json.dumps(r,allow_nan=False))
if __name__=='__main__':main()
