"""Frozen observed-edge split; validation-only config choice; no validation refit."""
import json,math
from pathlib import Path
import numpy as np
from .dataset import load,split
from .core import fit,prediction,objective,gradient,block_witness
ROOT=Path(__file__).parent

def metrics(model,edges):
    z=np.asarray(edges);p=prediction(model,z[:,0],z[:,1]);e=p-z[:,2];clip=np.clip(p,0,2)-z[:,2]
    return dict(n=len(z),rmse=float(np.sqrt(np.mean(e*e))),mae=float(np.mean(np.abs(e))),clipped_rmse=float(np.sqrt(np.mean(clip*clip))),outside_rating_range=int(np.count_nonzero((p<0)|(p>2))))

def run(data=None,max_sweeps=80):
    z,nu,ni,sha=load();z=z if data is None else np.asarray(data);ids=split(len(z));train=z[ids['train']];candidates=[]
    for rank in [2,4]:
        for lam in [.3,3.]:
            for seed in [0,1]:
                c=fit(train,nu,ni,rank,lam,seed,max_sweeps);c['id']=f'rank-{rank}-lambda-{lam:g}-seed-{seed}';c['train']=metrics(c['model'],train);c['validation']=metrics(c['model'],z[ids['validation']]);candidates.append(c)
    selected=min(candidates,key=lambda c:(c['validation']['rmse'],c['id']))['id']
    bias=[]
    for lam in [.3,3.,30.]:
        c=fit(train,nu,ni,0,lam,0,max_sweeps);c['id']=f'bias-lambda-{lam:g}';c['validation']=metrics(c['model'],z[ids['validation']]);c['train']=metrics(c['model'],train);bias.append(c)
    selected_bias=min(bias,key=lambda c:(c['validation']['rmse'],c['id']))['id']
    # Test evaluation occurs only after BOTH selections, without refitting.
    for c in candidates+bias:c['test']=metrics(c['model'],z[ids['test']])
    mean=float(train[:,2].mean());global_errors=z[ids['test'],2]-mean
    uc=np.bincount(train[:,0],minlength=nu);ic=np.bincount(train[:,1],minlength=ni)
    cold={k:dict(n=len(rows),warm=int(sum(uc[u]>0 and ic[i]>0 for u,i,_ in rows)),cold_user=int(sum(uc[u]==0 for u,_,_ in rows)),cold_item=int(sum(ic[i]==0 for _,i,_ in rows))) for k in ('validation','test') for rows in [z[ids[k]]]}
    return dict(version=1,source_sha256=sha,shape=[nu,ni],edges=z.tolist(),split=ids,candidates=candidates,bias_candidates=bias,selected_id=selected,selected_bias_id=selected_bias,global_test_rmse=float(np.sqrt(np.mean(global_errors**2))),cold=cold,protocol='seed20261108 60/20/20 observed edges; train-only mean and factors; separate validation-RMSE model and bias choices; no refit; all eight candidates and three bias fits retained; ratings0/1/2, missing is not zero',scope='small historical observed-rating prediction only, not top-N ranking, causal preferences or missing-pair accuracy')

def saved():
    r=json.loads((ROOT/'artifacts/report.json').read_text())
    for group in ('candidates','bias_candidates'):r[group]=[json.loads((ROOT/'artifacts'/name).read_text()) for name in r[group]]
    return r

def audit(report):
    source,nu,ni,sha=load()
    if report['shape']!=[nu,ni] or report['source_sha256']!=sha or report['edges']!=source.tolist() or report['split']!=split(len(source)):raise ValueError('source/split differs')
    z=np.asarray(report['edges']);train=z[report['split']['train']];checks=[]
    for c in report['candidates']+report['bias_candidates']:
        m=c['model'];o=objective(m,train);g=gradient(m,train);maxg=max(float(np.max(np.abs(v),initial=0)) for v in g.values());h=c['history']
        if abs(o['objective']-h[-1]['objective'])>1e-9*(1+o['objective']):raise ValueError('objective mismatch')
        if abs(maxg-h[-1]['max_gradient'])>1e-9*(1+maxg):raise ValueError('gradient mismatch')
        if any(b['objective']>a['objective']+1e-8*(1+a['objective']) for a,b in zip(h,h[1:])):raise ValueError('history increased')
        for part in ('train','validation','test'):
            current=metrics(m,z[report['split'][part]])
            if any(abs(current[k]-c[part][k])>1e-10 for k in current):raise ValueError('metric mismatch')
        checks.append(dict(id=c['id'],objective=o['objective'],max_gradient=maxg,status=c['status']))
    if min(report['candidates'],key=lambda c:(c['validation']['rmse'],c['id']))['id']!=report['selected_id']:raise ValueError('selection mismatch')
    if min(report['bias_candidates'],key=lambda c:(c['validation']['rmse'],c['id']))['id']!=report['selected_bias_id']:raise ValueError('bias selection mismatch')
    return dict(valid=True,candidates=checks,scope=report['scope'])
