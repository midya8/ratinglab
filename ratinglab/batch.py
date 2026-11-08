"""Finite conditional batch ridge response, numerical not full ALS retraining."""
import math
from numbers import Real
import numpy as np
from .influence import SCOPE


def _numeric(values):
    if any(isinstance(v,(bool,np.bool_)) or not isinstance(v,Real) for v in values):
        raise ValueError('real numeric values, not Booleans required')


def response(design, target, regularization, rows, query, deltas=None, delete=False):
    """Joint edit or joint deletion using a small Schur system, never an inverse.

    All rows refer to the original design. Separate deletion responses share
    the same original baseline; their sum is a contrast, not the joint answer.
    """
    raw=np.asarray(design,dtype=object)
    if raw.ndim!=2 or not raw.shape[0] or not raw.shape[1] or raw.shape[1]>33:
        raise ValueError('nonempty design with at most 33 columns required')
    _numeric(raw.flat);_numeric(np.asarray(target,dtype=object).flat);_numeric(np.asarray(query,dtype=object).flat)
    A=np.asarray(design,dtype=float);t=np.asarray(target,dtype=float);q=np.asarray(query,dtype=float)
    if t.shape!=(len(A),) or q.shape!=(A.shape[1],) or not all(np.isfinite(v).all() for v in (A,t,q)):
        raise ValueError('finite matching design/target/query required')
    if isinstance(regularization,bool) or not isinstance(regularization,Real) or not math.isfinite(regularization) or regularization<=0 or type(delete) is not bool:
        raise ValueError('positive finite penalty and Boolean deletion required')
    if not isinstance(rows,list) or len(rows)>64 or any(type(k) is not int or not 0<=k<len(A) for k in rows) or len(rows)!=len(set(rows)):
        raise ValueError('at most 64 distinct original integer row indices required')
    if delete:
        if deltas is not None:
            raise ValueError('deletion cannot also edit targets')
        changes=np.zeros(len(rows))
    else:
        if deltas is None:deltas=[0.]*len(rows)
        _numeric(np.asarray(deltas,dtype=object).flat);changes=np.asarray(deltas,dtype=float)
        if changes.shape!=(len(rows),) or not np.isfinite(changes).all():
            raise ValueError('one finite target delta per selected row required')
    lam=float(regularization);d=A.shape[1];H=A.T@A+lam*np.eye(d);rhs=A.T@t
    if not np.isfinite(H).all() or not np.isfinite(rhs).all():raise ArithmeticError('normal system overflow')
    try:
        base=np.linalg.solve(H,rhs);B=A[rows];Z=np.linalg.solve(H,B.T)
    except np.linalg.LinAlgError as exc:raise ArithmeticError('unusable numerical baseline system') from exc
    if not np.isfinite(base).all() or not np.isfinite(Z).all():raise ArithmeticError('nonfinite baseline response')
    K=B@Z;schur=np.eye(len(rows))-K;residual=t[rows]-B@base
    schur=(schur+schur.T)/2
    if len(rows):
        eigen=np.linalg.eigvalsh(schur);minimum=float(eigen[0]);condition=float(np.linalg.cond(schur,np.inf))
        # The true Schur matrix is SPD for positive ridge. Lost positivity or
        # severe conditioning is a numerical refusal, never repaired by jitter.
        if delete and (minimum<=0 or condition>1e12):
            raise ArithmeticError('Schur system lost numerical positivity or conditioning')
        if delete and np.any(np.diag(schur)<=0):raise ArithmeticError('invalid single-deletion denominator')
    else:minimum=None;condition=None
    coefficient=-np.linalg.solve(schur,residual) if delete and rows else changes
    point=base+Z@coefficient
    individual=(-residual/np.diag(schur)) if delete else changes
    separate=base+Z@individual
    retained=np.ones(len(A),dtype=bool);retained[rows]=not delete
    # Direct normal system built from remaining rows, not H minus Gram.
    RA=A[retained];rt=t[retained].copy()
    if not delete:rt[rows]+=changes
    changed_H=RA.T@RA+lam*np.eye(d);changed_rhs=RA.T@rt
    norm_residual=float(np.max(np.abs(changed_H@point-changed_rhs)))
    scale=1+float(np.max(np.abs(changed_rhs)))+float(np.max(np.abs(changed_H)))*d*float(np.max(np.abs(point)))
    relative=norm_residual/scale
    if not np.isfinite(point).all() or not all(math.isfinite(v) for v in (norm_residual,scale,relative)) or relative>1e-10:
        raise ArithmeticError('updated normal equation residual too large')
    gram_upper=float(np.sum(A*A))
    if not math.isfinite(gram_upper):raise ArithmeticError('nonfinite norm bound')
    return {'schema':1,'rows':rows.copy(),'delete':delete,'deltas':None if delete else changes.tolist(),
            'design':A.tolist(),'target':t.tolist(),'regularization':lam,'query':q.tolist(),
            'normal':H.tolist(),'rhs':rhs.tolist(),'baseline':base.tolist(),'directions':Z.tolist(),
            'leverage_block':K.tolist(),'schur':schur.tolist(),'minimum_schur_eigenvalue':minimum,
            'schur_eigenvalue_lower_bound':lam/(lam+gram_upper),'schur_condition':condition,
            'selected_residuals':residual.tolist(),'coefficient':coefficient.tolist(),
            'point':point.tolist(),'separate_sum_point':separate.tolist(),
            'baseline_score':float(q@base),'score':float(q@point),
            'score_change':float(q@(point-base)),'separate_score_change':float(q@(separate-base)),
            'interaction':float(q@(point-separate)),
            'maximum_normal_residual':norm_residual,'scaled_normal_residual':relative,
            'scope':SCOPE,'arithmetic':'binary64 numerical solves; residual checks, not exact arithmetic'}


def block_response(data,model,side,index,source_rows,query_other,deltas=None,delete=False):
    """Full source IDs bind only to training observations in ONE selected block."""
    if side not in ('user','item'):
        raise ValueError('side must be user or item')
    own='users' if side=='user' else 'items';opposite='items' if side=='user' else 'users'
    ownbias='user_bias' if side=='user' else 'item_bias';otherbias='item_bias' if side=='user' else 'user_bias'
    if type(index) is not int or not 0<=index<len(model[own]) or type(query_other) is not int or not 0<=query_other<len(model[opposite]):
        raise ValueError('known block and query required')
    train=data['split']['train'];sources=[j for j in train if data['edges'][j][0 if side=='user' else 1]==index]
    if not isinstance(source_rows,list) or any(type(j) is not int or j not in sources for j in source_rows) or len(source_rows)!=len(set(source_rows)):
        raise ValueError('distinct training source IDs within selected block required')
    others=[data['edges'][j][1 if side=='user' else 0] for j in sources]
    A=[[1.,*model[opposite][j]] for j in others]
    t=[data['edges'][j][2]-model['mean']-model[otherbias][o] for j,o in zip(sources,others)]
    if not delete:
        if deltas is None:deltas=[0.]*len(source_rows)
        _numeric(np.asarray(deltas,dtype=object).flat)
        if not isinstance(deltas,list) or len(deltas)!=len(source_rows) or any(not math.isfinite(v) or not 0<=data['edges'][j][2]+v<=2 for j,v in zip(source_rows,deltas)):
            raise ValueError('edited ratings must remain in [0,2]')
    q=[1.,*model[opposite][query_other]]
    out=response(A,t,model['regularization'],[sources.index(j) for j in source_rows],q,deltas,delete)
    offset=model['mean']+model[otherbias][query_other];saved=np.array([model[ownbias][index],*model[own][index]])
    out.update(side=side,index=index,source_rows=source_rows.copy(),block_source_rows=sources,query_other=query_other,
               original_ratings=[data['edges'][j][2] for j in source_rows],offset=offset,
               saved_score=float(offset+np.dot(q,saved)),baseline_score=out['baseline_score']+offset,
               score=out['score']+offset,saved_to_baseline_max_change=float(np.max(np.abs(saved-np.asarray(out['baseline'])))))
    return out
