"""Alternating exact ridge subproblems for one explicit observed-only objective."""
import math
import numpy as np

def validate(edges,n_users,n_items):
    z=np.asarray(edges,dtype=float)
    if z.ndim!=2 or z.shape[1]!=3 or not len(z) or not np.isfinite(z).all():raise ValueError('nonempty finite [user,item,rating] triples required')
    if type(n_users) is not int or type(n_items) is not int or min(n_users,n_items)<1:raise ValueError('positive dimensions required')
    if np.any(z[:,:2]!=np.floor(z[:,:2])) or np.any(z[:,:2]<0) or np.any(z[:,0]>=n_users) or np.any(z[:,1]>=n_items):raise ValueError('invalid user/item index')
    if len(set(map(tuple,z[:,:2])))!=len(z):raise ValueError('duplicate user/item pair')
    return z

def prediction(model,u,i):
    bu=np.asarray(model['user_bias']);bi=np.asarray(model['item_bias']);P=np.asarray(model['users']);Q=np.asarray(model['items']);rank=model['rank']
    u=np.asarray(u);i=np.asarray(i)
    if u.shape!=i.shape or not np.issubdtype(u.dtype,np.integer) or not np.issubdtype(i.dtype,np.integer):raise ValueError('matching integer index arrays required')
    if np.any(u<0) or np.any(u>=len(bu)) or np.any(i<0) or np.any(i>=len(bi)):raise ValueError('unknown user/item index')
    return model['mean']+bu[u]+bi[i]+np.sum(P[u].reshape((-1,rank))*Q[i].reshape((-1,rank)),axis=1) if rank else model['mean']+bu[u]+bi[i]

def arrays(model):return [np.asarray(model[k],dtype=float) for k in ('user_bias','item_bias','users','items')]

def objective(model,edges):
    z=np.asarray(edges);u=z[:,0].astype(int);i=z[:,1].astype(int);e=prediction(model,u,i)-z[:,2]
    sse=float(e@e);pen=model['regularization']*sum(float(np.sum(a*a)) for a in arrays(model))
    return dict(objective=sse+pen,sse=sse,penalty=pen)

def gradient(model,edges):
    z=np.asarray(edges);u=z[:,0].astype(int);i=z[:,1].astype(int);bu,bi,P,Q=arrays(model);lam=model['regularization'];e=prediction(model,u,i)-z[:,2]
    gb=2*lam*bu;gi=2*lam*bi;gp=2*lam*P;gq=2*lam*Q
    np.add.at(gb,u,2*e);np.add.at(gi,i,2*e)
    if model['rank']:
        np.add.at(gp,u,2*e[:,None]*Q[i]);np.add.at(gq,i,2*e[:,None]*P[u])
    return dict(zip(('user_bias','item_bias','users','items'),[gb,gi,gp,gq]))

def ridge(A,target,regularization):
    H=A.T@A+regularization*np.eye(A.shape[1]);rhs=A.T@target
    return np.linalg.solve(H,rhs),H,rhs

def fit(edges,n_users,n_items,rank=2,regularization=3.,seed=0,max_sweeps=80,tolerance=1e-7):
    z=validate(edges,n_users,n_items)
    if type(rank) is not int or not 0<=rank<=32 or isinstance(regularization,bool) or not math.isfinite(regularization) or regularization<=0 or type(max_sweeps) is not int or max_sweeps<0 or not math.isfinite(tolerance) or tolerance<=0:raise ValueError('invalid configuration')
    u=z[:,0].astype(int);i=z[:,1].astype(int);y=z[:,2];rng=np.random.default_rng(seed);P=rng.normal(0,.05,(n_users,rank));Q=rng.normal(0,.05,(n_items,rank));bu=np.zeros(n_users);bi=np.zeros(n_items);mean=float(np.mean(y));lam=float(regularization)
    us=[np.flatnonzero(u==j) for j in range(n_users)];its=[np.flatnonzero(i==j) for j in range(n_items)]
    # Empty entities have an exact zero optimum, including at budget zero.
    for j,ix in enumerate(us):
        if not len(ix):P[j]=0
    for j,ix in enumerate(its):
        if not len(ix):Q[j]=0
    def model():return dict(mean=mean,rank=rank,regularization=lam,user_bias=bu.tolist(),item_bias=bi.tolist(),users=P.tolist(),items=Q.tolist())
    def summary(stage,sweep,residual=0.,decrease=0.):
        m=model();o=objective(m,z);g=gradient(m,z);return dict(stage=stage,sweep=sweep,**o,max_gradient=max(float(np.max(np.abs(v),initial=0)) for v in g.values()),max_block_normal_residual=residual,block_decrease_sum=decrease)
    history=[summary('initial',0)];status='budget_exhausted'
    for sweep in range(1,max_sweeps+1):
        for side,groups in [('user',us),('item',its)]:
            residual=0.;decrease=0.;before=objective(model(),z)['objective']
            for j,ix in enumerate(groups):
                other=(Q[i[ix]] if side=='user' else P[u[ix]]);bias=(bi[i[ix]] if side=='user' else bu[u[ix]]);A=np.column_stack([np.ones(len(ix)),other]);target=y[ix]-mean-bias
                value,H,rhs=ridge(A,target,lam);old=np.r_[bu[j],P[j]] if side=='user' else np.r_[bi[j],Q[j]]
                oldloss=float(np.sum((A@old-target)**2)+lam*(old@old));newloss=float(np.sum((A@value-target)**2)+lam*(value@value));delta=oldloss-newloss
                if delta < -1e-9*(1+oldloss):raise ArithmeticError('block objective increased')
                decrease+=delta;residual=max(residual,float(np.max(np.abs(H@value-rhs),initial=0)))
                if side=='user':bu[j]=value[0];P[j]=value[1:]
                else:bi[j]=value[0];Q[j]=value[1:]
            after=summary(side,sweep,residual,decrease)
            if abs(before-after['objective']-decrease)>1e-8*(1+before):raise ArithmeticError('objective telescope differs')
            history.append(after)
        if history[-1]['max_gradient']<=tolerance:status='gradient_tolerance';break
    return dict(model=model(),history=history,status=status,sweeps=history[-1]['sweep'],seed=seed,tolerance=tolerance,scope='nonconvex observed-rating objective; exact ridge blocks up to numerical error, not global optimum or top-N utility')

def block_witness(model,edges,side,index):
    z=np.asarray(edges);u=z[:,0].astype(int);i=z[:,1].astype(int);bu,bi,P,Q=arrays(model);r=model['rank'];ix=np.flatnonzero((u if side=='user' else i)==index)
    A=np.column_stack([np.ones(len(ix)),Q[i[ix]] if side=='user' else P[u[ix]]]);t=z[ix,2]-model['mean']-(bi[i[ix]] if side=='user' else bu[u[ix]])
    H=A.T@A+model['regularization']*np.eye(r+1);rhs=A.T@t;point=np.r_[bu[index],P[index]] if side=='user' else np.r_[bi[index],Q[index]]
    return dict(side=side,index=index,observations=ix.tolist(),normal=H.tolist(),rhs=rhs.tolist(),point=point.tolist(),gradient=(2*(H@point-rhs)).tolist())
