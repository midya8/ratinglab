"""Supplied mathematical controls, not causal component ablations."""
import numpy as np
from .core import objective,gradient,prediction

def controls():
    edges=np.array([[0,0,2],[0,1,0],[1,0,0],[1,1,2]])
    zero=dict(mean=1.,rank=1,regularization=.1,user_bias=[0.,0.],item_bias=[0.,0.],users=[[0.],[0.]],items=[[0.],[0.]])
    better=dict(zero,users=[[1.],[-1.]],items=[[1.],[-1.]])
    angle=np.pi/4;R=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    P=np.array([[1.,2.],[-1.,.5]]);Q=np.array([[2.,-1.],[.3,1.2]])
    first=dict(mean=1.,rank=2,regularization=.1,user_bias=[.1,0.],item_bias=[0.,.2],users=P.tolist(),items=Q.tolist());second=dict(first,users=(P@R).tolist(),items=(Q@R).tolist())
    return dict(stationary_not_optimal=dict(zero_gradient=max(float(np.max(np.abs(v),initial=0)) for v in gradient(zero,edges).values()),stationary_objective=objective(zero,edges)['objective'],lower_objective=objective(better,edges)['objective'],zero=zero,better=better,edges=edges.tolist()),rotation=dict(first=first,second=second,first_terms=(P[0]*Q[0]).tolist(),second_terms=((P@R)[0]*(Q@R)[0]).tolist(),max_prediction_difference=float(np.max(np.abs(prediction(first,edges[:,0],edges[:,1])-prediction(second,edges[:,0],edges[:,1]))))),scope='supplied descriptive algebra controls, not causal ablations or data-generating truth')
