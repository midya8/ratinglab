import unittest,copy,random
import numpy as np
from fractions import Fraction as F
from ratinglab.core import fit,objective,gradient,prediction,ridge,block_witness
from ratinglab.controls import controls
from oracle import solve

class CoreTests(unittest.TestCase):
 def test_sixty_exact_ridge_references(self):
    rng=random.Random(4581)
    for case in range(60):
     d=1+case%5;A=[[rng.randint(-3,3) for _ in range(d)] for _ in range(2+case%7)];y=[F(rng.randint(-5,5),2) for _ in A];lam=F(rng.choice([1,3,5]),4);truth=solve(A,y,lam);actual,H,b=ridge(np.array(A,dtype=float),np.array(y,dtype=float),float(lam));np.testing.assert_allclose(actual,list(map(float,truth)),rtol=1e-11,atol=1e-11)
 def test_every_gradient_coordinate_central_difference(self):
    edges=[[0,0,2],[0,1,0],[1,0,1]];m=fit(edges,2,2,2,1.,seed=2,max_sweeps=1)['model'];g=gradient(m,edges)
    for key in g:
     a=np.asarray(m[key]);
     for ix in np.ndindex(a.shape):
      plus=copy.deepcopy(m);minus=copy.deepcopy(m);p=np.asarray(plus[key]);q=np.asarray(minus[key]);p[ix]+=1e-5;q[ix]-=1e-5;plus[key]=p.tolist();minus[key]=q.tolist();num=(objective(plus,edges)['objective']-objective(minus,edges)['objective'])/2e-5;self.assertAlmostEqual(num,g[key][ix],places=7)
 def test_blocks_monotone_and_telescope(self):
    c=fit([[0,0,2],[0,1,0],[1,0,0],[1,1,2]],2,2,2,.3,max_sweeps=20)
    for a,b in zip(c['history'],c['history'][1:]):self.assertLessEqual(b['objective'],a['objective']+1e-12);self.assertAlmostEqual(a['objective']-b['objective'],b['block_decrease_sum'],places=10);self.assertLess(b['max_block_normal_residual'],1e-12)
 def test_final_item_normal_equations(self):
    edges=[[0,0,2],[0,1,0],[1,0,1]];m=fit(edges,2,2,2,1.,max_sweeps=3)['model']
    for j in range(2):
     w=block_witness(m,edges,'item',j);np.testing.assert_allclose(np.array(w['normal'])@w['point'],w['rhs'],atol=1e-12)
 def test_zero_is_observed_not_missing(self):
    c=fit([[0,0,0],[0,1,2]],1,2,0,1.,max_sweeps=1);self.assertEqual(c['model']['mean'],1.);self.assertNotEqual(c['model']['item_bias'][0],0)
 def test_cold_start_exact_zero_optimum(self):
    m=fit([[0,0,2],[0,1,0]],2,3,2,1.,max_sweeps=3)['model'];self.assertEqual(m['users'][1],[0.,0.]);self.assertEqual(m['items'][2],[0.,0.]);self.assertEqual(m['user_bias'][1],0.);self.assertEqual(m['item_bias'][2],0.);self.assertEqual(prediction(m,[1],[2])[0],m['mean'])
 def test_zero_budget_is_explicit(self):
    c=fit([[0,0,2]],1,1,max_sweeps=0);self.assertEqual(c['status'],'budget_exhausted');self.assertEqual(len(c['history']),1)
 def test_stationary_is_not_global_minimum(self):
    c=controls()['stationary_not_optimal'];self.assertEqual(c['zero_gradient'],0.);self.assertEqual(c['stationary_objective'],4.);self.assertAlmostEqual(c['lower_objective'],.4)
 def test_rotation_preserves_predictions_not_terms(self):
    c=controls()['rotation'];self.assertLess(c['max_prediction_difference'],1e-14);self.assertNotEqual(c['first_terms'],c['second_terms']);self.assertAlmostEqual(sum(c['first_terms']),sum(c['second_terms']))
 def test_strict_bad_edges_and_configs(self):
    for edges in ([],[[.1,0,1]],[[0,-1,1]],[[2,0,1]],[[0,0,float('nan')]],[[0,0,1],[0,0,2]]):
     with self.assertRaises(ValueError):fit(edges,2,2)
    for kw in ({'rank':-1},{'rank':True},{'regularization':0},{'regularization':float('inf')},{'max_sweeps':-1},{'tolerance':0}):
     with self.assertRaises(ValueError):fit([[0,0,1]],1,1,**kw)
 def test_prediction_rejects_bad_indices(self):
    m=fit([[0,0,1]],1,1)['model']
    for u,i in [([.5],[0]),([True],[0]),([0],[-1]),([1],[0]),([0,0],[0])]:
     with self.assertRaises(ValueError):prediction(m,u,i)
