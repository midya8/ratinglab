import copy,json,math,subprocess,sys,tempfile,unittest
from pathlib import Path
from fractions import Fraction as F
import numpy as np
from oracle import solve as exact
from ratinglab.influence import response,block_response,references
from ratinglab.experiment import saved
from ratinglab.core import fit,ridge

class InfluenceTests(unittest.TestCase):
    def test_sixty_independent_fraction_edits_and_deletions(self):
        rng=np.random.default_rng(9811)
        for k in range(60):
            n=2+k%7;d=1+k%4;A=rng.integers(-3,4,(n,d));t=rng.integers(-4,5,n);lam=1+k%3;row=k%n;q=rng.integers(-2,3,d);delta=F(1,2)
            for delete in [False,True]:
                r=response(A,t,lam,row,q,0 if delete else float(delta),delete)
                rows=[list(map(int,a)) for j,a in enumerate(A) if not delete or j!=row];target=[F(int(v))+(delta if not delete and j==row else 0) for j,v in enumerate(t) if not delete or j!=row]
                want=exact(rows,target,lam);np.testing.assert_allclose(r['point'],list(map(float,want)),atol=2e-12,rtol=2e-12)
                self.assertAlmostEqual(r['score'],float(sum(F(int(v))*w for v,w in zip(q,want))),places=10)
                self.assertGreaterEqual(r['deletion_denominator']+2e-14,r['denominator_lower_bound']);self.assertLess(r['leverage'],1);self.assertGreaterEqual(r['leverage'],-1e-14)
                self.assertLess(r['max_normal_residual'],1e-10)
    def test_single_row_deletion_and_zero_design(self):
        r=response([[1,2]],[3],.3,0,[1,-2],delete=True);np.testing.assert_allclose(r['point'],[0,0],atol=3e-14);self.assertAlmostEqual(r['deleted_row_residual'],3)
        z=response([[0,0]],[9],.3,0,[1,2],delete=True);self.assertEqual(z['leverage'],0);self.assertEqual(z['score'],0)
    def test_finite_edit_derivative_and_deleted_residual(self):
        A=np.array([[1,2],[1,-1],[1,3.] ]);t=np.array([2.,0.,1.]);q=[1.,4.]
        r=response(A,t,2.,1,q,delta=.7);base=response(A,t,2.,1,q);self.assertAlmostEqual(r['score_change'],.7*r['query_derivative'])
        plus=response(A,t,2.,1,q,delta=1e-5);minus=response(A,t,2.,1,q,delta=-1e-5);self.assertAlmostEqual((plus['score']-minus['score'])/2e-5,r['query_derivative'],places=9)
        deletion=response(A,t,2.,1,A[1],delete=True);self.assertAlmostEqual(t[1]-deletion['score'],deletion['deleted_row_residual'])
    def test_all_2784_real_deletions_against_fresh_block_solves(self):
        data=saved();ids={data['selected_id'],data['selected_bias_id']};n=0
        for fit_ in data['candidates']+data['bias_candidates']:
            if fit_['id'] not in ids:continue
            m=fit_['model']
            for source in data['split']['train']:
                u,i,y=data['edges'][source]
                for side,index,query in [('user',u,i),('item',i,u)]:
                    r=block_response(data,m,side,index,source,query,delete=True);rows=[j for j in r['source_rows'] if j!=source];other='items' if side=='user' else 'users';bias='item_bias' if side=='user' else 'user_bias';o=1 if side=='user' else 0
                    A=np.array([[1,*m[other][data['edges'][j][o]]] for j in rows]).reshape((-1,m['rank']+1));t=np.array([data['edges'][j][2]-m['mean']-m[bias][data['edges'][j][o]] for j in rows]);point,H,b=ridge(A,t,m['regularization']);np.testing.assert_allclose(point,r['point'],atol=2e-12,rtol=2e-12);self.assertGreaterEqual(r['deletion_denominator']+1e-14,r['denominator_lower_bound']);n+=1
        self.assertEqual(n,2784)
    def test_saved_source_unchanged_and_reference_reproduction(self):
        root=Path(__file__).resolve().parents[1];before={str(p):p.read_bytes() for p in (root/'ratinglab/artifacts').glob('*.json')};refs=json.loads(before[str(root/'ratinglab/artifacts/influence-reference.json')])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'refs.json';actual=references(path)
            self.assertEqual(json.loads(path.read_text()),actual)
            metadata={'model_id','side','index','source_row','query_other','delete','delta'}
            def compare(got,want):
                self.assertEqual(len(got),len(want))
                for a,b in zip(got,want):
                    self.assertEqual(a.keys(),b.keys())
                    for key in a:
                        if key in metadata:
                            self.assertIs(type(a[key]),type(b[key]));self.assertEqual(a[key],b[key])
                        else:
                            self.assertTrue(np.isfinite(np.asarray(a[key])).all(),key)
                            np.testing.assert_allclose(a[key],b[key],atol=2e-10,rtol=2e-10,err_msg=key)
            compare(actual,refs)
            # Independent platform rounding is tolerated; meaningful drift,
            # changed discrete metadata and non-finite results must still fail.
            for key,value in [('source_row',-1),('score',actual[0]['score']+1e-5),('score',float('nan'))]:
                broken=copy.deepcopy(actual);broken[0][key]=value
                with self.assertRaises(AssertionError):compare(broken,refs)
            rounded=copy.deepcopy(actual);rounded[0]['score']+=1e-12
            compare(rounded,refs)
        self.assertEqual(len(refs),576);self.assertEqual(before,{str(p):p.read_bytes() for p in (root/'ratinglab/artifacts').glob('*.json')})
        # User blocks can differ from the saved state after the last item pass.
        self.assertGreater(max(r['saved_to_baseline_max_change'] for r in refs if r['side']=='user'),1e-12)
    def test_not_full_retraining(self):
        edges=[[0,0,2],[0,1,0],[1,0,0],[1,1,2]];a=fit(edges,2,2,max_sweeps=8);data={'edges':edges,'split':{'train':[0,1,2,3]}};r=block_response(data,a['model'],'user',0,0,0,delta=-1)
        changed=copy.deepcopy(edges);changed[0][2]=1;b=fit(changed,2,2,max_sweeps=8)
        self.assertNotEqual(b['model']['mean'],a['model']['mean']);self.assertNotEqual(b['model']['items'],a['model']['items']);self.assertIn('Not full ALS',r['scope'])
    def test_invalid_and_module_commands(self):
        for kw in [dict(row=True),dict(row=-1),dict(row=1),dict(regularization=0),dict(regularization=True),dict(delta=True),dict(delta=float('nan')),dict(delete=1),dict(delete=True,delta=.1)]:
            args=dict(A=[[1]],target=[1],regularization=1.,row=0,query=[1]);args.update(kw)
            with self.assertRaises(ValueError):response(**args)
        data=saved();m=data['candidates'][0]['model'];j=data['split']['test'][0]
        for side,index,row,query in [('other',0,0,0),('user',True,0,0),('user',0,j,0),('user',0,0,True)]:
            with self.assertRaises(ValueError):block_response(data,m,side,index,row,query)
        cp=subprocess.run([sys.executable,'-m','ratinglab.influence','demo'],capture_output=True,text=True,check=True);r=json.loads(cp.stdout);self.assertTrue(r['delete']);self.assertLess(r['max_normal_residual'],1e-10)
if __name__=='__main__':unittest.main()
