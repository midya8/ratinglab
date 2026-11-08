import math
import copy,itertools,math,random,unittest
from fractions import Fraction as F
import numpy as np
from ratinglab.batch import response,block_response
from ratinglab.influence import response as single
from ratinglab.experiment import saved


def exact(A,t,lam):
    d=len(A[0]);H=[[sum(F(row[i])*F(row[j]) for row in A)+F(lam)*int(i==j) for j in range(d)] for i in range(d)]
    rhs=[sum(F(row[i])*F(v) for row,v in zip(A,t)) for i in range(d)]
    a=[row+[v] for row,v in zip(H,rhs)]
    for j in range(d):
        pivot=next(k for k in range(j,d) if a[k][j]);a[j],a[pivot]=a[pivot],a[j]
        div=a[j][j];a[j]=[v/div for v in a[j]]
        for k in range(d):
            if k!=j:
                mul=a[k][j];a[k]=[v-mul*w for v,w in zip(a[k],a[j])]
    return [row[-1] for row in a]


class BatchTests(unittest.TestCase):
    def test_320_rational_direct_deletion_and_edit_oracles(self):
        rng=random.Random(1943);count=0
        for _ in range(10):
            A=[[rng.randrange(-4,5)/2 for _ in range(3)] for _ in range(4)]
            t=[rng.randrange(-4,5)/2 for _ in A];q=[rng.randrange(-4,5)/2 for _ in range(3)];lam=.5
            for flags in itertools.product([False,True],repeat=4):
                rows=[k for k,v in enumerate(flags) if v]
                for delete in [False,True]:
                    deltas=[(k+1)/4 for k in rows]
                    w=response(A,t,lam,rows,q,None if delete else deltas,delete)
                    if delete:
                        # Add explicit zero design row so the oracle still knows d.
                        keep=[k for k in range(len(A)) if k not in rows]
                        AA=[A[k] for k in keep] or [[0,0,0]];tt=[t[k] for k in keep] or [0]
                    else:
                        AA=A;tt=[v+(deltas[rows.index(k)] if k in rows else 0) for k,v in enumerate(t)]
                    truth=exact(AA,tt,lam)
                    np.testing.assert_allclose(w['point'],list(map(float,truth)),rtol=1e-11,atol=2e-12)
                    if rows:self.assertGreaterEqual(w['minimum_schur_eigenvalue']+1e-13,w['schur_eigenvalue_lower_bound'])
                    self.assertLess(w['scaled_normal_residual'],1e-12)
                    if not delete:self.assertAlmostEqual(w['interaction'],0,places=13)
                    count+=1
        self.assertEqual(count,320)

    def test_nonadditivity_all_deleted_empty_and_single_equivalence(self):
        A=[[1.,1.],[1.,1.],[1.,0.]];t=[1.,2.,0.];q=[1.,1.]
        w=response(A,t,1.,[0,1],q,delete=True)
        self.assertAlmostEqual(w['interaction'],-.675,places=13)
        np.testing.assert_allclose(response(A,t,1.,[0,1,2],q,delete=True)['point'],[0,0],atol=1e-14)
        np.testing.assert_allclose(response(A,t,1.,[],q,delete=True)['point'],response(A,t,1.,[],q)['baseline'])
        for delete in [False,True]:
            w=response(A,t,1.,[1],q,None if delete else [.5],delete)
            old=single(A,t,1.,1,q,0 if delete else .5,delete)
            np.testing.assert_allclose(w['point'],old['point'],atol=1e-14)
        perm=response(A,t,1.,[1,0],q,delete=True)
        np.testing.assert_allclose(w['baseline'],perm['baseline'],atol=1e-14)
        np.testing.assert_allclose(perm['point'],response(A,t,1.,[0,1],q,delete=True)['point'],atol=1e-14)

    def test_source_bound_models_and_direct_remaining_row_solves(self):
        data=saved();before=copy.deepcopy(data);count=0
        for fit in data['candidates']+data['bias_candidates']:
            if fit['id'] not in [data['selected_id'],data['selected_bias_id']]:continue
            m=fit['model']
            for anchor in data['split']['train'][:16]:
                u,i,_=data['edges'][anchor]
                for side,index in [('user',u),('item',i)]:
                    sources=[j for j in data['split']['train'] if data['edges'][j][0 if side=='user' else 1]==index]
                    rows=sources[:3]
                    for delete in [False,True]:
                        w=block_response(data,m,side,index,rows,0,None if delete else [1-data['edges'][j][2] for j in rows],delete)
                        A=np.asarray(w['design']);t=np.asarray(w['target']);mask=np.ones(len(A),dtype=bool)
                        if delete:mask[w['rows']]=False
                        else:t[w['rows']]+=w['deltas']
                        direct=np.linalg.lstsq(np.vstack([A[mask],np.sqrt(m['regularization'])*np.eye(A.shape[1])]),np.r_[t[mask],np.zeros(A.shape[1])],rcond=None)[0]
                        np.testing.assert_allclose(w['point'],direct,rtol=1e-11,atol=1e-11)
                        count+=1
        self.assertEqual(count,128);self.assertEqual(data,before)

    def test_export_reproduction_and_source_label_poisoning(self):
        import json
        from pathlib import Path
        from ratinglab.batch_export import export
        root=Path(__file__).resolve().parents[1]
        def check(a,b):
            if isinstance(b,dict):
                self.assertEqual(a.keys(),b.keys())
                for k in b:check(a[k],b[k])
            elif isinstance(b,list):
                self.assertEqual(len(a),len(b))
                for x,y in zip(a,b):check(x,y)
            elif type(b) is float:self.assertTrue(math.isclose(a,b,rel_tol=2e-10,abs_tol=2e-10),(a,b))
            else:self.assertEqual(a,b)
        check(export(),json.loads((root/'ratinglab/artifacts/batch-reference.json').read_text()))
        data=saved();fit=next(c for c in data['candidates'] if c['id']==data['selected_id']);m=fit['model']
        source=data['split']['train'][0];u,_,_=data['edges'][source]
        before=block_response(data,m,'user',u,[source],0,delete=True)
        poisoned=copy.deepcopy(data)
        for j in data['split']['validation']+data['split']['test']:
            poisoned['edges'][j][2]=987654
        self.assertEqual(before,block_response(poisoned,m,'user',u,[source],0,delete=True))
        for rows in [[True],[source,source],[data['split']['test'][0]],[-1]]:
            with self.assertRaises(ValueError):block_response(data,m,'user',u,rows,0,delete=True)

    def test_numerically_lost_deletion_positivity_refuses_without_jitter(self):
        with self.assertRaises(ArithmeticError):response([[1],[1]],[1,2],1e-16,[0,1],[1],delete=True)
        with self.assertRaises(ArithmeticError):response([[1.,1.],[1.,1.]],[1,2],1e-300,[],[1,1])
        # Target-only edits do not invert the nearly singular deletion Schur.
        self.assertAlmostEqual(response([[1],[1]],[1,2],1e-16,[0,1],[1],[1,0])['point'][0],2.)

    def test_installed_style_module_commands_and_strict_json(self):
        import tempfile,subprocess,sys,json
        from pathlib import Path
        problem=dict(design=[[1,1],[1,1],[1,0]],target=[1,2,0],regularization=1,rows=[0,1],query=[1,1],delete=True)
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'problem.json';p.write_text(json.dumps(problem))
            run=subprocess.run([sys.executable,'-m','ratinglab','batch','--input',str(p)],capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr);self.assertAlmostEqual(json.loads(run.stdout)['interaction'],-.675)
            p.write_text('{"design":[],"design":[]}')
            run=subprocess.run([sys.executable,'-m','ratinglab','batch','--input',str(p)],capture_output=True,text=True)
            self.assertEqual(run.returncode,2);self.assertIn('duplicate key',run.stderr)
        run=subprocess.run([sys.executable,'-m','ratinglab','batch-export'],capture_output=True,text=True)
        self.assertEqual(run.returncode,0,run.stderr);self.assertEqual(len(json.loads(run.stdout)['cases']),320)

    def test_refusals_and_nonmutation(self):
        A=[[1,2],[3,4]];t=[1,0];q=[1,1];old=copy.deepcopy((A,t,q))
        response(A,t,1.,[0],q,[.5]);self.assertEqual((A,t,q),old)
        for rows in [[0,0],[True],[-1],[2],(0,)]:
            with self.assertRaises(ValueError):response(A,t,1.,rows,q)
        for lam in [0,-1,True,math.nan]:
            with self.assertRaises(ValueError):response(A,t,lam,[],q)
        for delta in [[True],[math.nan],[],[1,2]]:
            with self.assertRaises(ValueError):response(A,t,1.,[0],q,delta)
        with self.assertRaises(ValueError):response(A,t,1.,[0],q,[0],True)
        with self.assertRaises(ValueError):response([[True,2],[3,4]],t,1.,[],q)


if __name__=='__main__':unittest.main()
