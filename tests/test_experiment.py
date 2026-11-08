import unittest,copy,json,tempfile,importlib.util
from pathlib import Path
import numpy as np
from ratinglab.dataset import load,split
from ratinglab.experiment import saved,run,audit,metrics
from ratinglab.core import prediction
from ratinglab.export import build

class ExperimentTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.r=saved();cls.fresh=run()
 def test_source_hash_shape_and_split(self):
    z,nu,ni,sha=load();self.assertEqual((len(z),nu,ni),(1161,138,130));self.assertEqual(sha,'fc994ea4e872365fc63e7f0bc970b36f86edeb4e80218d6a674a3d0e51181901');ids=split(len(z));self.assertEqual(list(map(len,ids.values())),[696,232,233]);self.assertEqual(len(set(sum(ids.values(),[]))),1161);self.assertEqual(set(z[:,2]),{0,1,2})
 def test_all_eleven_fresh_fits_and_audits(self):
    a=self.r;b=self.fresh;self.assertEqual(a['selected_id'],b['selected_id']);self.assertEqual(a['selected_bias_id'],b['selected_bias_id']);self.assertTrue(audit(a)['valid']);self.assertTrue(audit(b)['valid'])
    for x,y in zip(a['candidates']+a['bias_candidates'],b['candidates']+b['bias_candidates']):
     self.assertEqual(x['id'],y['id']);self.assertEqual(x['status'],y['status']);self.assertAlmostEqual(x['test']['rmse'],y['test']['rmse'],places=6);self.assertAlmostEqual(x['history'][-1]['objective'],y['history'][-1]['objective'],places=5)
 def test_test_poison_does_not_change_fits_or_selection(self):
    z=load()[0].copy();z[split(len(z))['test'],2]=99;other=run(data=z)
    self.assertEqual(other['selected_id'],self.fresh['selected_id']);self.assertEqual(other['selected_bias_id'],self.fresh['selected_bias_id'])
    for a,b in zip(other['candidates']+other['bias_candidates'],self.fresh['candidates']+self.fresh['bias_candidates']):self.assertEqual(a['model'],b['model']);self.assertEqual(a['history'],b['history']);self.assertEqual(a['validation'],b['validation'])
 def test_stronger_bias_control_retained_and_wins_this_test(self):
    r=self.r;c=next(c for c in r['candidates'] if c['id']==r['selected_id']);b=next(c for c in r['bias_candidates'] if c['id']==r['selected_bias_id']);self.assertLess(b['validation']['rmse'],c['validation']['rmse']);self.assertLess(b['test']['rmse'],c['test']['rmse']);self.assertEqual(len(r['candidates']),8);self.assertEqual(len(r['bias_candidates']),3)
 def test_actual_cold_counts_and_original_range(self):
    for d in self.r['cold'].values():self.assertEqual(d['n'],d['warm']);self.assertEqual(d['cold_user'],0);self.assertEqual(d['cold_item'],0)
    self.assertTrue(all(c['status']=='budget_exhausted' for c in self.r['candidates']))
 def test_browser_export_is_exact_and_reference_matches_native(self):
    root=Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as d:
     target=Path(d);(target/'docs').mkdir();build(target);self.assertEqual((target/'docs/data.js').read_bytes(),(root/'docs/data.js').read_bytes())
    refs=json.loads((root/'ratinglab/artifacts/browser-reference.json').read_text());z=np.array(self.r['edges'])
    for ref,c in zip(refs,self.r['candidates']+self.r['bias_candidates']):np.testing.assert_allclose(ref['scores'],prediction(c['model'],z[:,0],z[:,1]),atol=1e-12)
 def test_tampered_report_audit_rejects(self):
    p=copy.deepcopy(self.r);p['candidates'][0]['test']['rmse']+=1
    with self.assertRaises(ValueError):audit(p)

 def test_source_split_and_bias_selection_tampering_rejected(self):
    for change in ('source','split','bias'):
     r=copy.deepcopy(self.r)
     if change=='source':r['edges'][0][2]=99
     elif change=='split':r['split']['train'][0]=r['split']['test'][0]
     else:r['selected_bias_id']='bias-lambda-30'
     with self.assertRaises(ValueError):audit(r)
