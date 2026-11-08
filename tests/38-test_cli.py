import unittest,subprocess,sys,tempfile,json
from pathlib import Path
class CLITests(unittest.TestCase):
 def call(self,*args):return subprocess.run([sys.executable,'-m','ratinglab',*args],capture_output=True,text=True)
 def test_report_audit_controls_witness(self):
    for command in ['report','audit','controls','witness']:
     p=self.call(command);self.assertEqual(p.returncode,0,p.stderr);json.loads(p.stdout)
 def test_predict_and_strict_errors(self):
    with tempfile.TemporaryDirectory() as d:
     f=Path(d)/'input.json';f.write_text('{"pairs":[[0,0],[137,129]]}');a=self.call('predict','--input',str(f));self.assertEqual(a.returncode,0,a.stderr);self.assertEqual(len(json.loads(a.stdout)['predictions']),2)
     for body in ['{"pairs":[[true,0]]}','{"pairs":[[.5,0]]}','{"pairs":[[-1,0]]}','{"pairs":[]}','{"pairs":[[138,0]]}','{"pairs":[[0,0]],"extra":1}','{"pairs":[],"pairs":[]}','{"pairs":[[NaN,0]]}']:
      f.write_text(body);self.assertEqual(self.call('predict','--input',str(f)).returncode,2)
 def test_budget_zero_fit_json_roundtrip(self):
    with tempfile.TemporaryDirectory() as d:
     f=Path(d)/'fresh.json';a=self.call('fit','--budget','0','--output',str(f));self.assertEqual(a.returncode,0,a.stderr);self.assertEqual(self.call('audit','--report',str(f)).returncode,0)
