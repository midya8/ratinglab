import unittest,importlib.util,re
from pathlib import Path
from ratinglab.experiment import saved
class FigureTests(unittest.TestCase):
 def test_exact_figure_reproduction_and_same_scale(self):
    root=Path(__file__).resolve().parents[1];spec=importlib.util.spec_from_file_location('figure',root/'examples/figure.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);s=m.svg(saved());self.assertEqual(s,(root/'assets/test-rmse.svg').read_text());bars=re.findall(r'data-rmse="([^"]+)" x="390"[^>]+width="([^"]+)"',s);self.assertEqual(len(bars),12)
    for value,width in bars:self.assertAlmostEqual(float(value)*580,float(width),places=10)
