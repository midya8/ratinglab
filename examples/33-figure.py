"""Same-scale score figure, regenerated from frozen metrics without fitting."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ratinglab.experiment import saved
ROOT=Path(__file__).resolve().parents[1]
def svg(r):
 rows=[(c['id']+(' *' if c['id']==r['selected_id'] else ' + bias pick' if c['id']==r['selected_bias_id'] else ''),c['test']['rmse']) for c in r['candidates']+r['bias_candidates']]+[('Training mean',r['global_test_rmse'])]
 s=['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="940" viewBox="0 0 1100 940"><rect width="1100" height="940" fill="#fbfcf8"/><g fill="#193b36" font-family="sans-serif"><text x="40" y="52" font-size="29">The stronger bias-only control stays.</text><text x="40" y="85" font-size="17">Test RMSE on the same 233 observed pairs. Lower is better. Zero-based common scale.</text>']
 for j,(name,value) in enumerate(rows):
  y=135+j*54;color='#925593' if name.endswith('*') else '#286b5b';s.append(f'<text x="40" y="{y+21}" font-size="16">{name}</text><rect data-rmse="{value!r}" x="390" y="{y}" height="27" width="{value*580:.12f}" fill="{color}"/><text x="950" y="{y+21}" font-size="17">{value:.6f}</text>')
 s+=['<text x="390" y="815" font-size="16">0</text><text x="935" y="815" font-size="16">1.0 RMSE</text><text x="40" y="860" font-size="17">* Validation-selected factor model. Bias regularization separately validation-selected. No test choice.</text><text x="40" y="897" font-size="16">UCI Restaurant &amp; Consumer Data, CC BY 4.0. Observed-rating prediction, not ranking utility.</text></g></svg>'];return '\n'.join(s)+'\n'
if __name__=='__main__':(ROOT/'assets/test-rmse.svg').write_text(svg(saved()))
