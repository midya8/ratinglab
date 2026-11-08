#!/usr/bin/env bash
set -euo pipefail
installed=$1
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cd "$work"
export PYTHONPATH="$installed"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python - <<'PY'
import importlib,pkgutil,ratinglab
mods=list(pkgutil.walk_packages(ratinglab.__path__,ratinglab.__name__+'.'))
for m in mods:importlib.import_module(m.name)
assert len(mods)==7,[m.name for m in mods]
print('all eight modules including root imported outside checkout')
PY
python -m ratinglab report > report.json
python -m ratinglab audit > audit.json
python -m ratinglab controls > controls.json
python -m ratinglab witness > witness.json
printf '%s\n' '{"pairs":[[0,0],[137,129]]}' > samples.json
python -m ratinglab predict --input samples.json > predictions.json
python -m ratinglab fit --output fresh.json > summary.json
python -m ratinglab audit --report fresh.json > fresh-audit.json
python - <<'PY'
import json
from ratinglab.experiment import saved
fresh=json.load(open('fresh.json'));old=saved()
assert old['selected_id']==fresh['selected_id'] and old['selected_bias_id']==fresh['selected_bias_id']
assert len(json.load(open('predictions.json'))['predictions'])==2
assert len(json.load(open('fresh-audit.json'))['candidates'])==11
print('all six installed commands, eleven fresh fits and both complete-fit audits pass')
PY

python -m ratinglab.influence demo > influence.json
python -m ratinglab.influence references --output influence-reference.json > influence-summary.json
python - <<'PY2'
import json,math
from importlib.resources import files
r=json.load(open('influence.json'));assert r['delete'] and r['max_normal_residual']<1e-10
def compare(actual,expected,path='references'):
    # LAPACK/BLAS platforms may differ in their last binary64 bits.
    # Structures and discrete metadata stay exact; all floating values must be
    # finite and agree within the independent scalar/browser witness tolerance.
    assert type(actual) is type(expected),(path,type(actual),type(expected))
    if isinstance(expected,dict):
        assert actual.keys()==expected.keys(),path
        for key in expected:compare(actual[key],expected[key],path+'.'+key)
    elif isinstance(expected,list):
        assert len(actual)==len(expected),path
        for i,(a,b) in enumerate(zip(actual,expected)):compare(a,b,path+'['+str(i)+']')
    elif type(expected) is float:
        assert math.isfinite(actual) and math.isfinite(expected),path
        assert math.isclose(actual,expected,rel_tol=2e-10,abs_tol=2e-10),(path,actual,expected)
    else:assert actual==expected,(path,actual,expected)
actual=json.load(open('influence-reference.json'))
expected=json.loads(files('ratinglab').joinpath('artifacts/influence-reference.json').read_text())
compare(actual,expected)
# Check that this comparator is not an always-passing packaging gate.
from copy import deepcopy
for key,value in [('source_row',-1),('score',actual[0]['score']+1e-5),('score',float('nan'))]:
    broken=deepcopy(actual);broken[0][key]=value
    try:compare(broken,expected)
    except AssertionError:pass
    else:raise AssertionError('mutated reference accepted: '+key)
rounded=deepcopy(actual);rounded[0]['score']+=1e-12
compare(rounded,expected)
assert json.load(open('influence-summary.json'))['references']==576
print('both conditional-response module commands reproduce 576 references outside checkout')
PY2
