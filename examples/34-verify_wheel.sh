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
assert len(mods)==6,[m.name for m in mods]
print('all seven modules including root imported outside checkout')
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
