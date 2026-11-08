#!/usr/bin/env bash
set -euo pipefail
python -m ratinglab report >/dev/null
python -m ratinglab audit >/dev/null
python -m ratinglab controls >/dev/null
python -m ratinglab witness --side item --index 84 >/dev/null
