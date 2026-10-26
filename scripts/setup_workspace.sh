#!/bin/bash
# Build the grace2ace workspace: ONE conda env with ASSYST, GRACE (grace-tensorpotential),
# pacemaker (python-ace) and grace2ace, all installed from git clones next to this repo:
#
#   <workspace>/
#     grace2ace/               this repo
#     assyst/                  github.com/eisenforschung/assyst          (editable install)
#     grace-tensorpotential/   github.com/ICAMS/grace-tensorpotential    (editable install)
#     python-ace/              github.com/ICAMS/python-ace               (built + installed)
#
# Usage:   bash grace2ace/scripts/setup_workspace.sh [env-name]      (default env name: grace2ace)
# Re-running is safe: existing clones are kept, the env is reused.
#
# Why these versions:
#   * pacemaker releases (<= 0.2.8) and master import the OLD stand-alone `tensorpotential`
#     package, whose import name clashes with GRACE's. The branch feature/grace_fs instead
#     uses the copy of the ACE fitting backend that GRACE ships (tensorpotential.compat.pace),
#     so both live in one env. Commit d0f7a0b reproduced an old-env fit to all printed digits.
#   * The branch misses upstream fix 4ed2809 (duplicated rows after joining train and test data);
#     it is cherry-picked onto a local branch `grace2ace-fixes`.
#   * Its setup.py pins numpy<2.2 only because of one float() call; grace2ace works around
#     that at run time (grace2ace.pacemaker_compat), so pacemaker is installed with --no-deps.
set -euo pipefail

ENV_NAME="${1:-grace2ace}"
ASSYST_REF="${ASSYST_REF:-main}"
GRACE_REF="${GRACE_REF:-0.6.2.1}"
PACE_REF="${PACE_REF:-d0f7a0b}"          # tip of feature/grace_fs, tested

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="$(dirname "$HERE")"
CONDA="${CONDA_EXE:-$(command -v conda)}"
[ -n "$CONDA" ] || { echo "conda not found"; exit 1; }
BASE="$("$CONDA" info --base)"

clone () {  # name url ref
    if [ ! -d "$WS/$1/.git" ]; then git clone "$2" "$WS/$1"; fi
    git -C "$WS/$1" fetch --all --tags -q
    git -C "$WS/$1" checkout -q "$3"
    echo "$1 at $(git -C "$WS/$1" describe --tags --always)"
}
clone assyst                https://github.com/eisenforschung/assyst.git        "$ASSYST_REF"
clone grace-tensorpotential https://github.com/ICAMS/grace-tensorpotential.git  "$GRACE_REF"
clone python-ace            https://github.com/ICAMS/python-ace.git             "$PACE_REF"
# feature/grace_fs branched off before upstream fix 4ed2809 (PR #80): pd.concat of train+test
# without reset_index duplicates the first len(test) training rows (double weight) and every test
# row. Put the fix on top, on a local branch, so the change stays visible in `git log`.
git -C "$WS/python-ace" checkout -q -B grace2ace-fixes "$PACE_REF"
git -C "$WS/python-ace" -c user.name=grace2ace -c user.email=grace2ace@localhost cherry-pick 4ed2809 >/dev/null
echo "python-ace + fix 4ed2809 at $(git -C "$WS/python-ace" describe --tags --always)"

if [ ! -x "$BASE/envs/$ENV_NAME/bin/python" ]; then
    "$CONDA" env create -q -n "$ENV_NAME" -f "$HERE/environment.yml"
fi
E="$BASE/envs/$ENV_NAME"

# Enter the env without `conda activate` (works in scripts; a leftover venv on PATH cannot win).
export CONDA_PREFIX="$E" PATH="$E/bin:/usr/bin:/bin"
set +u
for hook in "$E"/etc/conda/activate.d/*.sh; do [ -f "$hook" ] && . "$hook" >/dev/null 2>&1; done
set -u
PY="$E/bin/python"

"$PY" -m pip install -q -e "$WS/grace-tensorpotential"          # brings TensorFlow + pip CUDA libs
"$PY" -m pip install -q -e "$WS/assyst[grace]"
"$PY" -m pip install -q scikit-learn cython pytest              # pacemaker deps not pulled by the above

# pacemaker: C++ build. The bundled yaml-cpp misses <cstdint>, which gcc >= 13 no longer
# includes implicitly. Installed non-editable: the cmake-built modules are not found otherwise.
export CXXFLAGS="${CXXFLAGS:-} -include cstdint"
( cd "$WS/python-ace" && rm -rf build && "$PY" -m pip install -q --no-deps --no-build-isolation . && rm -rf build )
"$PY" -m pip install -q --no-deps --no-build-isolation "$WS/python-ace/lib/maxvolpy"

"$PY" -m pip install -q --no-deps -e "$HERE"

"$PY" - <<'EOF'
import numpy, tensorflow as tf, pyace, assyst, tensorpotential
from tensorpotential.compat.pace.fit import FitTensorPotential   # pacemaker's GPU backend
from tensorpotential.calculator import grace_fm                  # GRACE
import grace2ace
print(f"OK  numpy {numpy.__version__} | TF {tf.__version__} | GRACE {tensorpotential.__version__} "
      f"| pyace {pyace.__version__} | grace2ace {grace2ace.__version__}")
EOF
echo "python for envs.generate / envs.fit:  $PY"
