#!/bin/bash
#SBATCH --job-name=@@JOB_NAME@@
#SBATCH --output=@@LOG@@
@@SBATCH_LINES@@

# Written by grace2ace @@VERSION@@ (regenerated on every submit; edit the run config instead).
# The env is entered without ~/.bashrc: its bin/ goes FIRST on PATH and its
# activate.d hooks run (what `conda activate` does), so a venv leaked from the
# submitting shell cannot replace the interpreter. Python is called by full path.
set -eo pipefail
@@ACTIVATE@@
@@SETUP@@
export OMP_NUM_THREADS=@@CPUS@@
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONNOUSERSITE=1

echo "grace2ace stage @@STAGE@@ on $(hostname) at $(date)"
@@COMMAND@@
echo "finished at $(date)"
