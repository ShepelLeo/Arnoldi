#!/bin/bash
#SBATCH --job-name=complex-cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --time=00:05:00
#SBATCH --partition=c24m256
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -e

echo "Start date: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Submit dir: $SLURM_SUBMIT_DIR"

cd "$SLURM_SUBMIT_DIR/complex-cpu"

export LD_LIBRARY_PATH=/opt/ohpc/pub/libs/gnu15/openblas/0.3.30/lib:/opt/ohpc/pub/compiler/gcc/15.2.0/lib64:/usr/lib64:${LD_LIBRARY_PATH:-}

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export RAYON_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS"
echo "RAYON_NUM_THREADS=$RAYON_NUM_THREADS"

echo "Checking binary libraries:"
ldd ./target/release/complex-cpu | grep -E "openblas|lapack|not found" || true

echo "Running program:"

./target/release/complex-cpu \
    --matrix-file matrices/quantum.mtx \
    --nev 4 \
    --ncv 100 \
    --max-restarts=100 \
    --tol=1e-12 \
    --ritz-inflation=1.3 \
    --target largest-magnitude \
    --output "convdiff_report_${SLURM_JOB_ID}.txt"

echo "Report:"
cat convdiff_report_${SLURM_JOB_ID}.txt

echo "End date: $(date)"