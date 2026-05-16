#!/bin/bash
#SBATCH --job-name=complex-cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --time=01:00:00
#SBATCH --partition=c24m256
#SBATCH --output=slurm-%x.out
#SBATCH --error=slurm-%x.err

set -e

echo "Start date: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Submit dir: $SLURM_SUBMIT_DIR"

cd "$SLURM_SUBMIT_DIR/complex-iram"

# Make Environment Modules available inside non-interactive Slurm jobs.
if [ -f /etc/profile.d/modules.sh ]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh
fi

# Load the same EasyBuild stack used for building/running MAGMA.
module load magma/2.9.0-foss-2025b-CUDA-12.9.1 2>/dev/null || true
module load CUDA/12.9.1 2>/dev/null || true

MAGMA_ROOT="${EBROOTMAGMA:-/home/l.shepel/.local/easybuild/software/magma/2.9.0-foss-2025b-CUDA-12.9.1}"
CUDA_ROOT="${EBROOTCUDA:-/home/l.shepel/.local/easybuild/software/CUDA/12.9.1}"

if [ ! -f "${MAGMA_ROOT}/lib/libmagma.so" ]; then
    echo "ERROR: libmagma.so not found at ${MAGMA_ROOT}/lib/libmagma.so" >&2
    echo "EBROOTMAGMA=${EBROOTMAGMA:-<unset>}" >&2
    exit 1
fi

export LD_LIBRARY_PATH="${MAGMA_ROOT}/lib:${CUDA_ROOT}/lib64:${CUDA_ROOT}/targets/x86_64-linux/lib:/opt/ohpc/pub/libs/gnu15/openblas/0.3.30/lib:/opt/ohpc/pub/compiler/gcc/15.2.0/lib64:/usr/lib64:${LD_LIBRARY_PATH:-}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export RAYON_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS"
echo "RAYON_NUM_THREADS=$RAYON_NUM_THREADS"

echo "Checking binary libraries:"
ldd ./target/release/complex-iram | grep -E "openblas|lapack|not found" || true

echo "Running program:"

./target/release/complex-iram \
    --backend lapack \
    --matrix-file matrices/quantum.mtx \
    --nev 4 \
    --ncv 100 \
    --max-restarts=100 \
    --tol=1e-12 \
    --ritz-inflation=1.3 \
    --target largest-magnitude \
    --output "cpu_report_${SLURM_JOB_ID}.txt"

echo "Report:"
cat cpu_report_${SLURM_JOB_ID}.txt

echo "End date: $(date)"