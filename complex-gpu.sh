#!/bin/bash
#SBATCH --job-name=complex-gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#SBATCH --output=slurm-%x.out
#SBATCH --error=slurm-%x.err

# If your cluster uses newer Slurm GPU syntax instead of --gres,
# replace the line above with:
# #SBATCH --gpus=1

set -euo pipefail

echo "Start date: $(date)"
echo "Node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Submit dir: ${SLURM_SUBMIT_DIR:-$PWD}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-<unset>}"
echo "SLURM_GPUS=${SLURM_GPUS:-<unset>}"
echo "SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-<unset>}"

cd "${SLURM_SUBMIT_DIR:-$PWD}/complex-iram"

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

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export RAYON_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "MAGMA_ROOT=${MAGMA_ROOT}"
echo "CUDA_ROOT=${CUDA_ROOT}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
echo "OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS}"
echo "RAYON_NUM_THREADS=${RAYON_NUM_THREADS}"

echo "GPU visibility check:"
which nvidia-smi || true

echo "Runtime dependency check:"
ldd ./target/release/complex-iram | egrep 'magma|cuda|cublas|cusparse|openblas|gfortran|gomp|not found' || true

if ldd ./target/release/complex-iram| grep -q 'not found'; then
    echo "ERROR: some shared libraries are still missing; see ldd output above." >&2
    exit 1
fi

echo "Running program:"
# JOB_SUFFIX="${SLURM_JOB_ID:-manual}"
# REPORT_FILE="gpu_report_4_${JOB_SUFFIX}.txt"
# echo "========================================"
#     echo "Running with no inflation"
#     echo "Output: ${REPORT_FILE}"
#     echo "========================================"

#     ./target/release/complex-iram \
#         --backend magma \
#         --matrix-file matrices/atmosmodm.mtx \
#         --nev 4 \
#         --ncv 400 \
#         --max-restarts=10000 \
#         --tol=1e-12 \
#         --seed 234 \
#         --target largest-magnitude \
#         --output "${REPORT_FILE}"


for NEV in 4 8 16 32 64; do
    JOB_SUFFIX="${SLURM_JOB_ID:-manual}"
    REPORT_FILE="gpu_quantum_${NEV}_${JOB_SUFFIX}.txt"

    echo "========================================"
    echo "Running with ritz-inflation=${NEV}"
    echo "Output: ${REPORT_FILE}"
    echo "========================================"

    ./target/release/complex-iram \
        --backend magma \
        --matrix-file matrices/quantum.mtx \
        --nev "${NEV}" \
        --ncv 200 \
        --max-restarts=10000 \
        --tol=1e-12 \
        --seed 234 \
        --target largest-magnitude \
        --output "${REPORT_FILE}"

done

echo "End date: $(date)"
