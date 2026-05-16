# complex-iram

Merged version of the original `complex-cpu` and `complex-gpu` projects.

The code keeps one IRAM/Arnoldi implementation and selects the linear algebra backend at runtime from the CLI:

```bash
cargo run --release -- --backend lapack --operator grcar --dimension 256 --nev 4
cargo run --release --features magma -- --backend magma --operator grcar --dimension 256 --nev 4
```

## Backends

- `lapack` uses the original LAPACK/OpenBLAS path.
- `magma` uses the original MAGMA/CUDA path. Build with `--features magma` and set either `MAGMA_LIB_DIR` or `MAGMA_DIR`. Optional CUDA lookup variables: `CUDA_HOME`, `CUDA_LIB_DIR`, `CUBLAS_LIB_DIR`.

`build.rs` always links OpenBLAS because the common shifted QR filter uses LAPACK/BLAS primitives. MAGMA/CUDA libraries are linked only when the `magma` Cargo feature is enabled.

## Structure

```text
src/
├── arnoldi.rs
├── backend
│   ├── lapack.rs
│   ├── magma.rs
│   └── mod.rs
├── config.rs
├── error.rs
├── iram.rs
├── lib.rs
├── linalg
│   ├── lapack.rs
│   ├── magma.rs
│   ├── mod.rs
│   ├── ops.rs
│   ├── shifted_qr.rs
│   └── small.rs
├── main.rs
├── memory.rs
├── operator.rs
├── report.rs
└── selection.rs
```

## Design notes

The split follows the same broad idea as backend-oriented Rust frameworks: the high-level algorithm is generic over a backend interface, while execution details live in backend implementations. `iram.rs` and `arnoldi.rs` call operations such as `compute_ritz_values`, `retrieve_ritz_vectors`, `orthogonalize_arnoldi_candidate` and `zgemm_nn` through the `Backend` trait instead of directly calling LAPACK or MAGMA.

The implicit restart filter `shifted_qr_filter` is now common in `linalg/shifted_qr.rs`. Both backends use the same host-side fast shifted QR implementation, so the two paths cannot silently diverge there.

## MAGMA backend allocation/performance pass

This version includes a conservative MAGMA optimization pass:

- Arnoldi projection buffers are now stored in `MagmaArnoldiWorkspace` and reused across Arnoldi steps instead of allocating a fresh `DeviceVector` and host `Vec` on every orthogonalization.
- `DeviceVector` supports prefix copies, so a projection buffer with capacity `ncv + 1` can serve the logical projection lengths `1..=ncv`.
- `MagmaBackend::zgemm_nn` now uses the backend's long-lived `MagmaSession` via `zgemm_with_session`, avoiding a new MAGMA queue/session for restart GEMM calls.
- MAGMA device allocation, device peak memory, and host/device transfer counters are recorded and rendered in the report.

The CPU-side `LinearOperator::apply_into` contract is intentionally preserved in this pass. Moving matrix-vector products for Matrix Market operators to cuSPARSE/cuBLAS would be a larger backend-operator redesign and should be validated separately.
