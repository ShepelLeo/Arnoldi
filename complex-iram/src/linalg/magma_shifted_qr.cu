#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cusparse.h>
#include <stdint.h>
#include <math.h>

static __device__ __forceinline__ cuDoubleComplex zzero() {
    return make_cuDoubleComplex(0.0, 0.0);
}

static __device__ __forceinline__ cuDoubleComplex zone() {
    return make_cuDoubleComplex(1.0, 0.0);
}

static __device__ __forceinline__ cuDoubleComplex zreal(double x) {
    return make_cuDoubleComplex(x, 0.0);
}

static __device__ __forceinline__ cuDoubleComplex zadd(cuDoubleComplex a, cuDoubleComplex b) {
    return make_cuDoubleComplex(cuCreal(a) + cuCreal(b), cuCimag(a) + cuCimag(b));
}

static __device__ __forceinline__ cuDoubleComplex zsub(cuDoubleComplex a, cuDoubleComplex b) {
    return make_cuDoubleComplex(cuCreal(a) - cuCreal(b), cuCimag(a) - cuCimag(b));
}

static __device__ __forceinline__ cuDoubleComplex zmul(cuDoubleComplex a, cuDoubleComplex b) {
    return make_cuDoubleComplex(
        cuCreal(a) * cuCreal(b) - cuCimag(a) * cuCimag(b),
        cuCreal(a) * cuCimag(b) + cuCimag(a) * cuCreal(b));
}

static __device__ __forceinline__ cuDoubleComplex zdiv_real(cuDoubleComplex a, double x) {
    return make_cuDoubleComplex(cuCreal(a) / x, cuCimag(a) / x);
}

static __device__ __forceinline__ cuDoubleComplex zneg(cuDoubleComplex a) {
    return make_cuDoubleComplex(-cuCreal(a), -cuCimag(a));
}

static __device__ __forceinline__ cuDoubleComplex zconj(cuDoubleComplex a) {
    return make_cuDoubleComplex(cuCreal(a), -cuCimag(a));
}

static __device__ __forceinline__ double znorm(cuDoubleComplex a) {
    return hypot(cuCreal(a), cuCimag(a));
}

static __device__ __forceinline__ int idx(int n, int row, int column) {
    return row * n + column;
}

static __device__ void lartg_device(cuDoubleComplex f, cuDoubleComplex g, double* c, cuDoubleComplex* s, cuDoubleComplex* r) {
    const double f_abs = znorm(f);
    const double g_abs = znorm(g);

    if (g_abs == 0.0) {
        *c = 1.0;
        *s = zzero();
        *r = f;
        return;
    }

    if (f_abs == 0.0) {
        *c = 0.0;
        *s = zdiv_real(zconj(g), g_abs);
        *r = zreal(g_abs);
        return;
    }

    const double scale = f_abs + g_abs;
    const double fs = f_abs / scale;
    const double gs = g_abs / scale;
    const double norm = scale * sqrt(fs * fs + gs * gs);
    const cuDoubleComplex alpha = zdiv_real(f, f_abs);

    *c = f_abs / norm;
    *s = zdiv_real(zmul(alpha, zconj(g)), norm);
    *r = zreal(norm);
    *r = zmul(alpha, *r);
}

static __device__ void apply_givens_from_left(cuDoubleComplex* h, int n, int i, double c_raw, cuDoubleComplex s_raw) {
    const cuDoubleComplex c = zreal(c_raw);
    const cuDoubleComplex s_conj = zconj(s_raw);

    for (int column = i; column < n; ++column) {
        const int upper_index = idx(n, i, column);
        const int lower_index = idx(n, i + 1, column);
        const cuDoubleComplex upper = h[upper_index];
        const cuDoubleComplex lower = h[lower_index];

        h[upper_index] = zadd(zmul(c, upper), zmul(s_raw, lower));
        h[lower_index] = zadd(zneg(zmul(s_conj, upper)), zmul(c, lower));
    }
}

static __device__ void apply_givens_from_right(cuDoubleComplex* h, int n, int i, int iend, double c_raw, cuDoubleComplex s_raw) {
    const cuDoubleComplex c = zreal(c_raw);
    const cuDoubleComplex s_conj = zconj(s_raw);
    const int last_row = min(i + 2, iend);

    for (int row = 0; row <= last_row; ++row) {
        const int left_index = idx(n, row, i);
        const int right_index = idx(n, row, i + 1);
        const cuDoubleComplex left = h[left_index];
        const cuDoubleComplex right = h[right_index];

        h[left_index] = zadd(zmul(c, left), zmul(s_conj, right));
        h[right_index] = zadd(zneg(zmul(s_raw, left)), zmul(c, right));
    }
}

static __device__ void accumulate_givens(cuDoubleComplex* q, int n, int i, int shift_index, double c_raw, cuDoubleComplex s_raw) {
    const cuDoubleComplex c = zreal(c_raw);
    const cuDoubleComplex s_conj = zconj(s_raw);
    const int row_count = min(i + shift_index + 2, n);

    for (int row = 0; row < row_count; ++row) {
        const int left_index = idx(n, row, i);
        const int right_index = idx(n, row, i + 1);
        const cuDoubleComplex left = q[left_index];
        const cuDoubleComplex right = q[right_index];

        q[left_index] = zadd(zmul(c, left), zmul(s_conj, right));
        q[right_index] = zadd(zneg(zmul(s_raw, left)), zmul(c, right));
    }
}

static __device__ double hessenberg_one_norm(const cuDoubleComplex* h, int n) {
    double norm = 0.0;
    for (int column = 0; column < n; ++column) {
        const int last_row = min(column + 1, n - 1);
        double column_sum = 0.0;
        for (int row = 0; row <= last_row; ++row) {
            column_sum += znorm(h[idx(n, row, column)]);
        }
        norm = fmax(norm, column_sum);
    }
    return norm;
}

static __device__ bool should_deflate(const cuDoubleComplex* h, int n, int i, double safe_threshold, double h_norm) {
    const double subdiagonal_norm = znorm(h[idx(n, i + 1, i)]);
    if (subdiagonal_norm == 0.0) {
        return true;
    }

    double scale = znorm(h[idx(n, i, i)]) + znorm(h[idx(n, i + 1, i + 1)]);
    if (scale == 0.0) {
        scale = h_norm;
    }
    return subdiagonal_norm <= fmax(2.2204460492503131e-16 * scale, safe_threshold);
}

static __device__ void make_subdiagonal_real_nonnegative(cuDoubleComplex* h, cuDoubleComplex* q, int n) {
    for (int j = 0; j < n - 1; ++j) {
        const int sub_index = idx(n, j + 1, j);
        const cuDoubleComplex sub = h[sub_index];
        const double magnitude = znorm(sub);
        if (magnitude == 0.0 || (cuCimag(sub) == 0.0 && cuCreal(sub) >= 0.0)) {
            continue;
        }

        const cuDoubleComplex phase = zdiv_real(sub, magnitude);
        const cuDoubleComplex phase_conj = zconj(phase);

        for (int column = j; column < n; ++column) {
            h[idx(n, j + 1, column)] = zmul(h[idx(n, j + 1, column)], phase_conj);
        }

        const int last_row = min(j + 2, n - 1);
        for (int row = 0; row <= last_row; ++row) {
            h[idx(n, row, j + 1)] = zmul(h[idx(n, row, j + 1)], phase);
        }

        for (int row = 0; row < n; ++row) {
            q[idx(n, row, j + 1)] = zmul(q[idx(n, row, j + 1)], phase);
        }

        h[sub_index] = zreal(magnitude);
    }
}

static __device__ void cleanup_hessenberg_roundoff(cuDoubleComplex* h, int n) {
    for (int row = 2; row < n; ++row) {
        for (int column = 0; column < row - 1; ++column) {
            h[idx(n, row, column)] = zzero();
        }
    }
}

static __global__ void shifted_qr_filter_kernel(cuDoubleComplex* h, cuDoubleComplex* q, const cuDoubleComplex* shifts, int n, int shift_count) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }
    if (n < 2 || shift_count == 0) {
        cleanup_hessenberg_roundoff(h, n);
        return;
    }

    const double safe_threshold = 2.2250738585072014e-308 * ((double)max(n, 1) / 2.2204460492503131e-16);
    const double h_norm = fmax(hessenberg_one_norm(h, n), safe_threshold);

    for (int shift_index = 0; shift_index < shift_count; ++shift_index) {
        const cuDoubleComplex shift = shifts[shift_index];
        int istart = 0;

        while (istart + 1 < n) {
            int iend = n - 1;

            for (int i = istart; i < n - 1; ++i) {
                if (should_deflate(h, n, i, safe_threshold, h_norm)) {
                    h[idx(n, i + 1, i)] = zzero();
                    iend = i;
                    break;
                }
            }

            if (istart == iend) {
                istart = iend + 1;
                continue;
            }

            cuDoubleComplex f = zsub(h[idx(n, istart, istart)], shift);
            cuDoubleComplex g = h[idx(n, istart + 1, istart)];

            for (int i = istart; i < iend; ++i) {
                double c;
                cuDoubleComplex s;
                cuDoubleComplex r;
                lartg_device(f, g, &c, &s, &r);

                if (i > istart) {
                    h[idx(n, i, i - 1)] = r;
                    h[idx(n, i + 1, i - 1)] = zzero();
                }

                apply_givens_from_left(h, n, i, c, s);
                apply_givens_from_right(h, n, i, iend, c, s);
                accumulate_givens(q, n, i, shift_index, c, s);

                if (i > 0 && i + 1 < n) {
                    h[idx(n, i + 1, i - 1)] = zzero();
                }

                if (i + 1 < iend) {
                    f = h[idx(n, i + 1, i)];
                    g = h[idx(n, i + 2, i)];
                }
            }

            istart = iend + 1;
        }
    }

    make_subdiagonal_real_nonnegative(h, q, n);
    for (int i = 0; i < n - 1; ++i) {
        if (should_deflate(h, n, i, safe_threshold, h_norm)) {
            h[idx(n, i + 1, i)] = zzero();
        }
    }
    cleanup_hessenberg_roundoff(h, n);
}

extern "C" int complex_iram_shifted_qr_filter_cuda(
    int n,
    cuDoubleComplex* h_row_major,
    cuDoubleComplex* q_row_major,
    const cuDoubleComplex* shifts,
    int shift_count) {
    shifted_qr_filter_kernel<<<1, 1>>>(h_row_major, q_row_major, shifts, n, shift_count);
    return (int)cudaDeviceSynchronize();
}


extern "C" int complex_iram_cusparse_zcsrmv(
    cusparseHandle_t handle,
    int dimension,
    int nnz,
    const int* csr_row_offsets,
    const int* csr_columns,
    const cuDoubleComplex* csr_values,
    const cuDoubleComplex* x,
    cuDoubleComplex* y) {
    if (dimension < 0 || nnz < 0) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }

    const cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
    const cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);

    cusparseSpMatDescr_t matrix = nullptr;
    cusparseDnVecDescr_t vec_x = nullptr;
    cusparseDnVecDescr_t vec_y = nullptr;
    void* buffer = nullptr;
    size_t buffer_size = 0;

    cusparseStatus_t status = cusparseCreateCsr(
        &matrix,
        (int64_t)dimension,
        (int64_t)dimension,
        (int64_t)nnz,
        const_cast<int*>(csr_row_offsets),
        const_cast<int*>(csr_columns),
        const_cast<cuDoubleComplex*>(csr_values),
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_C_64F);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        return (int)status;
    }

    status = cusparseCreateDnVec(
        &vec_x,
        (int64_t)dimension,
        const_cast<cuDoubleComplex*>(x),
        CUDA_C_64F);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        cusparseDestroySpMat(matrix);
        return (int)status;
    }

    status = cusparseCreateDnVec(
        &vec_y,
        (int64_t)dimension,
        y,
        CUDA_C_64F);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        cusparseDestroyDnVec(vec_x);
        cusparseDestroySpMat(matrix);
        return (int)status;
    }

    status = cusparseSpMV_bufferSize(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        matrix,
        vec_x,
        &beta,
        vec_y,
        CUDA_C_64F,
        CUSPARSE_SPMV_ALG_DEFAULT,
        &buffer_size);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        cusparseDestroyDnVec(vec_y);
        cusparseDestroyDnVec(vec_x);
        cusparseDestroySpMat(matrix);
        return (int)status;
    }

    if (buffer_size != 0) {
        cudaError_t cuda_status = cudaMalloc(&buffer, buffer_size);
        if (cuda_status != cudaSuccess) {
            cusparseDestroyDnVec(vec_y);
            cusparseDestroyDnVec(vec_x);
            cusparseDestroySpMat(matrix);
            return CUSPARSE_STATUS_ALLOC_FAILED;
        }
    }

    status = cusparseSpMV(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        matrix,
        vec_x,
        &beta,
        vec_y,
        CUDA_C_64F,
        CUSPARSE_SPMV_ALG_DEFAULT,
        buffer);

    if (buffer != nullptr) {
        cudaFree(buffer);
    }
    cusparseDestroyDnVec(vec_y);
    cusparseDestroyDnVec(vec_x);
    cusparseDestroySpMat(matrix);

    return (int)status;
}
