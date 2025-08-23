#pragma once

#include <cblas.h>
#include <cassert>

/**
 * We use external libraries for less trivial operations.
 * BLAS for Matrix Multiplication
 */

template <typename T, bool TransposeM1 = false, bool TransposeM2 = false>
requires(std::is_same_v<T, double> || std::is_same_v<T, float>) void blas_mat_mul(
    const T *m1, const T *m2, T *res, int m1_rows, int m1_cols, int m2_rows, int m2_cols) {

    constexpr T alpha = static_cast<T>(1.0);
    constexpr T beta = static_cast<T>(0.0);

    constexpr CBLAS_TRANSPOSE transa = TransposeM1 ? CblasTrans : CblasNoTrans;
    constexpr CBLAS_TRANSPOSE transb = TransposeM2 ? CblasTrans : CblasNoTrans;

    int a_rows = TransposeM1 ? m1_cols : m1_rows;
    int a_cols = TransposeM1 ? m1_rows : m1_cols;

    int b_rows = TransposeM2 ? m2_cols : m2_rows;
    int b_cols = TransposeM2 ? m2_rows : m2_cols;

    assert(a_cols == b_rows);

    if constexpr (std::is_same_v<T, double>) {
        cblas_dgemm(CblasRowMajor,
                    transa,
                    transb,
                    a_rows,
                    b_cols,
                    a_cols,
                    alpha,
                    m1,
                    m1_cols,
                    m2,
                    m2_cols,
                    beta,
                    res,
                    b_cols);
    } else {
        cblas_sgemm(CblasRowMajor,
                    transa,
                    transb,
                    a_rows,
                    b_cols,
                    a_cols,
                    alpha,
                    m1,
                    m1_cols,
                    m2,
                    m2_cols,
                    beta,
                    res,
                    b_cols);
    }
}