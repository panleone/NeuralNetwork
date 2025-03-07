#pragma once

#include <cblas.h> // LAPACK C interface
#include <type_traits>

/**
 * Internals for tensor operations
 * TODO: implement multiplication, division, etc...
 */
namespace tensor_ops {
/**
 * performs
 * b1 = b1 + alpha * b2
 */
template <typename T> void bufferSum(size_t bufferSize, T alpha, T *b1, T *b2) {
  if constexpr (std::is_same_v<T, double>) {
    cblas_daxpy(bufferSize, alpha, b2, 1, b1, 1);
  } else if constexpr (std::is_same_v<T, float>) {
    cblas_saxpy(bufferSize, alpha, b2, 1, b1, 1);
  } else {
    // Condition that is always false...
    static_assert(std::is_same_v<T, double>);
  }
}

/**
 * Multiply each element of the buffer by the scalar
 */
template <typename T> void scalarMul(size_t bufferSize, T *b1, T scalar) {
  if constexpr (std::is_same_v<T, double>) {
    cblas_dscal(bufferSize, scalar, b1, 1);
  } else if constexpr (std::is_same_v<T, float>) {
    cblas_sscal(bufferSize, scalar, b1, 1);
  } else {
    // Condition that is always false...
    static_assert(std::is_same_v<T, double>);
  }
}

/**
 * perform matrix multiplication between b1 and b2.
 * @param b1 - M*K matrix (up to transposition)
 * @param b2 - KxN matrix (up to transposition)
 * @param res - MxN matrix where matmul(b1, b2) is stored
 */
template <bool TransposeB1 = false, bool TransposeB2 = false, typename T>
void matMul(size_t M, size_t N, size_t K, T *b1, T *b2, T *res) {

  constexpr auto t1TransposeFlag = TransposeB1 ? CblasTrans : CblasNoTrans;
  constexpr auto t2TransposeFlag = TransposeB2 ? CblasTrans : CblasNoTrans;

  int lda = (TransposeB1 ? K : M);
  int ldb = (TransposeB2 ? N : K);
  int ldc = M; // Output matrix leading dimension

  if constexpr (std::is_same_v<T, double>) {
    cblas_dgemm(CblasColMajor, t1TransposeFlag, t2TransposeFlag, M, N, K, 1.0,
                b1, lda, b2, ldb, 0.0, res, ldc);
  } else if constexpr (std::is_same_v<T, float>) {
    cblas_sgemm(CblasColMajor, t1TransposeFlag, t2TransposeFlag, M, N, K, 1.0f,
                b1, lda, b2, ldb, 0.0f, res, ldc);
  } else {
    // Condition that is always false...
    static_assert(std::is_same_v<T, double>);
  }
}
} // namespace tensor_ops