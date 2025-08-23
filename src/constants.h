#pragma once
#include <cstddef>
#include <immintrin.h>

/**
 * Common AVX wrapper for float and double
 */

template <typename T>
struct struct_simd_type;
template <>
struct struct_simd_type<float> {
    using type = __m256;
    static constexpr size_t size = 8;
};
template <>
struct struct_simd_type<double> {
    using type = __m256d;
    static constexpr size_t size = 4;
};
template <typename T>
using simd_type = typename struct_simd_type<T>::type;

namespace ops {
    template <bool tLeft, bool tRight>
    constexpr size_t MAT_MUL;

    constexpr size_t VARIABLE_OP = 0;
    constexpr size_t CONSTANT_OP = 1;
    constexpr size_t SUM_OP = 2;
    constexpr size_t DIFF_OP = 3;
    constexpr size_t MUL_OP = 4;
    constexpr size_t DIVIDE_OP = 5;
    // Fused operator FMA ((a * b) + c)
    constexpr size_t FMA_OP = 6;
    // Fused operator FAM a + (b * c)
    constexpr size_t FAM_OP = 7;
    // Generalized matrix multiplication
    template <>
    constexpr size_t MAT_MUL<false, false> = 8;
    template <>
    constexpr size_t MAT_MUL<false, true> = 9;
    template <>
    constexpr size_t MAT_MUL<true, false> = 10;
    template <>
    constexpr size_t MAT_MUL<true, true> = 11;

    // 1d and 2d convolution operators
    constexpr size_t CONV_1D = 12;
    constexpr size_t CONV_2D = 13;

    // Unary operators
    constexpr size_t RELU = 14;
    constexpr size_t TRANSPOSE = 15;
    constexpr size_t EXP = 16;
    constexpr size_t LOG = 17;
    constexpr size_t FLIP_SIGN = 18;
    constexpr size_t SQRT = 19;
    constexpr size_t FLATTEN = 20;
} // namespace ops

namespace avx_constants {
    template <typename T>
    constexpr simd_type<T> zero;

    template <>
    constexpr simd_type<double> zero<double> = {0.0, 0.0, 0.0, 0.0};

    template <>
    constexpr simd_type<float> zero<float> = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    template <typename T>
    constexpr simd_type<T> minus_one;

    template <>
    constexpr simd_type<double> minus_one<double> = {-1.0, -1.0, -1.0, -1.0};

    template <>
    constexpr simd_type<float> minus_one<float> = {
        -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f};

    template <typename T>
    constexpr size_t intrinsic_size = struct_simd_type<T>::size;
} // namespace avx_constants