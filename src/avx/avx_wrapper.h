#pragma once

#include <cassert>
#include <cstddef>
#include <immintrin.h>
#include <type_traits>
#include "avx_ops.h"




template <typename T> simd_type<T> _mm256_load_px(const T *x) {
  if constexpr (std::is_same_v<T, float>) {
    return _mm256_load_ps(x);
  } else if constexpr (std::is_same_v<T, double>) {
    return _mm256_load_pd(x);
  } else {
    static_assert(std::is_same_v<T, float>);
  }
}

template <typename T> simd_type<T> _mm256_loadu_px(const T *x) {
  if constexpr (std::is_same_v<T, float>) {
    return _mm256_loadu_ps(x);
  } else if constexpr (std::is_same_v<T, double>) {
    return _mm256_loadu_pd(x);
  } else {
    static_assert(std::is_same_v<T, float>);
  }
}

template <typename T> simd_type<T> _mm256_set1_px(T x) {
  if constexpr (std::is_same_v<T, float>) {
    return _mm256_set1_ps(x);
  } else if constexpr (std::is_same_v<T, double>) {
    return _mm256_set1_pd(x);
  } else {
    static_assert(std::is_same_v<T, float>);
  }
}

template <typename T>
simd_type<T> _mm256_mul_px(simd_type<T> x, simd_type<T> y) {
  if constexpr (std::is_same_v<T, float>) {
    return _mm256_mul_ps(x, y);
  } else if constexpr (std::is_same_v<T, double>) {
    return _mm256_mul_pd(x, y);
  } else {
    static_assert(std::is_same_v<T, float>);
  }
}

template <typename T>
simd_type<T> _mm256_div_px(simd_type<T> x, simd_type<T> y) {
  if constexpr (std::is_same_v<T, float>) {
    return _mm256_div_ps(x, y);
  } else if constexpr (std::is_same_v<T, double>) {
    return _mm256_div_pd(x, y);
  } else {
    static_assert(std::is_same_v<T, float>);
  }
}

template <typename T> void _mm256_store_px(T *res, simd_type<T> x) {
  if constexpr (std::is_same_v<T, float>) {
    _mm256_store_ps(res, x);
  } else if constexpr (std::is_same_v<T, double>) {
    _mm256_store_pd(res, x);
  } else {
    static_assert(std::is_same_v<T, float>);
  }
}

template <typename T> void _mm256_storeu_px(T *res, simd_type<T> x) {
  if constexpr (std::is_same_v<T, float>) {
    _mm256_storeu_ps(res, x);
  } else if constexpr (std::is_same_v<T, double>) {
    _mm256_storeu_pd(res, x);
  } else {
    static_assert(std::is_same_v<T, float>);
  }
}

template <typename T>
simd_type<T> _mm256_sub_px(simd_type<T> x, simd_type<T> y) {
  if constexpr (std::is_same_v<T, float>) {
    return _mm256_sub_ps(x, y);
  } else if constexpr (std::is_same_v<T, double>) {
    return _mm256_sub_pd(x, y);
  } else {
    static_assert(std::is_same_v<T, float>);
  }
}

template <typename T>
simd_type<T> _mm256_add_px(simd_type<T> x, simd_type<T> y) {
  if constexpr (std::is_same_v<T, float>) {
    return _mm256_add_ps(x, y);
  } else if constexpr (std::is_same_v<T, double>) {
    return _mm256_add_pd(x, y);
  } else {
    static_assert(std::is_same_v<T, float>);
  }
}

template <typename T>
simd_type<T> _mm256_max_px(simd_type<T> x, simd_type<T> y) {
  if constexpr (std::is_same_v<T, float>) {
    return _mm256_max_ps(x, y);
  } else if constexpr (std::is_same_v<T, double>) {
    return _mm256_max_pd(x, y);
  } else {
    static_assert(std::is_same_v<T, float>);
  }
}

template <typename T>
simd_type<T> _mm256_exp_px(simd_type<T> x) {
  if constexpr (std::is_same_v<T, float>) {
    return _mm256_fast_exp_ps(x);
  } else if constexpr (std::is_same_v<T, double>) {
    return _mm256_exp_pd(x);
  } else {
    static_assert(std::is_same_v<T, float>);
  }
}

template <typename T>
simd_type<T> _mm256_log_px(simd_type<T> x) {
  if constexpr (std::is_same_v<T, float>) {
    return _mm256_log_ps(x);
  } else if constexpr (std::is_same_v<T, double>) {
    return _mm256_log_pd(x);
  } else {
    static_assert(std::is_same_v<T, float>);
  }
}

template <typename T>
simd_type<T> _mm256_sqrt_px(simd_type<T> x) {
  if constexpr (std::is_same_v<T, float>) {
    return _mm256_sqrt_ps(x);
  } else if constexpr (std::is_same_v<T, double>) {
    return _mm256_sqrt_pd(x);
  } else {
    static_assert(std::is_same_v<T, float>);
  }
}

template<typename T>
simd_type<T> _mm256_flip_sign_px(simd_type<T> x) {
  if constexpr (std::is_same_v<T, float>) {
    return _mm256_flip_sign_ps(x);
  } else if constexpr (std::is_same_v<T, double>) {
    return _mm256_flip_sign_pd(x);
  } else {
    static_assert(std::is_same_v<T, float>);
  }
}

template <typename T>
simd_type<T> _mm256_fnmadd_px(simd_type<T> x, simd_type<T> y, simd_type<T> z) {
  if constexpr (std::is_same_v<T, float>) {
    return _mm256_fnmadd_ps(x, y, z);
  } else if constexpr (std::is_same_v<T, double>) {
    return _mm256_fnmadd_pd(x, y, z);
  } else {
    static_assert(std::is_same_v<T, float>);
  }
}

template <typename T>
simd_type<T> _mm256_fmadd_px(simd_type<T> x, simd_type<T> y, simd_type<T> z) {
  if constexpr (std::is_same_v<T, float>) {
    return _mm256_fmadd_ps(x, y, z);
  } else if constexpr (std::is_same_v<T, double>) {
    return _mm256_fmadd_pd(x, y, z);
  } else {
    static_assert(std::is_same_v<T, float>);
  }
}

template <typename T, int OP>
simd_type<T> _mm256_cmp_px(simd_type<T> x, simd_type<T> y) {
  if constexpr (std::is_same_v<T, float>) {
    return _mm256_cmp_ps(x, y, OP);
  } else if constexpr (std::is_same_v<T, double>) {
    return _mm256_cmp_pd(x, y, OP);
  } else {
    static_assert(std::is_same_v<T, float>);
  }
}

template <typename T>
simd_type<T> _mm256_and_px(simd_type<T> x, simd_type<T> y) {
  if constexpr (std::is_same_v<T, float>) {
    return _mm256_and_ps(x, y);
  } else if constexpr (std::is_same_v<T, double>) {
    return _mm256_and_pd(x, y);
  } else {
    static_assert(std::is_same_v<T, float>);
  }
}

template <typename T> unsigned _mm256_movemask_px(simd_type<T> x) {
  if constexpr (std::is_same_v<T, float>) {
    return _mm256_movemask_ps(x);
  } else if constexpr (std::is_same_v<T, double>) {
    return _mm256_movemask_pd(x);
  } else {
    static_assert(std::is_same_v<T, float>);
  }
}

