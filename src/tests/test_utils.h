#pragma once

#include "../matrix.h"
#include <functional>

template <typename Err>
void should_throw(const std::function<void()> &fn, std::string_view reason) {
  bool success = false;
  try {
    fn();
    success = true;
  } catch (const Err &e) {
  }
  if (success)
    throw std::runtime_error(reason.data());
}

template <typename T>
void check_vector_equality(const Vector<T> &v1, const Vector<T> &v2, T eps,
                           std::string_view reason) {
  if (v1.N != v2.N || v1.M != v2.M || v1.M != 1) {
    throw std::runtime_error(reason.data());
  }
  for (size_t i = 0; i < v1.N; i++) {
    if (abs(v1(i) - v2(i)) > eps) {
      throw std::runtime_error(reason.data());
    }
  }
}

template <typename T>
void check_matrix_equality(const Matrix<T> &m1, const Matrix<T> &m2, T eps,
                           std::string_view reason) {
  if (m1.N != m2.N || m1.M != m2.M) {
    throw std::runtime_error(reason.data());
  }
  for (size_t i = 0; i < m1.N; i++) {
    for (size_t j = 0; j < m1.M; j++) {
      if (abs(m1(i, j) - m2(i, j)) > eps) {
        throw std::runtime_error(reason.data());
      }
    }
  }
}

template <typename T>
void check_num_equality(T x, T y, T eps, std::string_view reason) {
  if (abs(x - y) > eps) {
    throw std::runtime_error(reason.data());
  }
}