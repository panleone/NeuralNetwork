#pragma once

#include "matrix.h"
#include <cassert>

enum class LOSS {
  /**
   * Mean Squared error
   * Best likelihood estimator when p(y|x) = N(f(x), sigma)
   */
  MSE,
};

template <typename T, LOSS Type> class Optimizer {
public:
  T loss(const Vector<T> &y, const Vector<T> &f);

  Vector<T> lossGradient(const Vector<T> &y, const Vector<T> &f);
};

// This shouldn't be needed since I am compiling with C++23 and issue 2518 has
// been approved https://cplusplus.github.io/CWG/issues/2518.html
template <LOSS> constexpr bool dependent_false_v = false;

template <typename T, LOSS Type>
T Optimizer<T, Type>::loss(const Vector<T> &y, const Vector<T> &f) {
  if constexpr (Type == LOSS::MSE) {
    return (f - y).norm2();
  } else {
    static_assert(dependent_false_v<Type>);
  }
}

template <typename T, LOSS Type>
Vector<T> Optimizer<T, Type>::lossGradient(const Vector<T> &y,
                                           const Vector<T> &f) {
  if constexpr (Type == LOSS::MSE) {
    Vector<T> diff = f - y;
    diff *= 2;
    return diff;
  } else {
    static_assert(dependent_false_v<Type>);
  }
}