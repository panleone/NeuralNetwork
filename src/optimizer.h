#pragma once

#include "matrix.h"
#include <cassert>
#include <math.h>

enum class LOSS {
  /**
   * Mean Squared error
   * Best likelihood estimator when p(y|x) = N(f(x), sigma)
   */
  MSE,
  /**
   * Best likelihood estimator for multiclass classification
   */
  SOFTMAX
};

template <typename T, typename Ty, LOSS Type> class Optimizer {
public:
  T loss(const Ty &y, const Vector<T> &f);

  Vector<T> lossGradient(const Ty &y, const Vector<T> &f);
};

// This shouldn't be needed since I am compiling with C++23 and issue 2518 has
// been approved https://cplusplus.github.io/CWG/issues/2518.html
template <LOSS> constexpr bool dependent_false_v = false;

template <typename T, typename Ty, LOSS Type>
T Optimizer<T, Ty, Type>::loss(const Ty &y, const Vector<T> &f) {
  if constexpr (Type == LOSS::MSE) {
    return (f - y).norm2();
  } else if constexpr (Type == LOSS::SOFTMAX) {
    static_assert(std::is_integral_v<Ty>);
    assert(y < f.N && y >= 0);

    T c1 = static_cast<T>(0);
    for (size_t i = 0; i < f.N; i++) {
      c1 += exp(f(i));
    }
    return -f(y) + log(c1);

  } else {
    static_assert(dependent_false_v<Type>);
  }
}

template <typename T, typename Ty, LOSS Type>
Vector<T> Optimizer<T, Ty, Type>::lossGradient(const Ty &y,
                                               const Vector<T> &f) {
  if constexpr (Type == LOSS::MSE) {
    Vector<T> diff = f - y;
    diff *= 2;
    return diff;
  } else if constexpr (Type == LOSS::SOFTMAX) {
    static_assert(std::is_integral_v<Ty>);
    assert(y < f.N && y >= 0);

    Vector<T> res(f.N);

    T commonDenom = static_cast<T>(0);
    for (size_t i = 0; i < f.N; i++) {
      res(i) = exp(f(i));
      commonDenom += exp(f(i));
    }
    res /= commonDenom;
    res(y) -= 1;
    return res;
  } else {
    static_assert(dependent_false_v<Type>);
  }
}