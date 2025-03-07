#pragma once

#include "../matrix.h"
#include <cblas.h> // LAPACK C interface
#include <cstring> // for memcpy
#include <type_traits>

#include "indices.h" // metaprogramming utils

template <typename T, size_t... Shape>
requires(std::is_arithmetic_v<T>) &&
    (std::is_same_v<T, std::decay_t<T>>)&&((Shape > 0) && ...) class Tensor {
public:
  // TODO: make those fields private
  T *data{nullptr};
  constexpr static size_t size = (Shape * ...);
  constexpr static size_t dim = sizeof...(Shape);

  template <typename U, size_t... S> friend class Tensor;

  template <bool TransposeT1, bool TransposeT2, typename U, typename V>
  friend auto tensorMul(const U &t1, const V &t2);
  using PackedShape = Pack<Shape...>;
  using Type = T;
  Tensor<T, Shape...> &operator+=(const Tensor<T, Shape...> &t);
  Tensor<T, Shape...> &operator-=(const Tensor<T, Shape...> &t);

  Tensor() { data = new T[size]; }
  // TODO: this is just for retro-compatibility.
  // Remove when the Vector class will be removed
  Tensor(const Vector<T> &vector) {
    assert(vector.N == size);
    data = new T[size];
    for (size_t i = 0; i < size; i++) {
      data[i] = vector(i);
    }
  }
  Tensor(const Tensor<T, Shape...> &t2) {
    data = new T[size];
    std::memcpy(data, t2.data, size * sizeof(T));
  }

  Tensor(Tensor<T, Shape...> &&t2) { std::swap(data, t2.data); }
  Tensor<T, Shape...> &operator=(Tensor<T, Shape...> t2) {
    std::swap(data, t2.data);
    return *this;
  }

  void setZero() { std::fill(data, data + size, static_cast<T>(0)); }

  template <typename... I>
  requires((std::is_integral_v<I> && std::is_convertible_v<I, size_t>)&&...) &&
      (sizeof...(I) == sizeof...(Shape)) T &operator[](I... idxs) {
    using cumProduct = CumulativeProduct<Shape...>::type;
    return data[cumProduct::reduce(static_cast<size_t>(idxs)...)];
  }

  template <typename... I>
  requires((std::is_integral_v<I> && std::is_convertible_v<I, size_t>)&&...) &&
      (sizeof...(I) == sizeof...(Shape)) const T &operator[](I... idxs) const {
    using cumProduct = CumulativeProduct<Shape...>::type;
    return data[cumProduct::reduce(static_cast<size_t>(idxs)...)];
  }
  ~Tensor() { delete[] data; }

  /**
   * Scalar multiply and stores result in *this
   */
  auto &scalarMul(T scalar) {
    if constexpr (std::is_same_v<T, double>) {
      cblas_dscal(size, scalar, data, 1);
    } else if constexpr (std::is_same_v<T, float>) {
      cblas_dscal(size, scalar, data, 1);
    } else {
      // Condition that is always false...
      static_assert(std::is_same_v<T, double>);
    }
    return *this;
  }

  /**
   * does the following:
   * *this = *this + alpha * t
   */
  void tensorSum(const Tensor<T, Shape...> &t, T alpha) {
    if constexpr (std::is_same_v<T, double>) {
      cblas_daxpy(t.size, alpha, t.data, 1, data, 1);
    } else if constexpr (std::is_same_v<T, float>) {
      cblas_saxpy(t.size, alpha, t.data, 1, data, 1);
    } else {
      // Condition that is always false...
      static_assert(std::is_same_v<T, double>);
    }
  }
};

template <typename T, size_t... Shape>
Tensor<T, Shape...> &
Tensor<T, Shape...>::operator+=(const Tensor<T, Shape...> &t) {
  this->tensorSum(t, static_cast<T>(1));
  return *this;
}

template <typename T, size_t... Shape>
Tensor<T, Shape...> &
Tensor<T, Shape...>::operator-=(const Tensor<T, Shape...> &t) {
  this->tensorSum(t, static_cast<T>(-1));
  return *this;
}

template <typename T, size_t... Shape>
Tensor<T, Shape...> operator+(const Tensor<T, Shape...> &t1,
                              const Tensor<T, Shape...> &t2) {
  auto res{t1};
  res += t2;
  return res;
}

template <typename T, size_t... Shape>
Tensor<T, Shape...> operator-(const Tensor<T, Shape...> &t1,
                              const Tensor<T, Shape...> &t2) {
  auto res{t1};
  res -= t1;
  return res;
}

/**
 * Generalization of the matrix product.
 * Given t1_{i1, ..., in} and t2_{ji,... ,jn} with in = j1
 * returns t3_{i1,...,in-1, j2, ..., jn} =
 * = sum_{k} t1_{i1, ..., k}*t2_{k,...,jn}
 */
template <bool TransposeT1 = false, bool TransposeT2 = false, typename U,
          typename V>
auto tensorMul(const U &t1, const V &t2) {
  static_assert(std::is_same_v<typename U::Type, typename V::Type>);

  // If both tensors are vector we must treat them as matrices with one column
  constexpr bool vectorCase = U::dim == 1 && V::dim == 1;
  using Uf = std::conditional_t<vectorCase,
                                typename U::PackedShape::append<
                                    1>::type::extract<Tensor, typename U::Type>,
                                U>;
  using Vf = std::conditional_t<vectorCase,
                                typename V::PackedShape::append<
                                    1>::type::extract<Tensor, typename V::Type>,
                                V>;

  constexpr size_t commonDim = TransposeT1
                                   ? PopFront<typename Uf::PackedShape>::value
                                   : PopBack<typename Uf::PackedShape>::value;

  static_assert(commonDim == (TransposeT2
                                  ? PopBack<typename Vf::PackedShape>::value
                                  : PopFront<typename Vf::PackedShape>::value));

  // 1) Use metaprogramming to compute the shape of the result
  constexpr size_t residualRows = Uf::size / commonDim;
  constexpr size_t residualColumns = Vf::size / commonDim;

  using tmp =
      std::conditional_t<TransposeT1,
                         typename PopFront<typename Uf::PackedShape>::type,
                         typename PopBack<typename Uf::PackedShape>::type>;
  using tmp2 =
      std::conditional_t<TransposeT2,
                         typename PopBack<typename Vf::PackedShape>::type,
                         typename PopFront<typename Vf::PackedShape>::type>;

  using resPackType = typename tmp::merge<tmp2>::type;
  using resType = typename resPackType::extract<Tensor, typename Uf::Type>;
  auto resTensor = resType{};

  // 2) Perform the actual multiplication
  constexpr auto t1TransposeFlag = TransposeT1 ? CblasTrans : CblasNoTrans;
  constexpr auto t2TransposeFlag = TransposeT2 ? CblasTrans : CblasNoTrans;

  constexpr int lda = (TransposeT1 ? commonDim : residualRows);
  constexpr int ldb = (TransposeT2 ? residualColumns : commonDim);
  constexpr int ldc = residualRows; // Output matrix leading dimension

  if constexpr (std::is_same_v<typename Uf::Type, double>) {
    cblas_dgemm(CblasColMajor, t1TransposeFlag, t2TransposeFlag, residualRows,
                residualColumns, commonDim, 1.0, t1.data, lda, t2.data, ldb,
                0.0, resTensor.data, ldc);
  } else if constexpr (std::is_same_v<typename Uf::Type, float>) {
    cblas_sgemm(CblasColMajor, t1TransposeFlag, t2TransposeFlag, residualRows,
                residualColumns, commonDim, 1.0, t1.data, lda, t2.data, ldb,
                0.0, resTensor.data, ldc);
  } else {
    // Condition that is always false...
    static_assert(std::is_same_v<typename Uf::Type, double>);
  }
  return resTensor;
}
