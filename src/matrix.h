#pragma once

#include "math.h"
#include <cassert>
#include <cblas.h> // LAPACK C interface
#include <initializer_list>
#include <ostream>
#include <stdexcept>
#include <type_traits>
#include <vector>

class MatrixException : public std::exception {
  std::string reason;

public:
  MatrixException(const std::string &reason) : reason{reason} {}
  const char *what() const noexcept override { return reason.c_str(); }
};

// N number of rows, M number of columns
template <typename T> class Matrix {
public:
  using Field = T;
  std::size_t N;
  std::size_t M;
  std::vector<T> matData;

private:
  Matrix(const Matrix<T> &mat) : N{mat.N}, M{mat.M}, matData{mat.matData} {};

public:
  explicit Matrix(std::vector<T> &&arr)
      : N{arr.size()}, M{1}, matData{std::move(arr)} {}
  explicit Matrix(std::initializer_list<std::initializer_list<T>> doubleList) {
    N = doubleList.size();
    M = 0;
    // 1) Check initializer list consistency
    for (const std::initializer_list<T> &list : doubleList) {
      if (M == 0) {
        M = list.size();
        if (M == 0) {
          throw MatrixException("Matrix constructor initializer empty row");
        }
      }
      matData.insert(matData.end(), list.begin(), list.end());
      if (M != list.size()) {
        throw MatrixException(
            "Matrix constructor initializer list row dimensions mismatch");
      }
    }
  }
  Matrix(std::size_t N, std::size_t M)
      : N{N}, M{M}, matData{std::vector<T>(N * M)} {};
  Matrix(Matrix<T> &&mat)
      : N{mat.N}, M{mat.M}, matData{std::move(mat.matData)} {};
  Matrix<T> &operator=(Matrix<T> mat);
  Matrix<T> clone() const;

  inline T &operator()(std::size_t x, std::size_t y);
  inline const T &operator()(std::size_t x, std::size_t y) const;

  // Basic operations
  // 1) matrix-scalar operations
  Matrix<T> &operator*=(const T &c1);
  Matrix<T> &operator/=(const T &c1);

  // 2) matrix-matrix operations
  Matrix<T> &operator+=(const Matrix<T> &m1);
  Matrix<T> &operator-=(const Matrix<T> &m1);
  Matrix<T> transposeMatMul(const Matrix<T> &m1);

  // 3) unary operators
  Matrix<T> operator-() const;
  T norm() const;
  T norm2() const;
  Matrix<T> transpose() const;

  // 4) More stuff
  void reshape(size_t newN, size_t newM);
};

// x column y row
template <typename T> T &Matrix<T>::operator()(std::size_t x, std::size_t y) {
  return matData[x * M + y];
}

// x column y row
template <typename T>
const T &Matrix<T>::operator()(std::size_t x, std::size_t y) const {
  return matData[x * M + y];
}

template <typename T> Matrix<T> &Matrix<T>::operator=(Matrix<T> mat) {
  matData = std::move(mat.matData);
  N = mat.N;
  M = mat.M;
  return *this;
};

template <typename T> Matrix<T> Matrix<T>::clone() const {
  return Matrix<T>{*this};
};

template <typename T>
std::ostream &operator<<(std::ostream &o, const Matrix<T> &m1) {
  for (size_t j = 0; j < m1.N; j++) {
    for (size_t i = 0; i < m1.M; i++) {
      o << m1(j, i) << " ";
    }
    o << "\n";
  }
  return o;
}

template <typename T> Matrix<T> Matrix<T>::operator-() const {
  Matrix<T> res = this->clone();
  for (T &el : res.matData) {
    el = -el;
  }
  return res;
}

template <typename T> Matrix<T> &Matrix<T>::operator+=(const Matrix<T> &m1) {
  if (m1.N != N || m1.M != M) {
    throw MatrixException("Matrix += operator dimension mismatch");
  }
  if constexpr (std::is_same_v<T, float>) {
    cblas_saxpy(N * M, 1.0, m1.matData.data(), 1, matData.data(), 1);
  } else if constexpr (std::is_same_v<T, double>) {
    cblas_daxpy(N * M, 1.0, m1.matData.data(), 1, matData.data(), 1);
  } else {
    for (size_t i = 0; i < N * M; i++) {
      matData[i] += m1.matData[i];
    }
  }
  return *this;
}

template <typename T> Matrix<T> &Matrix<T>::operator-=(const Matrix<T> &m1) {
  if (m1.N != N || m1.M != M) {
    throw MatrixException("Matrix -= operator dimension mismatch");
  }
  for (size_t i = 0; i < N * M; i++) {
    matData[i] -= m1.matData[i];
  }
  return *this;
}

template <typename T>
Matrix<T> operator+(const Matrix<T> &m1, const Matrix<T> &m2) {
  if (m1.N != m2.N || m1.M != m2.M) {
    throw MatrixException("Matrix sum dimensions mismatch");
  }
  Matrix<T> res = m1.clone();
  res += m2;
  return res;
}
template <typename T>
Matrix<T> operator-(const Matrix<T> &m1, const Matrix<T> &m2) {
  return m1 + (-m2);
}

template <typename T> T Matrix<T>::norm() const { return sqrt(norm2()); }

template <typename T> T Matrix<T>::norm2() const {
  T norm2 = 0;
  for (const T &el : matData) {
    norm2 += el * el;
  }
  return norm2;
}

template <typename T> Matrix<T> Matrix<T>::transpose() const {
  Matrix<T> res(this->M, this->N);
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < M; j++) {
      res(j, i) = (*this)(i, j);
    }
  }
  return res;
}

template <typename T> Matrix<T> &Matrix<T>::operator*=(const T &c1) {
  for (T &el : matData) {
    el *= c1;
  }
  return *this;
}

template <typename T> Matrix<T> &Matrix<T>::operator/=(const T &c1) {
  for (T &el : matData) {
    el /= c1;
  }
  return *this;
}

template <typename T> Matrix<T> operator*(const Matrix<T> &m1, const T &c) {
  Matrix<T> res = m1.clone();
  res *= c;
  return res;
}

template <typename T> Matrix<T> operator*(const T &c, const Matrix<T> &m1) {
  return m1 * c;
}

template <typename T>
Matrix<T> operator*(const Matrix<T> &m1, const Matrix<T> &m2) {
  if (m1.M != m2.N) {
    throw MatrixException("Matrix product size mismatch");
  }
  Matrix<T> res(m1.N, m2.M);
  // For float and doubles we can use the very fast implementation of openblas
  if constexpr (std::is_same_v<T, float>) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m1.N, m2.M, m1.M,
                1.0f, m1.matData.data(), m1.M, m2.matData.data(), m2.M, 0.0f,
                res.matData.data(), res.M);
  } else if constexpr (std::is_same_v<T, double>) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m1.N, m2.M, m1.M,
                1.0, m1.matData.data(), m1.M, m2.matData.data(), m2.M, 0.0,
                res.matData.data(), res.M);
  } else {
    size_t N = m1.N;
    size_t M = m1.M;
    size_t K = m2.M;
    // ... in other cases fallback to manual, slow, matrix multiplication
    for (size_t i = 0; i < N; i++) {
      for (size_t k = 0; k < M; k++) {
        for (size_t j = 0; j < K; j++) {
          res(i, j) += m1(i, k) * m2(k, j);
        }
      }
    }
  }
  return res;
}

template <typename T> void Matrix<T>::reshape(size_t newN, size_t newM) {
  if (this->N * this->M != newN * newM || newN == 0 || newM == 0) {
    throw MatrixException("Matrix reshape not possible");
  }
  this->N = newN;
  this->M = newM;
}

/**
 * TODO: generalize better... this can become the standard function to perform
 * matrix multiplications A*B, A^T*B, A^T*B^T...
 * @param m1
 * @return this.transpose()*m1
 */
template <typename T>
Matrix<T> Matrix<T>::transposeMatMul(const Matrix<T> &m1) {
  // this_ai m1_aj
  if (this->N != m1.N) {
    throw MatrixException("Matrix transpose product size mismatch");
  }
  Matrix<T> res(this->M, m1.M);
  if constexpr (std::is_same_v<T, float>) {
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, this->M, m1.M, this->N,
                1.0f, this->matData.data(), this->M, m1.matData.data(), m1.M,
                0.0f, res.matData.data(), res.M);

  } else if constexpr (std::is_same_v<T, double>) {
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, this->M, m1.M, this->N,
                1.0, this->matData.data(), this->M, m1.matData.data(), m1.M,
                0.0, res.matData.data(), res.M);
  } else {
    res = (*this).transpose() * m1;
  }
  return res;
}

template <typename T> class Vector : public Matrix<T> {
public:
  // TODO: THROWS HAPPEN AFTER MOVE!!! FIX
  Vector(Matrix<T> &&mat) : Matrix<T>{std::move(mat)} {
    if (mat.M != 1) {
      throw MatrixException("Cannot convert given matrix to vector!");
    }
  };

  explicit Vector(std::vector<T> &&arr) : Matrix<T>{std::move(arr)} {}
  // We dont care that this is inefficient since it will be used basically only
  // for small vectors crafted by hand
  explicit Vector(std::initializer_list<T> data) : Matrix<T>{{data}} {
    *this = this->transpose();
  }

  explicit Vector(size_t N) : Matrix<T>{N, 1} {};
  T &operator()(std::size_t x);
  const T &operator()(std::size_t x) const;

  using Matrix<T>::operator();
};

template <typename T> T &Vector<T>::operator()(std::size_t x) {
  return Matrix<T>::operator()(x, 0);
}

template <typename T> const T &Vector<T>::operator()(std::size_t x) const {
  return Matrix<T>::operator()(x, 0);
}

template <typename T>
Matrix<T> outerProduct(const Matrix<T> &v1, const Matrix<T> &v2) {
  if (v1.M != 1 || v2.M != 1) {
    throw MatrixException("outer product shape error!");
  }
  // Mij = vi*vj
  Matrix<T> res(v1.N, v2.N);
  if constexpr (std::is_same_v<T, float>) {
    cblas_sger(CblasRowMajor, v1.N, v2.N, 1.0f, v1.matData.data(), 1,
               v2.matData.data(), 1, res.matData.data(), v2.N);
  } else if constexpr (std::is_same_v<T, double>) {
    cblas_dger(CblasRowMajor, v1.N, v2.N, 1.0f, v1.matData.data(), 1,
               v2.matData.data(), 1, res.matData.data(), v2.N);
  } else {
    for (size_t i = 0; i < v1.N; i++) {
      for (size_t j = 0; j < v2.N; j++) {
        res(i, j) = v1(i, 0) * v2(j, 0);
      }
    }
  }
  return res;
}