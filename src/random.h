#pragma once

#include <random>

#include "matrix.h"
#include <algorithm>

template <typename T>
T randomGaussianInternal(std::mt19937 &gen, const T &mu, const T &sigma) {
  std::normal_distribution d(mu, sigma);
  return d(gen);
}

template <typename T>
Matrix<T> randomMatrix(size_t N, size_t M, const T &mu, const T &sigma) {
  std::random_device rd{};
  std::mt19937 gen{rd()};
  Matrix<T> res(N, M);
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < M; j++) {
      res(i, j) = randomGaussianInternal(gen, mu, sigma);
    }
  }
  return res;
}

template <typename T>
Matrix<T> randomVector(size_t N, const T &mu, const T &sigma) {
  std::random_device rd{};
  std::mt19937 gen{rd()};
  Vector<T> res(N);
  for (size_t i = 0; i < N; i++) {
    res(i) = randomGaussianInternal(gen, mu, sigma);
  }
  return res;
}

template <typename T>
Matrix<T> deterministicConstantMatrix(size_t N, size_t M, const T &val) {
  Matrix<T> res(N, M);
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < M; j++) {
      res(i, j) = val;
    }
  }
  return res;
}

template <typename RandomIt> void shuffle(RandomIt first, RandomIt last) {
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(first, last, g);
}

template <typename T> T randomNumber(const T &mu, const T &sigma) {
  std::random_device rd{};
  std::mt19937 gen{rd()};
  return randomGaussianInternal(gen, mu, sigma);
}