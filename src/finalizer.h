#pragma once

#include "math.h"
#include <iostream>

enum class FINALIZER {
  /**
   * Standard update rule for gradient descent
   */
  STANDARD,
  /**
   * Gradient descent with momentum term
   */
  MOMENTUM,
  /**
   * ADAM optimizer.
   */
  ADAM
};

template <typename T> class FinalizerBase {
protected:
  T &watchedRef;

public:
  FinalizerBase(T &watched) : watchedRef{watched} {};
  virtual ~FinalizerBase() = default;
  virtual void finalize(const T &gradient) = 0;
};

template <typename T> class StandardFinalizer : public FinalizerBase<T> {
private:
  T::Field alpha;

public:
  StandardFinalizer(T &watched, T::Field alpha)
      : FinalizerBase<T>{watched}, alpha{std::move(alpha)} {};
  void finalize(const T &gradient) override {
    this->watchedRef -= alpha * gradient;
  }
};

template <typename T> class MomentumFinalizer : public FinalizerBase<T> {
private:
  using K = T::Field;
  T momentum;
  K alpha;
  K beta;

public:
  // TODO: avoid the clone
  MomentumFinalizer(T &watched, K alpha, K beta)
      : FinalizerBase<T>{watched}, momentum{watched.clone()},
        alpha{std::move(alpha)}, beta{std::move(beta)} {
    momentum *= 0;
  };
  void finalize(const T &gradient) override {
    momentum = beta * momentum + (static_cast<K>(1) - beta) * gradient;
    this->watchedRef -= alpha * momentum;
  }
};

template <typename T> class AdamFinalizer : public FinalizerBase<T> {
private:
  using K = T::Field;
  T momentum;
  T momentumSquared;
  K alpha;
  K beta;
  K gamma;
  K epsilon;
  size_t timeStamp{0};

public:
  // TODO: avoid the clone
  AdamFinalizer(T &watched, K alpha, K beta, K gamma, K epsilon)
      : FinalizerBase<T>{watched}, momentum{watched.clone()},
        momentumSquared{watched.clone()}, alpha{std::move(alpha)},
        beta{std::move(beta)}, gamma{std::move(gamma)}, epsilon{std::move(
                                                            epsilon)} {
    momentum *= 0;
    momentumSquared *= 0;
  };
  // TODO: implement matrix broadcasting
  void finalize(const T &gradient) override {
    momentum = beta * momentum + (1 - beta) * gradient;

    for (size_t i = 0; i < this->watchedRef.N; i++) {
      for (size_t j = 0; j < this->watchedRef.M; j++) {
        momentumSquared(i, j) =
            gamma * momentumSquared(i, j) +
            (static_cast<K>(1) - gamma) * gradient(i, j) * gradient(i, j);
      }
    }

    auto momentumRescaled = momentum.clone();
    momentumRescaled /= (static_cast<K>(1) - pow(beta, timeStamp + 1));

    auto momentumSquaredRescaled = momentumSquared.clone();
    momentumSquaredRescaled /= (static_cast<K>(1) - pow(gamma, timeStamp + 1));
    for (size_t i = 0; i < this->watchedRef.N; i++) {
      for (size_t j = 0; j < this->watchedRef.M; j++) {
        this->watchedRef(i, j) -=
            alpha * momentumRescaled(i, j) /
            (std::sqrt(momentumSquaredRescaled(i, j)) + epsilon);
      }
    }
    timeStamp += 1;
  }
};
