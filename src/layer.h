#pragma once

#include "math.h"
#include "matrix.h"
#include "random.h"
#include <cassert>
#include <memory>
#include <span>

#include "finalizer.h"

template <typename T> class BaseLayer {
public:
  virtual void forwardStep(Vector<T> &prevLayerOutput) = 0;
  virtual ~BaseLayer() = default;
  virtual void backwardStep(Vector<T> &prevGradient) = 0;
  virtual void setFinalizer(FINALIZER finalizer, std::span<T> params) = 0;
  virtual void finalize(size_t batchSize) = 0;
};

template <typename T> class ReluLayer : public BaseLayer<T> {
  Vector<int> reLUMask;

public:
  ReluLayer(std::size_t in) : reLUMask(in){};
  void forwardStep(Vector<T> &prevLayerOutput) override {
    if (prevLayerOutput.N != reLUMask.N) {
      throw std::runtime_error("ReLU activation bad size");
    }
    // ReLU activation unit
    for (size_t i = 0; i < prevLayerOutput.N; i++) {
      if (prevLayerOutput(i) <= static_cast<T>(0.0)) {
        prevLayerOutput(i) = static_cast<T>(0.0);
        reLUMask(i) = 0;
      } else {
        reLUMask(i) = 1;
      }
    }
  }

  void backwardStep(Vector<T> &prevGradient) override {
    if (prevGradient.N != reLUMask.N) {
      throw std::runtime_error("ReLU activation bad size");
    }
    for (size_t i = 0; i < prevGradient.N; i++) {
      if (reLUMask(i) == 0) {
        prevGradient(i) = static_cast<T>(0.0);
      }
    }
  }
  // Nothing to update
  void setFinalizer([[maybe_unused]] FINALIZER finalizer,
                    [[maybe_unused]] std::span<T> params) override{};
  void finalize([[maybe_unused]] size_t batchSize) override{};
};

template <typename T> class FullyConnectedLayer : public BaseLayer<T> {
private:
  Matrix<T> weights;
  Vector<T> bias;

  // the gradient of each batch is stored here
  Matrix<T> weightsGradient;
  std::unique_ptr<FinalizerBase<Matrix<T>>> weightsFinalizer;

  Vector<T> biasGradient;
  std::unique_ptr<FinalizerBase<Vector<T>>> biasFinalizer;

  Vector<T> prevLayerOutputCache;

public:
  // HE initialization
  FullyConnectedLayer(std::size_t in, std::size_t out, bool isUnitTest = false)
      : weights{randomMatrix<T>(
            out, in, static_cast<T>(0.0),
            sqrt(static_cast<T>(4.0) / (static_cast<T>(in + out))))},
        bias(out), weightsGradient(out, in), biasGradient(out),
        prevLayerOutputCache(in) {
    // For testing purposes we just set all elements of the matrix to the
    // constant value one
    if (isUnitTest) {
      weights = deterministicConstantMatrix(out, in, static_cast<T>(1.0));
    }
  }

  void forwardStep(Vector<T> &prevLayerOutput) override {
    // Cache the output of the previous layer (Will be used in backpropagation)
    this->prevLayerOutputCache = prevLayerOutput.clone();
    prevLayerOutput = this->weights * prevLayerOutput;
    prevLayerOutput += bias;
  }

  void backwardStep(Vector<T> &prevGradient) override {
    // Cache the gradient
    this->biasGradient += prevGradient;
    this->weightsGradient +=
        outerProduct(prevGradient, this->prevLayerOutputCache);

    // Update the gradient
    prevGradient = this->weights.transposeMatMul(prevGradient);
  }

  void setFinalizer(FINALIZER finalizer, std::span<T> params) override {
    if (finalizer == FINALIZER::STANDARD) {
      // params[0] = alpha
      assert(params.size() == 1);
      weightsFinalizer =
          std::make_unique<StandardFinalizer<Matrix<T>>>(weights, params[0]);
      biasFinalizer =
          std::make_unique<StandardFinalizer<Vector<T>>>(bias, params[0]);
    } else if (finalizer == FINALIZER::MOMENTUM) {
      // params[0] = alpha, params[1] = beta
      assert(params.size() == 2);
      weightsFinalizer = std::make_unique<MomentumFinalizer<Matrix<T>>>(
          weights, params[0], params[1]);
      biasFinalizer = std::make_unique<MomentumFinalizer<Vector<T>>>(
          bias, params[0], params[1]);
    } else if (finalizer == FINALIZER::ADAM) {
      // params[0] = alpha, params[1] = beta,
      // params[2] = gamma, params[3] = epsilon
      assert(params.size() == 4);
      weightsFinalizer = std::make_unique<AdamFinalizer<Matrix<T>>>(
          weights, params[0], params[1], params[2], params[3]);
      biasFinalizer = std::make_unique<AdamFinalizer<Vector<T>>>(
          bias, params[0], params[1], params[2], params[3]);
    }
  }

  void finalize(size_t batchSize) override {
    T tBatchSize = static_cast<T>(batchSize);
    weightsGradient /= tBatchSize;
    weightsFinalizer->finalize(weightsGradient);

    biasGradient /= tBatchSize;
    biasFinalizer->finalize(biasGradient);

    weightsGradient *= static_cast<T>(0.0);
    biasGradient *= static_cast<T>(0.0);
  }

  // For tests/debugging only
  const Vector<T> &getBias() const { return bias; }
  const Matrix<T> &getWeights() const { return weights; }
  const Matrix<T> &getWeightsGradient() const { return weightsGradient; }
  const Vector<T> &getBiasGradient() const { return biasGradient; }
};