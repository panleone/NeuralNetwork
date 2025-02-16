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
  virtual void forwardStep(Matrix<T> &prevLayerOutput) = 0;
  virtual ~BaseLayer() = default;
  virtual void backwardStep(Matrix<T> &prevGradient) = 0;
  virtual void setFinalizer(FINALIZER finalizer, std::span<T> params) = 0;
  virtual void finalize(size_t batchSize) = 0;
};

template <typename T> class ReluLayer : public BaseLayer<T> {
  Matrix<int> reLUMask;

public:
  ReluLayer(std::size_t inN, size_t inM = 1) : reLUMask(inN, inM){};
  void forwardStep(Matrix<T> &prevLayerOutput) override {
    if (prevLayerOutput.N != reLUMask.N || prevLayerOutput.M != reLUMask.M) {
      throw std::runtime_error("ReLU activation bad size");
    }
    // ReLU activation unit
    for (size_t i = 0; i < prevLayerOutput.N; i++) {
      for (size_t j = 0; j < prevLayerOutput.M; j++) {
        if (prevLayerOutput(i, j) <= static_cast<T>(0.0)) {
          prevLayerOutput(i, j) = static_cast<T>(0.0);
          reLUMask(i, j) = 0;
        } else {
          reLUMask(i, j) = 1;
        }
      }
    }
  }

  void backwardStep(Matrix<T> &prevGradient) override {
    if (prevGradient.N != reLUMask.N || prevGradient.M != reLUMask.M) {
      throw std::runtime_error("ReLU activation bad size");
    }
    for (size_t i = 0; i < prevGradient.N; i++) {
      for (size_t j = 0; j < prevGradient.M; j++) {
        if (reLUMask(i, j) == 0) {
          prevGradient(i, j) = static_cast<T>(0.0);
        }
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
  size_t prevLayerN;
  size_t prevLayerM;

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

  void forwardStep(Matrix<T> &prevLayerOutput) override {
    // Cache the dimension and reshape the input to vector shape
    this->prevLayerN = prevLayerOutput.N;
    this->prevLayerM = prevLayerOutput.M;
    prevLayerOutput.reshape(prevLayerOutput.N * prevLayerOutput.M, 1);
    // Cache the output of the previous layer (Will be used in backpropagation)
    this->prevLayerOutputCache = prevLayerOutput.clone();
    prevLayerOutput = this->weights * prevLayerOutput;
    prevLayerOutput += bias;
  }

  void backwardStep(Matrix<T> &prevGradient) override {
    // Cache the gradient
    this->biasGradient += prevGradient;
    this->weightsGradient +=
        outerProduct(prevGradient, this->prevLayerOutputCache);

    // Update the gradient
    prevGradient = this->weights.transposeMatMul(prevGradient);
    prevGradient.reshape(this->prevLayerN, this->prevLayerM);
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
    } else {
      assert(false);
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

// For the moment 1D convolution
template <typename T> class ConvolutionLayer : public BaseLayer<T> {
private:
  std::vector<Matrix<T>> convolutionMatrices;
  Vector<T> bias;

  std::vector<Matrix<T>> convolutionMatricesGradient;
  std::vector<std::unique_ptr<FinalizerBase<Matrix<T>>>>
      convolutionMatricesFinalizer;

  Vector<T> biasGradient;
  std::unique_ptr<FinalizerBase<Vector<T>>> biasFinalizer;

  size_t stride;
  size_t outChannel;
  size_t inChannel;
  size_t size;

  // TODO: avoid this initialization
  Matrix<T> prevLayerOutputCache{{0}};

public:
  ConvolutionLayer(size_t size, size_t inChannel, size_t outChannel,
                   size_t stride, bool isUnitTest = false)
      : bias(outChannel), convolutionMatricesFinalizer(outChannel),
        biasGradient(outChannel), stride{stride},
        outChannel{outChannel}, inChannel{inChannel}, size{size} {
    for (size_t i = 0; i < outChannel; i++) {
      Matrix<T> iConvolution = randomMatrix<T>(
          size, inChannel, static_cast<T>(0.0),
          sqrt(static_cast<T>(4.0) / (static_cast<T>(size + inChannel))));
      if (isUnitTest) {
        iConvolution =
            deterministicConstantMatrix(size, inChannel, static_cast<T>(i + 1));
      }
      convolutionMatrices.push_back(std::move(iConvolution));
      convolutionMatricesGradient.push_back(Matrix<T>(size, inChannel));
    }
  }

  virtual void forwardStep(Matrix<T> &prevLayerOutput) {
    assert(prevLayerOutput.M == inChannel);
    assert(size < prevLayerOutput.N);

    size_t maxSteps = prevLayerOutput.N - size;
    size_t outN = 1 + (maxSteps - maxSteps % stride) / stride;

    Matrix<T> layerOutput(outN, outChannel);
    for (size_t j = 0; j < outN; j += 1) {
      for (size_t i = 0; i < outChannel; i++) {
        for (size_t l = 0; l < size; l++) {
          for (size_t k = 0; k < inChannel; k++) {

            layerOutput(j, i) += convolutionMatrices[i](l, k) *
                                 prevLayerOutput(j * stride + l, k);
          }
        }
        layerOutput(j, i) += bias(i);
      }
    }
    prevLayerOutputCache = prevLayerOutput.clone();
    prevLayerOutput = std::move(layerOutput);
  }
  virtual void backwardStep(Matrix<T> &prevGradient) {
    assert(prevGradient.M == outChannel);
    // 1) Bias gradient
    for (size_t j = 0; j < prevGradient.N; j++) {
      for (size_t i = 0; i < outChannel; i++) {
        biasGradient(i) += prevGradient(j, i);
      }
    }
    // 2) Weights gradient
    for (size_t i = 0; i < prevGradient.N; i++) {
      for (size_t j = 0; j < outChannel; j++) {
        for (size_t alpha = 0; alpha < size; alpha++) {
          for (size_t beta = 0; beta < inChannel; beta++) {

            convolutionMatricesGradient[j](alpha, beta) +=
                prevGradient(i, j) *
                prevLayerOutputCache(alpha + i * stride, beta);
          }
        }
      }
    }
    // 3) Update the flowing gradient
    Matrix<T> updatedGradient(prevLayerOutputCache.N, prevLayerOutputCache.M);
    for (size_t alpha = 0; alpha < prevLayerOutputCache.N; alpha++) {
      size_t rMin = alpha % stride;
      int qMax = std::min((alpha - rMin) / stride, prevGradient.N - 1);
      for (size_t beta = 0; beta < prevLayerOutputCache.M; beta++) {
        for (size_t j = 0; j < outChannel; j++) {
          size_t r = rMin;
          int q = qMax;
          while (q >= 0 and r < size) {
            updatedGradient(alpha, beta) +=
                prevGradient(q, j) * convolutionMatrices[j](r, beta);
            r += stride;
            q -= 1;
          }
        }
      }
    }
    prevGradient = std::move(updatedGradient);
  };
  virtual void setFinalizer(FINALIZER finalizer, std::span<T> params) {
    if (finalizer == FINALIZER::STANDARD) {
      // params[0] = alpha
      assert(params.size() == 1);
      for (size_t i = 0; i < outChannel; i++) {
        convolutionMatricesFinalizer[i] =
            std::make_unique<StandardFinalizer<Matrix<T>>>(
                convolutionMatrices[i], params[0]);
      }
      biasFinalizer =
          std::make_unique<StandardFinalizer<Vector<T>>>(bias, params[0]);
    } else if (finalizer == FINALIZER::ADAM) {
      assert(params.size() == 4);
      for (size_t i = 0; i < outChannel; i++) {
        convolutionMatricesFinalizer[i] =
            std::make_unique<AdamFinalizer<Matrix<T>>>(convolutionMatrices[i],
                                                       params[0], params[1],
                                                       params[2], params[3]);
      }
      biasFinalizer = std::make_unique<AdamFinalizer<Vector<T>>>(
          bias, params[0], params[1], params[2], params[3]);
    } else {
      assert(false);
    }
  };
  virtual void finalize(size_t batchSize) {
    T tBatchSize = static_cast<T>(batchSize);

    biasGradient /= tBatchSize;
    biasFinalizer->finalize(biasGradient);
    biasGradient *= static_cast<T>(0.0);

    for (size_t i = 0; i < outChannel; i++) {
      convolutionMatricesGradient[i] /= tBatchSize;
      convolutionMatricesFinalizer[i]->finalize(convolutionMatricesGradient[i]);
      convolutionMatricesGradient[i] *= static_cast<T>(0.0);
    }
  };
};