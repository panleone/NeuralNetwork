#pragma once

#include <algorithm> // for std::for_each
#include <iostream>
#include <memory>

#include "layer.h"
#include "optimizer.h"
#include <span>

template <typename T, typename Ty, LOSS LossType> class NeuralNetwork {
private:
  T accumulatedGradient = static_cast<T>(0.0);
  size_t batchedElements = 0;

public:
  template <typename... Layers> NeuralNetwork(Layers &&...layers) {
    initialize_layers(std::forward<Layers &&>(layers)...);
  }

  Vector<T> predict(Vector<T> input) {
    std::for_each(nnLayers.begin(), nnLayers.end(),
                  [&](auto &layer) { layer->forwardStep(input); });
    return input;
  }

  T forwardOne(const Vector<T> &input, const Ty &out) {
    // input is processed layer by layer
    Vector<T> res = input.clone();
    std::for_each(nnLayers.begin(), nnLayers.end(),
                  [&](auto &layer) { layer->forwardStep(res); });

    Vector<T> gradient{optimizer.lossGradient(out, res)};
    std::for_each(nnLayers.rbegin(), nnLayers.rend(),
                  [&](auto &layer) { layer->backwardStep(gradient); });

    batchedElements += 1;
    accumulatedGradient += gradient(0);
    return optimizer.loss(out, res);
  }

  using DataPair = std::pair<Vector<T>, Ty>;

  T forwardBatch(std::span<DataPair> batch) {
    T averageLoss = static_cast<T>(0.0);
    for (auto &[x, y] : batch) {
      averageLoss += forwardOne(x, y);
    }
    return averageLoss / batch.size();
  }
  void backward(const T &alpha, const T &beta) {
    std::for_each(nnLayers.begin(), nnLayers.end(), [&](auto &layer) {
      layer->finalize(alpha, beta, batchedElements);
    });

    accumulatedGradient = static_cast<T>(0.0);
    batchedElements = 0;
  }

  const T &getGradient() const { return accumulatedGradient; }

private:
  Optimizer<T, Ty, LossType> optimizer;
  std::vector<std::unique_ptr<BaseLayer<T>>> nnLayers;

  template <typename V, typename... Args>
  void initialize_layers(V &&first, Args &&...args) {
    nnLayers.push_back(std::forward<V &&>(first));
    initialize_layers(std::forward<Args &&>(args)...);
  }
  void initialize_layers(){};
};