#pragma once

#include <algorithm> // for std::for_each
#include <iostream>
#include <memory>

#include "layer.h"

template <typename T> T lossFunction(const T &f, const T &y) {
  return (f - y) * (f - y);
}

template <typename T> T lossFunctionDerivative(const T &f, const T &y) {
  return 2 * (f - y);
}

template <typename T> class NeuralNetwork {
private:
  T accumulatedGradient = static_cast<T>(0.0);
  T accumulatedLoss = static_cast<T>(0.0);
  size_t batchedElements = 0;

public:
  template <typename... Layers> NeuralNetwork(Layers &&...layers) {
    initialize_layers(std::forward<Layers &&>(layers)...);
  }

  T forward(Vector<T> input, const T &out) {
    // input is processed layer by layer
    std::for_each(nnLayers.begin(), nnLayers.end(),
                  [&](auto &layer) { layer->forwardStep(input); });

    Vector<T> gradient{lossFunctionDerivative(input(0), out)};
    std::for_each(nnLayers.rbegin(), nnLayers.rend(),
                  [&](auto &layer) { layer->backwardStep(gradient); });

    batchedElements += 1;
    accumulatedGradient += gradient(0);
    accumulatedLoss += lossFunction(input(0), out);
    return input(0);
  }
  void backward(const T &alpha) {
    std::for_each(nnLayers.begin(), nnLayers.end(), [&](auto &layer) {
      layer->finalize(alpha, batchedElements);
    });

    // std::cout <<  "[DEBUG] Average loss function is: " <<
    // accumulatedLoss/batchedElements << " on a batch of size: " <<
    // batchedElements << std::endl; std::cout << "[DEBUG] Average gradient is:
    // " << accumulatedGradient / static_cast<T>(batchedElements) << std::endl;
    accumulatedGradient = static_cast<T>(0.0);
    accumulatedLoss = static_cast<T>(0.0);
    batchedElements = 0;
  }

  const T &getGradient() const { return accumulatedGradient; }
  size_t getBatchedElements() const { return batchedElements; }

private:
  std::vector<std::unique_ptr<BaseLayer<T>>> nnLayers;

  template <typename V, typename... Args>
  void initialize_layers(V &&first, Args &&...args) {
    nnLayers.push_back(std::forward<V &&>(first));
    initialize_layers(std::forward<Args &&>(args)...);
  }
  void initialize_layers(){};
};