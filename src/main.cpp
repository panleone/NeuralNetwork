#include <fenv.h>
#include <iostream>

#include "layer.h"
#include "neural_network.h"
#include "random.h"

#include "tests/test_runner.h"

#include "data_loader.h"

int main() {
  feenableexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW);

  // Run tests
  try {
    run_tests();
  } catch (std::runtime_error &err) {
    std::cerr << "Some tests failed! Is not safe to continue... quitting"
              << std::endl;
    std::cerr << err.what() << std::endl;
    exit(1);
  }

  NeuralNetwork<float, float, LOSS::MSE> model{
      std::make_unique<FullyConnectedLayer<float>>(1, 10),
      std::make_unique<ReluLayer<float>>(10),
      std::make_unique<FullyConnectedLayer<float>>(10, 10),
      std::make_unique<ReluLayer<float>>(10),
      std::make_unique<FullyConnectedLayer<float>>(10, 1),
  };

  NeuralNetwork<float, size_t, LOSS::SOFTMAX> model_classification{
      std::make_unique<FullyConnectedLayer<float>>(1, 10),
      std::make_unique<ReluLayer<float>>(10),
      std::make_unique<FullyConnectedLayer<float>>(10, 10),
      std::make_unique<ReluLayer<float>>(10),
      std::make_unique<FullyConnectedLayer<float>>(10, 10),
      std::make_unique<ReluLayer<float>>(10),
      std::make_unique<FullyConnectedLayer<float>>(10, 10),
      std::make_unique<ReluLayer<float>>(10),
      std::make_unique<FullyConnectedLayer<float>>(10, 10),
      std::make_unique<ReluLayer<float>>(10),
      std::make_unique<FullyConnectedLayer<float>>(10, 5),
  };
  DataLoader<float, size_t> loader;
  for (int i = 0; i < 10000; i++) {
    float x = static_cast<float>(i) / 5000.0f;
    size_t res = 0;

    if (i > 2000) {
      res = 1;
    }
    if (i > 4000) {
      res = 2;
    }
    if (i > 6000) {
      res = 3;
    }
    if (i > 8000) {
      res = 4;
    }
    loader.push(Vector<float>{x}, Vector<size_t>{res});
  }
  size_t epoch = 50;
  size_t batchSize = 100;
  float alpha = 0.1f;
  float beta = 0.9f;
  for (size_t i = 0; i < epoch; i++) {
    loader.randomIter(batchSize, [&](auto batch) {
      auto loss = model_classification.forwardBatch(batch);
      model_classification.backward(alpha, beta);
      std::cout << "Batch handled, average loss: " << loss << std::endl;
    });

    if (i > 0 && i % 20 == 0) {
      alpha /= 2.0f;
    }
  }
  for (size_t i = 0; i < 100; i++) {
    std::cout << model_classification.predict(
        Vector<float>{((float)i) / 5000.0f});
  }
  std::cout << "---" << std::endl;
  for (size_t i = 6900; i < 7000; i++) {
    std::cout << model_classification.predict(
        Vector<float>{((float)i) / 5000.0f});
  }
  return 0;
}