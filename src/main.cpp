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

  NeuralNetwork<float, LOSS::MSE> model{
      std::make_unique<FullyConnectedLayer<float>>(1, 10),
      std::make_unique<ReluLayer<float>>(10),
      std::make_unique<FullyConnectedLayer<float>>(10, 10),
      std::make_unique<ReluLayer<float>>(10),
      std::make_unique<FullyConnectedLayer<float>>(10, 1),
  };

  DataLoader<float> loader;
  for (int i = 0; i < 10000; i++) {
    float x = static_cast<float>(i) / 5000.0f;
    loader.push(Vector<float>{x}, Vector<float>{x * x + 5.0f * x + 4.0f});
  }
  size_t epoch = 50;
  size_t batchSize = 100;
  float alpha = 0.01f;
  float beta = 0.9f;
  for (size_t i = 0; i < epoch; i++) {
    loader.randomIter(batchSize, [&](auto batch) {
      auto loss = model.forwardBatch(batch);
      model.backward(alpha, beta);
      std::cout << "Batch handled, average loss: " << loss << std::endl;
    });

    if (i > 0 && i % 20 == 0) {
      alpha /= 2.0f;
    }
  }
  return 0;
}