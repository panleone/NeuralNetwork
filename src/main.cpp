#include <fenv.h>
#include <iostream>

#include "layer.h"
#include "neural_network.h"
#include "random.h"

#include "tests/test_runner.h"

#include "../datasets/mnist1d/load_mnist1d.h"
#include "data_loader.h"
#include "finalizer.h"
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
  std::cout << "Test passed" << std::endl;

  auto [trainDataset, testDataset] = loadMNIST1D();
  auto testData = testDataset.getData();

  {
    // This fully connected NN achieves around 26.0% error rate on the MNIST1D
    // dataset, by using ADAM Finalizer:
    // model.setAdamFinalizer(0.005f, 0.9f, 0.999f, 1.0e-6);
    // And batchSize = 100
    NeuralNetwork<float, size_t, LOSS::SOFTMAX> model{
        std::make_unique<FullyConnectedLayer<float>>(40, 100),
        std::make_unique<ReluLayer<float>>(100),
        std::make_unique<FullyConnectedLayer<float>>(100, 100),
        std::make_unique<ReluLayer<float>>(100),
        std::make_unique<FullyConnectedLayer<float>>(100, 100),
        std::make_unique<ReluLayer<float>>(100),
        std::make_unique<FullyConnectedLayer<float>>(100, 10),
    };
  }

  // This convolution neural network achieves 5.5% error rate on the MNIST1D dataset
  size_t outChannels = 50;
  NeuralNetwork<float, size_t, LOSS::SOFTMAX> cnnModel{
      std::make_unique<ConvolutionLayer<float>>(3, 1, outChannels, 2),
      std::make_unique<ReluLayer<float>>(19, outChannels),
      std::make_unique<ConvolutionLayer<float>>(3, outChannels, outChannels, 2),
      std::make_unique<ReluLayer<float>>(9, outChannels),
      std::make_unique<ConvolutionLayer<float>>(3, outChannels, outChannels, 2),
      std::make_unique<ReluLayer<float>>(4, outChannels),
      std::make_unique<FullyConnectedLayer<float>>(4 * outChannels, 10),
  };

  cnnModel.setAdamFinalizer(0.005f, 0.9f, 0.999f, 1.0e-6);
  size_t epoch = 2500;
  size_t batchSize = 100;

  for (size_t i = 0; i < epoch; i++) {
    float lossPerEpoch{0.0f};

    trainDataset.randomIter(batchSize, [&](auto batch) {
      lossPerEpoch += cnnModel.forwardBatch(batch);
      cnnModel.backward();
    });
    std::cout << "Epoch " << i + 1 << " completed. Loss was: " << lossPerEpoch
              << std::endl;

    size_t goodPredictions{0};
    for (const auto &[xTest, yTest] : testData) {
      auto pred = cnnModel.predict(xTest.clone());
      auto argmaxIt = max_element(pred.matData.begin(), pred.matData.end());
      size_t argmax =
          static_cast<size_t>(std::distance(pred.matData.begin(), argmaxIt));
      if (yTest == argmax) {
        goodPredictions += 1;
      }
    }
    std::cout << "Epoch " << i + 1 << " error on test data is: "
              << 100.0f * (1.0f - static_cast<float>(goodPredictions) /
                                      static_cast<float>(testData.size()))
              << "%" << std::endl;
  }

  return 0;
}