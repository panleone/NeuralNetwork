#include "load_mnist1d.h"

#include <fstream>
#include <sstream>

std::ifstream openFile(std::string_view fileName) {
  std::ifstream file(fileName.data());
  if (!file.is_open()) {
    // TODO: std::format, but my compiler doesnt support it yet...
    throw std::runtime_error("Cannot open file");
  }
  return file;
}
std::vector<Vector<float>> loadXfile(std::string_view fileName) {
  std::ifstream file = openFile(fileName);

  std::vector<Vector<float>> res;
  std::string line;
  while (std::getline(file, line)) {
    std::istringstream stream(line);
    std::vector<float> row;
    double value;

    while (stream >> value) {
      row.push_back(value);
    }
    res.push_back(Vector<float>(std::move(row)));
  }
  return res;
}

std::vector<size_t> loadYfile(std::string_view fileName) {
  std::ifstream file = openFile(fileName);
  std::vector<size_t> res;
  std::string line;
  while (std::getline(file, line)) {
    try {
      res.push_back(std::stoull(line));
    } catch (const std::invalid_argument &e) {
      throw std::runtime_error("Invalid data encountered while parsing file");
    }
  }
  return res;
}

resType loadMNIST1D() {
  DataLoader<Vector<float>, size_t> trainDataset;
  DataLoader<Vector<float>, size_t> testDataset;

  auto xTrain = loadXfile("./datasets/mnist1d/x_train.txt");
  auto yTrain = loadYfile("./datasets/mnist1d/y_train.txt");
  assert(xTrain.size() == yTrain.size());
  for (size_t i = 0; i < xTrain.size(); i++) {
    trainDataset.push(std::move(xTrain[i]), std::move(yTrain[i]));
  }

  auto xTest = loadXfile("./datasets/mnist1d/x_test.txt");
  auto yTest = loadYfile("./datasets/mnist1d/y_test.txt");
  assert(xTest.size() == yTest.size());
  for (size_t i = 0; i < xTest.size(); i++) {
    testDataset.push(std::move(xTest[i]), std::move(yTest[i]));
  }
  return std::pair(std::move(trainDataset), std::move(testDataset));
}