#pragma once

#include "../../src/data_loader.h"
#include "../../src/tensor.h"
#include <utility>

#include <fstream>
#include <sstream>

#include <iostream>

inline std::ifstream openFile(std::string_view fileName) {
    std::ifstream file(fileName.data());
    if (!file.is_open()) {
        // TODO: std::format, but my compiler doesnt support it yet...
        std::cout << fileName << std::endl;
        throw std::runtime_error("Cannot open file");
    }
    return file;
}

template <typename DType>
using resType = std::pair<DataLoader<Tensor<DType>, size_t>, DataLoader<Tensor<DType>, size_t>>;

template <typename DType>
std::vector<Tensor<DType>> loadXfile(std::string_view fileName) {
    std::ifstream file = openFile(fileName);

    std::vector<Tensor<DType>> res;
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream stream(line);
        std::vector<double> row;
        double value;

        while (stream >> value) {
            row.push_back(value);
        }

        Tensor<DType> t_row{{row.size()}};
        for (size_t i = 0; i < row.size(); i++) {
            t_row[i] = row[i];
        }
        res.push_back(t_row);
    }
    return res;
}

inline std::vector<size_t> loadYfile(std::string_view fileName) {
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

template <typename DType>
resType<DType> loadMNIST1D() {

    DataLoader<Tensor<DType>, size_t> trainDataset;
    DataLoader<Tensor<DType>, size_t> testDataset;

    auto xTrain = loadXfile<DType>("./datasets/mnist1d/x_train.txt");
    auto yTrain = loadYfile("./datasets/mnist1d/y_train.txt");
    assert(xTrain.size() == yTrain.size());
    for (size_t i = 0; i < xTrain.size(); i++) {
        trainDataset.push(std::move(xTrain[i]), std::move(yTrain[i]));
    }

    auto xTest = loadXfile<DType>("./datasets/mnist1d/x_test.txt");
    auto yTest = loadYfile("./datasets/mnist1d/y_test.txt");
    assert(xTest.size() == yTest.size());
    for (size_t i = 0; i < xTest.size(); i++) {
        testDataset.push(std::move(xTest[i]), std::move(yTest[i]));
    }
    return std::pair(std::move(trainDataset), std::move(testDataset));
}