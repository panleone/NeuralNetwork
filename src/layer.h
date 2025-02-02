#pragma once

#include <cassert>
#include <memory>

#include "math.h"
#include "matrix.h"
#include "random.h"

template<typename T>
class BaseLayer {
public:
    virtual void forwardStep(Vector<T>& prevLayerOutput) = 0;
    virtual ~BaseLayer() = default;
    virtual void backwardStep(Vector<T>& prevGradient) = 0;
    virtual void finalize(const T& alpha, size_t batchSize) = 0;
};

template<typename T>
class ReluLayer : public BaseLayer<T>{
    Vector<int> reLUMask;
public:
    ReluLayer(std::size_t in) : reLUMask(in) {};
    void forwardStep(Vector<T>& prevLayerOutput) override{
        if(prevLayerOutput.N != reLUMask.N){
            throw std::runtime_error("ReLU activation bad size");
        }
        // ReLU activation unit
        for (size_t i = 0; i < prevLayerOutput.N; i++) {
            if(prevLayerOutput(i) <= static_cast<T>(0.0)){
                prevLayerOutput(i) = static_cast<T>(0.0);
                reLUMask(i) = 0;
            } else {
                reLUMask(i) = 1;
            }
        }
    }

    void backwardStep(Vector<T>& prevGradient) override {
        if (prevGradient.N != reLUMask.N){
            throw std::runtime_error("ReLU activation bad size");
        }
        for (size_t i = 0; i < prevGradient.N; i++) {
            if (reLUMask(i) == 0) {
                prevGradient(i) = static_cast<T>(0.0);
            }
        }
    }
    // Nothing to update
    void finalize([[maybe_unused]] const T& alpha, [[maybe_unused]] size_t batchSize) override{};
};

template<typename T>
class FullyConnectedLayer : public BaseLayer<T>{
private:
    Matrix<T> weights;
    Vector<T> bias;

    // the gradient of each batch is stored here
    Matrix<T> weightsGradient;
    Vector<T> biasGradient;

    Vector<T> prevLayerOutputCache;
public:
    // HE initialization
    FullyConnectedLayer(std::size_t in, std::size_t out, bool isUnitTest = false) :
        weights{randomMatrix<T>(out, in, static_cast<T>(0.0), sqrt( static_cast<T>(4.0) / (static_cast<T>(in + out))))}, bias(out),
        weightsGradient(out, in), biasGradient(out),
        prevLayerOutputCache(in) {
        // For testing purposes we just set all elements of the matrix to the constant value one
        if(isUnitTest){
            weights = deterministicConstantMatrix(out, in, static_cast<T>(1.0));
        }
    }

    void forwardStep(Vector<T>& prevLayerOutput) override {
        // Cache the output of the previous layer (Will be used in backpropagation)
        this->prevLayerOutputCache = prevLayerOutput.clone();
        prevLayerOutput = this->bias + this->weights * prevLayerOutput;
    }

    void backwardStep(Vector<T>& prevGradient) override {
        // Cache the gradient
        this->biasGradient += prevGradient;
        this->weightsGradient += outerProduct(prevGradient, this->prevLayerOutputCache);

        // Update the gradient
        prevGradient = this->weights.transpose() * prevGradient;
    }

    void finalize(const T& alpha, size_t batchSize) override {
        weightsGradient /= batchSize;
        weightsGradient *= alpha;
        weights -= weightsGradient;

        biasGradient /= batchSize;
        biasGradient *= alpha;
        bias -= biasGradient;

        weightsGradient *= static_cast<T>(0.0);
        biasGradient *= static_cast<T>(0.0);
    }

    // For tests/debugging only
    const Vector<T>& getBias() const { return bias;}
    const Matrix<T>& getWeights() const { return weights;}
    const Matrix<T>& getWeightsGradient() const { return weightsGradient;}
    const Vector<T>& getBiasGradient() const { return biasGradient;}
};