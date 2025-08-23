#include <iostream>

#include "expressions/expression.h"

#include "debug_utils.h"

#include <memory>

#include "metaprogramming/stack.h"
#include "interpreter.h"

#include "tensor.h"

#include "optimizer.h"

#include "random.h"

#include "weight_initializer.h"
#include "loss.h"
#include <cmath>

#include "../datasets/mnist1d/load_mnist1d.h"

#include "tests/test_runner.h"

#include "avx/avx_wrapper.h"

#include "layers/linear_layer.h"
#include "layers/convolution_layer.h"
#include "layers/relu_layer.h"
#include "layers/flattener_layer.h"

#include "tensor_variable.h"

#include <chrono>

template <typename DType>
class MyModel {
  private:
    ConvolutionLayer1D<DType> l1;
    ConvolutionLayer1D<DType> l2;
    ConvolutionLayer1D<DType> l3;
    LinearLayer<DType> l4;

  public:
    MyModel(size_t out_channels)
        : l1{ConvolutionLayer1D<DType>{1, out_channels, 3, 2}},
          l2{ConvolutionLayer1D<DType>{out_channels, out_channels, 3, 2}},
          l3{ConvolutionLayer1D<DType>{out_channels, out_channels, 3, 2}}, l4{LinearLayer<DType>(
                                                                               4 * out_channels,
                                                                               10)} {}

    template <typename Expr>
    auto forward(const Expr &x) {
        auto y1 = relu(l1.forward(x));
        auto y2 = relu(l2.forward(y1));
        auto y3 = relu(l3.forward(y2));
        return l4.forward(flatten(y3));
    }
    auto get_parameters() {
        // Dummy tensor
        auto comp_graph = forward(no_grad(Tensor<DType>{{1}}));
        return comp_graph.get_parameters();
    }
};

int main() {
    // Run tests
    run_tests();

    using NNType = float;
    auto [train_set, test_set] = loadMNIST1D<NNType>();

    auto start_time = std::chrono::high_resolution_clock::now();

    auto model = MyModel<NNType>(/*out_channels=*/50);

    auto optimizer = AdamOptimizer<NNType>(0.01, 0.9, 0.999, 1.0e-6, model.get_parameters());
    auto loss_computer = SoftMaxLoss<NNType>{};

    NNType accumulated_loss = 0.0;

    size_t batch_size = 100;
    Variable<NNType, false> x_input_train({batch_size, 1, 40});
    Variable<NNType, false> x_input_test({1, 1, 40});

    std::vector<size_t> y_input_train(batch_size);

    he_initialization(model.get_parameters());

    for (size_t epoch = 0; epoch < 6 * 50; epoch++) {
        size_t good_preds = 0;
        size_t total_preds = 0;

        train_set.randomIter(batch_size, [&](auto batch) {
            size_t b = 0;
            for (const auto &[vx, vy] : batch) {
                for (size_t i = 0; i < 40; i++) {
                    x_input_train.tensor(b, 0, i) = vx[i];
                }
                y_input_train[b] = vy;
                b += 1;
            }

            auto y_predicted = model.forward(x_input_train);
            loss_computer.forward<true>(y_predicted);
            loss_computer.backward(y_predicted, y_input_train);

            optimizer.optimize(batch_size);
        });

        test_set.randomIter(1, [&](auto batch) {
            for (const auto &[vx, vy] : batch) {
                for (size_t i = 0; i < 40; i++) {
                    x_input_test.tensor(0, 0, i) = vx[i];
                }

                auto y_predicted = model.forward(x_input_test);
                auto idx = loss_computer.forward<false>(y_predicted);
                good_preds += static_cast<size_t>(idx == vy);
                total_preds += 1;
            }
        });
        std::cout << "Epoch " << epoch << ", error rate: "
                  << 100.0f -
                         static_cast<float>(good_preds) / static_cast<float>(total_preds) * 100.0f
                  << "%" << std::endl;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;
    std::cout << "Total training time: " << elapsed_seconds.count() << " seconds\n";

    return 0;
}