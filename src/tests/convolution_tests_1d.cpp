#include "convolution_tests_1d.h"
#include "../expressions/expression.h"
#include "../weight_initializer.h"

#include "test_utils.h"
#include <sstream>
#include <tuple>

static void convolution_operator_1d_tests();


void convolution_tests_1d() {
    convolution_operator_1d_tests();
}

/**
 * To test the correctness of the convolution operator, We check the result against this manual naive implementation
 */
static Tensor<double> naive_1d_convolution_forward(ConstTensor<double> kernel, ConstTensor<double> x, ConstTensor<double> bias, size_t STRIDE){
    assert(kernel.get_shape().get_dimension() == 3);
    assert(x.get_shape().get_dimension() == 3);
    assert(bias.get_shape().get_dimension() == 1);

    const auto& kernel_shape = kernel.get_shape().get_shape();
    const auto& x_shape = x.get_shape().get_shape();

    size_t OUT_CHANNELS = kernel_shape[0];
    assert(OUT_CHANNELS == bias.get_shape().get_shape()[0]);

    size_t IN_CHANNELS = kernel_shape[1];
    size_t KERNEL_SIZE = kernel_shape[2];

    size_t BATCH_SIZE = x_shape[0];
    assert(IN_CHANNELS == x_shape[1]);
    size_t FEATURE_SIZE = x_shape[2];

    assert(FEATURE_SIZE >= KERNEL_SIZE);
    size_t EFFECTIVE_WIDTH = (FEATURE_SIZE - KERNEL_SIZE) / STRIDE + 1;

    Tensor<double> res{{BATCH_SIZE, OUT_CHANNELS, EFFECTIVE_WIDTH}};

    for(size_t b = 0; b < BATCH_SIZE; ++b){
        for(size_t oc = 0; oc < OUT_CHANNELS; ++oc){
            for(size_t w = 0; w < EFFECTIVE_WIDTH; ++w){
                res(b, oc, w) = bias(oc);
                for(size_t k = 0; k < KERNEL_SIZE; ++k){
                    for(size_t ic = 0; ic < IN_CHANNELS; ++ic){
                        res(b, oc, w) += kernel(oc, ic, k)*x(b, ic, w*STRIDE + k);
                    }
                }
            }
        }
    }
    return res;
}

/**
 * And same for the back propagation
 */
static std::tuple<Tensor<double>, Tensor<double>, Tensor<double>> naive_1d_convolution_backward(
    ConstTensor<double> kernel,
    ConstTensor<double> x,
    ConstTensor<double> grad_out,
    size_t STRIDE)
{
    const auto& kernel_shape = kernel.get_shape().get_shape();
    const auto& x_shape      = x.get_shape().get_shape();
    const auto& g_shape      = grad_out.get_shape().get_shape();

    size_t OUT_CHANNELS  = kernel_shape[0];
    size_t IN_CHANNELS   = kernel_shape[1];
    size_t KERNEL_SIZE   = kernel_shape[2];

    size_t BATCH_SIZE    = x_shape[0];
    size_t FEATURE_SIZE  = x_shape[2];

    size_t EFFECTIVE_WIDTH = g_shape[2];

    Tensor<double> grad_kernel{{OUT_CHANNELS, IN_CHANNELS, KERNEL_SIZE}};
    grad_kernel.set_zero();

    Tensor<double> grad_bias{{OUT_CHANNELS}};
    grad_bias.set_zero();

    Tensor<double> grad_x{{BATCH_SIZE, IN_CHANNELS, FEATURE_SIZE}};
    grad_x.set_zero();

    for (size_t b = 0; b < BATCH_SIZE; ++b) {
        for (size_t oc = 0; oc < OUT_CHANNELS; ++oc) {
            for (size_t w = 0; w < EFFECTIVE_WIDTH; ++w) {
                double grad = grad_out(b, oc, w);

                grad_bias(oc) += grad;

                for (size_t k = 0; k < KERNEL_SIZE; ++k) {
                    for (size_t ic = 0; ic < IN_CHANNELS; ++ic) {
                        grad_kernel(oc, ic, k) += grad * x(b, ic, w * STRIDE + k);
                        grad_x(b, ic, w * STRIDE + k) += grad * kernel(oc, ic, k);
                    }
                }
            }
        }
    }

    return {grad_kernel, grad_x, grad_bias};
}

static void convolution_operator_1d_tests(){
    constexpr size_t test_runs = 100;
    constexpr double eps_threshold = 1e-4;

    for(size_t i = 0; i < test_runs; ++i){
        // Generate a random convolution
        size_t IN_CHANNELS = random_size_t(1, 10);
        size_t OUT_CHANNELS = random_size_t(1,10);
        size_t KERNEL_SIZE = random_size_t(1, 10);
        size_t STRIDE = random_size_t(1, 10);
        size_t BATCH_SIZE = random_size_t(1, 100);
        size_t FEATURES = random_size_t(KERNEL_SIZE, 100);

        Variable<double, true> kernel({OUT_CHANNELS, IN_CHANNELS, KERNEL_SIZE});
        Variable<double, true> x_data({BATCH_SIZE, IN_CHANNELS, FEATURES});
        Variable<double, true> bias({OUT_CHANNELS});

        auto res = conv_1d(kernel, x_data, bias).set_stride(STRIDE);
        const auto& conv_parameters = res.get_parameters();
        random_test_initialization(conv_parameters);

        auto layer_res = res.forward();
        auto layer_res_simulated = naive_1d_convolution_forward(kernel.tensor, x_data.tensor, bias.tensor, STRIDE);

        if(!check_tensor_equality<double>(layer_res, layer_res_simulated, eps_threshold)){
            std::ostringstream oss;
            oss << "[CONV_1D_TEST]: forward pass error mismatch (actual, simulated)=(" << layer_res << ", " << layer_res_simulated << ")";
            throw std::runtime_error(oss.str());
        }

        // Backpropagation
        Tensor<double> gradient = layer_res.clone();
        gradient.set_constant(1.0);
        const auto& [kernel_grad, x_grad, bias_grad] = naive_1d_convolution_backward(kernel.tensor, x_data.tensor, gradient, STRIDE);
        res.backward(gradient);
        if(!check_tensor_equality<double>(kernel_grad, conv_parameters[0].gradient, eps_threshold)){
            std::ostringstream oss;
            oss << "[CONV_1D_TEST]: kernel gradient mismatch (actual, simulated)=(" << kernel_grad << ", " << conv_parameters[0].gradient << ")";
            throw std::runtime_error(oss.str());
        }

        if(!check_tensor_equality<double>(x_grad, conv_parameters[1].gradient, eps_threshold)){
            std::ostringstream oss;
            oss << "[CONV_1D_TEST]: x gradient mismatch (actual, simulated)=(" << x_grad << ", " << conv_parameters[1].gradient << ")";
            throw std::runtime_error(oss.str());
        }

        if(!check_tensor_equality<double>(bias_grad, conv_parameters[2].gradient, eps_threshold)){
            std::ostringstream oss;
            oss << "[CONV_1D_TEST]: bias gradient mismatch (actual, simulated)=(" << bias_grad << ", " << conv_parameters[2].gradient << ")";
            throw std::runtime_error(oss.str());
        }
    }
}