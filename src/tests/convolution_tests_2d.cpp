#include "convolution_tests_2d.h"
#include "../expressions/expression.h"
#include "../weight_initializer.h"

#include "test_utils.h"
#include <sstream>
#include <tuple>

static void convolution_operator_2d_tests();

void convolution_tests_2d() { convolution_operator_2d_tests(); }

static Tensor<double>
add_x_padding(ConstTensor<double> x, size_t PADDING_HEIGHT, size_t PADDING_WIDTH) {
    const auto &x_shape = x.get_shape().get_shape();

    const size_t BATCH_SIZE = x_shape[0];
    const size_t IN_CHANNELS = x_shape[1];
    const size_t DATA_HEIGHT = x_shape[2];
    const size_t DATA_WIDTH = x_shape[3];

    Tensor<double> x_padded{{BATCH_SIZE,
                             IN_CHANNELS,
                             DATA_HEIGHT + 2 * PADDING_HEIGHT,
                             DATA_WIDTH + 2 * PADDING_WIDTH}};
    x_padded.set_zero();
    for (size_t b = 0; b < BATCH_SIZE; ++b) {
        for (size_t ic = 0; ic < IN_CHANNELS; ++ic) {
            for (size_t h = 0; h < DATA_HEIGHT; ++h) {
                for (size_t w = 0; w < DATA_WIDTH; ++w) {
                    x_padded(b, ic, h + PADDING_HEIGHT, w + PADDING_WIDTH) = x(b, ic, h, w);
                }
            }
        }
    }

    return x_padded;
}
/**
 * To test the correctness of the convolution operator, We check the result against this manual
 * naive implementation
 */
static Tensor<double> naive_2d_convolution_forward(ConstTensor<double> kernel,
                                                   ConstTensor<double> x,
                                                   ConstTensor<double> bias,
                                                   size_t STRIDE_HEIGHT,
                                                   size_t STRIDE_WIDTH,
                                                   size_t PADDING_HEIGHT,
                                                   size_t PADDING_WIDTH) {
    assert(kernel.get_shape().get_dimension() == 4);
    assert(x.get_shape().get_dimension() == 4);

    const auto &kernel_shape = kernel.get_shape().get_shape();
    const auto &x_shape = x.get_shape().get_shape();

    const size_t OUT_CHANNELS = kernel_shape[0];
    const size_t IN_CHANNELS = kernel_shape[1];
    const size_t KERNEL_HEIGHT = kernel_shape[2];
    const size_t KERNEL_WIDTH = kernel_shape[3];

    const size_t BATCH_SIZE = x_shape[0];
    assert(IN_CHANNELS == x_shape[1]);

    const size_t DATA_HEIGHT = x_shape[2];
    const size_t DATA_WIDTH = x_shape[3];

    assert(DATA_HEIGHT + 2 * PADDING_HEIGHT >= KERNEL_HEIGHT);
    assert(DATA_WIDTH + 2 * PADDING_WIDTH >= KERNEL_WIDTH);

    const size_t EFFECTIVE_WIDTH =
        (DATA_WIDTH - KERNEL_WIDTH + 2 * PADDING_WIDTH) / STRIDE_WIDTH + 1;
    const size_t EFFECTIVE_HEIGHT =
        (DATA_HEIGHT - KERNEL_HEIGHT + 2 * PADDING_HEIGHT) / STRIDE_HEIGHT + 1;

    Tensor<double> x_padded = add_x_padding(x, PADDING_HEIGHT, PADDING_WIDTH);

    Tensor<double> res{{BATCH_SIZE, OUT_CHANNELS, EFFECTIVE_HEIGHT, EFFECTIVE_WIDTH}};
    res.set_zero();

    for (size_t b = 0; b < BATCH_SIZE; ++b) {
        for (size_t oc = 0; oc < OUT_CHANNELS; ++oc) {
            for (size_t eff_h = 0; eff_h < EFFECTIVE_HEIGHT; ++eff_h) {
                for (size_t eff_w = 0; eff_w < EFFECTIVE_WIDTH; ++eff_w) {
                    res(b, oc, eff_h, eff_w) = bias(oc);
                    for (size_t ker_h = 0; ker_h < KERNEL_HEIGHT; ++ker_h) {
                        for (size_t ker_w = 0; ker_w < KERNEL_WIDTH; ++ker_w) {
                            for (size_t ic = 0; ic < IN_CHANNELS; ++ic) {
                                res(b, oc, eff_h, eff_w) += kernel(oc, ic, ker_h, ker_w) *
                                                            x_padded(b,
                                                                     ic,
                                                                     eff_h * STRIDE_HEIGHT + ker_h,
                                                                     eff_w * STRIDE_WIDTH + ker_w);
                            }
                        }
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
static std::tuple<Tensor<double>, Tensor<double>, Tensor<double>>
naive_2d_convolution_backward(ConstTensor<double> kernel,
                              ConstTensor<double> x,
                              ConstTensor<double> grad_out,
                              size_t STRIDE_HEIGHT,
                              size_t STRIDE_WIDTH,
                              size_t PADDING_HEIGHT,
                              size_t PADDING_WIDTH) {
    const auto &kernel_shape = kernel.get_shape().get_shape();
    const auto &x_shape = x.get_shape().get_shape();
    const auto &g_shape = grad_out.get_shape().get_shape();

    const size_t OUT_CHANNELS = kernel_shape[0];
    const size_t IN_CHANNELS = kernel_shape[1];
    const size_t KERNEL_HEIGHT = kernel_shape[2];
    const size_t KERNEL_WIDTH = kernel_shape[3];

    const size_t BATCH_SIZE = x_shape[0];
    const size_t FEATURE_HEIGHT = x_shape[2];
    const size_t FEATURE_WIDTH = x_shape[3];

    const size_t EFFECTIVE_HEIGHT = g_shape[2];
    const size_t EFFECTIVE_WIDTH = g_shape[3];

    Tensor<double> x_padded = add_x_padding(x, PADDING_HEIGHT, PADDING_WIDTH);

    Tensor<double> grad_kernel{{OUT_CHANNELS, IN_CHANNELS, KERNEL_HEIGHT, KERNEL_WIDTH}};
    grad_kernel.set_zero();

    Tensor<double> grad_bias{{OUT_CHANNELS}};
    grad_bias.set_zero();

    Tensor<double> grad_x_padded{{BATCH_SIZE,
                                  IN_CHANNELS,
                                  FEATURE_HEIGHT + 2 * PADDING_HEIGHT,
                                  FEATURE_WIDTH + 2 * PADDING_WIDTH}};
    grad_x_padded.set_zero();

    for (size_t b = 0; b < BATCH_SIZE; ++b) {
        for (size_t oc = 0; oc < OUT_CHANNELS; ++oc) {
            for (size_t eff_h = 0; eff_h < EFFECTIVE_HEIGHT; ++eff_h) {
                for (size_t eff_w = 0; eff_w < EFFECTIVE_WIDTH; ++eff_w) {
                    double grad = grad_out(b, oc, eff_h, eff_w);
                    grad_bias(oc) += grad;

                    for (size_t ker_h = 0; ker_h < KERNEL_HEIGHT; ++ker_h) {
                        for (size_t ker_w = 0; ker_w < KERNEL_WIDTH; ++ker_w) {
                            for (size_t ic = 0; ic < IN_CHANNELS; ++ic) {

                                grad_kernel(oc, ic, ker_h, ker_w) +=
                                    grad * x_padded(b,
                                                    ic,
                                                    eff_h * STRIDE_HEIGHT + ker_h,
                                                    eff_w * STRIDE_WIDTH + ker_w);
                                grad_x_padded(b,
                                              ic,
                                              eff_h * STRIDE_HEIGHT + ker_h,
                                              eff_w * STRIDE_WIDTH + ker_w) +=
                                    grad * kernel(oc, ic, ker_h, ker_w);
                            }
                        }
                    }
                }
            }
        }
    }

    Tensor<double> grad_x{{BATCH_SIZE, IN_CHANNELS, FEATURE_HEIGHT, FEATURE_WIDTH}};
    for (size_t b = 0; b < BATCH_SIZE; ++b) {
        for (size_t ic = 0; ic < IN_CHANNELS; ++ic) {
            for (size_t h = 0; h < FEATURE_HEIGHT; ++h) {
                for (size_t w = 0; w < FEATURE_WIDTH; ++w) {
                    grad_x(b, ic, h, w) =
                        grad_x_padded(b, ic, h + PADDING_HEIGHT, w + PADDING_WIDTH);
                }
            }
        }
    }

    return {grad_kernel, grad_x, grad_bias};
}

static void convolution_operator_2d_tests() {
    constexpr size_t test_runs = 100;
    constexpr double eps_threshold = 1e-4;

    for (size_t i = 0; i < test_runs; ++i) {
        // Generate a random convolution
        const size_t IN_CHANNELS = random_size_t(1, 10);
        const size_t OUT_CHANNELS = random_size_t(1, 10);
        const size_t KERNEL_HEIGHT = random_size_t(1, 10);
        const size_t KERNEL_WIDTH = random_size_t(1, 10);

        const size_t STRIDE_HEIGHT = random_size_t(1, 10);
        const size_t STRIDE_WIDTH = random_size_t(1, 10);

        const size_t PADDING_HEIGHT = random_size_t(1, 10);
        const size_t PADDING_WIDTH = random_size_t(1, 10);

        const size_t BATCH_SIZE = random_size_t(1, 100);
        const size_t FEATURES_HEIGHT = random_size_t(KERNEL_HEIGHT + 2 * PADDING_HEIGHT, 100);
        const size_t FEATURES_WIDTH = random_size_t(KERNEL_WIDTH + 2 * PADDING_WIDTH, 100);

        Variable<double, true> kernel({OUT_CHANNELS, IN_CHANNELS, KERNEL_HEIGHT, KERNEL_WIDTH});
        Variable<double, true> x_data({BATCH_SIZE, IN_CHANNELS, FEATURES_HEIGHT, FEATURES_WIDTH});
        Variable<double, true> bias({OUT_CHANNELS});

        auto res = conv_2d(kernel, x_data, bias)
                       .set_stride(STRIDE_HEIGHT, STRIDE_WIDTH)
                       .set_padding(PADDING_HEIGHT, PADDING_WIDTH);
        const auto &conv_parameters = res.get_parameters();
        he_initialization(conv_parameters);

        auto layer_res = res.forward();
        auto layer_res_simulated = naive_2d_convolution_forward(kernel.tensor,
                                                                x_data.tensor,
                                                                bias.tensor,
                                                                STRIDE_HEIGHT,
                                                                STRIDE_WIDTH,
                                                                PADDING_HEIGHT,
                                                                PADDING_WIDTH);

        if (!check_tensor_equality<double>(layer_res, layer_res_simulated, eps_threshold)) {
            std::ostringstream oss;
            oss << "[CONV_2D_TEST]: forward pass error mismatch (actual, simulated)=(" << layer_res
                << ", " << layer_res_simulated << ")";
            throw std::runtime_error(oss.str());
        }

        // Backpropagation
        Tensor<double> gradient = layer_res.clone();
        gradient.set_constant(1.0);
        const auto &[kernel_grad, x_grad, bias_grad] = naive_2d_convolution_backward(kernel.tensor,
                                                                                     x_data.tensor,
                                                                                     gradient,
                                                                                     STRIDE_HEIGHT,
                                                                                     STRIDE_WIDTH,
                                                                                     PADDING_HEIGHT,
                                                                                     PADDING_WIDTH);
        res.backward(gradient);
        if (!check_tensor_equality<double>(
                kernel_grad, conv_parameters[0].gradient, eps_threshold)) {
            std::ostringstream oss;
            oss << "[CONV_2D_TEST]: kernel gradient mismatch (actual, simulated)=(" << kernel_grad
                << ", " << conv_parameters[0].gradient << ")";
            throw std::runtime_error(oss.str());
        }

        if (!check_tensor_equality<double>(x_grad, conv_parameters[1].gradient, eps_threshold)) {
            std::ostringstream oss;
            oss << "[CONV_2D_TEST]: x gradient mismatch (actual, simulated)=(" << x_grad << ", "
                << conv_parameters[1].gradient << ")";
            throw std::runtime_error(oss.str());
        }

        if (!check_tensor_equality<double>(bias_grad, conv_parameters[2].gradient, eps_threshold)) {
            std::ostringstream oss;
            oss << "[CONV_2D_TEST]: bias gradient mismatch (actual, simulated)=(" << bias_grad
                << ", " << conv_parameters[2].gradient << ")";
            throw std::runtime_error(oss.str());
        }
    }
}