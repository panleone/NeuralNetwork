#pragma once

#include "../binary_operators/binary_operator.h"
#include "../../blas_wrapper.h"

#include <cassert>

/**
 * Partial specialization for 2d convolution
 */
template <typename A, typename B, typename C>
requires(std::is_same_v<typename A::DType, typename B::DType>) class DTernExprOp<A, B, C, DApConv2d>
    : public DTernaryExprCommonData<A, B, C, DApConv2d>,
      public DExpr<DTernExprOp<A, B, C, DApConv2d>> {
  public:
    using DType = typename A::DType;
    using DTernaryExprCommonData<A, B, C, DApConv2d>::traverse;
    using Operator = DApConv2d;

  private:
    // a_ is the kernel
    using DTernaryExprCommonData<A, B, C, DApConv2d>::a_;
    // b_ is the data buffer on which we apply the kernel
    using DTernaryExprCommonData<A, B, C, DApConv2d>::b_;
    // c_ is the bias vector
    using DTernaryExprCommonData<A, B, C, DApConv2d>::c_;
    // We also cache the kernel and x in their im2col version
    ConstTensor<DType> kernel_data_im2col;
    ConstTensor<DType> x_data_im2col;

    using This = DTernExprOp<A, B, C, DApConv2d>;

    // Stride of the convolution
    size_t STRIDE_HEIGHT = 1;
    size_t STRIDE_WIDTH = 1;

    // padding of the convolution
    // For the moment only zero-padding supported
    size_t PADDING_HEIGHT{0};
    size_t PADDING_WIDTH{0};

    // those variables get a non-zero value in the forward step.
    // We need to cache them for the backpropagation.
    size_t KERNEL_HEIGHT{0};
    size_t KERNEL_WIDTH{0};

    size_t IN_CHANNELS{0};
    size_t OUT_CHANNELS{0};

    size_t BATCH_SIZE{0};
    size_t DATA_HEIGHT{0};
    size_t DATA_WIDTH{0};
    size_t EFFECTIVE_WIDTH{0};
    size_t EFFECTIVE_HEIGHT{0};

  public:
    using Left = A;
    using Middle = B;
    using Right = C;

    DTernExprOp(const A &a, const B &b, const C &c)
        : DTernaryExprCommonData<A, B, C, DApConv2d>{a, b, c} {}

    template <bool recursive>
    struct Flatten {
        using Type = Stack<ops::VARIABLE_OP>;
    };

    struct Simplify {
        using Type = DTernExprOp<typename A::Simplify::Type,
                                 typename B::Simplify::Type,
                                 typename C::Simplify::Type,
                                 DApConv2d>;
    };

    void compute_temporaries_for_eval() {
        using SimplifiedT = Simplify::Type;
        a_.compute_temporaries_for_eval();
        b_.compute_temporaries_for_eval();
        c_.compute_temporaries_for_eval();

        auto kernel_matrix =
            kernel_im2col(Interpreter<typename SimplifiedT::Left>::const_interpret(a_),
                          Interpreter<typename SimplifiedT::Right>::const_interpret(c_));
        auto x_data_matrix =
            x_im2col(Interpreter<typename SimplifiedT::Middle>::const_interpret(b_));

        auto res_shape = Shape::get_matmul_shape<false, true>(x_data_matrix.get_shape(),
                                                              kernel_matrix.get_shape());

        this->res = res_col2im(
            mat_mul_wrapper<DType, false, true>(x_data_matrix, kernel_matrix, res_shape));
    }

    template <bool use_cache>
    ConstTensor<DType> compute_temporaries_for_backprop() {
        if constexpr (!use_cache) {
            ConstTensor<DType> kernel = a_.template compute_temporaries_for_backprop<use_cache>();
            ConstTensor<DType> x_data = b_.template compute_temporaries_for_backprop<use_cache>();
            ConstTensor<DType> bias = c_.template compute_temporaries_for_backprop<use_cache>();

            kernel_data_im2col = kernel_im2col(kernel, bias);
            x_data_im2col = x_im2col(x_data);

            auto res_shape = Shape::get_matmul_shape<false, true>(x_data_im2col.get_shape(),
                                                                  kernel_data_im2col.get_shape());

            this->res = res_col2im(
                mat_mul_wrapper<DType, false, true>(x_data_im2col, kernel_data_im2col, res_shape));
        }
        return this->res;
    }

    void backward_internal(const Tensor<DType> &grad) {
        Tensor<DType> grad_im2col = res_im2col(grad);

        Tensor<DType> b_grad = x_col2im(mat_mul_wrapper<DType, false, false>(
            grad_im2col, kernel_data_im2col, x_data_im2col.get_shape()));
        auto [a_grad, c_grad] = kernel_col2im(mat_mul_wrapper<DType, true, false>(
            grad_im2col, x_data_im2col, kernel_data_im2col.get_shape()));

        a_.backward_internal(a_grad);
        b_.backward_internal(b_grad);
        c_.backward_internal(c_grad);
    }

    // we implement the convolution with the im2col transformation
    Tensor<DType> kernel_im2col(const ConstTensor<DType> &kernel, const ConstTensor<DType> &bias) {
        const Shape &kernel_shape = kernel.get_shape();
        const auto &kernel_shape_data = kernel_shape.get_shape();

        // By convention we assume that the kernel must have the following shape
        // [OUT_CHANNELS, IN_CHANNELS, KERNEL_HEIGHT, KERNEL_WIDTH]
        assert(kernel_shape.get_dimension() == 4);

        OUT_CHANNELS = kernel_shape_data[0];
        IN_CHANNELS = kernel_shape_data[1];
        KERNEL_HEIGHT = kernel_shape_data[2];
        KERNEL_WIDTH = kernel_shape_data[3];

        const size_t KERNEL_SIZE = KERNEL_HEIGHT * KERNEL_WIDTH;

        Tensor<DType> res({OUT_CHANNELS, 1 + IN_CHANNELS * KERNEL_SIZE});

        for (size_t oc = 0; oc < OUT_CHANNELS; ++oc) {
            res(oc, 0) = bias(oc);
            for (size_t ic = 0; ic < IN_CHANNELS; ++ic) {
                size_t effective_ic = ic * KERNEL_SIZE;
                for (size_t kh = 0; kh < KERNEL_HEIGHT; ++kh) {
                    size_t effective_kh = kh * KERNEL_WIDTH;
                    for (size_t kw = 0; kw < KERNEL_WIDTH; ++kw) {
                        res(oc, 1 + effective_ic + effective_kh + kw) = kernel(oc, ic, kh, kw);
                    }
                }
            }
        }

        res.wrap_for_broadcasting();
        return res;
    }

    Tensor<DType> x_im2col(const ConstTensor<DType> &tensor) {
        const Shape &t_shape = tensor.get_shape();
        const auto &t_shape_data = t_shape.get_shape();

        // By convention we assume that the input x must have the following shape
        // [BATCH_SIZE, IN_CHANNELS, DATA_HEIGHT, DATA_WIDTH]
        assert(t_shape.get_dimension() == 4);
        BATCH_SIZE = t_shape_data[0];
        assert(IN_CHANNELS == t_shape_data[1]);
        DATA_HEIGHT = t_shape_data[2];
        DATA_WIDTH = t_shape_data[3];

        assert(DATA_WIDTH + 2 * PADDING_WIDTH >= KERNEL_WIDTH);
        assert(DATA_HEIGHT + 2 * PADDING_HEIGHT >= KERNEL_HEIGHT);

        // Residual number of features after the application of the 1d convolution
        EFFECTIVE_HEIGHT = (DATA_HEIGHT - KERNEL_HEIGHT + 2 * PADDING_HEIGHT) / STRIDE_HEIGHT + 1;
        EFFECTIVE_WIDTH = (DATA_WIDTH - KERNEL_WIDTH + 2 * PADDING_WIDTH) / STRIDE_WIDTH + 1;
        const size_t EFFECTIVE_SIZE = EFFECTIVE_HEIGHT * EFFECTIVE_WIDTH;
        const size_t KERNEL_SIZE = KERNEL_HEIGHT * KERNEL_WIDTH;

        Tensor<DType> tensor_padded{{BATCH_SIZE,
                                     IN_CHANNELS,
                                     DATA_HEIGHT + 2 * PADDING_HEIGHT,
                                     DATA_WIDTH + 2 * PADDING_WIDTH}};
        tensor_padded.set_zero();

        for (size_t b = 0; b < BATCH_SIZE; ++b) {
            for (size_t ic = 0; ic < IN_CHANNELS; ++ic) {
                for (size_t h = 0; h < DATA_HEIGHT; ++h) {
                    for (size_t w = 0; w < DATA_WIDTH; ++w) {
                        tensor_padded(b, ic, h + PADDING_HEIGHT, w + PADDING_WIDTH) =
                            tensor(b, ic, h, w);
                    }
                }
            }
        }
        Tensor<DType> tensor_im2col({BATCH_SIZE * EFFECTIVE_SIZE, IN_CHANNELS * KERNEL_SIZE + 1});
        for (size_t b = 0; b < BATCH_SIZE; ++b) {
            for (size_t ic = 0; ic < IN_CHANNELS; ++ic) {
                for (size_t eff_h = 0; eff_h < EFFECTIVE_HEIGHT; ++eff_h) {
                    for (size_t eff_w = 0; eff_w < EFFECTIVE_WIDTH; ++eff_w) {
                        for (size_t kh = 0; kh < KERNEL_HEIGHT; ++kh) {
                            for (size_t kw = 0; kw < KERNEL_WIDTH; ++kw) {
                                tensor_im2col(b * EFFECTIVE_SIZE + eff_h * EFFECTIVE_WIDTH + eff_w,
                                              1 + ic * KERNEL_SIZE + kh * KERNEL_WIDTH + kw) =
                                    tensor_padded(b,
                                                  ic,
                                                  eff_h * STRIDE_HEIGHT + kh,
                                                  eff_w * STRIDE_WIDTH + kw);
                            }
                        }
                    }
                }
            }
        }
        // fill the first row with 1 (so we can add the convolution bias in a single operation)
        for (size_t i = 0; i < BATCH_SIZE * EFFECTIVE_SIZE; ++i) {
            tensor_im2col(i, 0) = static_cast<DType>(1.0);
        }

        tensor_im2col.wrap_for_broadcasting();
        return tensor_im2col;
    }

    Tensor<DType> res_im2col(const ConstTensor<DType> &res_grad) {
        const Shape &t_shape = res_grad.get_shape();
        const auto &t_shape_data = t_shape.get_shape();

        // By convention we assume that the gradient must have the following shape
        // [BATCH_SIZE, OUT_CHANNELS, EFFECTIVE_DATA_HEIGHT, EFFECTIVE_DATA_WIDTH]
        assert(t_shape.get_dimension() == 4);
        assert(BATCH_SIZE == t_shape_data[0]);
        assert(OUT_CHANNELS == t_shape_data[1]);
        assert(EFFECTIVE_HEIGHT == t_shape_data[2]);
        assert(EFFECTIVE_WIDTH == t_shape_data[3]);

        const size_t EFFECTIVE_SIZE = EFFECTIVE_HEIGHT * EFFECTIVE_WIDTH;

        Tensor<DType> res_grad_im2col({BATCH_SIZE * EFFECTIVE_SIZE, OUT_CHANNELS});
        for (size_t b = 0; b < BATCH_SIZE; ++b) {
            for (size_t eff_h = 0; eff_h < EFFECTIVE_HEIGHT; ++eff_h) {
                for (size_t eff_w = 0; eff_w < EFFECTIVE_WIDTH; ++eff_w) {
                    for (size_t oc = 0; oc < OUT_CHANNELS; ++oc) {
                        res_grad_im2col(b * EFFECTIVE_SIZE + eff_h * EFFECTIVE_WIDTH + eff_w, oc) =
                            res_grad(b, oc, eff_h, eff_w);
                    }
                }
            }
        }

        res_grad_im2col.wrap_for_broadcasting();
        return res_grad_im2col;
    }

    // Inverse trasnformations for backpropagation
    std::pair<Tensor<DType>, Tensor<DType>>
    kernel_col2im(const ConstTensor<DType> &grad_matrix) const {
        const Shape &t_shape = grad_matrix.get_shape();
        const auto &t_shape_data = t_shape.get_shape();

        // By convention we assume that the grad_matrix must have the following shape
        // [OUT_CHANNELS, IN_CHANNELS * KERNEL_HEIGHT * KERNEL_WIDTH]
        const size_t KERNEL_SIZE = KERNEL_WIDTH * KERNEL_HEIGHT;
        assert(t_shape.get_dimension() == 2);

        assert(OUT_CHANNELS == t_shape_data[0]);
        assert(1 + IN_CHANNELS * KERNEL_SIZE == t_shape_data[1]);

        Tensor<DType> grad_kernel({OUT_CHANNELS, IN_CHANNELS, KERNEL_HEIGHT, KERNEL_WIDTH});
        Tensor<DType> bias_kernel({OUT_CHANNELS});

        for (size_t oc = 0; oc < OUT_CHANNELS; ++oc) {
            bias_kernel(oc) = grad_matrix(oc, 0);
            for (size_t ic = 0; ic < IN_CHANNELS; ++ic) {
                for (size_t kh = 0; kh < KERNEL_HEIGHT; ++kh) {
                    for (size_t kw = 0; kw < KERNEL_WIDTH; ++kw) {
                        grad_kernel(oc, ic, kh, kw) =
                            grad_matrix(oc, 1 + ic * KERNEL_SIZE + kh * KERNEL_WIDTH + kw);
                    }
                }
            }
        }

        grad_kernel.wrap_for_broadcasting();
        bias_kernel.wrap_for_broadcasting();
        return {grad_kernel, bias_kernel};
    }

    Tensor<DType> x_col2im(const ConstTensor<DType> &grad_x_matrix) const {
        const Shape &t_shape = grad_x_matrix.get_shape();
        const auto &t_shape_data = t_shape.get_shape();

        // By convention we assume that the grad_matrix must have the following shape
        // [BATCH_SIZE * EFFECTIVE_DATA_HEIGHT * EFFECTIVE_DATA_WIDTH,
        // IN_CHANNELS*KERNEL_HEIGHT*KERNEL_WIDTH]
        assert(t_shape.get_dimension() == 2);
        const size_t KERNEL_SIZE = KERNEL_WIDTH * KERNEL_HEIGHT;
        const size_t EFFECTIVE_SIZE = EFFECTIVE_HEIGHT * EFFECTIVE_WIDTH;

        assert(BATCH_SIZE * EFFECTIVE_SIZE == t_shape_data[0]);
        assert(1 + IN_CHANNELS * KERNEL_SIZE == t_shape_data[1]);

        Tensor<DType> grad_x_padded({BATCH_SIZE,
                                     IN_CHANNELS,
                                     DATA_HEIGHT + 2 * PADDING_HEIGHT,
                                     DATA_WIDTH + 2 * PADDING_WIDTH});
        grad_x_padded.set_zero();

        for (size_t b = 0; b < BATCH_SIZE; ++b) {
            for (size_t ic = 0; ic < IN_CHANNELS; ++ic) {
                for (size_t eff_h = 0; eff_h < EFFECTIVE_HEIGHT; ++eff_h) {
                    for (size_t eff_w = 0; eff_w < EFFECTIVE_WIDTH; ++eff_w) {
                        for (size_t kh = 0; kh < KERNEL_HEIGHT; ++kh) {
                            for (size_t kw = 0; kw < KERNEL_WIDTH; ++kw) {
                                grad_x_padded(
                                    b, ic, eff_h * STRIDE_HEIGHT + kh, eff_w * STRIDE_WIDTH + kw) +=
                                    grad_x_matrix(b * EFFECTIVE_HEIGHT * EFFECTIVE_WIDTH +
                                                      eff_h * EFFECTIVE_WIDTH + eff_w,
                                                  1 + ic * KERNEL_HEIGHT * KERNEL_WIDTH +
                                                      kh * KERNEL_WIDTH + kw);
                            }
                        }
                    }
                }
            }
        }

        Tensor<DType> grad_x({BATCH_SIZE, IN_CHANNELS, DATA_HEIGHT, DATA_WIDTH});
        for (size_t b = 0; b < BATCH_SIZE; ++b) {
            for (size_t ic = 0; ic < IN_CHANNELS; ++ic) {
                for (size_t h = 0; h < DATA_HEIGHT; ++h) {
                    for (size_t w = 0; w < DATA_WIDTH; ++w) {
                        grad_x(b, ic, h, w) =
                            grad_x_padded(b, ic, h + PADDING_HEIGHT, w + PADDING_WIDTH);
                    }
                }
            }
        }

        grad_x.wrap_for_broadcasting();
        return grad_x;
    }

    Tensor<DType> res_col2im(const ConstTensor<DType> &res_matrix) {
        const Shape &t_shape = res_matrix.get_shape();
        const auto &t_shape_data = t_shape.get_shape();

        // By convention we assume that the input has shape
        // [BATCH_SIZE * EFFECTIVE_HEIGHT * EFFECTIVE_WIDTH, OUT_CHANNELS]
        assert(t_shape.get_dimension() == 2);
        const size_t EFFECTIVE_SIZE = EFFECTIVE_HEIGHT * EFFECTIVE_WIDTH;
        assert(BATCH_SIZE * EFFECTIVE_SIZE == t_shape_data[0]);
        assert(OUT_CHANNELS == t_shape_data[1]);

        Tensor<DType> res({BATCH_SIZE, OUT_CHANNELS, EFFECTIVE_HEIGHT, EFFECTIVE_WIDTH});
        for (size_t b = 0; b < BATCH_SIZE; ++b) {
            for (size_t eff_h = 0; eff_h < EFFECTIVE_HEIGHT; ++eff_h) {
                for (size_t eff_w = 0; eff_w < EFFECTIVE_WIDTH; ++eff_w) {
                    for (size_t oc = 0; oc < OUT_CHANNELS; ++oc) {
                        res(b, oc, eff_h, eff_w) =
                            res_matrix(b * EFFECTIVE_SIZE + eff_h * EFFECTIVE_WIDTH + eff_w, oc);
                    }
                }
            }
        }

        res.wrap_for_broadcasting();
        return res;
    }

    This &set_stride(size_t stride_height, size_t stride_width) {
        STRIDE_HEIGHT = stride_height;
        STRIDE_WIDTH = stride_width;
        return *this;
    }

    This &set_padding(size_t padding_height, size_t padding_width) {
        PADDING_HEIGHT = padding_height;
        PADDING_WIDTH = padding_width;

        return *this;
    }
};
