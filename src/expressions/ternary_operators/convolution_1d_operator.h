#pragma once

#include "ternary_operator.h"
#include "../../blas_wrapper.h"

/**
 * Partial specialization for 1d convolution
 */
template <typename A, typename B, typename C>
requires(std::is_same_v<typename A::DType, typename B::DType>) class DTernExprOp<A, B, C, DApConv1d>
    : public DExpr<DTernExprOp<A, B, C, DApConv1d>> {
  public:
    using DType = typename A::DType;

  private:
    // a_ is the kernel
    A a_;
    // b_ is the data buffer on which we apply the kernel
    B b_;
    // c_ is the bias vector
    C c_;
    // Result of the 1d convolution
    ConstTensor<DType> res{};
    // We also cache the kernel and x in their im2col version
    ConstTensor<DType> kernel_data_im2col;
    ConstTensor<DType> x_data_im2col;

    using This = DTernExprOp<A, B, C, DApConv1d>;

    // Stride of the convolution
    size_t STRIDE{1};

    // those variables get a non-zero value in the forward step.
    // We need to cache them for the backpropagation.
    size_t KERNEL_SIZE{0};
    size_t IN_CHANNELS{0};
    size_t OUT_CHANNELS{0};

    size_t BATCH_SIZE{0};
    size_t FEATURE_SIZE{0};
    size_t EFFECTIVE_WIDTH{0};

  public:
    using Left = A;
    using Middle = B;
    using Right = C;

    DTernExprOp(const A &a, const B &b, const C &c) : a_{a}, b_{b}, c_{c} {}

    static consteval size_t get_num_tensors() { return 1; }

    void collect_tensor_handles(auto &current_stack) const {
        current_stack.push_back_variable(res);
    }
    void get_parameters_internal(auto &res) const {
        a_.get_parameters_internal(res);
        b_.get_parameters_internal(res);
        c_.get_parameters_internal(res);
    }

    template <bool recursive>
    struct Flatten {
        using Type = Stack<ops::VARIABLE_OP>;
    };

    struct Simplify {
        using Type = DTernExprOp<typename A::Simplify::Type,
                                 typename B::Simplify::Type,
                                 typename C::Simplify::Type,
                                 DApConv1d>;
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

        res = res_col2im(
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

            res = res_col2im(
                mat_mul_wrapper<DType, false, true>(x_data_im2col, kernel_data_im2col, res_shape));
        }
        return res;
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

        const Shape &bias_shape = bias.get_shape();
        const auto &bias_shape_data = bias_shape.get_shape();

        // By convention we assume that the kernel must have the following shape
        // [OUT_CHANNELS, IN_CHANNELS, 1 + KERNEL_SIZE]
        assert(bias_shape.get_dimension() == 1);
        assert(kernel_shape.get_dimension() == 3);
        OUT_CHANNELS = kernel_shape_data[0];
        IN_CHANNELS = kernel_shape_data[1];
        KERNEL_SIZE = kernel_shape_data[2];

        assert(OUT_CHANNELS = bias_shape_data[0]);

        Tensor<DType> res({OUT_CHANNELS, 1 + IN_CHANNELS * KERNEL_SIZE});

        for (size_t oc = 0; oc < OUT_CHANNELS; ++oc) {
            res(oc, 0) = bias(oc);

            for (size_t ic = 0; ic < IN_CHANNELS; ++ic) {
                size_t ic_times_kernel = ic * KERNEL_SIZE;
                for (size_t k = 0; k < KERNEL_SIZE; ++k) {
                    res(oc, 1 + k + ic_times_kernel) = kernel(oc, ic, k);
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
        // [BATCH_SIZE, IN_CHANNELS, FEATURE_SIZE]
        assert(t_shape.get_dimension() == 3);
        BATCH_SIZE = t_shape_data[0];
        assert(IN_CHANNELS == t_shape_data[1]);
        FEATURE_SIZE = t_shape_data[2];

        // Residual number of features after the application of the 1d convolution
        assert(FEATURE_SIZE >= KERNEL_SIZE);
        EFFECTIVE_WIDTH = (FEATURE_SIZE - KERNEL_SIZE) / STRIDE + 1;

        Tensor<DType> tensor_im2col({BATCH_SIZE * EFFECTIVE_WIDTH, IN_CHANNELS * KERNEL_SIZE + 1});
        for (size_t b = 0; b < BATCH_SIZE; b++) {
            for (size_t ic = 0; ic < IN_CHANNELS; ic++) {
                for (size_t k = 0; k < KERNEL_SIZE; k++) {
                    for (size_t w = 0; w < EFFECTIVE_WIDTH; w++) {
                        tensor_im2col(b * EFFECTIVE_WIDTH + w, 1 + k + ic * KERNEL_SIZE) =
                            tensor(b, ic, k + w * STRIDE);
                    }
                }
            }
        }
        // fill the first row with 1 (so we can add the convolution bias in a single operation)
        for (size_t i = 0; i < BATCH_SIZE * EFFECTIVE_WIDTH; ++i) {
            tensor_im2col(i, 0) = static_cast<DType>(1.0);
        }

        tensor_im2col.wrap_for_broadcasting();
        return tensor_im2col;
    }

    Tensor<DType> res_im2col(const ConstTensor<DType> &res_grad) {
        const Shape &t_shape = res_grad.get_shape();
        const auto &t_shape_data = t_shape.get_shape();

        // By convention we assume that the gradient must have the following shape
        // [BATCH_SIZE, OUT_CHANNELS, EFFECTIVE_WIDTH]
        assert(t_shape.get_dimension() == 3);
        assert(BATCH_SIZE == t_shape_data[0]);
        assert(OUT_CHANNELS == t_shape_data[1]);
        assert(EFFECTIVE_WIDTH == t_shape_data[2]);

        Tensor<DType> res_grad_im2col{{BATCH_SIZE * EFFECTIVE_WIDTH, OUT_CHANNELS}};
        for (size_t b = 0; b < BATCH_SIZE; b++) {
            for (size_t oc = 0; oc < OUT_CHANNELS; oc++) {
                for (size_t w = 0; w < EFFECTIVE_WIDTH; w++) {
                    res_grad_im2col(b * EFFECTIVE_WIDTH + w, oc) = res_grad(b, oc, w);
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
        // [OUT_CHANNELS, IN_CHANNELS*KERNEL_SIZE]
        assert(t_shape.get_dimension() == 2);
        assert(OUT_CHANNELS == t_shape_data[0]);
        assert(1 + IN_CHANNELS * KERNEL_SIZE == t_shape_data[1]);

        Tensor<DType> grad_kernel({OUT_CHANNELS, IN_CHANNELS, KERNEL_SIZE});
        Tensor<DType> bias_kernel({OUT_CHANNELS});

        for (size_t oc = 0; oc < OUT_CHANNELS; ++oc) {
            bias_kernel(oc) = grad_matrix(oc, 0);
            for (size_t ic = 0; ic < IN_CHANNELS; ++ic) {
                for (size_t k = 0; k < KERNEL_SIZE; ++k) {
                    grad_kernel(oc, ic, k) = grad_matrix(oc, 1 + k + ic * KERNEL_SIZE);
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
        // [BATCH_SIZE * EFFECTIVE_WIDTH, IN_CHANNELS*KERNEL_SIZE]

        assert(t_shape.get_dimension() == 2);
        assert(BATCH_SIZE * EFFECTIVE_WIDTH == t_shape_data[0]);
        assert(1 + IN_CHANNELS * KERNEL_SIZE == t_shape_data[1]);

        Tensor<DType> grad_x{{BATCH_SIZE, IN_CHANNELS, FEATURE_SIZE}};
        grad_x.set_zero();

        for (size_t b = 0; b < BATCH_SIZE; b++) {
            for (size_t ic = 0; ic < IN_CHANNELS; ic++) {
                for (size_t k = 0; k < KERNEL_SIZE; k++) {
                    for (size_t w = 0; w < EFFECTIVE_WIDTH; w++) {
                        grad_x(b, ic, k + w * STRIDE) +=
                            grad_x_matrix(b * EFFECTIVE_WIDTH + w, 1 + k + ic * KERNEL_SIZE);
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
        // [BATCH_SIZE * EFFECTIVE_WIDTH, OUT_CHANNELS]
        assert(t_shape.get_dimension() == 2);
        assert(BATCH_SIZE * EFFECTIVE_WIDTH == t_shape_data[0]);
        assert(OUT_CHANNELS == t_shape_data[1]);

        Tensor<DType> res{{BATCH_SIZE, OUT_CHANNELS, EFFECTIVE_WIDTH}};
        for (size_t b = 0; b < BATCH_SIZE; b++) {
            for (size_t oc = 0; oc < OUT_CHANNELS; oc++) {
                for (size_t w = 0; w < EFFECTIVE_WIDTH; w++) {
                    res(b, oc, w) = res_matrix(b * EFFECTIVE_WIDTH + w, oc);
                }
            }
        }

        res.wrap_for_broadcasting();
        return res;
    }

    This &set_stride(size_t stride) {
        STRIDE = stride;
        return *this;
    }
};
