#pragma once

#include "../expression_base.h"
#include "binary_operator_simplifier.h"
#include "../../interpreter.h"

template <typename A, typename B, typename Op>
requires(std::is_same_v<typename A::DType, typename B::DType>) class DBinExprOp
    : public DExpr<DBinExprOp<A, B, Op>> {
  public:
    using DType = typename A::DType;

  private:
    A a_;
    B b_;
    // For backpropagation
    ConstTensor<DType> res{};
    using This = DBinExprOp<A, B, Op>;

  public:
    using Operator = Op;
    using Left = A;
    using Right = B;

    DBinExprOp(const A &a, const B &b) : a_{a}, b_{b} {}

    template <bool recursive>
    struct Flatten {
        using tmp1 = std::conditional_t<recursive,
                                        typename A::template Flatten<true>::Type,
                                        Stack<ops::VARIABLE_OP>>;
        using tmp2 = std::conditional_t<recursive,
                                        typename B::template Flatten<true>::Type,
                                        Stack<ops::VARIABLE_OP>>;
        using tmp3 = Stack<Op::STACK_VAL>;
        using Type = MergeStacksT<MergeStacksT<tmp1, tmp2>, tmp3>;
    };

    static consteval size_t get_num_tensors() {
        return A::get_num_tensors() + B::get_num_tensors();
    }

    void collect_tensor_handles(auto &current_stack) const {
        a_.collect_tensor_handles(current_stack);
        b_.collect_tensor_handles(current_stack);
    }

    void get_parameters_internal(auto &res) const {
        a_.get_parameters_internal(res);
        b_.get_parameters_internal(res);
    }

    struct Simplify {
        using Type = typename BinarySimplifier<This>::Type;
    };

    void compute_temporaries_for_eval() {
        a_.compute_temporaries_for_eval();
        b_.compute_temporaries_for_eval();
    }
    template <bool use_cache>
    ConstTensor<DType> compute_temporaries_for_backprop() {
        if constexpr (!use_cache) {
            ConstTensor<DType> a_res = a_.template compute_temporaries_for_backprop<use_cache>();
            ConstTensor<DType> b_res = b_.template compute_temporaries_for_backprop<use_cache>();

            assert(Shape::are_broadcastable(a_res.get_shape(), b_res.get_shape()));
            res = InterpretInternal<DType, typename Flatten<false>::Type>::const_eval(
                make_data_buffer<DType>(a_res, b_res));
        }
        return res;
    }

    void backward_internal(const Tensor<DType> &grad) {
        Tensor<DType> a_grad = grad.clone();
        Tensor<DType> b_grad = grad.clone();

        ConstTensor<DType> a_prev =
            a_.template compute_temporaries_for_backprop</*use_cache=*/true>();
        ConstTensor<DType> b_prev =
            b_.template compute_temporaries_for_backprop</*use_cache=*/true>();

        if constexpr (std::is_same_v<Op, DApSum>) {
            // Nothing to do
        } else if constexpr (std::is_same_v<Op, DApDiff>) {
            InterpretInternal<DType, Stack<ops::VARIABLE_OP, ops::FLIP_SIGN>>::eval(
                make_data_buffer(b_grad), b_grad);
        } else if constexpr (std::is_same_v<Op, DApMul>) {
            InterpretInternal<DType, Stack<ops::VARIABLE_OP, ops::VARIABLE_OP, ops::MUL_OP>>::eval(
                make_data_buffer(a_grad, b_prev), a_grad);
            InterpretInternal<DType, Stack<ops::VARIABLE_OP, ops::VARIABLE_OP, ops::MUL_OP>>::eval(
                make_data_buffer(b_grad, a_prev), b_grad);
        } else {
            static_assert(std::is_same_v<Op, DApSum>);
        }

        a_.backward_internal(reduce_axis(a_grad, a_prev.get_shape()));
        b_.backward_internal(reduce_axis(b_grad, b_prev.get_shape()));
    }
};