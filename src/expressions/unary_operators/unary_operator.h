#pragma once

#include "../expression_base.h"

#include "../../metaprogramming/stack.h"
#include "../../interpreter.h"

template <typename T, typename U>
requires(std::is_same_v<T, double> || std::is_same_v<T, float>) struct InterpretInternal;

// Common data to all unary operators
template <typename A, typename Op>
class DUnaryExprCommonData {
  public:
    A a_;
    ConstTensor<typename A::DType> res{};
    using Operator = Op;

  public:
    DUnaryExprCommonData(const A &a) : a_{a} {}
    template <typename Visitor>
    void traverse(Visitor &v) {
        v(*this);
        if constexpr (!Visitor::template END_RECURSION<Op>) {
            a_.traverse(v);
        }
    }
    template <typename Visitor>
    void traverse(Visitor &v) const {
        v(*this);
        if constexpr (!Visitor::template END_RECURSION<Op>) {
            a_.traverse(v);
        }
    }
};

template <typename A, typename Op>
class DUnaryExprOp : public DUnaryExprCommonData<A, Op>, public DExpr<DUnaryExprOp<A, Op>> {
  public:
    using DType = typename A::DType;
    using DUnaryExprCommonData<A, Op>::traverse;

  private:
    using DUnaryExprCommonData<A, Op>::a_;

  public:
    using Operand = A;
    using Operator = Op;

    DUnaryExprOp(const A &a) : DUnaryExprCommonData<A, Op>{a} {}

    template <bool recursive>
    struct Flatten {
        using tmp1 = std::conditional_t<recursive,
                                        typename A::template Flatten<true>::Type,
                                        Stack<ops::VARIABLE_OP>>;
        using tmp2 = Stack<Op::STACK_VAL>;
        using Type = MergeStacksT<tmp1, tmp2>;
    };

    static consteval size_t get_num_tensors() { return A::get_num_tensors(); }

    struct Simplify {
        using Type = DUnaryExprOp<typename A::Simplify::Type, Op>;
    };

    void compute_temporaries_for_eval() { a_.compute_temporaries_for_eval(); }
    template <bool use_cache>
    ConstTensor<DType> compute_temporaries_for_backprop() {
        if constexpr (!use_cache) {
            ConstTensor<DType> operand = a_.template compute_temporaries_for_backprop<use_cache>();

            this->res = InterpretInternal<DType, typename Flatten<false>::Type>::const_eval(
                make_data_buffer<DType>(operand));
        }
        return this->res;
    }

    void backward_internal(const Tensor<DType> &grad) {
        Tensor<DType> a_grad = grad.clone();
        ConstTensor<DType> a_res =
            a_.template compute_temporaries_for_backprop</*use_cache=*/true>();

        if constexpr (std::is_same_v<Op, DApRELU>) {
            relu_backprop(a_grad, a_res);
        } else if constexpr (std::is_same_v<Op, DApExp>) {
            InterpretInternal<DType, Stack<ops::VARIABLE_OP, ops::VARIABLE_OP, ops::MUL_OP>>::eval(
                make_data_buffer<DType>(a_grad, a_res), a_grad);
        } else if constexpr (std::is_same_v<Op, DApLog>) {
            InterpretInternal<DType, Stack<ops::VARIABLE_OP, ops::VARIABLE_OP, ops::DIVIDE_OP>>::
                eval(make_data_buffer<DType>(a_grad, a_res), a_grad);
        } else if constexpr (std::is_same_v<Op, DApFlipSign>) {
            InterpretInternal<DType, Stack<ops::VARIABLE_OP, ops::FLIP_SIGN>>::eval(
                make_data_buffer<DType>(a_grad), a_grad);
        } else {
            static_assert(std::is_same_v<Op, DApRELU>);
        }

        a_.backward_internal(a_grad);
    }
};