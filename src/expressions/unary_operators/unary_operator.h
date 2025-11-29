#pragma once

#include "../expression_base.h"
#include "../expression_common_data.h"

#include "../../metaprogramming/stack.h"
#include "../../interpreter.h"

template <typename T, typename U>
requires(std::is_same_v<T, double> || std::is_same_v<T, float>) struct InterpretInternal;

template <typename A, typename Op>
class DUnaryExprOp : public DExprCommonData<Op, A>, public DExpr<DUnaryExprOp<A, Op>> {
  private:
    using CommonData = DExprCommonData<Op, A>;
    using CommonData::a_;

  public:
    using Operand = A;
    using CommonData::traverse;
    using typename CommonData::DType;
    using typename CommonData::Operator;
    template <bool recursive>
    using Flatten = typename CommonData::Flatten<recursive>;
    DUnaryExprOp(const A &a) : CommonData{a} {}

    void compute_temporaries_for_eval() { a_().compute_temporaries_for_eval(); }
    template <bool use_cache>
    ConstTensor<DType> compute_temporaries_for_backprop() {
        if constexpr (!use_cache) {
            ConstTensor<DType> operand =
                a_().template compute_temporaries_for_backprop<use_cache>();

            this->res = InterpretInternal<DType, typename Flatten<false>::Type>::const_eval(
                make_data_buffer<DType>(operand));
        }
        return this->res;
    }

    void backward_internal(const Tensor<DType> &grad) {
        Tensor<DType> a_grad = grad.clone();
        ConstTensor<DType> a_res =
            a_().template compute_temporaries_for_backprop</*use_cache=*/true>();

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

        a_().backward_internal(a_grad);
    }

    struct Simplify {
        using Type = DUnaryExprOp<typename A::Simplify::Type, Op>;
    };
};