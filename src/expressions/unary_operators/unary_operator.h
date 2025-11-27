#pragma once

#include "../expression_base.h"

#include "../../metaprogramming/stack.h"
#include "../../interpreter.h"

template <typename T, typename U>
requires(std::is_same_v<T, double> || std::is_same_v<T, float>) struct InterpretInternal;

template <typename A, typename Op>
class DUnaryExprOp;

// Common data to all unary operators
template <typename A, typename Op>
class DUnaryExprCommonData {
  public:
    using Operator = Op;
    using Operand = A;
    using DType = typename A::DType;

    A a_;
    ConstTensor<DType> res{};

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

    template <typename Visitor>
    static consteval auto traverse() {
        using This = DUnaryExprCommonData<A, Op>;
        constexpr auto node_res = Visitor::template Visit<This>();
        if constexpr (!Visitor::template END_RECURSION<Op>) {
            constexpr auto a_res = A::template traverse<Visitor>();
            return Visitor::template Aggregate(node_res, a_res);
        } else {
            return node_res;
        }
    }

    struct Simplify {
        using Type = DUnaryExprOp<typename A::Simplify::Type, Operator>;
    };

    template <bool recursive>
    struct FlattenOpNoTemporary {
        using tmp1 = std::conditional_t<recursive,
                                        typename A::template Flatten<true>::Type,
                                        Stack<ops::VARIABLE_OP>>;
        using tmp2 = Stack<Op::STACK_VAL>;
        using Type = MergeStacksT<tmp1, tmp2>;
    };

    template <bool recursive>
    struct Flatten {
        using Type = std::conditional_t<Op::NEEDS_TEMPORARY_FOR_EVAL,
                                        Stack<ops::VARIABLE_OP>,
                                        typename FlattenOpNoTemporary<recursive>::Type>;
    };
};

template <typename A, typename Op>
class DUnaryExprOp : public DUnaryExprCommonData<A, Op>, public DExpr<DUnaryExprOp<A, Op>> {
  private:
    using CommonData = DUnaryExprCommonData<A, Op>;
    using CommonData::a_;

  public:
    using CommonData::Operand;
    using CommonData::Operator;
    using CommonData::traverse;
    using typename CommonData::DType;
    using typename CommonData::Simplify;
    template <bool recursive>
    using Flatten = typename CommonData::Flatten<recursive>;
    DUnaryExprOp(const A &a) : CommonData{a} {}

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