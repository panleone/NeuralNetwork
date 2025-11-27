#pragma once

#include "../expression_base.h"
#include "binary_operator_simplifier.h"
#include "../../interpreter.h"

// Common data to all binary operators
template <typename A, typename B, typename Op>
class DBinaryExprCommonData {
  public:
    using Operator = Op;
    using DType = typename A::DType;

    A a_;
    B b_;
    ConstTensor<DType> res{};

  public:
    DBinaryExprCommonData(const A &a, const B &b) : a_{a}, b_{b} {}
    template <typename Visitor>
    void traverse(Visitor &v) {
        v(*this);
        if constexpr (!Visitor::template END_RECURSION<Op>) {
            a_.traverse(v);
            b_.traverse(v);
        }
    }
    template <typename Visitor>
    void traverse(Visitor &v) const {
        v(*this);
        if constexpr (!Visitor::template END_RECURSION<Op>) {
            a_.traverse(v);
            b_.traverse(v);
        }
    }

    template <typename Visitor>
    static consteval auto traverse() {
        using This = DBinaryExprCommonData<A, B, Op>;
        constexpr auto node_res = Visitor::template Visit<This>();
        if constexpr (!Visitor::template END_RECURSION<Op>) {
            constexpr auto a_res = A::template traverse<Visitor>();
            constexpr auto b_res = B::template traverse<Visitor>();

            return Visitor::template Aggregate(node_res, a_res, b_res);
        } else {
            return node_res;
        }
    }

    template <bool recursive>
    struct FlattenOpNoTemporary {
        using tmp1 = std::conditional_t<recursive,
                                        typename A::template Flatten<true>::Type,
                                        Stack<ops::VARIABLE_OP>>;
        using tmp2 = std::conditional_t<recursive,
                                        typename B::template Flatten<true>::Type,
                                        Stack<ops::VARIABLE_OP>>;
        using tmp3 = Stack<Op::STACK_VAL>;
        using Type = MergeStacksT<MergeStacksT<tmp1, tmp2>, tmp3>;
    };
    template <bool recursive>
    struct Flatten {
        using Type = std::conditional_t<Op::NEEDS_TEMPORARY_FOR_EVAL,
                                        Stack<ops::VARIABLE_OP>,
                                        typename FlattenOpNoTemporary<recursive>::Type>;
    };
};

template <typename A, typename B, typename Op>
requires(std::is_same_v<typename A::DType, typename B::DType>) class DBinExprOp
    : public DBinaryExprCommonData<A, B, Op>,
      public DExpr<DBinExprOp<A, B, Op>> {
  private:
    using CommonData = DBinaryExprCommonData<A, B, Op>;
    using CommonData::a_;
    using CommonData::b_;
    using This = DBinExprOp<A, B, Op>;

  public:
    using CommonData::Operator;
    using CommonData::traverse;
    using typename CommonData::DType;
    template <bool recursive>
    using Flatten = typename CommonData::Flatten<recursive>;

    using Left = A;
    using Right = B;

    DBinExprOp(const A &a, const B &b) : CommonData{a, b} {}

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
            this->res = InterpretInternal<DType, typename Flatten<false>::Type>::const_eval(
                make_data_buffer<DType>(a_res, b_res));
        }
        return this->res;
    }

    void backward_internal(const Tensor<DType> &grad) {
        Tensor<DType> a_grad = grad.clone();
        Tensor<DType> b_grad = grad.clone();

        ConstTensor<DType> ac_grad = ConstTensor<DType>(a_grad);
        ConstTensor<DType> bc_grad = ConstTensor<DType>(b_grad);

        ConstTensor<DType> a_prev =
            a_.template compute_temporaries_for_backprop</*use_cache=*/true>();
        ConstTensor<DType> b_prev =
            b_.template compute_temporaries_for_backprop</*use_cache=*/true>();

        if constexpr (std::is_same_v<Op, DApSum>) {
            // Nothing to do
        } else if constexpr (std::is_same_v<Op, DApDiff>) {
            InterpretInternal<DType, Stack<ops::VARIABLE_OP, ops::FLIP_SIGN>>::eval(
                make_data_buffer<DType>(b_grad), b_grad);
        } else if constexpr (std::is_same_v<Op, DApMul>) {
            InterpretInternal<DType, Stack<ops::VARIABLE_OP, ops::VARIABLE_OP, ops::MUL_OP>>::eval(
                make_data_buffer<DType>(ac_grad, b_prev), a_grad);
            InterpretInternal<DType, Stack<ops::VARIABLE_OP, ops::VARIABLE_OP, ops::MUL_OP>>::eval(
                make_data_buffer<DType>(bc_grad, a_prev), b_grad);
        } else {
            static_assert(std::is_same_v<Op, DApSum>);
        }

        a_.backward_internal(reduce_axis(a_grad, a_prev.get_shape()));
        b_.backward_internal(reduce_axis(b_grad, b_prev.get_shape()));
    }
};