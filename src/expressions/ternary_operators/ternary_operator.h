#pragma once

#include "../expression_base.h"

#include "../../metaprogramming/stack.h"

// Common data to all ternary operators
template <typename A, typename B, typename C, typename Op>
class DTernaryExprCommonData {
  public:
    using Operator = Op;
    using DType = typename A::DType;

    A a_;
    B b_;
    C c_;
    ConstTensor<DType> res{};

  public:
    DTernaryExprCommonData(const A &a, const B &b, const C &c) : a_{a}, b_{b}, c_{c} {}
    template <typename Visitor>
    void traverse(Visitor &v) {
        v(*this);
        if constexpr (!Visitor::template END_RECURSION<Op>) {
            a_.traverse(v);
            b_.traverse(v);
            c_.traverse(v);
        }
    }
    template <typename Visitor>
    void traverse(Visitor &v) const {
        v(*this);
        if constexpr (!Visitor::template END_RECURSION<Op>) {
            a_.traverse(v);
            b_.traverse(v);
            c_.traverse(v);
        }
    }

    template <typename Visitor>
    static consteval auto traverse() {
        using This = DTernaryExprCommonData<A, B, C, Op>;
        constexpr auto node_res = Visitor::template Visit<This>();
        if constexpr (!Visitor::template END_RECURSION<Op>) {
            constexpr auto a_res = A::template traverse<Visitor>();
            constexpr auto b_res = B::template traverse<Visitor>();
            constexpr auto c_res = B::template traverse<Visitor>();

            return Visitor::template Aggregate(node_res, a_res, b_res, c_res);
        } else {
            return node_res;
        }
    }

    struct Simplify {
        using Type = DTernExprOp<typename A::Simplify::Type,
                                 typename B::Simplify::Type,
                                 typename C::Simplify::Type,
                                 Op>;
    };
};

template <typename A, typename B, typename C, typename Op>
requires(std::is_same_v<typename A::DType, typename B::DType>
             &&std::is_same_v<typename B::DType, typename C::DType>) class DTernExprOp
    : public DTernaryExprCommonData<A, B, C, Op> {
  private:
    using CommonData = DTernaryExprCommonData<A, B, C, Op>;
    using CommonData::a_;
    using CommonData::b_;
    using CommonData::c_;

  public:
    using CommonData::Operator;
    using CommonData::traverse;
    using typename CommonData::DType;
    using typename CommonData::Simplify;

    DTernExprOp(const A &a, const B &b, const C &c) : CommonData{a, b, c} {}

    template <bool recursive>
    struct Flatten {
        using tmp1 = std::conditional_t<recursive,
                                        typename A::template Flatten<true>::Type,
                                        Stack<ops::VARIABLE_OP>>;
        using tmp2 = std::conditional_t<recursive,
                                        typename B::template Flatten<true>::Type,
                                        Stack<ops::VARIABLE_OP>>;
        using tmp3 = std::conditional_t<recursive,
                                        typename C::template Flatten<true>::Type,
                                        Stack<ops::VARIABLE_OP>>;

        using tmp4 = Stack<Op::STACK_VAL>;
        using Type = MergeStacksT<MergeStacksT<MergeStacksT<tmp1, tmp2>, tmp3>, tmp4>;
    };

    void compute_temporaries_for_eval() {
        a_.compute_temporaries_for_eval();
        b_.compute_temporaries_for_eval();
        c_.compute_temporaries_for_eval();
    }
};