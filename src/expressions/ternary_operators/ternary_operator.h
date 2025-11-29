#pragma once

#include "../expression_base.h"
#include "../expression_common_data.h"
#include "../../metaprogramming/stack.h"

template <typename A, typename B, typename C, typename Op>
class DTernExprOp : public DExprCommonData<Op, A, B, C> {
  private:
    using CommonData = DExprCommonData<Op, A, B, C>;
    using CommonData::a_;
    using CommonData::b_;
    using CommonData::c_;

  public:
    using CommonData::traverse;
    using typename CommonData::DType;
    using typename CommonData::Operator;
    template <bool recursive>
    using Flatten = typename CommonData::Flatten<recursive>;

    DTernExprOp(const A &a, const B &b, const C &c) : CommonData{a, b, c} {}

    void compute_temporaries_for_eval() {
        a_().compute_temporaries_for_eval();
        b_().compute_temporaries_for_eval();
        c_().compute_temporaries_for_eval();
    }

    struct Simplify {
        using Type = DTernExprOp<typename A::Simplify::Type,
                                 typename B::Simplify::Type,
                                 typename C::Simplify::Type,
                                 Operator>;
    };
};