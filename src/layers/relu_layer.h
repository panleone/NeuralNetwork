#pragma once

#include "../expressions/expression.h"
#include "../tensor.h"

template <typename DType>
class ReluLayer {
  public:
    ReluLayer() {}

    template <typename Expr>
    auto forward(const Expr &x) {
        return relu(x);
    }
};