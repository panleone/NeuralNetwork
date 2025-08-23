#pragma once

#include "../expressions/expression.h"
#include "../tensor.h"

template<typename DType>
class FlattenerLayer{
public:
    FlattenerLayer() {}

    template<typename Expr>
    auto forward(const Expr& x){
        return flatten(x);
    }
};