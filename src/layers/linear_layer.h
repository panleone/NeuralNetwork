#pragma once

#include "../expressions/expression.h"
#include "../tensor_variable.h"

template<typename DType>
class LinearLayer{
    // weight matrix
    Variable<DType, true> m;
    // bias vector
    Variable<DType, true> q;
public:
    LinearLayer(size_t in_size, size_t out_size) : m{{in_size, out_size}}, q{{out_size}}{}

    template<typename Expr>
    auto forward(const Expr& x){
        return matmul(x, m) + q;
    }
};