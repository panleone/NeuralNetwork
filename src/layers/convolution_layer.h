#pragma once

#include "../expressions/expression.h"
#include "../tensor_variable.h"

template<typename DType>
class ConvolutionLayer1D{
    // kernel matrix
    Variable<DType, true> k;
    // bias vector
    Variable<DType, true> q;

    size_t stride;
public:
    ConvolutionLayer1D(size_t in_channels, size_t out_channels, size_t kernel_size, size_t stride) : k{{out_channels, in_channels, kernel_size}}, q{{out_channels}}, stride{stride}{}

    template<typename Expr>
    auto forward(const Expr& x){
        return conv_1d(k, x, q).set_stride(stride);
    }
};