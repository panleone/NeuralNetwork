#pragma once

#include <vector>

#include "binary_operators/binary_operator.h"
#include "binary_operators/matmul_operator.h"

#include "ternary_operators/ternary_operator.h"

#include "binary_operators/convolution_1d_operator.h"
#include "binary_operators/convolution_2d_operator.h"

#include "unary_operators/unary_operator.h"
#include "unary_operators/flattener_operator.h"
#include "variable.h"

template <typename Expr>
auto DExpr<Expr>::get_parameters() const {
    using T = IntrinsicType::Type;
    std::vector<Variable<T, true>> res{};
    static_cast<const Expr &>(*this).get_parameters_internal(res);
    return res;
}

template <typename Expr>
auto DExpr<Expr>::collect_tensor_handles() const {
    using T = IntrinsicType::Type;

    constexpr size_t num_tensors = DExpr<Expr>::get_num_tensors();
    DataBuffer<T, num_tensors> data_buffer{};
    static_cast<const Expr &>(*this).collect_tensor_handles(data_buffer);
    return data_buffer;
}