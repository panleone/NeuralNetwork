#pragma once

#include <vector>

#include "binary_operators/binary_operator.h"
#include "binary_operators/matmul_operator.h"

#include "ternary_operators/ternary_operator.h"
#include "ternary_operators/convolution_1d_operator.h"
#include "ternary_operators/convolution_2d_operator.h"

#include "unary_operators/unary_operator.h"
#include "unary_operators/flattener_operator.h"
#include "unary_operators/indexing_operator.h"
#include "variable.h"

#include "visitors/visitors.h"

template <typename Expr>
auto DExpr<Expr>::get_parameters() const {
    using T = IntrinsicType::Type;
    GetParametersVisitor<T> visitor{};

    static_cast<const Expr &>(*this).traverse(visitor);
    return visitor.res;
}

template <typename Expr>
auto DExpr<Expr>::collect_tensor_handles() const {
    using T = IntrinsicType::Type;
    constexpr size_t num_tensors = DExpr<Expr>::get_num_tensors();
    GetTensorHandlesVisitor<T, num_tensors> visitor{};

    static_cast<const Expr &>(*this).traverse(visitor);
    return visitor.res;
}