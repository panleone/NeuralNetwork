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
#include "unary_operators/shared_node_operator.h"

#include "variable.h"

#include "visitors/runtime_visitors.h"
#include "visitors/compile_time_visitors.h"

template <typename Expr>
auto DExpr<Expr>::get_parameters() const {
    using T = IntrinsicType::Type;
    return visitor_manager.visit<GetParametersVisitor<T>>(static_cast<const Expr &>(*this));
}

template <typename Expr>
auto DExpr<Expr>::collect_tensor_handles() const {
    using T = IntrinsicType::Type;
    constexpr size_t num_tensors = Expr::template traverse<GetNumTensorHandlesVisitor<T>>();

    return visitor_manager.visit<GetTensorHandlesVisitor<T, num_tensors>>(
        static_cast<const Expr &>(*this));
}

template <typename Expr>
void DExpr<Expr>::post_backprop_cleanup() {
    visitor_manager.visit<ResetSharedNodesVisitor>(static_cast<Expr &>(*this));
}