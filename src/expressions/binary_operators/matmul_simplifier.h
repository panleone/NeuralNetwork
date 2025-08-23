#pragma once

#include <concepts>

#include "../operations.h"
#include "common_simplifier.h"


template<typename T>
concept IsMatmulOp = (
    std::is_same_v<OpT<T>, DApMatMul<false, false>>
);

template<typename T1, typename T2>
concept IsMatmulSimplifiable = IsMatmulOp<T1> && (
    std::is_same_v<typename T2::Operator, DApTranspose>
);

template<typename Expr>
struct MatMulSimplifier;

/**
 * mat_mul(x.T, y.T) -> mat_mul<T,T>(x, y) 
 */
template<typename T>
requires (IsMatmulOp<T> && IsMatmulSimplifiable<T, LeftT<T>> && IsMatmulSimplifiable<T, RightT<T>>)
struct MatMulSimplifier<T> {
    using LeftOperand = typename LeftT<T>::Operand;
    using RightOperand = typename RightT<T>::Operand;
    using Type = DBinExprOp<typename LeftOperand::Simplify::Type, typename RightOperand::Simplify::Type, DApMatMul<true, true>>;
};

/**
 * mat_mul(x, y.T) -> mat_mul<,T>(x, y) 
 */
template<typename T>
requires (IsMatmulOp<T> && !IsMatmulSimplifiable<T, LeftT<T>> && IsMatmulSimplifiable<T, RightT<T>>)
struct MatMulSimplifier<T> {
    using LeftOperand = LeftT<T>;
    using RightOperand = typename RightT<T>::Operand;
    using Type = DBinExprOp<typename LeftOperand::Simplify::Type, typename RightOperand::Simplify::Type, DApMatMul<false, true>>;
};

/**
 * mat_mul(x.T, y) -> mat_mul<T,>(x, y) 
 */
template<typename T>
requires (IsMatmulOp<T> && IsMatmulSimplifiable<T, LeftT<T>> && !IsMatmulSimplifiable<T, RightT<T>>)
struct MatMulSimplifier<T> {
    using LeftOperand = typename LeftT<T>::Operand;
    using RightOperand = RightT<T>;
    using Type = DBinExprOp<typename LeftOperand::Simplify::Type, typename RightOperand::Simplify::Type, DApMatMul<true, false>>;
};

/**
 * mat_mul(x, y) -> mat_mul<,>(x, y)
 */
template<typename T>
requires (IsMatmulOp<T> && !IsMatmulSimplifiable<T, LeftT<T>> && !IsMatmulSimplifiable<T, RightT<T>>)
struct MatMulSimplifier<T> {
    using LeftOperand = LeftT<T>;
    using RightOperand = RightT<T>;
    using Type = DBinExprOp<typename LeftOperand::Simplify::Type, typename RightOperand::Simplify::Type, DApMatMul<false, false>>;
};