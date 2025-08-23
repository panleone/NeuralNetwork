#pragma once

#include <concepts>

#include "../operations.h"
#include "common_simplifier.h"


/**
 * Utils for simplifying a binary expression
 */

template<typename Op1, typename Op2>
struct OperatorRules;

template<typename Op>
struct FlipRules;

template<typename Op1, typename Op2>
using OperatorRulesT = typename OperatorRules<Op1, Op2>::Type;

template<typename Op>
using FlipRulesT = typename FlipRules<Op>::Type;


template<typename T>
concept IsBinaryOp = requires {
    typename T::Left;
    typename T::Right;
};


template<typename Expr>
struct BinarySimplifier;

/**
 * Require that both operators are Binary and they are simplifiable
 */
template<typename T1, typename T2>
concept AreBinaryAndSimplifiable = IsBinaryOp<T1> && IsBinaryOp<T2> && requires {
    typename OperatorRules<OpT<T1>, OpT<T2>>::Type;
};

/**
 * Simplify the parent with its left children
 */
template<typename T>
requires (AreBinaryAndSimplifiable<T, LeftT<T>>)
struct BinarySimplifier<T> {
    using ResOp = OperatorRulesT<OpT<T>, OpT<LeftT<T>>>;
    using Type = DTernExprOp<typename LeftT<LeftT<T>>::Simplify::Type, typename RightT<LeftT<T>>::Simplify::Type, typename RightT<T>::Simplify::Type, ResOp>;
};


/**
 * Simplify the parent with its right children
 */
template<typename T>
requires (!AreBinaryAndSimplifiable<T, LeftT<T>> && AreBinaryAndSimplifiable<T, RightT<T>>)
struct BinarySimplifier<T> {
    using ResOp = FlipRulesT<OperatorRulesT<OpT<T>, OpT<RightT<T>>>>;
    using Type = DTernExprOp<typename LeftT<T>::Simplify::Type, typename LeftT<RightT<T>>::Simplify::Type, typename RightT<RightT<T>>::Simplify::Type, ResOp>;    
};


/**
 * No simplification is possible
 */
template<typename T>
requires (!AreBinaryAndSimplifiable<T, LeftT<T>> && !AreBinaryAndSimplifiable<T, RightT<T>>)
struct BinarySimplifier<T> {
    using Type = DBinExprOp<typename LeftT<T>::Simplify::Type, typename RightT<T>::Simplify::Type, OpT<T>>;
};

/**
 * Simplification rules
 */
template<>
struct OperatorRules<DApSum, DApMul>{
    using Type = DApFMA;
};


template<>
struct FlipRules<DApFMA>{
    using Type = DApFAM;
};