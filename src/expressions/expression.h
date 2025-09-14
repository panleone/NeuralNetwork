#pragma once

/**
 * To use the expressions system importing this header is enough.
 * TODO: find a smarter way?
 */
#include "operations.h"
#include "../tensor.h"

#include "../metaprogramming/stack.h"
#include "expression_base_impl.h"

#include "../tensor_variable.h"

template <typename T, bool require_gradient>
DExprTensor<T, require_gradient> to_dexpr_internal(const Variable<T, require_gradient> &x) {
    return DExprTensor<T, require_gradient>(x);
}

inline DExprTensor<double, false> no_grad(const Tensor<double> &x) {
    return DExprTensor<double, false>(x);
}

inline DExprTensor<float, false> no_grad(const Tensor<float> &x) {
    return DExprTensor<float, false>(x);
}

inline DExprTensor<float, false> no_grad(float x) {
    Tensor<float> x_tensor{1};
    x_tensor[0] = x;
    return DExprTensor<float, false>(x_tensor);
}

inline DExprTensor<double, false> no_grad(double x) {
    Tensor<double> x_tensor{1};
    x_tensor[0] = x;
    return DExprTensor<double, false>(x_tensor);
}

// For variables

inline DExprTensor<double, true> to_dexpr(const Variable<double, true> &x) {
    return to_dexpr_internal<double, true>(x);
}

inline DExprTensor<double, false> to_dexpr(const Variable<double, false> &x) {
    return to_dexpr_internal<double, false>(x);
}

inline DExprTensor<float, true> to_dexpr(const Variable<float, true> &x) {
    return to_dexpr_internal<float, true>(x);
}

inline DExprTensor<float, false> to_dexpr(const Variable<float, false> &x) {
    return to_dexpr_internal<float, false>(x);
}

template <typename T>
concept HasToDexpr = requires(const T &t) {
    {to_dexpr(t)};
};

template <typename Expr>
const DExpr<Expr> &to_dexpr(const DExpr<Expr> &x) {
    return x;
}

template <typename A, typename B>
requires(HasToDexpr<A>) && (HasToDexpr<B>)auto operator+(const A &x, const B &y) {
    return to_dexpr(x) + to_dexpr(y);
}

template <typename A, typename B>
requires(HasToDexpr<A>) && (HasToDexpr<B>)auto operator-(const A &x, const B &y) {
    return to_dexpr(x) - to_dexpr(y);
}

template <typename A, typename B>
requires(HasToDexpr<A>) && (HasToDexpr<B>)auto operator/(const A &x, const B &y) {
    return to_dexpr(x) / to_dexpr(y);
}

template <typename A, typename B>
requires(HasToDexpr<A>) && (HasToDexpr<B>)auto operator*(const A &x, const B &y) {
    return to_dexpr(x) * to_dexpr(y);
}

template <typename A, typename B>
requires(HasToDexpr<A>) && (HasToDexpr<B>)auto matmul(const A &x, const B &y) {
    return matmul(to_dexpr(x), to_dexpr(y));
}

template <typename A, typename B, typename C>
requires(HasToDexpr<A>) &&
    (HasToDexpr<B>)&&(HasToDexpr<C>)auto conv_1d(const A &x, const B &y, const C &z) {
    return conv_1d(to_dexpr(x), to_dexpr(y), to_dexpr(z));
}

template <typename A, typename B, typename C>
requires(HasToDexpr<A>) &&
(HasToDexpr<B>)&&(HasToDexpr<C>)auto conv_2d(const A &x, const B &y, const C &z) {
    return conv_2d(to_dexpr(x), to_dexpr(y), to_dexpr(z));
}

template <typename A>
requires(HasToDexpr<A>) auto relu(const A &x) { return relu(to_dexpr(x)); }

template <typename A>
requires(HasToDexpr<A>) auto transpose(const A &x) { return transpose(to_dexpr(x)); }

template <typename A>
requires(HasToDexpr<A>) auto exp(const A &x) { return exp(to_dexpr(x)); }

template <typename A>
requires(HasToDexpr<A>) auto log(const A &x) { return log(to_dexpr(x)); }

template <typename A>
requires(HasToDexpr<A>) auto sqrt(const A &x) { return sqrt(to_dexpr(x)); }

template <typename A>
requires(HasToDexpr<A>) auto flatten(const A &x) { return flatten(to_dexpr(x)); }

template <typename A>
requires(HasToDexpr<A>) auto operator-(const A &x) { return -to_dexpr(x); }

template <typename A, typename B>
auto operator/(const DExpr<A> &x, const DExpr<B> &y) {
    return DBinExprOp<A, B, DApDivide>(static_cast<const A &>(x), static_cast<const B &>(y));
}

template <typename A, typename B>
auto operator*(const DExpr<A> &x, const DExpr<B> &y) {
    return DBinExprOp<A, B, DApMul>(static_cast<const A &>(x), static_cast<const B &>(y));
}

template <typename A, typename B>
auto operator+(const DExpr<A> &x, const DExpr<B> &y) {
    return DBinExprOp<A, B, DApSum>(static_cast<const A &>(x), static_cast<const B &>(y));
}

template <typename A, typename B>
auto operator-(const DExpr<A> &x, const DExpr<B> &y) {
    return DBinExprOp<A, B, DApDiff>(static_cast<const A &>(x), static_cast<const B &>(y));
}

template <typename A, typename B>
auto matmul(const DExpr<A> &x, const DExpr<B> &y) {
    return DBinExprOp<A, B, DApMatMul<false, false>>(static_cast<const A &>(x),
                                                     static_cast<const B &>(y));
}

template <typename A, typename B, typename C>
auto conv_1d(const DExpr<A> &x, const DExpr<B> &y, const DExpr<C> &z) {
    return DTernExprOp<A, B, C, DApConv1d>(
        static_cast<const A &>(x), static_cast<const B &>(y), static_cast<const C &>(z));
}

template <typename A, typename B, typename C>
auto conv_2d(const DExpr<A> &x, const DExpr<B> &y, const DExpr<C> &z) {
    return DTernExprOp<A, B, C, DApConv2d>(
            static_cast<const A &>(x), static_cast<const B &>(y), static_cast<const C &>(z));
}

template <typename A>
auto relu(const DExpr<A> &x) {
    return DUnaryExprOp<A, DApRELU>(static_cast<const A &>(x));
}

template <typename A>
auto transpose(const DExpr<A> &x) {
    return DUnaryExprOp<A, DApTranspose>(static_cast<const A &>(x));
}

template <typename A>
auto exp(const DExpr<A> &x) {
    return DUnaryExprOp<A, DApExp>(static_cast<const A &>(x));
}

/**
 * natural logarithm
 */
template <typename A>
auto log(const DExpr<A> &x) {
    return DUnaryExprOp<A, DApLog>(static_cast<const A &>(x));
}

template <typename A>
auto sqrt(const DExpr<A> &x) {
    return DUnaryExprOp<A, DApSqrt>(static_cast<const A &>(x));
}

template <typename A>
auto flatten(const DExpr<A> &x) {
    return DUnaryExprOp<A, DApFlatten>(static_cast<const A &>(x));
}

template <typename A>
auto operator-(const DExpr<A> &x) {
    return DUnaryExprOp<A, DApFlipSign>(static_cast<const A &>(x));
}

template <typename A>
void operator+=(const auto &x, const DExpr<A> &y) {
    (no_grad(x) + y).eval(x);
}

template <typename A>
void operator-=(const auto &x, const DExpr<A> &y) {
    (no_grad(x) - y).eval(x);
}
