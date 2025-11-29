#pragma once

template <typename A, typename B, typename C, typename Op>
class DTernExprOp;

template <typename A, typename B, typename Op>
class DBinExprOp;

template <typename T>
using LeftT = typename T::Left;

template <typename T>
using RightT = typename T::Right;

template <typename T>
using OpT = typename T::Operator;