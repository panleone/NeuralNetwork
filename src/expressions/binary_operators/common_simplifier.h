#pragma once

template <typename A, typename B, typename C, typename Op>
requires(std::is_same_v<typename A::DType, typename B::DType>
             &&std::is_same_v<typename B::DType, typename C::DType>) class DTernExprOp;

template <typename A, typename B, typename Op>
requires(std::is_same_v<typename A::DType, typename B::DType>) class DBinExprOp;

template <typename T>
using LeftT = typename T::Left;

template <typename T>
using RightT = typename T::Right;

template <typename T>
using OpT = typename T::Operator;