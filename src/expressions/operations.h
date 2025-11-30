#pragma once

#include "../constants.h"

/*
 * TODO: add more informations here, for example the number of operands of each operation, etc...
 */

class DApSum {
  public:
    /**
     * Unique identifier for the operator
     */
    static constexpr size_t STACK_VAL = ops::SUM_OP;
    /**
     * Tue iff the nodes requires to cache the temporary result for evaluation.
     * (Note: not for forward/backpropagation, in such case all nodes store their temporary)
     */
    static constexpr bool NEEDS_TEMPORARY_FOR_EVAL = false;
};

class DApDiff {
  public:
    static constexpr size_t STACK_VAL = ops::DIFF_OP;
    static constexpr bool NEEDS_TEMPORARY_FOR_EVAL = false;
};

class DApFMA {
  public:
    static constexpr size_t STACK_VAL = ops::FMA_OP;
    static constexpr bool NEEDS_TEMPORARY_FOR_EVAL = false;
};

class DApFAM {
  public:
    static constexpr size_t STACK_VAL = ops::FAM_OP;
    static constexpr bool NEEDS_TEMPORARY_FOR_EVAL = false;
};

class DApMul {
  public:
    static constexpr size_t STACK_VAL = ops::MUL_OP;
    static constexpr bool NEEDS_TEMPORARY_FOR_EVAL = false;
};

class DApDivide {
  public:
    static constexpr size_t STACK_VAL = ops::DIVIDE_OP;
    static constexpr bool NEEDS_TEMPORARY_FOR_EVAL = false;
};

template <bool tLeft, bool tRight>
class DApMatMul {
  public:
    static constexpr size_t STACK_VAL = ops::MAT_MUL<tLeft, tRight>;
    static constexpr bool NEEDS_TEMPORARY_FOR_EVAL = true;
};

class DApConv1d {
  public:
    static constexpr size_t STACK_VAL = ops::CONV_1D;
    static constexpr bool NEEDS_TEMPORARY_FOR_EVAL = true;
};

class DApConv2d {
  public:
    static constexpr size_t STACK_VAL = ops::CONV_2D;
    static constexpr bool NEEDS_TEMPORARY_FOR_EVAL = true;
};

class DApRELU {
  public:
    static constexpr size_t STACK_VAL = ops::RELU;
    static constexpr bool NEEDS_TEMPORARY_FOR_EVAL = false;
};

class DApTranspose {
  public:
    static constexpr size_t STACK_VAL = ops::TRANSPOSE;
    static constexpr bool NEEDS_TEMPORARY_FOR_EVAL = true;
};

class DApExp {
  public:
    static constexpr size_t STACK_VAL = ops::EXP;
    static constexpr bool NEEDS_TEMPORARY_FOR_EVAL = false;
};

class DApLog {
  public:
    static constexpr size_t STACK_VAL = ops::LOG;
    static constexpr bool NEEDS_TEMPORARY_FOR_EVAL = false;
};

class DApFlipSign {
  public:
    static constexpr size_t STACK_VAL = ops::FLIP_SIGN;
    static constexpr bool NEEDS_TEMPORARY_FOR_EVAL = false;
};

class DApSqrt {
  public:
    static constexpr size_t STACK_VAL = ops::SQRT;
    static constexpr bool NEEDS_TEMPORARY_FOR_EVAL = false;
};

class DApFlatten {
  public:
    static constexpr size_t STACK_VAL = ops::FLATTEN;
    static constexpr bool NEEDS_TEMPORARY_FOR_EVAL = true;
};

class DApIndexer {
  public:
    static constexpr size_t STACK_VAL = ops::INDEXER;
    static constexpr bool NEEDS_TEMPORARY_FOR_EVAL = true;
};

class DApShared {
  public:
    static constexpr size_t STACK_VAL = ops::SHARED;
    static constexpr bool NEEDS_TEMPORARY_FOR_EVAL = true;
};