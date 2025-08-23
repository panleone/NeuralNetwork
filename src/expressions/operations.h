#pragma once

#include "../constants.h"

/*
 * TODO: add more informations here, for example the number of operands of each operation, if they
 * need the partial evaluation etc...
 */

class DApSum {
  public:
    static constexpr size_t STACK_VAL = ops::SUM_OP;
};

class DApDiff {
  public:
    static constexpr size_t STACK_VAL = ops::DIFF_OP;
};

class DApFMA {
  public:
    static constexpr size_t STACK_VAL = ops::FMA_OP;
};

class DApFAM {
  public:
    static constexpr size_t STACK_VAL = ops::FAM_OP;
};

class DApMul {
  public:
    static constexpr size_t STACK_VAL = ops::MUL_OP;
};

class DApDivide {
  public:
    static constexpr size_t STACK_VAL = ops::DIVIDE_OP;
};

template <bool tLeft, bool tRight>
class DApMatMul {
  public:
    static constexpr size_t STACK_VAL = ops::MAT_MUL<tLeft, tRight>;
};

class DApConv1d {
  public:
    static constexpr size_t STACK_VAL = ops::CONV_1D;
};

class DApConv2d {
  public:
    static constexpr size_t STACK_VAL = ops::CONV_2D;
};

class DApRELU {
  public:
    static constexpr size_t STACK_VAL = ops::RELU;
};

class DApTranspose {
  public:
    static constexpr size_t STACK_VAL = ops::TRANSPOSE;
};

class DApExp {
  public:
    static constexpr size_t STACK_VAL = ops::EXP;
};

class DApLog {
  public:
    static constexpr size_t STACK_VAL = ops::LOG;
};

class DApFlipSign {
  public:
    static constexpr size_t STACK_VAL = ops::FLIP_SIGN;
};

class DApSqrt {
  public:
    static constexpr size_t STACK_VAL = ops::SQRT;
};

class DApFlatten {
  public:
    static constexpr size_t STACK_VAL = ops::FLATTEN;
};