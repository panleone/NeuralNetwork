#pragma once

#include "metaprogramming/stack.h"
#include "expressions/expression.h"
#include "constants.h"

#include <immintrin.h>

#include "avx/avx_ops.h"
#include "avx/avx_wrapper.h"

#include "blas_wrapper.h"
template <typename DType, size_t N>
class DataStack {
    size_t stack_index{0};
    simd_type<DType> avx_stack[N];

  public:
    void push(simd_type<DType> val) {
        avx_stack[stack_index] = val;
        ++stack_index;
    }
    simd_type<DType> pop() {
        --stack_index;
        return avx_stack[stack_index];
    }
    void reset() { stack_index = 0; }
};

template <typename DType, size_t instruction, typename RegisterType, typename DataBuffer>
inline void execute_instruction_avx(DataBuffer &data_pointers, RegisterType &registers, size_t i) {
    if constexpr (instruction == ops::VARIABLE_OP) {
        registers.push(_mm256_loadu_px(&data_pointers.get_next_variable()[i]));
    } else if constexpr (instruction == ops::SUM_OP) {
        auto r2 = registers.pop();
        auto r1 = registers.pop();
        registers.push(_mm256_add_px<DType>(r1, r2));
    } else if constexpr (instruction == ops::DIFF_OP) {
        auto r2 = registers.pop();
        auto r1 = registers.pop();
        registers.push(_mm256_sub_px<DType>(r1, r2));
    } else if constexpr (instruction == ops::MUL_OP) {
        auto r2 = registers.pop();
        auto r1 = registers.pop();
        registers.push(_mm256_mul_px<DType>(r1, r2));
    } else if constexpr (instruction == ops::DIVIDE_OP) {
        auto r2 = registers.pop();
        auto r1 = registers.pop();
        registers.push(_mm256_div_px<DType>(r1, r2));
    } else if constexpr (instruction == ops::FMA_OP) {
        auto r3 = registers.pop();
        auto r2 = registers.pop();
        auto r1 = registers.pop();
        registers.push(_mm256_fmadd_px<DType>(r1, r2, r3));
    } else if constexpr (instruction == ops::FAM_OP) {
        auto r3 = registers.pop();
        auto r2 = registers.pop();
        auto r1 = registers.pop();
        registers.push(_mm256_fmadd_px<DType>(r2, r3, r1));
    } else if constexpr (instruction == ops::RELU) {
        auto r1 = registers.pop();
        registers.push(_mm256_max_px<DType>(r1, avx_constants::zero<DType>));
    } else if constexpr (instruction == ops::EXP) {
        auto r1 = registers.pop();
        registers.push(_mm256_exp_px<DType>(r1));
    } else if constexpr (instruction == ops::LOG) {
        auto r1 = registers.pop();
        registers.push(_mm256_log_px<DType>(r1));
    } else if constexpr (instruction == ops::FLIP_SIGN) {
        auto r1 = registers.pop();
        registers.push(_mm256_flip_sign_px<DType>(r1));
    } else if constexpr (instruction == ops::SQRT) {
        auto r1 = registers.pop();
        registers.push(_mm256_sqrt_px<DType>(r1));
    } else {
        static_assert(instruction == ops::VARIABLE_OP);
    }
}

template <typename T, typename U>
requires(std::is_same_v<T, double> || std::is_same_v<T, float>) struct InterpretInternal;

template <typename DType, size_t... indices>
requires(std::is_same_v<DType, double> ||
         std::is_same_v<DType, float>) struct InterpretInternal<DType, Stack<indices...>> {
    InterpretInternal() = delete;
    static void eval(auto &&data_pointers, const Tensor<DType> &res) {

        // TODO: this is in an overestimate of the actual stack size needed
        constexpr size_t registers_stack_size =
            CountStack<Stack<indices...>, ops::VARIABLE_OP>::value;
        DataStack<DType, registers_stack_size> registers;

        for (size_t i = 0; i < res.get_size(); i += avx_constants::intrinsic_size<DType>) {
            data_pointers.reset();
            (execute_instruction_avx<DType, indices>(data_pointers, registers, i), ...);
            _mm256_storeu_px(&res[i], registers.pop());
        }
        res.wrap_for_broadcasting();
    }
    static ConstTensor<DType> const_eval(auto &&data_pointers) {
        // In the trivial case in which the operation is trivial (the identity)
        // Then just return a shallow copy of the input
        if constexpr (sizeof...(indices) == 1) {
            return data_pointers.get_next_variable().t_ref;
        } else {
            Tensor<DType> res{data_pointers.get_max_shape()};
            eval(std::forward<decltype(data_pointers)>(data_pointers), res);
            res.wrap_for_broadcasting();
            return res;
        }
    }
};

template <typename SimplifiedExpr>
class Interpreter {
    Interpreter() = delete;
    using DExprStack = typename DExpr<SimplifiedExpr>::template Flatten<true>::Type;

  public:
    template <typename Expr>
    static Tensor<typename Expr::DType> interpret(const DExpr<Expr> &expression) {
        auto data_pointers = expression.collect_tensor_handles();
        Tensor<typename Expr::DType> res{data_pointers.get_max_shape()};

        InterpretInternal<typename Expr::DType, DExprStack>::eval(data_pointers, res);
        return res;
    }
    template <typename Expr>
    static void interpret(const DExpr<Expr> &expression, const Tensor<typename Expr::DType> &res) {
        auto data_pointers = expression.collect_tensor_handles();
        InterpretInternal<typename Expr::DType, DExprStack>::eval(data_pointers, res);
    }
    template <typename Expr>
    static ConstTensor<typename Expr::DType> const_interpret(const DExpr<Expr> &expression) {
        auto data_pointers = expression.collect_tensor_handles();
        return InterpretInternal<typename Expr::DType, DExprStack>::const_eval(data_pointers);
    }
};

/**
 * From here backpropagation utils that I am not sure where to put yet
 */

// TODO: optimize
// TODO: put this function in another file?
template <typename DType>
inline Tensor<DType> reduce_axis(Tensor<DType> tensor, Shape target_shape) {
    Tensor<DType> res{std::move(target_shape)};
    res.set_zero();
    size_t res_size = res.get_size();
    for (size_t i = 0; i < tensor.get_size(); i++) {
        res[i % res_size] += tensor[i];
    }
    res.wrap_for_broadcasting();
    return res;
}

template <typename DType>
inline void relu_backprop(Tensor<DType> input_grad, ConstTensor<DType> tensor) {
    assert(input_grad.get_size() == tensor.get_size());
    for (size_t i = 0; i < input_grad.get_size(); i += avx_constants::intrinsic_size<DType>) {
        simd_type<DType> v_input_grad = _mm256_loadu_px(&input_grad[i]);
        simd_type<DType> v_tensor = _mm256_loadu_px(&tensor[i]);

        simd_type<DType> cmp =
            _mm256_cmp_px<DType, _CMP_GT_OS>(v_tensor, avx_constants::zero<DType>);
        simd_type<DType> res = _mm256_and_px<DType>(v_input_grad, cmp);

        _mm256_storeu_px(&input_grad[i], res);
    }
    input_grad.wrap_for_broadcasting();
}

template <typename DType>
inline DType get_max(ConstTensor<DType> t) {

    // This chould crash for tensors with less than intrinsic_size elements, BUT
    // our tensors have all more than intrinsic_size elements due to broadcast logic
    simd_type<DType> v_res = _mm256_loadu_px(&t[0]);
    for (size_t i = avx_constants::intrinsic_size<DType>; i < t.get_size();
         i += avx_constants::intrinsic_size<DType>) {
        simd_type<DType> v_t = _mm256_loadu_px(&t[i]);
        v_res = _mm256_max_px<DType>(v_res, v_t);
    }
    alignas(32) DType res[avx_constants::intrinsic_size<DType>];
    _mm256_store_px(res, v_res);

    // TODO: can we do better here?
    DType max_val = res[0];
    for (size_t i = 1; i < avx_constants::intrinsic_size<DType>; ++i) {
        max_val = std::max(max_val, res[i]);
    }
    return max_val;
}
template <typename DType>
inline void softmax_max_shift(const Tensor<DType> &t) {
    const auto &t_shape = t.get_shape();
    const auto &t_shape_data = t_shape.get_shape();

    assert(t_shape.get_dimension() == 2);

    for (size_t b = 0; b < t_shape_data[0]; ++b) {
        DType batch_max = t(b, 0);
        for (size_t i = 1; i < t_shape_data[1]; ++i) {
            batch_max = std::max(batch_max, t(b, i));
        }
        for (size_t i = 0; i < t_shape_data[1]; ++i) {
            t(b, i) -= batch_max;
        }
    }
    t.wrap_for_broadcasting();
}

template <typename DType>
inline void softmax_normalization(const Tensor<DType> &t) {
    const auto &t_shape = t.get_shape();
    const auto &t_shape_data = t_shape.get_shape();

    assert(t_shape.get_dimension() == 2);

    for (size_t b = 0; b < t_shape_data[0]; ++b) {
        DType batch_sum = t(b, 0);
        for (size_t i = 1; i < t_shape_data[1]; ++i) {
            batch_sum += t(b, i);
        }
        DType batch_sum_inv = static_cast<DType>(1.0) / batch_sum;
        for (size_t i = 0; i < t_shape_data[1]; ++i) {
            t(b, i) *= batch_sum_inv;
        }
    }
    t.wrap_for_broadcasting();
}

template <typename DType>
inline DType get_sum(ConstTensor<DType> t) {
    simd_type<DType> v_sum = avx_constants::zero<DType>;
    size_t i = 0;
    for (; i + avx_constants::intrinsic_size<DType> <= t.get_size();
         i += avx_constants::intrinsic_size<DType>) {
        simd_type<DType> v_t = _mm256_loadu_px(&t[i]);
        v_sum = _mm256_add_px<DType>(v_sum, v_t);
    }

    alignas(32) DType tmp[avx_constants::intrinsic_size<DType>];
    _mm256_store_px(tmp, v_sum);

    DType res = tmp[0];
    for (size_t j = 1; j < avx_constants::intrinsic_size<DType>; ++j) {
        res += tmp[j];
    }

    for (; i < t.get_size(); i++) {
        res += t[i];
    }
    return res;
}

// TODO: move somewhere else?
template <typename DType, bool transpose_t1 = false, bool transpose_t2 = false>
static inline Tensor<DType> mat_mul_wrapper(const ConstTensor<DType> &t1,
                                            const ConstTensor<DType> &t2,
                                            const Shape &res_shape) {
    Tensor<DType> res{res_shape};

    const auto &t1_s = t1.get_shape();
    const auto &t2_s = t2.get_shape();

    // Convention: Treat 1D vectors as a Matrix with only one column (i.e. shape [N, 1])
    size_t t1_row = t1_s.get_dimension() == 1 ? t1_s.get_size() : t1_s.get_size() / t1_s.last();
    size_t t1_col = t1_s.get_size() / t1_row;

    size_t t2_row = t2_s.first();
    size_t t2_col = t2_s.get_size() / t2_row;

    blas_mat_mul<DType, transpose_t1, transpose_t2>(
        &t1[0], &t2[0], &res[0], t1_row, t1_col, t2_row, t2_col);

    res.wrap_for_broadcasting();
    return res;
}