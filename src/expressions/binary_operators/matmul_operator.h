#pragma once

#include "binary_operator.h"
#include "matmul_simplifier.h"
#include "../../blas_wrapper.h"


/**
 * Partial specialization for matrix multiplication
 */
template<typename A, typename B, bool  tLeft, bool tRight>
requires (std::is_same_v<typename A::DType, typename B::DType>)
class DBinExprOp<A, B, DApMatMul<tLeft, tRight>> : public DExpr<DBinExprOp<A, B, DApMatMul<tLeft, tRight>>> {
public:
    using DType = typename A::DType;
private:
    A a_;
    B b_;
    // Result of the mat mul
    ConstTensor<DType> res{};
    using This = DBinExprOp<A, B, DApMatMul<tLeft, tRight>>;
public:
    using Left = A;
    using Right = B;
    using Operator = DApMatMul<tLeft, tRight>;

    static constexpr bool transpose_left = tLeft;
    static constexpr bool transpose_right = tRight;

    DBinExprOp(const A& a, const B& b) : a_{a}, b_{b}{}

    static consteval size_t get_num_tensors() {
        return 1;
    }

    void collect_tensor_handles(auto& current_stack) const {
        current_stack.push_back_variable(res);
    }
    void get_parameters_internal(auto& res) const {
        a_.get_parameters_internal(res);
        b_.get_parameters_internal(res);
    }

    template<bool recursive>
    struct Flatten {
        using Type = Stack<ops::VARIABLE_OP>;
    };

    struct Simplify {
        using Type = typename MatMulSimplifier<This>::Type;
    };

    void compute_temporaries_for_eval() {
        using SimplifiedT = Simplify::Type;

        a_.compute_temporaries_for_eval();
        b_.compute_temporaries_for_eval();

        auto t1 = Interpreter<typename SimplifiedT::Left>::const_interpret(a_);
        auto t2 = Interpreter<typename SimplifiedT::Right>::const_interpret(b_);

        auto res_shape = Shape::get_matmul_shape<SimplifiedT::transpose_left, SimplifiedT::transpose_right>(t1.get_shape(), t2.get_shape());

        res = mat_mul_wrapper<DType, SimplifiedT::transpose_left, SimplifiedT::transpose_right>(t1, t2, res_shape);
    }

    template<bool use_cache>
    ConstTensor<DType> compute_temporaries_for_backprop() {
        if constexpr(!use_cache){
            ConstTensor<DType> t1 = a_.template compute_temporaries_for_backprop<use_cache>();
            ConstTensor<DType> t2 = b_.template compute_temporaries_for_backprop<use_cache>();

            res = mat_mul_wrapper<DType, false, false>(t1, t2, Shape::get_matmul_shape<false, false>(t1.get_shape(), t2.get_shape()));
        }
        return res;
    }

    void backward_internal(const Tensor<DType>& grad){
        ConstTensor<DType> a_res = a_.template compute_temporaries_for_backprop</*use_cache=*/true>();
        ConstTensor<DType> b_res = b_.template compute_temporaries_for_backprop</*use_cache=*/true>();

        Tensor<DType> a_grad = mat_mul_wrapper<DType, false, true>(grad, b_res, a_res.get_shape());
        Tensor<DType> b_grad = mat_mul_wrapper<DType, true, false>(a_res, grad, b_res.get_shape());

        a_.backward_internal(a_grad);
        b_.backward_internal(b_grad);
    }
};


