#pragma once

#include "tensor.h"

#include "expressions/expression.h"
template<typename Expr>
class DExpr;

/**
 * SoftMax loss function (softmax + cross entropy).
 * Since the backward step can be simplified a lot (algebraically), it is better to write a custom class for it, instead of using the computational graph directly.
 */

template<typename DType>
class SoftMaxLoss {
public:
    Tensor<DType> softmax_probabilities;
    template<bool requires_grad, typename Expr>
    size_t forward(DExpr<Expr>& expr) {
        auto res = requires_grad ? expr.forward().clone() : expr.eval();
        softmax_max_shift(res);
        //std::cout << res.get_shape() << std::endl;
        //DType max = get_max<DType>(res);

        Tensor<DType> exp_shifted_res = exp(no_grad(res)).eval();
        softmax_normalization(exp_shifted_res);
        softmax_probabilities = exp_shifted_res;
        //DType sum_inv = static_cast<DType>(1.0) / get_sum<DType>(exp_shifted_res);
        //softmax_probabilities = (no_grad(exp_shifted_res) * no_grad(sum_inv)).eval();
        return get_softmax_argmax();
    }

    int get_softmax_argmax(){
        DType* ptr = &softmax_probabilities[0];
        return std::max_element(ptr, ptr + softmax_probabilities.get_size()) - ptr;
    }

    template<typename Expr>
    void backward(DExpr<Expr>& expr, const std::vector<size_t>& classes_idx){
        for(size_t b = 0; b < softmax_probabilities.get_shape().get_shape()[0]; ++b){
            // safety check
            size_t class_idx = classes_idx[b];
            assert(class_idx < softmax_probabilities.get_shape().get_shape()[1]);
            softmax_probabilities(b, class_idx) -= static_cast<DType>(1.0);
            
        }
        expr.backward(softmax_probabilities);
    }
};
