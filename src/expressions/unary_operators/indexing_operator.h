#pragma once

#include "unary_operator.h"

// Partial specialization for the Indexer operator
template <typename A>
class DUnaryExprOp<A, DApIndexer> : public DExpr<DUnaryExprOp<A, DApIndexer>> {
  public:
    using DType = typename A::DType;

  private:
    A a_;
    size_t index;
    // For back propagation
    Tensor<DType> res{1};
    Shape in_shape{};

  public:
    using Operand = A;
    using Operator = DApIndexer;

    DUnaryExprOp(const A &a, size_t index) : a_{a}, index{index} {}

    template <bool recursive>
    struct Flatten {
        using Type = Stack<ops::VARIABLE_OP>;
    };

    static consteval size_t get_num_tensors() { return 1; }
    void collect_tensor_handles(auto &current_stack) const {
        current_stack.push_back_variable(res);
    }

    void get_parameters_internal(auto &res) const { a_.get_parameters_internal(res); }

    struct Simplify {
        using Type = DUnaryExprOp<typename A::Simplify::Type, DApFlatten>;
    };

    void compute_temporaries_for_eval() {
        using SimplifiedT = Simplify::Type;

        a_.compute_temporaries_for_eval();

        res[0] = Interpreter<typename Simplify::Type::Operand>::const_interpret(a_)[index];
        res.wrap_for_broadcasting();
    }

    template <bool use_cache>
    ConstTensor<DType> compute_temporaries_for_backprop() {
        if constexpr (!use_cache) {
            ConstTensor<DType> operand = a_.template compute_temporaries_for_backprop<use_cache>();
            auto tmp = InterpretInternal<DType, typename Flatten<false>::Type>::const_eval(
                make_data_buffer<DType>(operand));
            res[0] = tmp[index];
            res.wrap_for_broadcasting();
            in_shape = tmp.get_shape();
        }
        return res;
    }

    void backward_internal(const Tensor<DType> &grad) {
        Tensor<DType> grad_out{in_shape};
        grad_out.set_zero();
        grad_out[index] = grad[0];
        grad_out.wrap_for_broadcasting();
        a_.backward_internal(grad_out);
    }
};