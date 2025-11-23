#pragma once

#include "unary_operator.h"

// Partial specialization for the Indexer operator
template <typename A>
class DUnaryExprOp<A, DApIndexer> : public DUnaryExprCommonData<A, DApIndexer>,
                                    public DExpr<DUnaryExprOp<A, DApIndexer>> {
  public:
    using DType = typename A::DType;
    using DUnaryExprCommonData<A, DApIndexer>::traverse;

  private:
    using DUnaryExprCommonData<A, DApIndexer>::a_;
    size_t index;
    Shape in_shape{};

  public:
    using Operand = A;
    using Operator = DApIndexer;

    DUnaryExprOp(const A &a, size_t index) : DUnaryExprCommonData<A, DApIndexer>{a}, index{index} {}

    template <bool recursive>
    struct Flatten {
        using Type = Stack<ops::VARIABLE_OP>;
    };

    static consteval size_t get_num_tensors() { return 1; }

    struct Simplify {
        using Type = DUnaryExprOp<typename A::Simplify::Type, DApIndexer>;
    };

    ConstTensor<DType> extract_index(ConstTensor<DType> t) {
        Tensor<DType> t_res{{1}};
        t_res[0] = t[index];
        t_res.wrap_for_broadcasting();
        return t_res;
    }
    void compute_temporaries_for_eval() {
        using SimplifiedT = Simplify::Type;

        a_.compute_temporaries_for_eval();
        this->res =
            extract_index(Interpreter<typename Simplify::Type::Operand>::const_interpret(a_));
    }

    template <bool use_cache>
    ConstTensor<DType> compute_temporaries_for_backprop() {
        if constexpr (!use_cache) {
            ConstTensor<DType> operand = a_.template compute_temporaries_for_backprop<use_cache>();
            auto tmp = InterpretInternal<DType, typename Flatten<false>::Type>::const_eval(
                make_data_buffer<DType>(operand));
            this->res = extract_index(tmp);
            in_shape = tmp.get_shape();
        }
        return this->res;
    }

    void backward_internal(const Tensor<DType> &grad) {
        Tensor<DType> grad_out{in_shape};
        grad_out.set_zero();
        grad_out[index] = grad[0];
        grad_out.wrap_for_broadcasting();
        a_.backward_internal(grad_out);
    }
};