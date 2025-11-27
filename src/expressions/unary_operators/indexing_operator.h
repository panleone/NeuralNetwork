#pragma once

#include "unary_operator.h"

// Partial specialization for the Indexer operator
template <typename A>
class DUnaryExprOp<A, DApIndexer> : public DUnaryExprCommonData<A, DApIndexer>,
                                    public DExpr<DUnaryExprOp<A, DApIndexer>> {
  private:
    using CommonData = DUnaryExprCommonData<A, DApIndexer>;
    using CommonData::a_;
    size_t index;
    Shape in_shape{};

  public:
    using CommonData::Operand;
    using CommonData::Operator;
    using CommonData::traverse;
    using typename CommonData::DType;
    using typename CommonData::Simplify;
    template <bool recursive>
    using Flatten = typename CommonData::Flatten<recursive>;

    DUnaryExprOp(const A &a, size_t index) : CommonData{a}, index{index} {}

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