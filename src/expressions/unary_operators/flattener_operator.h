#pragma once

#include "unary_operator.h"

// Partial specialization for the Flatten operator
template <typename A>
class DUnaryExprOp<A, DApFlatten> : public DUnaryExprCommonData<A, DApFlatten>,
                                    public DExpr<DUnaryExprOp<A, DApFlatten>> {
  private:
    using CommonData = DUnaryExprCommonData<A, DApFlatten>;
    using CommonData::a_;
    Shape in_shape{};

  public:
    using CommonData::Operand;
    using CommonData::Operator;
    using CommonData::traverse;
    using typename CommonData::DType;
    using typename CommonData::Simplify;
    template <bool recursive>
    using Flatten = typename CommonData::Flatten<recursive>;

    DUnaryExprOp(const A &a) : CommonData{a} {}

    void compute_temporaries_for_eval() {
        using SimplifiedT = Simplify::Type;

        a_.compute_temporaries_for_eval();

        this->res = Interpreter<typename Simplify::Type::Operand>::const_interpret(a_);

        assert(this->res.get_shape().get_dimension() >= 2);

        size_t batch_size = this->res.get_shape().get_shape()[0];
        Shape res_shape{{batch_size, this->res.get_shape().get_size() / batch_size}};
        this->res.set_shape(res_shape);
    }

    template <bool use_cache>
    ConstTensor<DType> compute_temporaries_for_backprop() {
        if constexpr (!use_cache) {
            ConstTensor<DType> operand = a_.template compute_temporaries_for_backprop<use_cache>();

            this->res = InterpretInternal<DType, typename Flatten<false>::Type>::const_eval(
                make_data_buffer<DType>(operand));
            // We are assuming that we flatten a shape of the form [batch_size, (other_stuff ,)...]
            in_shape = this->res.get_shape();
            assert(in_shape.get_dimension() >= 2);
            size_t batch_size = in_shape.get_shape()[0];
            Shape res_shape{{batch_size, in_shape.get_size() / batch_size}};
            this->res.set_shape(res_shape);
        }
        return this->res;
    }

    void backward_internal(const Tensor<DType> &grad) {
        Tensor<DType> a_grad = grad.clone();

        a_grad.set_shape(in_shape);

        a_.backward_internal(a_grad);
    }
};