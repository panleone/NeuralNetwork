#pragma once

#include "unary_operator.h"

// Partial specialization for the Flatten operator
template <typename A>
class DUnaryExprOp<A, DApFlatten> : public DUnaryExprCommonData<A, DApFlatten>,
                                    public DExpr<DUnaryExprOp<A, DApFlatten>> {
  public:
    using DType = typename A::DType;
    using DUnaryExprCommonData<A, DApFlatten>::traverse;

  private:
    using DUnaryExprCommonData<A, DApFlatten>::a_;
    Shape in_shape{};

  public:
    using Operand = A;
    using Operator = DApFlatten;

    DUnaryExprOp(const A &a) : DUnaryExprCommonData<A, DApFlatten>{a} {}

    template <bool recursive>
    struct Flatten {
        using Type = Stack<ops::VARIABLE_OP>;
    };

    static consteval size_t get_num_tensors() { return 1; }

    struct Simplify {
        using Type = DUnaryExprOp<typename A::Simplify::Type, DApFlatten>;
    };

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