#pragma once

#include "expression_base.h"

#include "../tensor_variable.h"

template <typename T, bool require_gradient>
requires(std::is_same_v<T, double> || std::is_same_v<T, float>) class DExprTensor
    : public DExpr<DExprTensor<T, require_gradient>> {
  public:
    using This = DExprTensor<T, require_gradient>;
    using DType = T;

    Variable<DType, require_gradient> t_;

    // If we pass a raw tensor by convention, it must not require a gradient
    DExprTensor(const Tensor<DType> &t) requires(!require_gradient) : t_{t} {};

    // TODO: should we zero the gradient here? not sure
    DExprTensor(const Variable<DType, require_gradient> &t) : t_{t} {};

    template <bool recursive>
    struct Flatten {
        using Type = Stack<ops::VARIABLE_OP>;
    };

    static consteval size_t get_num_tensors() { return 1; }

    void collect_tensor_handles(auto &current_stack) const {
        // TODO: can we avoid the wrap_for_broadcating?
        t_.tensor.wrap_for_broadcasting();

        current_stack.push_back_variable(t_.tensor);
    }

    struct Simplify {
        using Type = This;
    };

    void compute_temporaries_for_eval() {}
    template <bool use_cache>
    ConstTensor<DType> compute_temporaries_for_backprop() {
        t_.tensor.wrap_for_broadcasting();
        return t_.tensor;
    }

    void backward_internal(const Tensor<DType> &gradient) {
        if constexpr (require_gradient) {
            InterpretInternal<DType, Stack<ops::VARIABLE_OP, ops::VARIABLE_OP, ops::SUM_OP>>::eval(
                make_data_buffer<DType>(t_.gradient, gradient), t_.gradient);
        }
    }

    template <typename Visitor>
    void traverse(Visitor &v) {
        v(*this);
    }
    template <typename Visitor>
    void traverse(Visitor &v) const {
        v(*this);
    }
};
