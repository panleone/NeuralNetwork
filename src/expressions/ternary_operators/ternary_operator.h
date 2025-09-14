#pragma once

#include "../expression_base.h"

#include "../../metaprogramming/stack.h"

template <typename A, typename B, typename C, typename Op>
requires(std::is_same_v<typename A::DType, typename B::DType>
             &&std::is_same_v<typename B::DType, typename C::DType>) class DTernExprOp {
  private:
    A a_;
    B b_;
    C c_;

    using DType = typename A::DType;

  public:
    using Operator = Op;
    DTernExprOp(const A &a, const B &b, const C &c) : a_{a}, b_{b}, c_{c} {}

    template <bool recursive>
    struct Flatten {
        using tmp1 = std::conditional_t<recursive,
                                        typename A::template Flatten<true>::Type,
                                        Stack<ops::VARIABLE_OP>>;
        using tmp2 = std::conditional_t<recursive,
                                        typename B::template Flatten<true>::Type,
                                        Stack<ops::VARIABLE_OP>>;
        using tmp3 = std::conditional_t<recursive,
                                        typename C::template Flatten<true>::Type,
                                        Stack<ops::VARIABLE_OP>>;

        using tmp4 = Stack<Op::STACK_VAL>;
        using Type = MergeStacksT<MergeStacksT<MergeStacksT<tmp1, tmp2>, tmp3>, tmp4>;
    };

    static consteval size_t get_num_tensors() {
        return A::get_num_tensors() + B::get_num_tensors() + C::get_num_tensors();
    }

    void collect_tensor_handles(auto &current_stack) const {
        a_.collect_tensor_handles(current_stack);
        b_.collect_tensor_handles(current_stack);
        c_.collect_tensor_handles(current_stack);
    };

    void get_parameters_internal(
        std::vector<std::pair<const Tensor<DType> &, const Tensor<DType> &>> &res) const {
        a_.get_parameters_internal(res);
        b_.get_parameters_internal(res);
        c_.get_parameters_internal(res);
    }

    void compute_temporaries_for_eval() {
        a_.compute_temporaries_for_eval();
        b_.compute_temporaries_for_eval();
        c_.compute_temporaries_for_eval();
    }
};