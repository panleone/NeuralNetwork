#pragma once

#include <memory>
#include <iostream>
#include "../expression_base.h"

#include "../../metaprogramming/stack.h"
#include "../../interpreter.h"

template <typename T, typename U>
requires(std::is_same_v<T, double> || std::is_same_v<T, float>) struct InterpretInternal;

template <typename A>
class DUnaryExprOp<A, DApShared> : public DExpr<DUnaryExprOp<A, DApShared>> {
  public:
    using DType = typename A::DType;

  private:
    const std::shared_ptr<A> a_;
    // Number of shared nodes in the current computational graph
    std::shared_ptr<size_t> num_nodes{std::make_shared<size_t>(0)};
    std::shared_ptr<bool> get_parameters_internal_flag{std::make_shared<bool>(false)};
    std::shared_ptr<bool> compute_temporaries_for_eval_flag{std::make_shared<bool>(false)};
    std::shared_ptr<bool> compute_temporaries_for_backprop_flag{std::make_shared<bool>(false)};

    const std::shared_ptr<ConstTensor<DType>> res{std::make_shared<ConstTensor<DType>>()};

  public:
    using Operand = A;
    using Operator = DApShared;

    DUnaryExprOp(const A &a) : a_{std::make_shared<A>(a)} {}

    /**
     * Since the node is shared, when we flatten it we want only the result of the operation.
     * For example consider:
     *
     * auto sum = x1 - x2;
     * auto tmp = shared(sum);
     * auto res = tmp * tmp
     *
     *  With this flattened version the interpreter will see
     *  [ops::VARIABLE_OP, ops::VARIABLE_OP, ops::MUL_OP].
     *  ops::VARIABLE_OP in this case will be the result of x1 - x2, which is returned by
     * collect_tensor_handles. (consistently, get_num_tensors returns 1).
     *
     *  Thus we effectively avoid computing x1 - x2 twice.
     */
    template <bool recursive>
    struct Flatten {
        using Type = Stack<ops::VARIABLE_OP>;
    };

    // The tensor here is the (shared) result of the referenced node.
    static consteval size_t get_num_tensors() { return 1; }

    void collect_tensor_handles(auto &current_stack) const {
        current_stack.push_back_variable(*res);
    }

    /**
     * Make sure the shared parameters are returned only once.
     */
    void get_parameters_internal(auto &res) const {
        if (!*get_parameters_internal_flag) {
            a_->get_parameters_internal(res);
            *get_parameters_internal_flag = true;
        }
    }

    struct Simplify {
        using Type = DUnaryExprOp<typename A::Simplify::Type, DApShared>;
    };

    void compute_temporaries_for_eval() {
        if (!*compute_temporaries_for_eval_flag) {
            a_->compute_temporaries_for_eval();
            *res = Interpreter<typename Simplify::Type::Operand>::const_interpret(*a_);
            *compute_temporaries_for_eval_flag = true;
        }
    }

    template <bool use_cache>
    ConstTensor<DType> compute_temporaries_for_backprop() {
        if (*compute_temporaries_for_backprop_flag) {
            return *res;
        }
        if constexpr (!use_cache) {
            ConstTensor<DType> operand = a_->template compute_temporaries_for_backprop<use_cache>();

            *res = InterpretInternal<DType, typename Flatten<false>::Type>::const_eval(
                make_data_buffer<DType>(operand));

            *compute_temporaries_for_backprop_flag = true;
        }
        return *res;
    }

    // TODO: optimize, we should just accumulate grad and call backward_internal only once!
    void backward_internal(const Tensor<DType> &grad) { a_->backward_internal(grad); }
};