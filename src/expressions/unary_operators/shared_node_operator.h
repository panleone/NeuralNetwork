#pragma once

#include <memory>
#include <iostream>
#include "../expression_base.h"

#include "../../metaprogramming/stack.h"
#include "../../metaprogramming/metaprogramming_utils.h"

#include "../../interpreter.h"

template <typename T, typename U>
requires(std::is_same_v<T, double> || std::is_same_v<T, float>) struct InterpretInternal;

// This is the only class that doesn't inherits from DExprCommonData, as it is a bit special
template <typename A>
class DUnaryExprOp<A, DApShared> : public DExpr<DUnaryExprOp<A, DApShared>> {
  public:
    using DType = typename A::DType;

  private:
    using This = DUnaryExprOp<A, DApShared>;
    const std::shared_ptr<A> a_;
    /**
     * Idea: in the forward pass we just count how many times the node is queried.
     * By symmetry this number will also be the number of times the node will be queried in the
     * backward pass. Therefore, we can accumulate the gradient forward_call_counts times and
     * perform the actual backward pass only when backward_call_counts == forward_call_counts
     */
    std::shared_ptr<size_t> forward_call_counts{std::make_shared<size_t>(0)};
    std::shared_ptr<size_t> backward_call_counts{std::make_shared<size_t>(0)};

    std::shared_ptr<bool> compute_temporaries_for_eval_flag{std::make_shared<bool>(false)};
    std::shared_ptr<size_t> last_visitor_id{std::make_shared<size_t>(0)};

    const std::shared_ptr<ConstTensor<DType>> res{std::make_shared<ConstTensor<DType>>()};
    const std::shared_ptr<Tensor<DType>> accumulated_grad{std::make_shared<Tensor<DType>>()};

  public:
    using Operand = A;
    using Operator = DApShared;

    DUnaryExprOp(const A &a) : a_{std::make_shared<A>(a)} {}
    ConstTensor<DType> get_res() const { return *res; }
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
        if constexpr (!use_cache) {
            *forward_call_counts += 1;
            if (*forward_call_counts > 1) {
                return *res;
            }
            ConstTensor<DType> operand = a_->template compute_temporaries_for_backprop<use_cache>();

            *res = InterpretInternal<DType, typename Flatten<false>::Type>::const_eval(
                make_data_buffer<DType>(operand));
        }
        return *res;
    }

    void backward_internal(const Tensor<DType> &grad) {
        if (*backward_call_counts == 0) {
            *accumulated_grad = grad.clone();
        } else {
            InterpretInternal<DType, Stack<ops::VARIABLE_OP, ops::VARIABLE_OP, ops::SUM_OP>>::eval(
                make_data_buffer<DType>(grad, *accumulated_grad), *accumulated_grad);
        }
        *backward_call_counts += 1;
        if (*backward_call_counts == *forward_call_counts) {
            a_->backward_internal(*accumulated_grad);
            accumulated_grad->set_zero();
        }
    }

    template <typename Visitor>
    void traverse(Visitor &v) {
        if (*last_visitor_id == v.visitor_id) {
            return;
        }
        v(*this);
        assert(v.visitor_id > *last_visitor_id);
        *last_visitor_id = v.visitor_id;

        if constexpr (!Visitor::template END_RECURSION<Operator>) {
            a_->traverse(v);
        }
    }
    template <typename Visitor>
    void traverse(Visitor &v) const {
        if (*last_visitor_id == v.visitor_id) {
            return;
        }
        v(*this);
        assert(v.visitor_id > *last_visitor_id);
        *last_visitor_id = v.visitor_id;

        if constexpr (!Visitor::template END_RECURSION<Operator>) {
            a_->traverse(v);
        }
    }

    template <typename Visitor>
    static consteval auto traverse() {
        constexpr auto node_res = Visitor::template Visit<This>();
        if constexpr (!Visitor::template END_RECURSION<Operator>) {
            // Shared data must be traversed either 1 time or 0.
            // At compile time it has to be 0, since we cannot count the number of visits.
            static_assert(is_always_false_v<Visitor>);
            return Visitor::template Aggregate(node_res, A::template traverse<Visitor>());
        } else {
            return node_res;
        }
    }

    void reset_shared_counters() {
        assert(*forward_call_counts == *backward_call_counts && *forward_call_counts > 0);
        *forward_call_counts = 0;
        *backward_call_counts = 0;
        *compute_temporaries_for_eval_flag = false;
    }
};