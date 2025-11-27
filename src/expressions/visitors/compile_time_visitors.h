#pragma once

/**
 * Returns the number of effective tensors. See GetTensorHandlesVisitor
 */
template <typename T>
struct GetNumTensorHandlesVisitor {
    template <typename Operator>
    static constexpr bool END_RECURSION = Operator::NEEDS_TEMPORARY_FOR_EVAL;

    template <typename Node>
    requires requires { typename Node::Operator; }
    static consteval size_t Visit() {
        if constexpr (Node::Operator::NEEDS_TEMPORARY_FOR_EVAL) {
            return 1;
        } else {
            return 0;
        }
    }
    template <typename Node>
    static consteval size_t Visit() {
        static_assert(std::is_same_v<Node, DExprTensor<T, true>> ||
                      std::is_same_v<Node, DExprTensor<T, false>>);
        return 1;
    }

    template <typename... I>
    static consteval size_t Aggregate(I... i) {
        return (i + ...);
    }
};
