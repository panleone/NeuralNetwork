#pragma once

/**
 * Returns the internal parameters of an expression tree, for which we require a gradient.
 */
template <typename T>
struct GetParametersVisitor {
    /**
     * Decide whether to end the recursion based on compile time information about the Operator
     */
    template <typename Operator>
    static constexpr bool END_RECURSION = false;

    std::vector<Variable<T, true>> res{};
    template <typename Node>
    void operator()(const Node &node) {
        if constexpr (std::is_same_v<Node, DExprTensor<T, /*require_gradient*/ true>>) {
            res.push_back(node.t_);
        }
    }
};

/**
 * Returns the internal parameters of an expression tree, for which we require a gradient.
 */
template <typename T, size_t num_tensors>
struct GetTensorHandlesVisitor {
    template <typename Operator>
    static constexpr bool END_RECURSION = Operator::NEEDS_TEMPORARY_FOR_EVAL;

    DataBuffer<T, num_tensors> res{};
    template <typename Node>
    void operator()(const Node &node) {
        if constexpr (Node::Operator::NEEDS_TEMPORARY_FOR_EVAL) {
            res.push_back_variable(node.res);
        }
    }
    void operator()(const DExprTensor<T, true> &node) {
        node.t_.tensor.wrap_for_broadcasting();
        res.push_back_variable(node.t_.tensor);
    }
    void operator()(const DExprTensor<T, false> &node) {
        node.t_.tensor.wrap_for_broadcasting();
        res.push_back_variable(node.t_.tensor);
    }
};