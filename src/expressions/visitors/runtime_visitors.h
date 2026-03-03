#pragma once

/**
 * Returns the internal parameters of an expression tree, for which we require a gradient.
 */
template <typename T>
struct GetParametersVisitor {
    const size_t visitor_id;
    GetParametersVisitor(size_t visitor_id) : visitor_id{visitor_id} {};
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
 * Returns the effective tensors of the expression tree. Which are:
 * 1) The leaves of the tree
 * 2) The partial results of nodes that require NEEDS_TEMPORARY_FOR_EVAL
 */
template <typename T, size_t num_tensors>
struct GetTensorHandlesVisitor {
    const size_t visitor_id;
    GetTensorHandlesVisitor(size_t visitor_id) : visitor_id{visitor_id} {};

    template <typename Operator>
    static constexpr bool END_RECURSION = Operator::NEEDS_TEMPORARY_FOR_EVAL;

    DataBuffer<T, num_tensors> res{};
    template <typename Node>
    void operator()(const Node &node) {
        if constexpr (Node::Operator::NEEDS_TEMPORARY_FOR_EVAL) {
            res.push_back_variable(node.get_res());
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

/**
 * Returns the internal parameters of an expression tree, for which we require a gradient.
 */
struct ResetSharedNodesVisitor {
    const size_t visitor_id;
    ResetSharedNodesVisitor(size_t visitor_id) : visitor_id{visitor_id} {};
    /**
     * Decide whether to end the recursion based on compile time information about the Operator
     */
    template <typename Operator>
    static constexpr bool END_RECURSION = false;
    size_t res = 0;
    template <typename Node>
    void operator()(Node &node) {
        if constexpr (requires(Node & n) { n.reset_shared_counters(); }) {
            node.reset_shared_counters();
        }
    }
};