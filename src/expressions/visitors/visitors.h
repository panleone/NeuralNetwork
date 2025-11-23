#pragma once

/**
 * Returns the internal parameters of an expression tree, for which we require a gradient.
 */
template <typename T>
struct GetParametersVisitor {
    std::vector<Variable<T, true>> res{};
    template <typename Node>
    void operator()(const Node &node) {
        if constexpr (std::is_same_v<Node, DExprTensor<T, /*require_gradient*/ true>>) {
            res.push_back(node.t_);
        }
    }
};