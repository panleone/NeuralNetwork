#pragma once
#include <tuple>

/**
 * Data and structures that are common to all Expressions
 */

template <typename T0, typename... T>
struct ExtractFirst {
    using Type = T0;
};

template <typename Op, typename... ChildNodes>
requires(sizeof...(ChildNodes) > 0) &&
    ((std::is_same_v<typename ExtractFirst<ChildNodes...>::Type::DType,
                     typename ChildNodes::DType>)&&...) class DExprCommonData {
  public:
    static constexpr size_t n_childs = sizeof...(ChildNodes);
    std::tuple<ChildNodes...> child_nodes;

    using Operator = Op;
    using DType = typename ExtractFirst<ChildNodes...>::Type::DType;
    ConstTensor<DType> res{};

    DExprCommonData(const ChildNodes &...childs) : child_nodes{childs...} {}
    template <typename Visitor>
    void traverse(Visitor &v) {
        v(*this);
        if constexpr (!Visitor::template END_RECURSION<Op>) {
            std::apply([&](auto &...nodes) { (nodes.traverse(v), ...); }, child_nodes);
        }
    }
    template <typename Visitor>
    void traverse(Visitor &v) const {
        v(*this);
        if constexpr (!Visitor::template END_RECURSION<Op>) {
            std::apply([&](const auto &...nodes) { (nodes.traverse(v), ...); }, child_nodes);
        }
    }

    template <typename Visitor>
    static consteval auto traverse() {
        using This = DExprCommonData<Op, ChildNodes...>;
        constexpr auto node_res = Visitor::template Visit<This>();
        if constexpr (!Visitor::template END_RECURSION<Op>) {
            return Visitor::template Aggregate(node_res,
                                               ChildNodes::template traverse<Visitor>()...);
        } else {
            return node_res;
        }
    }

    template <bool recursive>
    struct FlattenOpNoTemporary {
        using tmp2 = Stack<Op::STACK_VAL>;
        using Type =
            MergeStacksT<std::conditional_t<recursive,
                                            typename ChildNodes::template Flatten<recursive>::Type,
                                            Stack<ops::VARIABLE_OP>>...,
                         tmp2>;
    };

    template <bool recursive>
    struct Flatten {
        using Type = std::conditional_t<Op::NEEDS_TEMPORARY_FOR_EVAL,
                                        Stack<ops::VARIABLE_OP>,
                                        typename FlattenOpNoTemporary<recursive>::Type>;
    };

    // By convention, we name the first 3 childs as a, b, c
    auto &a_() requires(n_childs >= 1) { return std::get<0>(this->child_nodes); }
    auto &b_() requires(n_childs >= 2) { return std::get<1>(this->child_nodes); }
    auto &c_() requires(n_childs >= 3) { return std::get<2>(this->child_nodes); }
};