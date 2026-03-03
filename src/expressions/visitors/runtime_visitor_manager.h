#pragma once
#include <cstddef>

// Spawns Visitors with monotonically increasing visitor ids
class RuntimeVisitorManager {
    mutable size_t visitor_id{0};

  public:
    template <typename RuntimeVisitor>
    auto visit(const auto &expr) const {
        RuntimeVisitor visitor{++visitor_id};
        expr.traverse(visitor);
        return visitor.res;
    }
    template <typename RuntimeVisitor>
    auto visit(auto &expr) {
        RuntimeVisitor visitor{++visitor_id};
        expr.traverse(visitor);
        return visitor.res;
    }
};