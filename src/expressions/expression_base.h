#pragma once

template <typename T>
class Interpreter;

template <typename Expr>
class DExpr {
  public:
    /**
     * Expression intrinsic type, it can be float or double
     */
    struct IntrinsicType {
        using Type = typename Expr::DType;
    };

    /**
     * Flatten the expression tree in a list structure
     * For example x0 + (x1 + relu(x2)) is flattened to:
     * [ops::VARIABLE_OP, ops::VARIABLE_OP, ops::RELU, ops::SUM_OP, ops::VARIABLE_OP, ops::SUM_OP]
     *
     * This structure can then be used as input to the Interpreter, that will efficiently compute
     * the actual output in a single fused loop.
     *
     * - ops::VARIABLE_OP, will be interpreted as: "push the next variable on the stack" (x0, x1 or
     * x2)
     * - ops::RELU, as: "pop the last pushed variable on the stack and compute it's relu and push
     * the result back on the stack"
     * - ops::SUM_OP, as: "pop the last two pushed variables and compute their sum, and push the
     * result back"
     */
    template <bool recursive>
    struct Flatten {
        using Type = typename Expr::template Flatten<recursive>::Type;
    };

    /**
     * Simplify an expression:
     * for example
     * x0 * x1 + x2
     * is simplified to a single fused operation
     * FMA(x0, x1, x2)
     */
    struct Simplify {
        using Type = typename Expr::Simplify::Type;
    };

    /**
     * Returns a DataBuffer with the actual Tensor in the trees.
     */
    auto collect_tensor_handles() const;

    // Vector of pairs <Tensor, Gradient>
    auto get_parameters() const;

    /**
     * simplify, evaluate the expression and return the result as a Tensor
     */
    auto eval() {
        // TODO: we know at compile time if compute_temporaries_for_eval is needed or not. Add such
        // optimization
        this->compute_temporaries_for_eval();
        return Interpreter<typename Simplify::Type>::interpret(*this);
    }

    /**
     * simplify, evaluate the expression and store the result into res
     */
    void eval(const auto &res) {
        // TODO: we know at compile time if compute_temporaries_for_eval is needed or not. Add such
        // optimization
        this->compute_temporaries_for_eval();
        Interpreter<typename Simplify::Type>::interpret(*this, res);
    }

    auto forward() {
        return static_cast<Expr &>(*this)
            .template compute_temporaries_for_backprop</*use_cache=*/false>();
    }

    /**
     * Backward step and gradients computation
     */
    void backward(auto gradient) { static_cast<Expr &>(*this).backward_internal(gradient); }

  private:
    /**
     * Even If we don't need the gradient there are some operations (matmul) for which we regardless
     * need to compute their intermediate value.
     *
     * For example in the expression
     * x0 * x1 + matmul(x2, x3)
     * we first compute
     * tmp = matmul(x2, x3)
     * and then let the interpreter calculate
     * x0 * x1  + tmp
     * with a single fused AVX loop.
     */
    void compute_temporaries_for_eval() {
        static_cast<Expr &>(*this).compute_temporaries_for_eval();
    }

    /**
     * When we need the gradient, for example in backpropagation,
     * the most efficient action is computing the temporary result for each OperatorNode
     * so for example
     * x0 * x1 + matmul(x2, x3)
     * becomes
     * tmp1 = x0*x1
     * tmp2 = matmul(x2, x3)
     * temp3 = tmp1 + tmp2
     *
     * Those temporaries will then be used for gradient computation.
     */
    template <bool use_cache>
    auto compute_temporaries_for_backprop() {
        return static_cast<Expr &>(*this).template compute_temporaries_for_backprop<use_cache>();
    }

    /**
     * Count the number of leaves in the computational graph
     */
    static consteval size_t get_num_tensors() { return Expr::get_num_tensors(); }

    template <typename Visitor>
    void traverse(Visitor &v) {
        return static_cast<Expr &>(*this).traverse(v);
    }
};