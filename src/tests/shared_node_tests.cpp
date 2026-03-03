#include "shared_node_tests.h"
#include "test_utils.h"
#include "../expressions/expression.h"
#include "../optimizer.h"
#include "../random.h"

#include <sstream>

static void nested_shared_network_tests();
/**
 * as a very stupid test, let's create the function
 * y = (relu(dot(x1,x2)) + relu(dot(x1,x2))) * (relu(dot(x1,x2)) + relu(dot(x1,x2)))
 *
 * We test that the smart approach with shared nodes gives the same results as standard
 * approach
 *
 * shared_node1
 */
void shared_node_tests() { nested_shared_network_tests(); }

static void nested_shared_network_tests() {

    constexpr size_t vec_len = 2;
    constexpr double eps_threshold = 1e-4;
    constexpr size_t test_runs = 1000;

    // Basic network, without shared logic and duplication of work
    Variable<double, true> x1_b{{1, vec_len}};
    Variable<double, true> x2_b{{vec_len, 1}};
    auto x1_b_expr = to_dexpr(x1_b);
    auto x2_b_expr = to_dexpr(x2_b);
    auto expr_basic = (relu(matmul(x1_b_expr, x2_b_expr)) + relu(matmul(x1_b_expr, x2_b_expr))) *
                      (relu(matmul(x1_b_expr, x2_b_expr)) + relu(matmul(x1_b_expr, x2_b_expr)));

    // Smart shared network
    Variable<double, true> x1_s{{1, vec_len}};
    Variable<double, true> x2_s{{vec_len, 1}};
    auto x1_s_expr = to_dexpr(x1_s);
    auto x2_s_expr = to_dexpr(x2_s);

    auto tmp1 = shared(relu(matmul(x1_s_expr, x2_s_expr)));
    auto tmp2 = shared(tmp1 + tmp1);
    auto expr_shared = tmp2 * tmp2;

    // 1) Attach an optimizer to the parameters of the network
    auto params_basic = expr_basic.get_parameters();
    auto params_shared = expr_shared.get_parameters();

    if (!check_integer_equality(params_basic.size(), 8ul)) {
        std::ostringstream oss;
        oss << "[SHARED_NODES_TEST]: number of parameters mismatch (actual, simulated)=(" << 8
            << ", " << params_basic.size() << ")";
        throw std::runtime_error(oss.str());
    }

    if (!check_integer_equality(params_shared.size(), 2ul)) {
        std::ostringstream oss;
        oss << "[SHARED_NODES_TEST]: number of parameters mismatch (actual, simulated)=(" << 2
            << ", " << params_shared.size() << ")";
        throw std::runtime_error(oss.str());
    }

    auto optimizer_basic = AdamOptimizer<double>(0.01, 0.9, 0.999, 1.0e-6, std::move(params_basic));
    auto optimizer_shared =
        AdamOptimizer<double>(0.01, 0.9, 0.999, 1.0e-6, std::move(params_shared));

    Tensor<double> grad{{1, 1}};
    grad[0] = 1.0;
    grad.wrap_for_broadcasting();
    for (size_t i = 0; i < test_runs; ++i) {
        // 2) Randomly initialize the tensors
        for (size_t j = 0; j < vec_len; ++j) {
            double x = random_number<double>(1.0, 3.0);
            x1_b.tensor[j] = x;
            x1_s.tensor[j] = x;

            x = random_number<double>(1.0, 3.0);
            x2_b.tensor[j] = x;
            x2_s.tensor[j] = x;
        }
        x1_b.tensor.wrap_for_broadcasting();
        x1_s.tensor.wrap_for_broadcasting();
        x2_b.tensor.wrap_for_broadcasting();
        x2_s.tensor.wrap_for_broadcasting();

        // 3) Check that the forward pass give the same result
        ConstTensor<double> expr_basic_res = expr_basic.forward();
        ConstTensor<double> expr_shared_res = expr_shared.forward();

        if (!check_tensor_equality<double>(expr_basic_res, expr_shared_res, eps_threshold)) {
            std::ostringstream oss;
            oss << "[SHARED_NODES_TEST]: forward pass mismatch (basic, shared)=(" << expr_basic_res
                << ", " << expr_shared_res << ")";
            throw std::runtime_error(oss.str());
        }

        // 4) Check that the backward pass give the same result
        expr_basic.backward(grad);
        expr_shared.backward(grad);
        if (!check_tensor_equality<double>(x1_b.gradient, x1_s.gradient, eps_threshold)) {
            std::ostringstream oss;
            oss << "[SHARED_NODES_TEST]: backward pass mismatch on x1 (basic, shared)=("
                << x1_b.gradient << ", " << x1_s.gradient << ")";
            throw std::runtime_error(oss.str());
        }
        if (!check_tensor_equality<double>(x2_b.gradient, x2_s.gradient, eps_threshold)) {
            std::ostringstream oss;
            oss << "[SHARED_NODES_TEST]: backward pass mismatch on x2 (basic, shared)=("
                << x2_b.gradient << ", " << x2_s.gradient << ")";
            throw std::runtime_error(oss.str());
        }

        // 5) Check that the optimizer modify the two paramters in the same way
        optimizer_basic.optimize(1);
        optimizer_shared.optimize(1);

        if (!check_tensor_equality<double>(x1_b.tensor, x1_s.tensor, eps_threshold)) {
            std::ostringstream oss;
            oss << "[SHARED_NODES_TEST]: optimizer pass mismatch on x1 (basic, shared)=("
                << x1_b.tensor << ", " << x1_s.tensor << ")";
            throw std::runtime_error(oss.str());
        }
        if (!check_tensor_equality<double>(x2_b.tensor, x2_s.tensor, eps_threshold)) {
            std::ostringstream oss;
            oss << "[SHARED_NODES_TEST]: optimizer pass mismatch on x2 (basic, shared)=("
                << x2_b.tensor << ", " << x2_s.tensor << ")";
            throw std::runtime_error(oss.str());
        }
    }
}