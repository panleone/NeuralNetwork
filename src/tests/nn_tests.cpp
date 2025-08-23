
#include "../expressions/expression.h"
#include "../random.h"
#include "../optimizer.h"
#include "../weight_initializer.h"

#include <sstream>

#include "../tensor_variable.h"

static void gradient_flow_tests();

void nn_tests() {
    gradient_flow_tests();
}

/**
 * Idea: In the backpropagation the neural network will output dl/dx, the
 * derivative of the loss function with respect to the input vector. We can
 * check that this derivative is indeed the correct value, by using the
 * approximated formula dl/dx = (l(x + eps) - l(x-eps))/(2*eps) for a small eps.
 * Any value of l(x) can be obtained by simply feeding x as input of the nn.
 * 
 * Furthermore, we can repeat the same operation for any parameters of the model
 * dl/d\theta ~ [l(x; \theta + eps) - l(x; \theta)] / (2 * eps)
 */
static void gradient_flow_tests() {

    size_t non_zero_gradients{0};

    // Create a neural network
    Variable<double, true> l1_m({1, 3});
    Variable<double, true> l1_a({3});

    Variable<double, true> l2_m({3, 8});
    Variable<double, true> l2_a({8});

    Variable<double, true> l3_m({8, 1});
    Variable<double, true> l3_a({1});

    Variable<double, true> x{{1, 1}};
    auto x_expr = to_dexpr(x);


    auto y_tmp = relu(matmul(x_expr, l1_m) + l1_a);
    auto y_tmp_2 = relu(matmul(y_tmp, l2_m) + l2_a);
    auto predicted = matmul(y_tmp_2, l3_m) + l3_a;

    constexpr double eps = 0.0001;
    
    
    Tensor<double> input_grad{{1}};
    input_grad[0] = 1.0;

    auto params = predicted.get_parameters();
    for (int i = 0; i < 100; i++) {
        random_test_initialization(params);

        for(const auto&[_, grad] : params) {
            grad.set_zero();
        }

        for(const auto& [param, grad] : params) {
            for(size_t j = 0; j < param.get_size(); j++){
                double param_cache = param[j];
                param[j] = param_cache + eps;
                double res_pos = predicted.forward()[0];

                param[j] = param_cache - eps;
                double res_neg = predicted.forward()[0];

                double init_grad = grad[j];
                double simulated_grad_increment = (res_pos - res_neg) / (2.0 * eps);

                param[j] = param_cache;
                predicted.forward();

                predicted.backward(input_grad);
                double grad_increment = grad[j] - init_grad;

                constexpr double tiny = 1e-12;
                if (std::abs(grad_increment) < tiny) {
                    if(std::abs(simulated_grad_increment) > 1e-8) {
                        std::ostringstream oss;
                        std::cout << i << std::endl;
                        oss << "[NN_TEST]: non-zero gradient (actual, simulated)=(" << grad_increment << ", " << simulated_grad_increment << ") at iteration " << i;
                        throw std::runtime_error(oss.str());
                    }
                } else {
                    double rel_error = std::abs((simulated_grad_increment - grad_increment) / grad_increment);
                    if(rel_error > 1e-5){
                        std::ostringstream oss;
                        oss << "[NN_TEST]: gradient mismatch (actual, simulated)=(" << grad_increment << ", " << simulated_grad_increment << ") at iteration " << i;
                        throw std::runtime_error(oss.str());
                    }
                    non_zero_gradients += 1;
                }
            }
        }
    }

    if(non_zero_gradients == 0){
        throw std::runtime_error("[NN_TEST]: gradients were all null");
    }
}