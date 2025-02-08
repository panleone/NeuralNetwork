#include "nn_tests.h"

#include <memory>

#include "test_utils.h"

#include "../layer.h"
#include "../neural_network.h"
#include "../random.h"

static void gradient_flow_tests();

void nn_tests() { gradient_flow_tests(); }

/**
 * Idea: In the backpropagation the neural network will output dl/dx, the
 * derivative of the loss function with respect to the input vector. We can
 * check that this derivative is indeed the correct value, by using the
 * approximated formula dl/dx = (l(x + eps) - l(x-eps))/(2*eps) for a small eps.
 * Any value of l(x) can be obtained by simply feeding x as input of the nn.
 */
static void gradient_flow_tests() {
  // Create a neural network
  NeuralNetwork<double, double, LOSS::MSE> model{
      std::make_unique<FullyConnectedLayer<double>>(1, 3),
      std::make_unique<ReluLayer<double>>(3),
      std::make_unique<FullyConnectedLayer<double>>(3, 8),
      std::make_unique<ReluLayer<double>>(8),
      std::make_unique<FullyConnectedLayer<double>>(8, 1)};

  // Small number to compute approximated derivative
  double eps = 0.0001;

  // We will test the function
  // y = 5*x*x + 2*x + 1
  for (int i = 0; i < 100; i++) {
    double x = randomNumber(0.0, 1.0);
    double y = 5 * x * x + 2 * x + 1;

    model.forwardOne(Vector<double>{x}, Vector<double>{y});
    // this must be the derivative of the loss function with respect to x
    double gradient = model.getGradient();

    // Compute l(x+eps)
    double resP = model.predict(Vector<double>{x + eps})(0);
    resP = (resP - y) * (resP - y);

    // Compute l(x-eps)
    double resM = model.predict(Vector<double>{x - eps})(0);
    resM = (resM - y) * (resM - y);

    check_num_equality(gradient, (resP - resM) / (2 * eps), eps,
                       "Backpropagation, gradient error");

    model.backward(0.01, 0.0);
  }
}