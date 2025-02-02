#include <functional>
#include <iostream>

#include "layer_tests.h"
#include "test_utils.h"

#include "../layer.h"
#include "../random.h"

static void relu_layer_tests();
static void fully_connected_layer_tests();
void layer_tests() {
  relu_layer_tests();
  fully_connected_layer_tests();
}

static void relu_layer_tests() {
  // 10**-4 float comparison precision
  float epsilon = 0.0001;
  ReluLayer<float> layer(5);

  // During the forward step the negative part of the input is zeroed, the
  // positive part remains invariant
  Vector<float> forward_step_input{1.0f, -101.0f, 74.0f, 3.2f, -6.1f};
  Vector<float> expected_forward_step_output{1.0f, 0.0f, 74.0f, 3.2f, 0.0f};
  layer.forwardStep(forward_step_input);
  check_vector_equality(forward_step_input, expected_forward_step_output,
                        epsilon, "ReluLayer forward step");

  // During backward step some elements of the gradient are zeroed
  Vector<float> gradient_input{7.0f, 2.5f, -3.0f, 1.0f, -3.0f};
  Vector<float> expected_gradient_output{7.0f, 0.0f, -3.0f, 1.0f, 0.0f};
  layer.backwardStep(gradient_input);
  check_vector_equality(gradient_input, expected_gradient_output, epsilon,
                        "ReluLayer backward step");

  Vector<float> badSizeVec{1.0f};
  should_throw<std::runtime_error>([&]() { layer.forwardStep(badSizeVec); },
                                   "ReLuLayer forward step");
  should_throw<std::runtime_error>([&]() { layer.backwardStep(badSizeVec); },
                                   "ReLuLayer backward step");

  // previous inputs of the forward_step are forgotten
  for (int i = 0; i < 50; i++) {
    forward_step_input = randomVector(5, 0.0f, 1.0f);
    layer.forwardStep(forward_step_input);
  }

  forward_step_input = Vector<float>{-1.0f, 17.0f, -2.0f, -5.4f, 7.0f};
  layer.forwardStep(forward_step_input);
  check_vector_equality(forward_step_input,
                        Vector<float>{0.0f, 17.0f, 0.0f, 0.0f, 7.0f}, epsilon,
                        "ReluLayer backward step");

  gradient_input = Vector<float>{7.0f, 2.5f, -3.0f, 1.0f, -3.0f};
  layer.backwardStep(gradient_input);
  check_vector_equality(gradient_input,
                        Vector<float>{0.0f, 2.5f, 0.0f, 0.0f, -3.0f}, epsilon,
                        "ReluLayer backward step");

  // Calling backwardStep twice use the same previous input
  gradient_input = Vector<float>{1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  layer.backwardStep(gradient_input);
  check_vector_equality(gradient_input,
                        Vector<float>{0.0f, 1.0f, 0.0f, 0.0f, 1.0f}, epsilon,
                        "ReluLayer backward step");
}

static void fully_connected_layer_tests() {
  // 10**-4 float comparison precision
  float epsilon = 0.0001;
  FullyConnectedLayer<float> layer(2, 1, true);

  auto initWeight = layer.getWeights().clone();
  Vector<float> initBias = layer.getBias().clone();

  // Correctly forward the input / backward the gradient and cache the
  // derivatives.
  Vector<float> input{1.0f, 2.0f};
  layer.forwardStep(input);
  check_vector_equality(input, Vector<float>{3.0f}, epsilon,
                        "Fully connected forward step");
  check_matrix_equality(initWeight, layer.getWeights(), epsilon,
                        "Fully connected forward step");
  check_vector_equality(initBias, layer.getBias(), epsilon,
                        "Fully connected forward step");
  check_matrix_equality(Matrix<float>{{0.0f, 0.0f}}, layer.getWeightsGradient(),
                        epsilon, "Fully connected forward step");
  check_vector_equality(Vector<float>{0.0f}, layer.getBiasGradient(), epsilon,
                        "Fully connected forward step");

  Vector<float> gradient{-0.5f};
  layer.backwardStep(gradient);
  check_vector_equality(gradient, Vector<float>{-0.5f, -0.5f}, epsilon,
                        "Fully connected backward step");
  check_matrix_equality(initWeight, layer.getWeights(), epsilon,
                        "Fully connected backward step");
  check_vector_equality(initBias, layer.getBias(), epsilon,
                        "Fully connected backward step");
  check_matrix_equality(Matrix<float>{{-0.5f, -1.0f}},
                        layer.getWeightsGradient(), epsilon,
                        "Fully connected backward step");
  check_vector_equality(Vector<float>{-0.5f}, layer.getBiasGradient(), epsilon,
                        "Fully connected backward step");

  // Gradient is additive
  input = Vector<float>{0.0f, 7.0f};
  layer.forwardStep(input);
  check_vector_equality(input, Vector<float>{7.0f}, epsilon,
                        "Fully connected forward step");
  check_matrix_equality(initWeight, layer.getWeights(), epsilon,
                        "Fully connected forward step");
  check_vector_equality(initBias, layer.getBias(), epsilon,
                        "Fully connected forward step");
  check_matrix_equality(Matrix<float>{{-0.5f, -1.0f}},
                        layer.getWeightsGradient(), epsilon,
                        "Fully connected forward step");
  check_vector_equality(Vector<float>{-0.5f}, layer.getBiasGradient(), epsilon,
                        "Fully connected forward step");

  gradient = Vector<float>{1.0f};
  layer.backwardStep(gradient);
  check_vector_equality(gradient, Vector<float>{1.0f, 1.0f}, epsilon,
                        "Fully connected backward step");
  check_matrix_equality(initWeight, layer.getWeights(), epsilon,
                        "Fully connected backward step");
  check_vector_equality(initBias, layer.getBias(), epsilon,
                        "Fully connected backward step");
  check_matrix_equality(Matrix<float>{{-0.5f, -1.0f + 7.0f}},
                        layer.getWeightsGradient(), epsilon,
                        "Fully connected backward step");
  check_vector_equality(Vector<float>{-0.5f + 1.0f}, layer.getBiasGradient(),
                        epsilon, "Fully connected backward step");

  // Correct finalization
  float alpha = 0.1f;
  Matrix<float> accumulatedWeightsGradient = layer.getWeightsGradient().clone();
  Vector<float> accumulatedBiasGradient = layer.getBiasGradient().clone();
  layer.finalize(alpha, 2);
  check_matrix_equality(Matrix<float>{{0.0f, 0.0f}}, layer.getWeightsGradient(),
                        epsilon, "Fully connected finalize");
  check_vector_equality(Vector<float>{0.0f}, layer.getBiasGradient(), epsilon,
                        "Fully connected finalize");
  check_matrix_equality(
      initWeight - (alpha / 2.0f) * accumulatedWeightsGradient,
      layer.getWeights(), epsilon, "Fully connected finalize");

  initBias -= (alpha / 2.0f) * accumulatedBiasGradient;
  check_vector_equality(initBias, layer.getBias(), epsilon,
                        "Fully connected finalize");
}