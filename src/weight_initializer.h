#pragma once

#include <vector>
#include "random.h"

/**
 * Initialize the weights in a neural network.
 */
template <typename DType>
inline void he_initialization(const std::vector<Variable<DType, true>> &params) {
    for (const auto &param : params) {
        const auto &tensor = param.tensor;
        const Shape &tensor_shape = tensor.get_shape();

        size_t t_dim = tensor_shape.get_dimension();
        // Scalars and Vectors can be set to zero.
        if (t_dim == 0 || t_dim == 1) {
            tensor.set_zero();
        } else {
            // TODO: not sure what to do for higher dimensional tensors ( > 4). Read some papers
            // about it.
            assert(t_dim <= 4);

            size_t fan_in;
            const auto &t_shape = tensor_shape.get_shape();
            if (t_dim == 2) {
                fan_in = t_shape[0] + t_shape[1];
            } else if (t_dim == 3) {
                fan_in = t_shape[1] * t_shape[2];
            } else if (t_dim == 4) {
                fan_in = t_shape[1] * t_shape[2] * t_shape[3];
            }

            double shape_sum = static_cast<double>(fan_in);
            GaussianGenerator generator{0.0, std::sqrt(4.0 / shape_sum)};

            for (size_t i = 0; i < tensor.get_size(); i++) {
                tensor[i] = generator.generate();
            }
        }
    }
}

/**
 * For some tests we need outputs that are far from zero (becauese the RELU is non differentiable at
 * zero).
 */
template <typename DType>
inline void random_test_initialization(const std::vector<Variable<DType, true>> &params) {
    for (const auto &param : params) {
        const auto &tensor = param.tensor;
        // Generate random numbers with non-zero mean
        GaussianGenerator generator{5.0, 1.0};

        for (size_t i = 0; i < tensor.get_size(); i++) {
            tensor[i] = generator.generate();
        }
    }
}
