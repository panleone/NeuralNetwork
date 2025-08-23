#pragma once

#include "tensor.h"

#include <vector>
#include "tensor_variable.h"

template <typename DType>
class StandardOptimizer {
  private:
    DType alpha;

    // Vector of pairs (parameters, gradient)
    std::vector<std::pair<const Tensor<DType> &, const Tensor<DType> &>> parameters;

  public:
    StandardOptimizer(DType alpha, size_t batch_size, decltype(parameters) &&params)
        : alpha{alpha}, parameters{std::move(params)} {};
    void optimize(size_t batch_size) {
        DType corrected_alpha = alpha / static_cast<DType>(batch_size);
        for (const auto &[param, gradient] : parameters) {
            param -= corrected_alpha * gradient;
            gradient.set_zero();
        }
    }
};

template <typename DType>
class AdamOptimizer {
  private:
    DType alpha;
    DType beta;
    DType one_minus_beta;
    DType gamma;
    DType one_minus_gamma;
    DType epsilon;
    size_t time_stamp{0};

    static constexpr DType one = static_cast<DType>(1.0);
    // Vector of pairs (parameters, gradient)
    std::vector<Variable<DType, true>> parameters;

    std::vector<Tensor<DType>> momentums;
    std::vector<Tensor<DType>> momentums_sq;

  public:
    AdamOptimizer(
        DType alpha, DType beta, DType gamma, DType epsilon, decltype(parameters) &&params)
        : alpha{alpha}, beta{beta}, one_minus_beta{one - beta}, gamma{gamma},
          one_minus_gamma{one - gamma}, epsilon{epsilon}, parameters{std::move(params)} {
        for (const auto &param : this->parameters) {
            const auto &gradient = param.gradient;
            auto momentum = Tensor<DType>{{gradient.get_shape()}};
            momentum.set_zero();
            momentums.push_back(std::move(momentum));

            auto momentum_sq = Tensor<DType>{{gradient.get_shape()}};
            momentum_sq.set_zero();
            momentums_sq.push_back(std::move(momentum_sq));
        }
    };
    void optimize(size_t batch_size) {

        DType batch_size_inv = one / static_cast<DType>(batch_size);
        for (size_t i = 0; i < parameters.size(); i++) {
            const auto &param = parameters[i].tensor;
            const auto &gradient = parameters[i].gradient;
            auto gradient_norm = (no_grad(gradient) * no_grad(batch_size_inv)).eval();

            const auto &momentum = momentums[i];
            const auto &momentum_sq = momentums_sq[i];

            momentum += no_grad(one_minus_beta) * (no_grad(gradient_norm) - no_grad(momentum));
            DType momentum_scale = one - pow(beta, time_stamp + 1);
            DType momentum_scale_inv = one / momentum_scale;
            auto momentum_norm = no_grad(momentum) * no_grad(momentum_scale_inv);

            momentum_sq += no_grad(one_minus_gamma) *
                           (no_grad(gradient_norm) * no_grad(gradient_norm) - no_grad(momentum_sq));
            DType momentum_sq_scale = one - pow(gamma, time_stamp + 1);
            DType momentum_sq_scale_inv = one / momentum_sq_scale;
            auto momentum_sq_norm = no_grad(momentum_sq) * no_grad(momentum_sq_scale_inv);

            param -= no_grad(alpha) * momentum_norm / (sqrt(momentum_sq_norm) + no_grad(epsilon));

            gradient.set_zero();
        }

        time_stamp += 1;
    }
};

template <typename DType>
class MomentumOptimizer {
  private:
    DType alpha;
    DType one_minus_beta;

    static constexpr DType one = static_cast<DType>(1.0);
    // Vector of pairs (parameters, gradient)
    std::vector<std::pair<const Tensor<DType> &, const Tensor<DType> &>> parameters;

    std::vector<Tensor<DType>> momentums;

  public:
    MomentumOptimizer(DType alpha, DType beta, decltype(parameters) &&params)
        : alpha{alpha}, one_minus_beta{one - beta}, parameters{std::move(params)} {
        for (const auto &[param, gradient] : this->parameters) {

            auto momentum = Tensor<DType>{{gradient.get_shape()}};
            momentum.set_zero();

            momentums.push_back(std::move(momentum));
        }
    };
    void optimize(size_t batch_size) {

        DType batch_size_inv = one / static_cast<DType>(batch_size);
        for (size_t i = 0; i < parameters.size(); i++) {
            const auto &[param, gradient] = parameters[i];

            const Tensor<DType> &momentum = momentums[i];

            auto gradient_norm = gradient * batch_size_inv;

            momentum += one_minus_beta * (gradient_norm - momentum);
            param -= alpha * momentum;
            gradient.set_zero();
        }
    }
};