#pragma once

#include "tensor.h"

/**
 * A Variable is just a useful wrapper that contains a tensor and (if required) its gradient.
 */
template <typename T, bool requires_gradient>
class Variable;

template <typename T>
class Variable<T, /*requires_gradient=*/false> {
  public:
    Tensor<T> tensor;
    Variable(const Tensor<T> &t) : tensor{t} {}
    Variable(std::initializer_list<size_t> shape) : tensor{shape} { tensor.set_zero(); }

    template <typename Stream>
    void serialize(Stream &stream) const {
        tensor.serialize(stream);
    }
    template <typename Stream>
    void deserialize(Stream &stream) {
        tensor.deserialize(stream);
    }
};

template <typename T>
class Variable<T, /*requires_gradient=*/true> {
  public:
    Tensor<T> tensor;
    Tensor<T> gradient;
    Variable(const Tensor<T> &t) : tensor{t}, gradient{t.get_shape()} { gradient.set_zero(); }
    Variable(std::initializer_list<size_t> shape) : tensor{shape}, gradient{shape} {
        tensor.set_zero();
        gradient.set_zero();
    }

    template <typename Stream>
    void serialize(Stream &stream) const {
        tensor.serialize(stream);
        gradient.serialize(stream);
    }
    template <typename Stream>
    void deserialize(Stream &stream) {
        tensor.deserialize(stream);
        gradient.deserialize(stream);
    }
};
