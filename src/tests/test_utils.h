#pragma once

#include "../tensor.h"

template <typename T>
bool check_tensor_equality(ConstTensor<T> t1, ConstTensor<T> t2, T epsilon) {
    // 1) The shape must be the same
    if (t1.get_shape() != t2.get_shape()) {
        return false;
    }

    // 2) check that the actual data is the same, up to some precision error
    for (size_t i = 0; i < t1.get_size(); ++i) {
        if (abs(t1[i] - t2[i]) > epsilon) {
            return false;
        }
    }

    return true;
}

template <typename T>
bool check_float_equality(T t1, T t2, T epsilon) {
    return abs(t1 - t2) < epsilon;
}

template <typename T>
bool check_integer_equality(T t1, T t2) {
    return t1 == t2;
}