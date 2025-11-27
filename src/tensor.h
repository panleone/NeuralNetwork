#pragma once

#include <algorithm>
#include <numeric>
#include <cassert>
#include <cstddef>
#include <span>
#include <cstring> // for std::memcpy

#include <iostream>
#include <array>

#include "constants.h"

class Shape {
    static constexpr size_t SHAPE_MAX_DIM = 10;
    size_t dimension;
    std::array<size_t, SHAPE_MAX_DIM> shape;
    // computed only once, and used for tensor indexing
    std::array<size_t, SHAPE_MAX_DIM> cumulative_shape;
    size_t size{1};
    // True if the tensor is a scalar (size = 1)
    bool is_a_scalar;

  public:
    Shape(const std::array<size_t, SHAPE_MAX_DIM> &in_shape, size_t in_dimension)
        : shape{in_shape}, dimension{in_dimension} {
        assert(in_dimension < SHAPE_MAX_DIM);

        for (size_t i = 0; i < in_dimension; ++i) {
            size *= shape[i];
        }
        size_t cum_prod{1};
        for (size_t i = in_dimension; i-- > 0;) {
            cumulative_shape[i] = cum_prod;
            cum_prod *= shape[i];
        }
        assert(size > 0);
        is_a_scalar = (size == 1);
    }

    Shape(std::initializer_list<size_t> in_shape) : dimension{in_shape.size()} {
        assert(dimension < SHAPE_MAX_DIM);
        size_t i = 0;
        for (size_t s : in_shape) {
            shape[i] = s;
            size *= s;
            i += 1;
        }
        size_t cum_prod{1};
        for (size_t i = dimension; i-- > 0;) {
            cumulative_shape[i] = cum_prod;
            cum_prod *= shape[i];
        }
        assert(size > 0);
        is_a_scalar = (size == 1);
    }

    std::span<const size_t> get_shape() const {
        return std::span<const size_t>(&shape[0], dimension);
    }
    std::span<const size_t> get_cumulative_shape() const {
        return std::span<const size_t>(&cumulative_shape[0], dimension);
    }

    size_t get_dimension() const { return dimension; }
    size_t get_size() const { return size; }

    size_t first() const {
        assert(get_dimension() > 0);
        return shape[0];
    }
    size_t last() const {
        assert(get_dimension() > 0);
        return shape[dimension - 1];
    }

    static bool are_broadcastable(const Shape &s1, const Shape &s2) {
        // Everything is broadcstable with a constant
        if (s1.is_a_scalar || s2.is_a_scalar) {
            return true;
        }
        size_t smallest_size = std::min(s1.dimension, s2.dimension);
        for (size_t i = 1; i <= smallest_size; ++i) {
            if (s1.shape[s1.dimension - i] != s2.shape[s2.dimension - i]) {
                return false;
            }
        }
        return true;
    }
    static const Shape &get_broadcasted_shape(const Shape &s1, const Shape &s2) {
        assert(Shape::are_broadcastable(s1, s2));
        return s1.dimension < s2.dimension ? s2 : s1;
    }

    static bool are_compatible(const Shape &s1, const Shape &s2) { return s1.size == s2.size; }

    template <bool transpose_s1, bool transpose_s2>
    static const Shape get_matmul_shape(const Shape &s1, const Shape &s2) {

        assert(s1.dimension > 0);
        assert(s2.dimension > 0);
        size_t s1_common_dimension = transpose_s1 ? s1.first() : s1.last();
        size_t s2_common_dimension = transpose_s2 ? s2.last() : s2.first();

        assert(s1_common_dimension == s2_common_dimension);

        const auto &shape1 = s1.shape;
        const auto &shape2 = s2.shape;

        size_t res_dimension = s1.dimension + s2.dimension - 2;
        assert(res_dimension <= SHAPE_MAX_DIM);

        std::array<size_t, SHAPE_MAX_DIM> res_shape;
        for (size_t i = 0; i < s1.dimension - 1; ++i) {
            res_shape[i] = shape1[transpose_s1 ? i + 1 : i];
        }

        for (size_t i = 0; i < s2.dimension - 1; ++i) {
            res_shape[i + s1.dimension - 1] = shape2[transpose_s2 ? i : i + 1];
        }

        return Shape{res_shape, res_dimension};
    }

    friend std::ostream &operator<<(std::ostream &o, const Shape &shape) {
        o << "( ";
        for (size_t i = 0; i < shape.get_dimension(); i++) {
            o << shape.shape[i] << ", ";
        }
        o << ")\n";
        return o;
    }
    bool operator==(const Shape &s) const {
        if (s.dimension != dimension) {
            return false;
        }
        for (size_t i = 0; i < dimension; ++i) {
            if (s.shape[i] != shape[i]) {
                return false;
            }
        }
        return true;
    }

    bool operator!=(const Shape &s) const { return !(*this == s); }

    const size_t &operator[](size_t idx) const { return shape[idx]; }

    template <typename Stream>
    void serialize(Stream &stream) const {
        stream.write(*this);
    }
    template <typename Stream>
    void deserialize(Stream &stream) {
        stream.read(*this);
    }
};

template <typename T>
class GenericTensorData {
  private:
    size_t data_size;
    size_t *ref_counter{nullptr};
    template <typename>
    friend class GenericTensorData;

  public:
    T *data{nullptr};

    GenericTensorData() = default;
    GenericTensorData(size_t size) : data_size{size} {
        data = new T[size];

        ref_counter = new size_t;
        *ref_counter = 1;
    }

    GenericTensorData(const GenericTensorData &t)
        : data_size{t.data_size}, ref_counter{t.ref_counter}, data{t.data} {
        if (ref_counter) {
            *ref_counter += 1;
        }
    }

    template <typename K>
    requires(std::is_same_v<std::remove_const_t<T>, K>)
        GenericTensorData(const GenericTensorData<K> &t)
        : data_size{t.data_size}, ref_counter{t.ref_counter}, data{t.data} {
        if (ref_counter) {
            *ref_counter += 1;
        }
    }

    GenericTensorData(GenericTensorData &&t) { swap(*this, t); }

    GenericTensorData &operator=(GenericTensorData t) {
        swap(*this, t);
        return *this;
    }

    auto clone() const {
        GenericTensorData<std::remove_const_t<T>> res{data_size};
        std::memcpy(&res.data[0], data, data_size * sizeof(data[0]));

        return res;
    }

    ~GenericTensorData() {
        if (ref_counter) {
            *ref_counter -= 1;
            if (*ref_counter == 0) {
                delete[] data;
                delete ref_counter;
            }
        }
    }

    void set_zero() const { std::fill(data, data + data_size, static_cast<T>(0)); }

    void set_constant(T x) const { std::fill(data, data + data_size, x); }

    friend void swap(GenericTensorData &t1, GenericTensorData &t2) {
        std::swap(t1.data_size, t2.data_size);
        std::swap(t1.data, t2.data);
        std::swap(t1.ref_counter, t2.ref_counter);
    }

    template <typename Stream>
    void serialize(Stream &stream) const {
        stream.write(data_size);
        stream.write(data, data_size);
    }
    template <typename Stream>
    void deserialize(Stream &stream) {
        stream.read(data_size);

        GenericTensorData deser_obj(data_size);
        stream.read(deser_obj.data, data_size);
        *this = deser_obj;
    }
};

template <typename T>
size_t get_avx_wrapped_size(size_t size) {
    constexpr size_t intrinsic_size = avx_constants::intrinsic_size<T>;
    return (size % intrinsic_size == 0) ? size : size + intrinsic_size;
}

template <typename T>
class GenericTensor;

template <typename T>
using Tensor = GenericTensor<T>;
template <typename T>
using ConstTensor = GenericTensor<const T>;

template <typename T>
class GenericTensor {
    Shape shape;
    // Effective size used needed for efficient broadcasting
    size_t avx_wrapped_size;
    GenericTensorData<T> tensor_data;

  public:
    template <typename>
    friend class GenericTensor;

    GenericTensor(Shape in_shape)
        : shape{std::move(in_shape)}, avx_wrapped_size{get_avx_wrapped_size<T>(shape.get_size())},
          tensor_data{avx_wrapped_size} {}

    GenericTensor(std::initializer_list<size_t> in_shape)
        : shape{in_shape}, avx_wrapped_size{get_avx_wrapped_size<T>(shape.get_size())},
          tensor_data{avx_wrapped_size} {}

    GenericTensor() : shape{}, avx_wrapped_size{0}, tensor_data{} {}
    ~GenericTensor() = default;
    GenericTensor(const GenericTensor &t) = default;
    GenericTensor(GenericTensor &&t) = default;

    template <typename K>
    requires(std::is_same_v<std::remove_const_t<T>, K>) GenericTensor(const GenericTensor<K> &t)
        : shape{t.shape}, avx_wrapped_size{t.avx_wrapped_size}, tensor_data{t.tensor_data} {}

    GenericTensor &operator=(GenericTensor t) {
        std::swap(shape, t.shape);
        swap(tensor_data, t.tensor_data);
        std::swap(avx_wrapped_size, t.avx_wrapped_size);

        return *this;
    }

    auto clone() const {
        GenericTensor<std::remove_const_t<T>> res{get_shape()};
        res.tensor_data = tensor_data.clone();
        res.avx_wrapped_size = avx_wrapped_size;
        return res;
    }

    void set_zero() const { tensor_data.set_zero(); }

    void set_constant(T x) const { tensor_data.set_constant(x); }

    T &operator[](size_t idx) { return tensor_data.data[idx]; }
    T &operator[](size_t idx) const { return tensor_data.data[idx]; }

    T &operator()(std::initializer_list<size_t> idxs) {
        assert(shape.get_dimension() == idxs.size());
        bool are_indices_in_bound = std::equal(
            idxs.begin(), idxs.end(), shape.get_shape().begin(), [](size_t idx, size_t shape_idx) {
                return idx < shape_idx;
            });
        assert(are_indices_in_bound);

        return operator[](std::inner_product(
            idxs.begin(), idxs.end(), shape.get_cumulative_shape().begin(), size_t{0}));
    }

    T &operator()(std::initializer_list<size_t> idxs) const {
        assert(shape.get_dimension() == idxs.size());
        bool are_indices_in_bound = std::equal(
            idxs.begin(), idxs.end(), shape.get_shape().begin(), [](size_t idx, size_t shape_idx) {
                return idx < shape_idx;
            });
        assert(are_indices_in_bound);

        return operator[](std::inner_product(
            idxs.begin(), idxs.end(), shape.get_cumulative_shape().begin(), size_t{0}));
    }

    template <typename... Indices>
    T &operator()(Indices... indices) {
        return operator()({static_cast<size_t>(indices)...});
    }

    template <typename... Indices>
    T &operator()(Indices... indices) const {
        return operator()({static_cast<size_t>(indices)...});
    }

    const Shape &get_shape() const { return shape; }

    void set_shape(Shape s2) {
        assert(Shape::are_compatible(shape, s2));
        shape = std::move(s2);
    }

    size_t get_size() const { return shape.get_size(); }

    void wrap_for_broadcasting() const {
        for (size_t i = 0; i < avx_wrapped_size - get_size(); i++) {
            tensor_data.data[get_size() + i] = tensor_data.data[i];
        }
    }
    void assert_ready_for_broadcasting() const {
        for (size_t i = 0; i < avx_wrapped_size - get_size(); i++) {
            assert(tensor_data.data[get_size() + i] == tensor_data.data[i]);
        }
    }

    friend std::ostream &operator<<(std::ostream &o, const GenericTensor<T> &t) {
        o << "(flattened t)[ ";
        for (size_t i = 0; i < t.get_size(); i++) {
            o << t[i] << ", ";
        }
        o << "]\n";
        return o;
    }

    template <typename Stream>
    void serialize(Stream &stream) const {
        shape.serialize(stream);
        tensor_data.serialize(stream);
    }

    template <typename Stream>
    void deserialize(Stream &stream) {
        shape.deserialize(stream);
        tensor_data.deserialize(stream);
        avx_wrapped_size = get_avx_wrapped_size<T>(shape.get_size());
    }
};

/**
 * Tensor wrapper for broadcasting logic
 */
template <typename DType>
class TensorBroadcastableRef {
    size_t t_size;

  public:
    ConstTensor<DType> t_ref;

    TensorBroadcastableRef() = default;
    TensorBroadcastableRef(ConstTensor<DType> tensor)
        : t_ref{std::move(tensor)}, t_size{tensor.get_size()} {
        t_ref.assert_ready_for_broadcasting();
    };
    const DType &operator[](size_t idx) const { return t_ref[idx % t_size]; }
};

template <typename DType, size_t N>
class DataBuffer {
    std::array<TensorBroadcastableRef<DType>, N> expression_variables;
    size_t expression_variables_idx{0};
    size_t push_back_idx{0};

  public:
    DataBuffer() : expression_variables{} {};

    template <typename... T>
    requires(sizeof...(T) == N && (std::constructible_from<TensorBroadcastableRef<DType>, T &&> &&
                                   ...)) DataBuffer(T &&...tensors)
        : expression_variables({TensorBroadcastableRef<DType>(std::forward<T>(tensors))...}){};
    void push_back_variable(const ConstTensor<DType> &variable) {
        expression_variables[push_back_idx++] = TensorBroadcastableRef<DType>(variable);
        assert(push_back_idx <= N);
    }

    const TensorBroadcastableRef<DType> &get_next_variable() {
        return expression_variables[expression_variables_idx++];
    }

    void reset() { expression_variables_idx = 0; }

    // Returns the biggest shape stored in the buffer
    const Shape &get_max_shape() {
        size_t i_arg_max{0};
        size_t max_dim = expression_variables[0].t_ref.get_shape().get_dimension();

        for (size_t i = 1; i < N; ++i) {
            size_t new_dim = expression_variables[i].t_ref.get_shape().get_dimension();
            if (new_dim > max_dim) {
                max_dim = new_dim;
                i_arg_max = i;
            }
        }

        return expression_variables[i_arg_max].t_ref.get_shape();
    }
};

/**
 * Utils to quickly create a data buffer
 */
template <typename DType, typename... T>
auto make_data_buffer(T &&...tensors) -> DataBuffer<DType, sizeof...(tensors)> {
    return DataBuffer<DType, sizeof...(tensors)>(std::forward<T>(tensors)...);
}
