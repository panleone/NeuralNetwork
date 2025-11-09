#pragma once

#include "../expressions/expression.h"
#include "../tensor_variable.h"

template <typename DType>
class ConvolutionLayer1D {
    // kernel matrix
    Variable<DType, true> k;
    // bias vector
    Variable<DType, true> q;

    size_t stride;

  public:
    ConvolutionLayer1D(size_t in_channels, size_t out_channels, size_t kernel_size, size_t stride)
        : k{{out_channels, in_channels, kernel_size}}, q{{out_channels}}, stride{stride} {}

    template <typename Expr>
    auto forward(const Expr &x) {
        return conv_1d(k, x, q).set_stride(stride);
    }

    template <typename Stream>
    void serialize(Stream &stream) const {
        k.serialize(stream);
        q.serialize(stream);
        stream.write(stride);
    }
    template <typename Stream>
    void deserialize(Stream &stream) {
        k.deserialize(stream);
        q.deserialize(stream);
        stream.read(stride);
    }
};

template <typename DType>
class ConvolutionLayer2D {
    // kernel matrix
    Variable<DType, true> k;
    // bias vector
    Variable<DType, true> q;

    size_t stride_x;
    size_t stride_y;

  public:
    ConvolutionLayer2D(size_t in_channels,
                       size_t out_channels,
                       size_t kernel_size_x,
                       size_t kernel_size_y,
                       size_t stride_x,
                       size_t stride_y)
        : k{{out_channels, in_channels, kernel_size_x, kernel_size_y}}, q{{out_channels}},
          stride_x{stride_x}, stride_y{stride_y} {}

    template <typename Expr>
    auto forward(const Expr &x) {
        return conv_2d(k, x, q).set_stride(stride_x, stride_y);
    }

    template <typename Stream>
    void serialize(Stream &stream) const {
        k.serialize(stream);
        q.serialize(stream);
        stream.write(stride_x);
        stream.write(stride_y);
    }
    template <typename Stream>
    void deserialize(Stream &stream) {
        k.deserialize(stream);
        q.deserialize(stream);
        stream.read(stride_x);
        stream.read(stride_y);
    }
};