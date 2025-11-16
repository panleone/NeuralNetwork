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
    size_t padding;

  public:
    ConvolutionLayer1D(
        size_t in_channels, size_t out_channels, size_t kernel_size, size_t stride, size_t padding)
        : k{{out_channels, in_channels, kernel_size}}, q{{out_channels}}, stride{stride},
          padding{padding} {}

    template <typename Expr>
    auto forward(const Expr &x) {
        return conv_1d(k, x, q).set_stride(stride).set_padding(padding);
    }

    template <typename Stream>
    void serialize(Stream &stream) const {
        k.serialize(stream);
        q.serialize(stream);
        stream.write(stride);
        stream.write(padding);
    }
    template <typename Stream>
    void deserialize(Stream &stream) {
        k.deserialize(stream);
        q.deserialize(stream);
        stream.read(stride);
        stream.read(padding);
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

    size_t padding_x;
    size_t padding_y;

  public:
    ConvolutionLayer2D(size_t in_channels,
                       size_t out_channels,
                       size_t kernel_size_x,
                       size_t kernel_size_y,
                       size_t stride_x,
                       size_t stride_y,
                       size_t padding_x,
                       size_t padding_y)
        : k{{out_channels, in_channels, kernel_size_x, kernel_size_y}}, q{{out_channels}},
          stride_x{stride_x}, stride_y{stride_y}, padding_x{padding_x}, padding_y{padding_y} {}

    template <typename Expr>
    auto forward(const Expr &x) {
        return conv_2d(k, x, q).set_stride(stride_x, stride_y).set_padding(padding_x, padding_y);
    }

    template <typename Stream>
    void serialize(Stream &stream) const {
        k.serialize(stream);
        q.serialize(stream);
        stream.write(stride_x);
        stream.write(stride_y);
        stream.write(padding_x);
        stream.write(padding_y);
    }
    template <typename Stream>
    void deserialize(Stream &stream) {
        k.deserialize(stream);
        q.deserialize(stream);
        stream.read(stride_x);
        stream.read(stride_y);
        stream.read(padding_x);
        stream.read(padding_y);
    }
};