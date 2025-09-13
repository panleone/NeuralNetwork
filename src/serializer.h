#pragma once

#include <iostream>
#include <string_view>
#include <fstream>
#include <type_traits>

/**
 * TODO: this implementation is endianess dependent.
 * In general works only if serialize and deserialize are done on the same machine
 */

template <typename T>
concept TriviallySerializable = (std::is_trivially_copyable_v<T> && !std::is_pointer_v<T>);

class Serializer {
    std::ofstream stream;

  public:
    explicit Serializer(std::string_view path) : stream{path.data(), std::ios::binary} {};
    template <typename T>
    requires(TriviallySerializable<T>) void write(const T &data) {
        stream.write(reinterpret_cast<const char *>(&data), sizeof(T));
    }

    template <typename T>
    requires(TriviallySerializable<T>) void write(const T *ptr, size_t n) {
        stream.write(reinterpret_cast<const char *>(&ptr[0]), n * sizeof(T));
    }
};

class Deserializer {
    std::ifstream stream;

  public:
    explicit Deserializer(std::string_view path) : stream{path.data(), std::ios::binary} {};
    template <typename T>
    requires(TriviallySerializable<T>) void read(T &data) {
        stream.read(reinterpret_cast<char *>(&data), sizeof(T));
    }

    template <typename T>
    requires(TriviallySerializable<T>) void read(T *ptr, size_t n) {
        stream.read(reinterpret_cast<char *>(&ptr[0]), n * sizeof(T));
    }
};