#pragma once

template <typename T>
struct is_always_false {
    static constexpr bool value = false;
};

template <typename T>
constexpr bool is_always_false_v = is_always_false<T>::value;