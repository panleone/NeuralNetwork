#pragma once

#include <cxxabi.h>
#include <typeinfo>
#include <memory>

template<typename T>
std::string type_name() {
    const char* mangled = typeid(T).name();
    int status = 0;
    std::unique_ptr<char, void(*)(void*)> demangled(
        abi::__cxa_demangle(mangled, nullptr, nullptr, &status),
        std::free
    );
    return (status == 0) ? demangled.get() : mangled;
}