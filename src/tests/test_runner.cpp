#include "convolution_tests_1d.h"
#include "convolution_tests_2d.h"
#include "nn_tests.h"
#include "shared_node_tests.h"
#include <stdexcept>
#include <iostream>

int main() {
    try{
        shared_node_tests();
        convolution_tests_1d();
        convolution_tests_2d();
        nn_tests();
    } catch (const std::runtime_error& e){
        std::cerr << "Unit tests failed: " << e.what() << std::endl;
        return 1;
    }
    std::cout << "All tests passed" << std::endl;
    return 0;
}