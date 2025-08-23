#include "convolution_tests_1d.h"
#include "convolution_tests_2d.h"

#include "nn_tests.h"

void run_tests(){
    convolution_tests_1d();
    convolution_tests_2d();
    nn_tests();
}