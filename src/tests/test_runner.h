#pragma once

#include "matrix_tests.h"
#include "nn_tests.h"
#include "layer_tests.h"

void run_tests(){
    matrix_tests();
    layer_tests();
    nn_tests();
}