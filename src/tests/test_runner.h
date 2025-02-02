#pragma once

#include "layer_tests.h"
#include "matrix_tests.h"
#include "nn_tests.h"

void run_tests() {
  matrix_tests();
  layer_tests();
  nn_tests();
}