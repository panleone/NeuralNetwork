CXX=g++
# CXXFLAGS = -std=c++23 -g -O2 -Wall -Wextra -fsanitize=undefined,address -march=native
# flags for performance testing
CXXFLAGS = -std=c++23 -Ofast -march=native -ffast-math -flto -DNDEBUG
LDFLAGS= -lopenblas
OBJ = src/main.o datasets/mnist1d/load_mnist1d.o

OBJ_TESTS = src/tests/convolution_tests_1d.o src/tests/convolution_tests_2d.o src/tests/nn_tests.o src/tests/test_utils.o src/tests/test_runner.o \

HEADERS =  src/blas_wrapper.h src/constants.h src/data_loader.h src/debug_utils.h src/interpreter.h src/loss.h src/random.h \
		   src/optimizer.h src/tensor.h src/tensor_variable.h src/weight_initializer.h src/serializer.h \
		   datasets/mnist1d/load_mnist1d.h \
		   src/avx/avx_ops.h src/avx/avx_wrapper.h \
		   src/layers/convolution_layer.h src/layers/flattener_layer.h src/layers/linear_layer.h src/layers/relu_layer.h \
		   src/metaprogramming/stack.h \
		   src/expressions/expression.h src/expressions/expression_base.h src/expressions/expression_base_impl.h src/expressions/operations.h src/expressions/variable.h \
		   src/expressions/unary_operators/flattener_operator.h src/expressions/unary_operators/unary_operator.h src/expressions/unary_operators/indexing_operator.h \
		   src/expressions/binary_operators/binary_operator.h src/expressions/binary_operators/binary_operator_simplifier.h src/expressions/binary_operators/common_simplifier.h \
		   src/expressions/binary_operators/matmul_operator.h src/expressions/binary_operators/matmul_simplifier.h \
		   src/expressions/ternary_operators/ternary_operator.h src/expressions/ternary_operators/convolution_1d_operator.h src/expressions/ternary_operators/convolution_2d_operator.h

HEADERS_TESTS = src/tests/convolution_tests_1d.h src/tests/convolution_tests_2d.h src/tests/nn_tests.h src/tests/test_runner.h src/tests/test_utils.h \

SRC = src/main.cpp \
	  datasets/mnist1d/load_mnist1d.cpp
SRC_TESTS = src/tests/convolution_tests_1d.cpp src/tests/convolution_tests_2d.cpp src/tests/nn_tests.cpp src/tests/test_utils.cpp src/tests/test_runner.cpp \

BIN = NeuralNetwork

$(BIN) : $(OBJ) $(OBJ_TESTS)
	$(CXX) $(CXXFLAGS)  $(OBJ) $(OBJ_TESTS) $(LDFLAGS) -o $(BIN)

$(OBJ) : $(HEADERS) Makefile
$(OBJ_TESTS) : $(HEADERS_TESTS) Makefile

.PHONY: clean format
clean:
	find . -type f -name '*.o' -exec rm {} +
	-rm "$(BIN)"

format:
	clang-format -i $(SRC) $(HEADERS) $(HEADERS_TESTS) $(SRC_TESTS)