CXX=g++
CXXFLAGS = -std=c++23 -g -O2 -Wall -Wextra -fsanitize=undefined,address
# flags for performanc testing
# CXXFLAGS = -std=c++23 -flto -Ofast

OBJ = src/main.o

OBJ_TESTS = src/tests/layer_tests.o src/tests/matrix_tests.o src/tests/nn_tests.o \
			datasets/mnist1d/load_mnist1d.o


HEADERS =  src/matrix.h src/layer.h src/data_loader.h src/neural_network.h src/random.h \
			src/optimizer.h \
			datasets/mnist1d/load_mnist1d.h
HEADERS_TESTS = src/tests/layer_tests.h src/tests/matrix_tests.h src/tests/nn_tests.h \
				src/tests/test_utils.h src/tests/test_runner.h

SRC = src/main.cpp \
	  datasets/mnist1d/load_mnist1d.cpp
SRC_TESTS = src/tests/layer_tests.cpp src/tests/matrix_tests.cpp src/tests/nn_tests.cpp

BIN = NeuralNetwork

$(BIN) : $(OBJ) $(OBJ_TESTS)
	$(CXX) $(CXXFLAGS) $(OBJ) $(OBJ_TESTS) -o $(BIN)

$(OBJ) : $(HEADERS) Makefile
$(OBJ_TESTS) : $(HEADERS_TESTS) Makefile

.PHONY: clean format
clean:
	find . -type f -name '*.o' -exec rm {} +
	-rm "$(BIN)"

format:
	clang-format -i $(SRC) $(HEADERS) $(HEADERS_TESTS) $(SRC_TESTS)