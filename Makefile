CXX=g++
CXXFLAGS = -std=c++20 -g -O2 -Wall -Wextra -g -fsanitize=undefined,address

OBJ = src/main.o

OBJ_TESTS = src/tests/layer_tests.o src/tests/matrix_tests.o src/tests/nn_tests.o


HEADERS =  src/matrix.h src/layer.h src/data_loader.h src/neural_network.h src/random.h
HEADERS_TESTS = src/tests/layer_tests.h src/tests/matrix_tests.h src/tests/nn_tests.h \
				src/tests/test_utils.h src/tests/test_runner.h

SRC = src/main.cpp
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