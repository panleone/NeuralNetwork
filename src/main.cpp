#include <iostream>
#include <fenv.h>

#include "layer.h"
#include "neural_network.h"
#include "random.h"

#include "tests/test_runner.h"

#include "data_loader.h"

int main(){
    feenableexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW);

    // Run tests
    try {
        run_tests();
    } catch(std::runtime_error& err){
        std::cerr << "Some tests failed! Is not safe to continue... quitting" << std::endl;
        std::cerr << err.what() << std::endl;
        exit(1);
    }

    NeuralNetwork<float> model{
        std::make_unique<FullyConnectedLayer<float>>(1,10),
        std::make_unique<ReluLayer<float>>(10),
        std::make_unique<FullyConnectedLayer<float>>(10,10),
        std::make_unique<ReluLayer<float>>(10),
        std::make_unique<FullyConnectedLayer<float>>(10, 1),
    };

    DataLoader<float> loader;
    for (int i = 0; i < 10000; i++){
        float x = static_cast<float>(i) / 10000.0f;
        loader.push(Vector<float>{x}, x*x + 5.0f*x + 4.0f);
    }
    size_t epoch = 10;
    size_t batchSize = 100;
    float alpha = 0.01f;
    for(size_t i = 0; i < epoch; i++) {
        size_t j = 0;
        for (auto &[x, y] : loader.randomIter()) {
            j += 1;
            auto res = model.forward(x.clone(), y);
            std::cout << res << " " << y << std::endl;
            if(j == batchSize){
                model.backward(alpha);
                j = 0;
            }
        }
        if(i > 0 && i % 20 == 0){
            alpha /= 2.0f;
        }
    }
    return 0;
}