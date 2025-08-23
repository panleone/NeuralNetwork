#pragma once

#include <algorithm>
#include <random>

template<typename T>
class GaussianGenerator {
    T mu;
    T sigma;
    std::normal_distribution<T> d;

    std::random_device rd{};
    std::mt19937 gen{rd()};

public:
    GaussianGenerator(const T& mu, const T& sigma) : d{mu, sigma} {};
    T generate(){ return d(gen); }
};

template <typename RandomIt> void shuffle(RandomIt first, RandomIt last) {
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(first, last, g);
}

/**
 * Generate a random number following gaussian distribution with mean mu and variance sigma
 */
template <typename T> T random_number(const T &mu, const T &sigma) {
  return GaussianGenerator(mu, sigma).generate();
}

/**
 * Random positive integer in the range min, max
 */
static size_t random_size_t(size_t min, size_t max) {
    static std::random_device rd;  // non-deterministic random number generator
    static std::mt19937_64 gen(rd()); // 64-bit Mersenne Twister
    std::uniform_int_distribution<size_t> dist(min, max);
    return dist(gen);
}