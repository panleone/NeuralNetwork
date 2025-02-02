#pragma once

#include <functional>
#include <span>

#include "matrix.h"
#include "random.h"

template <typename T> class DataLoader {

  using DataPair = std::pair<Vector<T>, Vector<T>>;
  std::vector<DataPair> data;

public:
  DataLoader(){};
  void push(const Vector<T> &x, const Vector<T> &y) {
    data.push_back(std::make_pair(x.clone(), y.clone()));
  }
  void push(Vector<T> &&x, Vector<T> &&y) {
    data.push_back(std::make_pair(std::move(x), std::move(y)));
  }

  void randomIter(size_t batchSize,
                  const std::function<void(std::span<DataPair>)> &fn) {
    shuffle(data.begin(), data.end());
    auto it = data.begin();
    while (it != data.end()) {
      size_t endOffset = static_cast<size_t>(std::distance(it, data.end()));
      size_t dist = std::min(batchSize, endOffset);
      fn(std::span(it, dist));
      std::advance(it, dist);
    }
  }
};