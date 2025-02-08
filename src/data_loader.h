#pragma once

#include <functional>
#include <span>

#include "matrix.h"
#include "random.h"

template <typename Tx, typename Ty> class DataLoader {

  using DataPair = std::pair<Vector<Tx>, Vector<Ty>>;
  std::vector<DataPair> data;

public:
  DataLoader(){};
  void push(const Vector<Tx> &x, const Vector<Ty> &y) {
    data.push_back(std::make_pair(x.clone(), y.clone()));
  }
  void push(Vector<Tx> &&x, Vector<Ty> &&y) {
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