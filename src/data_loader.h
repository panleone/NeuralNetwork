#pragma once

#include <functional>
#include <span>
#include <type_traits>

#include "matrix.h"
#include "random.h"

template <typename Tx, typename Ty> class DataLoader {

  using DataPair = std::pair<Tx, Ty>;
  std::vector<DataPair> data;

public:
  DataLoader(){};
  template <typename Ux, typename Uy> void push(Ux &&x, Uy &&y) {
    data.push_back(
        std::make_pair(std::forward<Ux &&>(x), std::forward<Uy &&>(y)));
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
  std::span<DataPair> getData() {
    return std::span<DataPair>(data.data(), data.size());
  }
};