#pragma once

#include <span>

#include "matrix.h"
#include "random.h"

template<typename T>
class DataLoader{

    using DataPair = std::pair<Vector<T>, T>;
    std::vector<DataPair> data;
public:
    DataLoader(){};
    void push(const Vector<T>& x, const T& y){
        data.push_back(std::make_pair(x.clone(), y));
    }
    void push(Vector<T>&& x, T&& y){
        data.push_back(std::make_pair(std::move(x), std::move(y)));
    }

    std::span<DataPair> randomIter(){
        shuffle(data.begin(), data.end());
        return data;
    }
};