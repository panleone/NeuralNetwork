#pragma once

#include "../../src/data_loader.h"

using resType = std::pair<DataLoader<Vector<float>, size_t>,
                          DataLoader<Vector<float>, size_t>>;
resType loadMNIST1D();