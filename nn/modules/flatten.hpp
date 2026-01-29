#pragma once

#include "module.hpp"
#include <infinicore/nn/module.hpp>
#include <infinicore/tensor.hpp>
#include <stdexcept>
#include <vector>

namespace infinidemo::nn::modules {
using namespace infinicore;

class Flatten : public infinidemo::nn::modules::Module {
public:
    Flatten(int start_dim = 1, int end_dim = -1)
        : start_dim_(start_dim), end_dim_(end_dim) {}

    inline Tensor forward(Tensor &input) const {
        const auto &shape = input->shape();
        const int ndim = static_cast<int>(shape.size());
        int actual_end_dim = end_dim_ < 0 ? ndim + end_dim_ : end_dim_;

        if (start_dim_ < 0 || start_dim_ >= ndim || actual_end_dim < 0 || actual_end_dim >= ndim || start_dim_ > actual_end_dim) {
            throw std::runtime_error("Flatten: invalid dimension range");
        }

        // Calculate flattened size
        size_t flattened_size = 1;
        for (int i = start_dim_; i <= actual_end_dim; ++i) {
            flattened_size *= shape[i];
        }

        // new shape: [dims before start_dim, flattened_size, dims after end_dim]
        std::vector<Size> new_shape;
        new_shape.reserve(start_dim_ + 1 + (ndim - actual_end_dim - 1));
        new_shape.insert(new_shape.end(), shape.begin(), shape.begin() + start_dim_);
        new_shape.push_back(flattened_size);
        new_shape.insert(new_shape.end(), shape.begin() + actual_end_dim + 1, shape.end());

        return input->view(new_shape);
    }

private:
    void to_device_(const Device &device) override {}

protected:
    int start_dim_;
    int end_dim_;
};

} // namespace infinidemo::nn::modules
