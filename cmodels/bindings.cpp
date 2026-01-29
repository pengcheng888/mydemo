#include "mnist/bindings_mnist.hpp"
#include "resnet/bindings_resnet.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;
PYBIND11_MODULE(_infinidemo, m) {
    infinidemo::models::bind_mnist(m);
    infinidemo::models::bind_resnet_model(m);
    infinidemo::models::bind_resnet_config(m);
}
