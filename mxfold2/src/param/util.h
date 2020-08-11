#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

template <int D>
auto
get_unchecked(py::object obj, const char* name)
{
    py::object v;
    if (py::isinstance<py::dict>(obj))
        v = py::cast<py::dict>(obj)[name];
    else
        v = obj.attr(name);
    if (py::hasattr(v, "detach")) // assume that v is a torch.tensor with require_grad
        v = v.attr("detach")().attr("numpy")();
    auto vv = v.cast<py::array_t<float>>();
    return vv.unchecked<D>();
}

template <int D>
auto
get_mutable_unchecked(py::object obj, const char* name)
{
    py::object v;
    if (py::isinstance<py::dict>(obj))
        v = py::cast<py::dict>(obj)[name];
    else
        v = obj.attr(name);
    if (py::hasattr(v, "numpy")) 
        v = v.attr("numpy")();
    auto vv = v.cast<py::array_t<float>>();
    return vv.mutable_unchecked<D>();
}
