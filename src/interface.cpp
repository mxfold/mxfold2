#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <string>
#include "fold.h"
#include "parameter.h"

namespace py = pybind11;

auto make_paren(const std::vector<u_int32_t>& p)
{
    std::string s(p.size()-1, '.');
    for (size_t i=1; i!=p.size(); ++i)
    {
        if (p[i] != 0)
            s[i-1] = p[i]>i ? '(' : ')';
    }
    return s;
}

auto predict(const std::string& seq, py::object pa)
{
    if (pa.is_none())
    {
        auto param = std::make_unique<MFE>();
        param->load_default();
        Fold<MFE> f(std::move(param));
        auto e = f.compute_viterbi(seq);
        auto p = f.traceback_viterbi();
        auto s = make_paren(p);
        return std::make_pair(e, s);
    }
    else
    {
        auto param = std::make_unique<PyMFE>(pa);
        Fold<PyMFE> f(std::move(param));
        auto e = f.compute_viterbi(seq);
        auto p = f.traceback_viterbi();
        auto s = make_paren(p);
        return std::make_pair(e, s);
    }
}

PYBIND11_MODULE(dnnfold, m)
{
    m.doc() = "module for RNA secondary predicton with DNN";
    m.def("predict", &predict, "predict RNA secondary structure", 
        py::arg("seq"), py::arg("param") = py::none());
}