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

auto predict(const std::string& seq, py::object pa, 
            int min_hairpin, int max_internal, std::string constraint, 
            std::string reference, float pos_penalty, float neg_penalty)
{
    FoldOptions options;
    options.min_hairpin_loop_length(min_hairpin)
        .max_internal_loop_length(max_internal)
        .constraints(constraint);
    if (!reference.empty())
        options.penalty(reference, pos_penalty, neg_penalty);
    
    auto param = std::make_unique<TurnerNearestNeighbor>(pa);
    Fold<TurnerNearestNeighbor> f(std::move(param));
    auto e = f.compute_viterbi(seq, options);
    auto p = f.traceback_viterbi();
    f.traceback_viterbi(seq, options);
    auto s = make_paren(p);
    return std::make_tuple(e, s, p);
}

PYBIND11_MODULE(interface, m)
{
    using namespace std::literals::string_literals;
    using namespace pybind11::literals;
    m.doc() = "module for RNA secondary predicton with DNN";
    m.def("predict", &predict, "predict RNA secondary structure", 
        "seq"_a, "param"_a=py::none(), 
        "min_hairpin_length"_a=3, 
        "max_internal_length"_a=30, 
        "constraint"_a=""s, 
        "reference"_a=""s, 
        "pos_penalty"_a=0.0, 
        "neg_penalty"_a=0.0);
}