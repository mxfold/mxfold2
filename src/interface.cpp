#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <string>
#include "fold/zuker.h"
#include "fold/nussinov.h"
#include "param/turner.h"
#include "param/positional.h"
#include "param/bpscore.h"

namespace py = pybind11;

template < class ParamClass >
auto predict_zuker(const std::string& seq, py::object pa, 
            int min_hairpin, int max_internal, std::string constraint, std::string reference, 
            float pos_paired, float neg_paired, float pos_unpaired, float neg_unpaired)
{
    typename Zuker<ParamClass>::Options options;
    options.min_hairpin_loop_length(min_hairpin)
        .max_internal_loop_length(max_internal)
        .constraints(constraint)
        .penalty(reference, pos_paired, neg_paired, pos_unpaired, neg_unpaired);
    
    auto param = std::make_unique<ParamClass>(seq, pa);
    Zuker<ParamClass> f(std::move(param));
    f.compute_viterbi(seq, options);
    auto [e, p] = f.traceback_viterbi(seq, options);
    auto s = Zuker<ParamClass>::make_paren(p);
    return std::make_tuple(e, s, p);
}

template < class ParamClass >
auto predict_nussinov(const std::string& seq, py::object pa, 
            int min_hairpin, std::string constraint, std::string reference, 
            float pos_paired, float neg_paired, float pos_unpaired, float neg_unpaired)
{
    typename Nussinov<ParamClass>::Options options;
    options.min_hairpin_loop_length(min_hairpin)
        .constraints(constraint)
        .penalty(reference, pos_paired, neg_paired, pos_unpaired, neg_unpaired);
    
    auto param = std::make_unique<ParamClass>(seq, pa);
    Nussinov<ParamClass> f(std::move(param));
    f.compute_viterbi(seq, options);
    auto [e, p] = f.traceback_viterbi(seq, options);
    auto s = Nussinov<ParamClass>::make_paren(p);
    return std::make_tuple(e, s, p);
}

PYBIND11_MODULE(interface, m)
{
    using namespace std::literals::string_literals;
    using namespace pybind11::literals;

    auto predict_turner = &predict_zuker<TurnerNearestNeighbor>;
    m.doc() = "module for RNA secondary predicton with DNN";
    m.def("predict_turner", predict_turner, 
        "predict RNA secondary structure with Turner Model", 
        "seq"_a, "param"_a, 
        "min_hairpin_length"_a=3, 
        "max_internal_length"_a=30, 
        "constraint"_a=""s, 
        "reference"_a=""s, 
        "loss_pos_paired"_a=0.0, 
        "loss_neg_paired"_a=0.0,
        "loss_pos_unpaired"_a=0.0, 
        "loss_neg_unpaired"_a=0.0);

    auto predict_zuker_positional = &predict_zuker<PositionalNearestNeighbor>;
    m.def("predict_zuker", predict_zuker_positional, 
        "predict RNA secondary structure with positional nearest neighbor model", 
        "seq"_a, "param"_a, 
        "min_hairpin_length"_a=3, 
        "max_internal_length"_a=30, 
        "constraint"_a=""s, 
        "reference"_a=""s, 
        "loss_pos_paired"_a=0.0, 
        "loss_neg_paired"_a=0.0,
        "loss_pos_unpaired"_a=0.0, 
        "loss_neg_unpaired"_a=0.0);

    auto predict_nussinov_positional = &predict_nussinov<PositionalBasePairScore>;
    m.def("predict_nussinov", predict_nussinov_positional, 
        "predict RNA secondary structure with positional nussinov model", 
        "seq"_a, "param"_a, 
        "min_hairpin_length"_a=3, 
        "constraint"_a=""s, 
        "reference"_a=""s, 
        "loss_pos_paired"_a=0.0, 
        "loss_neg_paired"_a=0.0,
        "loss_pos_unpaired"_a=0.0, 
        "loss_neg_unpaired"_a=0.0);
}