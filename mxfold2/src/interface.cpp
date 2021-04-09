#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <string>
#include "fold/zuker.h"
#include "fold/nussinov.h"
#include "param/turner.h"
#include "param/positional.h"
#include "param/bpscore.h"
#include "param/mix.h"

namespace py = pybind11;

static
auto
convert_constraints(py::list constraint)
{
    std::vector<u_int32_t> ret(constraint.size(), Fold::Options::ANY);
    for (auto i=0; i!=constraint.size(); i++)
    {
        if (py::isinstance<py::str>(constraint[i]))
        {
            std::string c = py::cast<py::str>(constraint[i]);
            if (c=="x")
                ret[i] = Fold::Options::UNPAIRED;
            else if (c=="<")
                ret[i] = Fold::Options::PAIRED_L;
            else if (c==">")
                ret[i] = Fold::Options::PAIRED_R;
            else if (c=="|")
                ret[i] = Fold::Options::PAIRED_LR;
            /* else  if (c==".") 
                ret[i] = Fold::Options::ANY; */
        }
        else if (py::isinstance<py::int_>(constraint[i]))
        {
            auto v = py::cast<py::int_>(constraint[i]);
            switch (static_cast<int>(v)) {
                case  0: ret[i] = Fold::Options::UNPAIRED; break;
                case -1: ret[i] = Fold::Options::ANY; break;
                case -2: ret[i] = Fold::Options::PAIRED_L; break;
                case -3: ret[i] = Fold::Options::PAIRED_R; break;
                case -4: ret[i] = Fold::Options::PAIRED_LR; break;
                default: 
                    if (static_cast<int>(v)>=0) ret[i] = v;
                    break;
            }
        }
    }
    return ret;
}

static
auto
convert_pairs(py::list pairs)
{
    std::vector<std::pair<u_int32_t, u_int32_t>> ret;
    for (auto pair: pairs)
    {
        if (py::isinstance<py::list>(pair))
        {
            auto p = py::cast<py::list>(pair);
            if (p.size()==2)
            {
                auto p0 = py::cast<py::int_>(p[0]);
                auto p1 = py::cast<py::int_>(p[1]);
                ret.emplace_back(p0, p1);
            }
        }
    }
    return ret;
}

static
auto
convert_reference(py::list reference)
{

    auto r = py::cast<py::list>(reference);
    if (r.size()>0 && py::isinstance<py::int_>(r[0]))
    {
        std::vector<u_int32_t> c(r.size());
        std::transform(std::begin(r), std::end(r), std::begin(c),
                    [](auto x) -> u_int32_t { return py::cast<py::int_>(x); });
        return c;
    }
    return std::vector<u_int32_t>();
}

template < class ParamClass >
auto predict_zuker(const std::string& seq, py::object pa, 
            int min_hairpin, int max_internal, int max_helix,
            py::object constraint, py::object reference, 
            float pos_paired, float neg_paired, float pos_unpaired, float neg_unpaired)
{
    typename Zuker<ParamClass>::Options options;
    options.min_hairpin_loop_length(min_hairpin)
        .max_internal_loop_length(max_internal)
        .max_helix_length(max_helix);
    if (/*!constraint.is_none()*/ py::isinstance<py::list>(constraint)) 
    {
        auto c = py::cast<py::list>(constraint);
        auto c1 = convert_constraints(c);
        if (c1.size()>0)
            options.constraints(c1);
        auto c2 = convert_pairs(c);
        if (c2.size()>0)
            options.constraints(c2);
    }
    if (/*!reference.is_none()*/ py::isinstance<py::list>(reference))
    {
        auto r = py::cast<py::list>(reference);
        auto r1 = convert_reference(r);
        if (r1.size() > 0)
            options.penalty(r1, pos_paired, neg_paired, pos_unpaired, neg_unpaired);
        auto r2 = convert_pairs(r);
        if (r2.size() > 0)
            options.penalty(r2, pos_paired, neg_paired, pos_unpaired, neg_unpaired);
    }

    auto param = std::make_unique<ParamClass>(seq, pa);
    Zuker<ParamClass> f(std::move(param));
    f.compute_viterbi(seq, options);
    auto [e, p] = f.traceback_viterbi(seq, options);
    auto s = Zuker<ParamClass>::make_paren(p);
    return std::make_tuple(e, s, p);
}

template < class ParamClass >
auto partfunc_zuker(const std::string& seq, py::object pa, 
            int min_hairpin, int max_internal, int max_helix,
            py::object constraint, py::object reference, 
            float pos_paired, float neg_paired, float pos_unpaired, float neg_unpaired)
{
    typename Zuker<ParamClass>::Options options;
    options.min_hairpin_loop_length(min_hairpin)
        .max_internal_loop_length(max_internal)
        .max_helix_length(max_helix);
    if (/*!constraint.is_none()*/ py::isinstance<py::list>(constraint)) 
    {
        auto c = py::cast<py::list>(constraint);
        auto c1 = convert_constraints(c);
        if (c1.size()>0)
            options.constraints(c1);
        auto c2 = convert_pairs(c);
        if (c2.size()>0)
            options.constraints(c2);
    }
    if (/*!reference.is_none()*/ py::isinstance<py::list>(reference))
    {
        auto r = py::cast<py::list>(reference);
        auto r1 = convert_reference(r);
        if (r1.size() > 0)
            options.penalty(r1, pos_paired, neg_paired, pos_unpaired, neg_unpaired);
        auto r2 = convert_pairs(r);
        if (r2.size() > 0)
            options.penalty(r2, pos_paired, neg_paired, pos_unpaired, neg_unpaired);
    }

    auto param = std::make_unique<ParamClass>(seq, pa);
    Zuker<ParamClass> f(std::move(param));
    auto ret = f.compute_inside(seq, options);
    f.compute_outside(seq, options);
    auto bpp = f.compute_basepairing_probabilities(seq, options);
    py::array_t<float> bpp_a({bpp.size(), bpp[0].size()});
    auto bpp_a2 = bpp_a.mutable_unchecked<2>();
    for (auto i=0; i<bpp.size(); i++)
        for (auto j=0; j<bpp[i].size(); j++)
            bpp_a2(i, j) = bpp[i][j];
    return std::make_pair(ret, bpp_a);
}

template < class ParamClass >
auto predict_nussinov(const std::string& seq, py::object pa, 
            int min_hairpin, int max_internal, int max_helix,
            py::object constraint, py::object reference, 
            float pos_paired, float neg_paired, float pos_unpaired, float neg_unpaired)
{
    typename Nussinov<ParamClass>::Options options;
    options.min_hairpin_loop_length(min_hairpin);
    if (/*!constraint.is_none()*/ py::isinstance<py::list>(constraint)) 
    {
        auto c = py::cast<py::list>(constraint);
        auto c1 = convert_constraints(c);
        if (c1.size()>0)
            options.constraints(c1);
        auto c2 = convert_pairs(c);
        if (c2.size()>0)
            options.constraints(c2);
    }
    if (/*!reference.is_none()*/ py::isinstance<py::list>(reference))
    {
        auto r = py::cast<py::list>(reference);
        auto r1 = convert_reference(r);
        if (r1.size() > 0)
            options.penalty(r1, pos_paired, neg_paired, pos_unpaired, neg_unpaired);
        auto r2 = convert_pairs(r);
        if (r2.size() > 0)
            options.penalty(r2, pos_paired, neg_paired, pos_unpaired, neg_unpaired);
    }

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
    m.doc() = "module for RNA secondary predicton with DNN";

    auto predict_turner = &predict_zuker<TurnerNearestNeighbor>;
    m.def("predict_turner", predict_turner, 
        "predict RNA secondary structure with Turner Model", 
        "seq"_a, "param"_a, 
        "min_hairpin_length"_a=3, 
        "max_internal_length"_a=30, 
        "max_helix_length"_a=30,
        "constraint"_a=py::none(), 
        "reference"_a=py::none(), 
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
        "max_helix_length"_a=30,
        "constraint"_a=py::none(), 
        "reference"_a=py::none(), 
        "loss_pos_paired"_a=0.0, 
        "loss_neg_paired"_a=0.0,
        "loss_pos_unpaired"_a=0.0, 
        "loss_neg_unpaired"_a=0.0);

    auto predict_zuker_mixed = &predict_zuker<MixedNearestNeighbor>;
    m.def("predict_mxfold", predict_zuker_mixed, 
        "predict RNA secondary structure with mixed nearest neighbor model", 
        "seq"_a, "param"_a, 
        "min_hairpin_length"_a=3, 
        "max_internal_length"_a=30, 
        "max_helix_length"_a=30,
        "constraint"_a=py::none(), 
        "reference"_a=py::none(), 
        "loss_pos_paired"_a=0.0, 
        "loss_neg_paired"_a=0.0,
        "loss_pos_unpaired"_a=0.0,
        "loss_neg_unpaired"_a=0.0);

    auto predict_nussinov_positional = &predict_nussinov<PositionalBasePairScore>;
    m.def("predict_nussinov", predict_nussinov_positional, 
        "predict RNA secondary structure with positional nussinov model", 
        "seq"_a, "param"_a, 
        "min_hairpin_length"_a=3, 
        "max_internal_length"_a=30, 
        "max_helix_length"_a=30,
        "constraint"_a=py::none(), 
        "reference"_a=py::none(), 
        "loss_pos_paired"_a=0.0, 
        "loss_neg_paired"_a=0.0,
        "loss_pos_unpaired"_a=0.0, 
        "loss_neg_unpaired"_a=0.0);

    auto partfunc_turner = &partfunc_zuker<TurnerNearestNeighbor>;
    m.def("partfunc_turner", partfunc_turner, 
        "Partition function with Turner Model", 
        "seq"_a, "param"_a, 
        "min_hairpin_length"_a=3, 
        "max_internal_length"_a=30, 
        "max_helix_length"_a=30,
        "constraint"_a=py::none(), 
        "reference"_a=py::none(), 
        "loss_pos_paired"_a=0.0, 
        "loss_neg_paired"_a=0.0,
        "loss_pos_unpaired"_a=0.0, 
        "loss_neg_unpaired"_a=0.0);

    auto partfunc_zuker_positional = &partfunc_zuker<PositionalNearestNeighbor>;
    m.def("partfunc_zuker", partfunc_zuker_positional, 
        "Partition function with positional nearest neighbor model", 
        "seq"_a, "param"_a, 
        "min_hairpin_length"_a=3, 
        "max_internal_length"_a=30, 
        "max_helix_length"_a=30,
        "constraint"_a=py::none(), 
        "reference"_a=py::none(), 
        "loss_pos_paired"_a=0.0, 
        "loss_neg_paired"_a=0.0,
        "loss_pos_unpaired"_a=0.0, 
        "loss_neg_unpaired"_a=0.0);

    auto partfunc_zuker_mixed = &partfunc_zuker<MixedNearestNeighbor>;
    m.def("partfunc_mxfold", partfunc_zuker_mixed, 
        "Partition function with mixed nearest neighbor model", 
        "seq"_a, "param"_a, 
        "min_hairpin_length"_a=3, 
        "max_internal_length"_a=30, 
        "max_helix_length"_a=30,
        "constraint"_a=py::none(), 
        "reference"_a=py::none(), 
        "loss_pos_paired"_a=0.0, 
        "loss_neg_paired"_a=0.0,
        "loss_pos_unpaired"_a=0.0,
        "loss_neg_unpaired"_a=0.0);
}