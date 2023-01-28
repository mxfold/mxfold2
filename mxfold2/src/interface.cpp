#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <string>
#include "fold/zuker.h"
#include "fold/nussinov.h"
#include "param/turner.h"
#include "param/positional.h"
#include "param/positional_bl.h"
#include "param/positional_1d.h"
#include "param/bpscore.h"
#include "param/mix.h"
#include "fold/linearfold/LinearFold.h"

namespace py = pybind11;

class FoldWrapper
{
protected:
    auto convert_constraints(py::list constraint) const
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

    auto convert_pairs(py::list pairs) const
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

    auto convert_reference(py::list reference) const
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
};

template < class ParamClass >
class ZukerWrapper : public FoldWrapper
{
public:
    ZukerWrapper() {}
    
    void set_param(const std::string& seq, py::object pa)
    {
        seq_ = seq;
        auto param = std::make_unique<ParamClass>(seq, pa);
        f_ = std::make_unique<Zuker<ParamClass>>(std::move(param));
    }

    auto set_options(int min_hairpin, int max_internal, int max_helix,
            const std::string& allowed_pairs,
            py::object constraint, py::object reference, 
            float pos_paired, float neg_paired, float pos_unpaired, float neg_unpaired)
    {
        typename Zuker<ParamClass>::Options options;
        options.min_hairpin_loop_length(min_hairpin)
            .max_internal_loop_length(max_internal)
            .max_helix_length(max_helix);

        for (auto i=0; i!=allowed_pairs.size(); i+=2)
            options.set_allowed_pair(allowed_pairs[i], allowed_pairs[i+1]);

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
        std::swap(options, options_);
        return options_;
    }

    auto compute_viterbi(const std::string& seq, py::object pa,
            int min_hairpin, int max_internal, int max_helix,
            const std::string& allowed_pairs,
            py::object constraint, py::object reference, 
            float pos_paired, float neg_paired, float pos_unpaired, float neg_unpaired)
    {
        set_param(seq, pa);
        set_options(min_hairpin, max_internal, max_helix, allowed_pairs, 
                constraint, reference, pos_paired, neg_paired, pos_unpaired, neg_unpaired);
        return f_->compute_viterbi(seq_, options_);
    }

    auto traceback_viterbi(int from_pos = 0)
    {
        auto [e, p] = f_->traceback_viterbi(seq_, options_);
        auto s = Zuker<ParamClass>::make_paren(p);
        return std::make_tuple(e, s, p);
    }

    auto compute_basepairing_probabilities(const std::string& seq, py::object pa, 
            int min_hairpin, int max_internal, int max_helix,
            const std::string& allowed_pairs,
            py::object constraint, py::object reference, 
            float pos_paired, float neg_paired, float pos_unpaired, float neg_unpaired)
    {
        set_param(seq, pa);
        set_options(min_hairpin, max_internal, max_helix, allowed_pairs, 
                constraint, reference, pos_paired, neg_paired, pos_unpaired, neg_unpaired);

        auto ret = f_->compute_inside(seq_, options_);
        f_->compute_outside(seq_, options_);
        auto bpp = f_->compute_basepairing_probabilities(seq_, options_);
        py::array_t<float> bpp_a({bpp.size(), bpp[0].size()});
        auto bpp_a2 = bpp_a.mutable_unchecked<2>();
        for (auto i=0; i<bpp.size(); i++)
            for (auto j=0; j<bpp[i].size(); j++)
                bpp_a2(i, j) = bpp[i][j];
        return std::make_pair(ret, bpp_a);
    }

private:
    std::string seq_;
    std::unique_ptr<Zuker<ParamClass>> f_;
    typename Zuker<ParamClass>::Options options_;
};

template < class ParamClass >
class NussinovWrapper : public FoldWrapper
{
public:
    NussinovWrapper() {}

    void set_param(const std::string& seq, py::object pa)
    {
        seq_ = seq;
        auto param = std::make_unique<ParamClass>(seq, pa);
        f_ = std::make_unique<Nussinov<ParamClass>>(std::move(param));
    }

    auto set_options(int min_hairpin, int max_internal, int max_helix,
                const std::string& allowed_pairs,
                py::object constraint, py::object reference, 
                float pos_paired, float neg_paired, float pos_unpaired, float neg_unpaired)
    {
        typename Nussinov<ParamClass>::Options options;
        options.min_hairpin_loop_length(min_hairpin);

        for (auto i=0; i!=allowed_pairs.size(); i+=2)
            options.set_allowed_pair(allowed_pairs[i], allowed_pairs[i+1]);

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
        std::swap(options, options_);
        return options_;
    }

    auto compute_viterbi(const std::string& seq, py::object pa, 
                int min_hairpin, int max_internal, int max_helix,
                const std::string& allowed_pairs,
                py::object constraint, py::object reference, 
                float pos_paired, float neg_paired, float pos_unpaired, float neg_unpaired)
    {
        set_param(seq, pa);
        set_options(min_hairpin, max_internal, max_helix, allowed_pairs, constraint, reference, 
            pos_paired, neg_paired, pos_unpaired, neg_unpaired);
        return f_->compute_viterbi(seq_, options_);
    }

    auto traceback_viterbi(int from_pos = 0)
    {
        auto [e, p] = f_->traceback_viterbi(seq_, options_);
        auto s = Nussinov<ParamClass>::make_paren(p);
        return std::make_tuple(e, s, p);
    }

private:
    std::string seq_;
    std::unique_ptr<Nussinov<ParamClass>> f_;
    typename Nussinov<ParamClass>::Options options_;
};

template < class ParamClass >
class LinearFoldWrapper : public FoldWrapper
{
public:
    LinearFoldWrapper(int beam_size = 100) : beam_size_(beam_size) {}

    void set_param(const std::string& seq, py::object pa,
        int beam, int min_hairpin, int max_internal, float pos_paired, float neg_paired)
    {
        seq_ = seq;
        std::transform(seq_.begin(), seq_.end(), seq_.begin(), ::toupper);
        auto param = std::make_unique<ParamClass>(seq, pa);
        f_ = std::make_unique<LinearFold::BeamCKYParser<ParamClass>>(std::move(param),
            beam, min_hairpin, max_internal, pos_paired, neg_paired);
    }

    void set_options(const std::string& allowed_pairs, py::object constraint, py::object reference)
                
    {     // TODO: support {pos, neg}_unpaired for LinearFold
    #if 0 // TODO: support allowed_pairs for LinearFold, which is needed for modifications
        typename LinearFold::BeamCKYParser<ParamClass>::Options options;
        for (auto i=0; i!=allowed_pairs.size(); i+=2)
            options.set_allowed_pair(allowed_pairs[i], allowed_pairs[i+1]);
    #endif
        cons_.clear();
        if (/*!constraint.is_none()*/ py::isinstance<py::list>(constraint)) 
        {
            auto c = py::cast<py::list>(constraint);
            cons_ = convert_constraints(c);
        }
        ref_.clear();
        if (/*!reference.is_none()*/ py::isinstance<py::list>(reference))
        {
            auto r = py::cast<py::list>(reference);
            ref_ = convert_reference(r);
        }
    }
    
    auto compute_viterbi(const std::string& seq, py::object pa, 
            int min_hairpin, int max_internal, int max_helix,
            const std::string& allowed_pairs,
            py::object constraint, py::object reference, 
            float pos_paired, float neg_paired, float pos_unpaired, float neg_unpaired)
    {
        set_param(seq, pa, beam_size_, min_hairpin, max_internal,pos_paired, neg_paired);
        set_options(allowed_pairs, constraint, reference);
        auto r = f_->parse(seq_, cons_.size() > 0 ? &cons_: NULL, ref_.size() > 0 ? &ref_ : NULL);
        return r.score;
    }

    auto traceback_viterbi(int from_pos=0)
    {
        auto [e, p] = f_->traceback(seq_, ref_.size() > 0 ? &ref_ : NULL, from_pos);
        auto s = LinearFold::BeamCKYParser<ParamClass>::make_paren(p);
        return std::make_tuple(e, s, p);
    }

protected:
    auto convert_constraints(py::list constraint) const
    {
        std::vector<int> ret(constraint.size()-1, LinearFold::C_ANY);
        for (auto i=1; i!=constraint.size(); i++)
        {
            if (py::isinstance<py::str>(constraint[i]))
            {
                std::string c = py::cast<py::str>(constraint[i]);
                if (c=="x")
                    ret[i-1] = LinearFold::C_UNPAIRED;
                else if (c=="<")
                    ret[i-1] = LinearFold::C_PAIRED_L;
                else if (c==">")
                    ret[i-1] = LinearFold::C_PAIRED_R;
                else if (c=="|")
                    ret[i-1] = LinearFold::C_PAIRED_LR;
                /* else  if (c==".") 
                    ret[i] = LinearFold::BeamCKYParser::C_ANY; */
            }
            else if (py::isinstance<py::int_>(constraint[i]))
            {
                auto v = static_cast<int>(py::cast<py::int_>(constraint[i]));
                switch (v) {
                    case  0: ret[i-1] = LinearFold::C_UNPAIRED; break;
                    case -1: ret[i-1] = LinearFold::C_ANY; break;
                    case -2: ret[i-1] = LinearFold::C_PAIRED_L; break;
                    case -3: ret[i-1] = LinearFold::C_PAIRED_R; break;
                    case -4: ret[i-1] = LinearFold::C_PAIRED_LR; break;
                    default: 
                        if (v>0) ret[i-1] = v-1;
                        break;
                }
            }
        }
        return ret;
    }

    auto convert_reference(py::list reference) const
    {

        auto r = py::cast<py::list>(reference);
        if (r.size()>0 && py::isinstance<py::int_>(r[0]))
        {
            std::vector<int> c(r.size()-1);
            std::transform(std::begin(r)+1, std::end(r), std::begin(c),
                        [](auto x) -> int { return static_cast<int>(py::cast<py::int_>(x))-1; });
            return c;
        }
        return std::vector<int>();
    }

private:
    std::string seq_;
    std::unique_ptr<LinearFold::BeamCKYParser<ParamClass>> f_;
    std::vector<int> cons_;
    std::vector<int> ref_;
    int beam_size_;
};

PYBIND11_MODULE(interface, m)
{
    using namespace std::literals::string_literals;
    using namespace pybind11::literals;
    m.doc() = "module for RNA secondary predicton with DNN";

    py::class_<ZukerWrapper<TurnerNearestNeighbor>>(m, "ZukerTurnerWrapper")
        .def(py::init<>())
        .def("compute_viterbi", &ZukerWrapper<TurnerNearestNeighbor>::compute_viterbi, 
            "predict RNA secondary structure with Turner model", 
            "seq"_a, "param"_a, 
            "min_hairpin_length"_a=3, 
            "max_internal_length"_a=30, 
            "max_helix_length"_a=30,
            "allowed_pairs"_a="aucggu",
            "constraint"_a=py::none(), 
            "reference"_a=py::none(), 
            "loss_pos_paired"_a=0.0, 
            "loss_neg_paired"_a=0.0,
            "loss_pos_unpaired"_a=0.0, 
            "loss_neg_unpaired"_a=0.0)
        .def("traceback_viterbi", &ZukerWrapper<TurnerNearestNeighbor>::traceback_viterbi,
            "traceback for Turner model",
            "from_pos"_a=0)
        .def("compute_basepairing_probabilities", &ZukerWrapper<TurnerNearestNeighbor>::compute_basepairing_probabilities,
            "Partition function with Turner model", 
            "seq"_a, "param"_a, 
            "min_hairpin_length"_a=3, 
            "max_internal_length"_a=30, 
            "max_helix_length"_a=30,
            "allowed_pairs"_a="aucggu",
            "constraint"_a=py::none(), 
            "reference"_a=py::none(), 
            "loss_pos_paired"_a=0.0, 
            "loss_neg_paired"_a=0.0,
            "loss_pos_unpaired"_a=0.0, 
            "loss_neg_unpaired"_a=0.0);

    py::class_<ZukerWrapper<PositionalNearestNeighbor>>(m, "ZukerPositionalWrapper")
        .def(py::init<>())
        .def("compute_viterbi", &ZukerWrapper<PositionalNearestNeighbor>::compute_viterbi, 
            "predict RNA secondary structure with positional nearest neighbor model", 
            "seq"_a, "param"_a, 
            "min_hairpin_length"_a=3, 
            "max_internal_length"_a=30, 
            "max_helix_length"_a=30,
            "allowed_pairs"_a="aucggu",
            "constraint"_a=py::none(), 
            "reference"_a=py::none(), 
            "loss_pos_paired"_a=0.0, 
            "loss_neg_paired"_a=0.0,
            "loss_pos_unpaired"_a=0.0, 
            "loss_neg_unpaired"_a=0.0)
        .def("traceback_viterbi", &ZukerWrapper<PositionalNearestNeighbor>::traceback_viterbi,
            "traceback for positional nearest neighbor model",
            "from_pos"_a=0)
        .def("compute_basepairing_probabilities", &ZukerWrapper<PositionalNearestNeighbor>::compute_basepairing_probabilities,
            "Partition function with positional nearest neighbor model", 
            "seq"_a, "param"_a, 
            "min_hairpin_length"_a=3, 
            "max_internal_length"_a=30, 
            "max_helix_length"_a=30,
            "allowed_pairs"_a="aucggu",
            "constraint"_a=py::none(), 
            "reference"_a=py::none(), 
            "loss_pos_paired"_a=0.0, 
            "loss_neg_paired"_a=0.0,
            "loss_pos_unpaired"_a=0.0, 
            "loss_neg_unpaired"_a=0.0);

    py::class_<ZukerWrapper<MixedNearestNeighbor>>(m, "ZukerMixedWrapper")
        .def(py::init<>())
        .def("compute_viterbi", &ZukerWrapper<MixedNearestNeighbor>::compute_viterbi, 
            "predict RNA secondary structure with mixed nearest neighbor model", 
            "seq"_a, "param"_a, 
            "min_hairpin_length"_a=3, 
            "max_internal_length"_a=30, 
            "max_helix_length"_a=30,
            "allowed_pairs"_a="aucggu",
            "constraint"_a=py::none(), 
            "reference"_a=py::none(), 
            "loss_pos_paired"_a=0.0, 
            "loss_neg_paired"_a=0.0,
            "loss_pos_unpaired"_a=0.0, 
            "loss_neg_unpaired"_a=0.0)
        .def("traceback_viterbi", &ZukerWrapper<MixedNearestNeighbor>::traceback_viterbi,
            "traceback for mixed nearest neighbor model",
            "from_pos"_a=0)
        .def("compute_basepairing_probabilities", &ZukerWrapper<MixedNearestNeighbor>::compute_basepairing_probabilities,
            "Partition function with mixed nearest neighbor model", 
            "seq"_a, "param"_a, 
            "min_hairpin_length"_a=3, 
            "max_internal_length"_a=30, 
            "max_helix_length"_a=30,
            "allowed_pairs"_a="aucggu",
            "constraint"_a=py::none(), 
            "reference"_a=py::none(), 
            "loss_pos_paired"_a=0.0, 
            "loss_neg_paired"_a=0.0,
            "loss_pos_unpaired"_a=0.0, 
            "loss_neg_unpaired"_a=0.0);

    py::class_<ZukerWrapper<PositionalNearestNeighborBL>>(m, "ZukerPositionalBLWrapper")
        .def(py::init<>())
        .def("compute_viterbi", &ZukerWrapper<PositionalNearestNeighborBL>::compute_viterbi, 
            "predict RNA secondary structure with positional nearest neighbor model", 
            "seq"_a, "param"_a, 
            "min_hairpin_length"_a=3, 
            "max_internal_length"_a=30, 
            "max_helix_length"_a=30,
            "allowed_pairs"_a="aucggu",
            "constraint"_a=py::none(), 
            "reference"_a=py::none(), 
            "loss_pos_paired"_a=0.0, 
            "loss_neg_paired"_a=0.0,
            "loss_pos_unpaired"_a=0.0, 
            "loss_neg_unpaired"_a=0.0)
        .def("traceback_viterbi", &ZukerWrapper<PositionalNearestNeighborBL>::traceback_viterbi,
            "traceback for positional nearest neighbor model")
        .def("compute_basepairing_probabilities", &ZukerWrapper<PositionalNearestNeighborBL>::compute_basepairing_probabilities,
            "Partition function with positional nearest neighbor model", 
            "seq"_a, "param"_a, 
            "min_hairpin_length"_a=3, 
            "max_internal_length"_a=30, 
            "max_helix_length"_a=30,
            "allowed_pairs"_a="aucggu",
            "constraint"_a=py::none(), 
            "reference"_a=py::none(), 
            "loss_pos_paired"_a=0.0, 
            "loss_neg_paired"_a=0.0,
            "loss_pos_unpaired"_a=0.0, 
            "loss_neg_unpaired"_a=0.0);

    py::class_<ZukerWrapper<MixedNearestNeighborBL>>(m, "ZukerMixedBLWrapper")
        .def(py::init<>())
        .def("compute_viterbi", &ZukerWrapper<MixedNearestNeighborBL>::compute_viterbi, 
            "predict RNA secondary structure with mixed nearest neighbor model", 
            "seq"_a, "param"_a, 
            "min_hairpin_length"_a=3, 
            "max_internal_length"_a=30, 
            "max_helix_length"_a=30,
            "allowed_pairs"_a="aucggu",
            "constraint"_a=py::none(), 
            "reference"_a=py::none(), 
            "loss_pos_paired"_a=0.0, 
            "loss_neg_paired"_a=0.0,
            "loss_pos_unpaired"_a=0.0, 
            "loss_neg_unpaired"_a=0.0)
        .def("traceback_viterbi", &ZukerWrapper<MixedNearestNeighborBL>::traceback_viterbi,
            "traceback for mixed nearest neighbor model")
        .def("compute_basepairing_probabilities", &ZukerWrapper<MixedNearestNeighborBL>::compute_basepairing_probabilities,
            "Partition function with mixed nearest neighbor model", 
            "seq"_a, "param"_a, 
            "min_hairpin_length"_a=3, 
            "max_internal_length"_a=30, 
            "max_helix_length"_a=30,
            "allowed_pairs"_a="aucggu",
            "constraint"_a=py::none(), 
            "reference"_a=py::none(), 
            "loss_pos_paired"_a=0.0, 
            "loss_neg_paired"_a=0.0,
            "loss_pos_unpaired"_a=0.0, 
            "loss_neg_unpaired"_a=0.0);

    py::class_<ZukerWrapper<MixedNearestNeighbor1D>>(m, "ZukerMixed1DWrapper")
        .def(py::init<>())
        .def("compute_viterbi", &ZukerWrapper<MixedNearestNeighbor1D>::compute_viterbi, 
            "predict RNA secondary structure with mixed nearest neighbor model", 
            "seq"_a, "param"_a, 
            "min_hairpin_length"_a=3, 
            "max_internal_length"_a=30, 
            "max_helix_length"_a=30,
            "allowed_pairs"_a="aucggu",
            "constraint"_a=py::none(), 
            "reference"_a=py::none(), 
            "loss_pos_paired"_a=0.0, 
            "loss_neg_paired"_a=0.0,
            "loss_pos_unpaired"_a=0.0, 
            "loss_neg_unpaired"_a=0.0)
        .def("traceback_viterbi", &ZukerWrapper<MixedNearestNeighbor1D>::traceback_viterbi,
            "traceback for mixed nearest neighbor model")
        .def("compute_basepairing_probabilities", &ZukerWrapper<MixedNearestNeighbor1D>::compute_basepairing_probabilities,
            "Partition function with mixed nearest neighbor model", 
            "seq"_a, "param"_a, 
            "min_hairpin_length"_a=3, 
            "max_internal_length"_a=30, 
            "max_helix_length"_a=30,
            "allowed_pairs"_a="aucggu",
            "constraint"_a=py::none(), 
            "reference"_a=py::none(), 
            "loss_pos_paired"_a=0.0, 
            "loss_neg_paired"_a=0.0,
            "loss_pos_unpaired"_a=0.0, 
            "loss_neg_unpaired"_a=0.0);

    py::class_<NussinovWrapper<PositionalBasePairScore>>(m, "NussinovWrapper")
        .def(py::init<>())
        .def("compute_viterbi", &NussinovWrapper<PositionalBasePairScore>::compute_viterbi, 
            "predict RNA secondary structure with positional Nussinov model", 
            "seq"_a, "param"_a, 
            "min_hairpin_length"_a=3, 
            "max_internal_length"_a=30, 
            "max_helix_length"_a=30,
            "allowed_pairs"_a="aucggu",
            "constraint"_a=py::none(), 
            "reference"_a=py::none(), 
            "loss_pos_paired"_a=0.0, 
            "loss_neg_paired"_a=0.0,
            "loss_pos_unpaired"_a=0.0, 
            "loss_neg_unpaired"_a=0.0)
        .def("traceback_viterbi", &NussinovWrapper<PositionalBasePairScore>::traceback_viterbi,
            "traceback for positional Nussinov model",
            "from_pos"_a=0);

    py::class_<LinearFoldWrapper<TurnerNearestNeighbor>>(m, "LinearFoldTurnerWrapper")
        .def(py::init<int>(), "constructor", "beam_size"_a=100)
        .def("compute_viterbi", &LinearFoldWrapper<TurnerNearestNeighbor>::compute_viterbi, 
            "predict RNA secondary structure with LinearFold-V Model", 
            "seq"_a, "param"_a, 
            "min_hairpin_length"_a=3, 
            "max_internal_length"_a=30, 
            "max_helix_length"_a=30,
            "allowed_pairs"_a="aucggu",
            "constraint"_a=py::none(), 
            "reference"_a=py::none(), 
            "loss_pos_paired"_a=0.0, 
            "loss_neg_paired"_a=0.0,
            "loss_pos_unpaired"_a=0.0, 
            "loss_neg_unpaired"_a=0.0)
        .def("traceback_viterbi", &LinearFoldWrapper<TurnerNearestNeighbor>::traceback_viterbi, 
            "traceback for LinearFold-V",
            "from_pos"_a=0);

    py::class_<LinearFoldWrapper<PositionalNearestNeighborBL>>(m, "LinearFoldPositionalWrapper")
        .def(py::init<int>(), "constructor", "beam_size"_a=100)
        .def("compute_viterbi", &LinearFoldWrapper<PositionalNearestNeighborBL>::compute_viterbi, 
            "predict RNA secondary structure with LinearFold Model", 
            "seq"_a, "param"_a, 
            "min_hairpin_length"_a=3, 
            "max_internal_length"_a=30, 
            "max_helix_length"_a=30,
            "allowed_pairs"_a="aucggu",
            "constraint"_a=py::none(), 
            "reference"_a=py::none(), 
            "loss_pos_paired"_a=0.0, 
            "loss_neg_paired"_a=0.0,
            "loss_pos_unpaired"_a=0.0, 
            "loss_neg_unpaired"_a=0.0)
        .def("traceback_viterbi", &LinearFoldWrapper<PositionalNearestNeighborBL>::traceback_viterbi,
            "traceback for LinearFold model",
            "from_pos"_a=0);

    py::class_<LinearFoldWrapper<MixedNearestNeighborBL>>(m, "LinearFoldMixedWrapper")
        .def(py::init<int>(), "constructor", "beam_size"_a=100)
        .def("compute_viterbi", &LinearFoldWrapper<MixedNearestNeighborBL>::compute_viterbi, 
            "predict RNA secondary structure with Mixed LinearFold Model", 
            "seq"_a, "param"_a, 
            "min_hairpin_length"_a=3, 
            "max_internal_length"_a=30, 
            "max_helix_length"_a=30,
            "allowed_pairs"_a="aucggu",
            "constraint"_a=py::none(), 
            "reference"_a=py::none(), 
            "loss_pos_paired"_a=0.0, 
            "loss_neg_paired"_a=0.0,
            "loss_pos_unpaired"_a=0.0, 
            "loss_neg_unpaired"_a=0.0)
        .def("traceback_viterbi", &LinearFoldWrapper<MixedNearestNeighborBL>::traceback_viterbi,
            "traceback for Mixed LinearFold model",
            "from_pos"_a=0);

    py::class_<LinearFoldWrapper<PositionalNearestNeighbor>>(m, "LinearFoldPositional2DWrapper")
        .def(py::init<int>(), "constructor", "beam_size"_a=100)
        .def("compute_viterbi", &LinearFoldWrapper<PositionalNearestNeighbor>::compute_viterbi, 
            "predict RNA secondary structure with LinearFold Model", 
            "seq"_a, "param"_a, 
            "min_hairpin_length"_a=3, 
            "max_internal_length"_a=30, 
            "max_helix_length"_a=30,
            "allowed_pairs"_a="aucggu",
            "constraint"_a=py::none(), 
            "reference"_a=py::none(), 
            "loss_pos_paired"_a=0.0, 
            "loss_neg_paired"_a=0.0,
            "loss_pos_unpaired"_a=0.0, 
            "loss_neg_unpaired"_a=0.0)
        .def("traceback_viterbi", &LinearFoldWrapper<PositionalNearestNeighbor>::traceback_viterbi,
            "traceback for LinearFold model");

    py::class_<LinearFoldWrapper<MixedNearestNeighbor>>(m, "MixedLinearFoldPositional2DWrapper")
        .def(py::init<int>(), "constructor", "beam_size"_a=100)
        .def("compute_viterbi", &LinearFoldWrapper<MixedNearestNeighbor>::compute_viterbi, 
            "predict RNA secondary structure with LinearFold Model", 
            "seq"_a, "param"_a, 
            "min_hairpin_length"_a=3, 
            "max_internal_length"_a=30, 
            "max_helix_length"_a=30,
            "allowed_pairs"_a="aucggu",
            "constraint"_a=py::none(), 
            "reference"_a=py::none(), 
            "loss_pos_paired"_a=0.0, 
            "loss_neg_paired"_a=0.0,
            "loss_pos_unpaired"_a=0.0, 
            "loss_neg_unpaired"_a=0.0)
        .def("traceback_viterbi", &LinearFoldWrapper<MixedNearestNeighbor>::traceback_viterbi,
            "traceback for LinearFold model");

    py::class_<LinearFoldWrapper<MixedNearestNeighbor1D>>(m, "MixedLinearFoldPositional1DWrapper")
        .def(py::init<int>(), "constructor", "beam_size"_a=100)
        .def("compute_viterbi", &LinearFoldWrapper<MixedNearestNeighbor1D>::compute_viterbi, 
            "predict RNA secondary structure with LinearFold Model", 
            "seq"_a, "param"_a, 
            "min_hairpin_length"_a=3, 
            "max_internal_length"_a=30, 
            "max_helix_length"_a=30,
            "allowed_pairs"_a="aucggu",
            "constraint"_a=py::none(), 
            "reference"_a=py::none(), 
            "loss_pos_paired"_a=0.0, 
            "loss_neg_paired"_a=0.0,
            "loss_pos_unpaired"_a=0.0, 
            "loss_neg_unpaired"_a=0.0)
        .def("traceback_viterbi", &LinearFoldWrapper<MixedNearestNeighbor1D>::traceback_viterbi,
            "traceback for LinearFold model");
}