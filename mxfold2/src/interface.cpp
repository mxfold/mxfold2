#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <string>
#ifdef USE_OPENMP
#include <omp.h>
#endif
#include "fold/zuker.h"
#include "fold/nussinov.h"
#include "fold/linearfold/LinearFold.h"
#include "fold/linfold.h"
#include "param/contrafold.h"
#include "param/turner.h"
#include "param/positional.h"
#include "param/positional_bl.h"
#include "param/positional_1d.h"
#include "param/bpscore.h"
#include "param/mix.h"

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

    template < typename V, typename W >
    auto convert_list(py::list r) const
    {
        if (r.size()>0 && py::isinstance<W>(r[0]))
        {
            std::vector<V> c(r.size());
            std::transform(std::begin(r), std::end(r), std::begin(c),
                        [](auto x) -> V { return py::cast<W>(x); });
            return c;
        }
        return std::vector<V>();
    }

    auto convert_reference(py::list reference) const
    {
        return convert_list<u_int32_t, py::int_>(reference);
    }

    auto convert_position_scores(py::list v) const
    {
        return convert_list<float, py::float_>(v);
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
            float pos_paired, float neg_paired, float pos_unpaired, float neg_unpaired,
            py::object paired_position_scores)
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
        }
        if (/*!reference.is_none()*/ py::isinstance<py::list>(reference))
        {
            auto r = py::cast<py::list>(reference);
            auto r1 = convert_reference(r);
            if (r1.size() > 0)
                options.margin_terms(r1, pos_paired, neg_paired, pos_unpaired, neg_unpaired);
        }
        if (py::isinstance<py::list>(paired_position_scores))
        {
            auto sc = convert_position_scores(py::cast<py::list>(paired_position_scores));
            options.score_paired_position(sc);
        }
        std::swap(options, options_);
        return options_;
    }

    auto compute_viterbi(const std::string& seq, py::object pa,
            int min_hairpin, int max_internal, int max_helix,
            const std::string& allowed_pairs,
            py::object constraint, py::object reference, 
            float pos_paired, float neg_paired, float pos_unpaired, float neg_unpaired,
            py::object paired_position_scores)
    {
        set_param(seq, pa);
        set_options(min_hairpin, max_internal, max_helix, allowed_pairs, 
                constraint, reference, pos_paired, neg_paired, pos_unpaired, neg_unpaired,
                paired_position_scores);
        return f_->compute_viterbi(seq_, options_);
    }

    auto traceback_viterbi()
    {
        auto [e, p] = f_->traceback_viterbi(seq_, options_);
        auto s = Zuker<ParamClass>::make_paren(p);
        return std::make_tuple(e, s, p);
    }

    auto compute_basepairing_probabilities(const std::string& seq, py::object pa, 
            int min_hairpin, int max_internal, int max_helix,
            const std::string& allowed_pairs,
            py::object constraint, py::object reference, 
            float pos_paired, float neg_paired, float pos_unpaired, float neg_unpaired,
            py::object paired_position_scores)
    {
        set_param(seq, pa);
        set_options(min_hairpin, max_internal, max_helix, allowed_pairs, 
                constraint, reference, pos_paired, neg_paired, pos_unpaired, neg_unpaired, 
                paired_position_scores);

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
        }
        if (/*!reference.is_none()*/ py::isinstance<py::list>(reference))
        {
            auto r = py::cast<py::list>(reference);
            auto r1 = convert_reference(r);
            if (r1.size() > 0)
                options.margin_terms(r1, pos_paired, neg_paired, pos_unpaired, neg_unpaired);
        }
        std::swap(options, options_);
        return options_;
    }

    auto compute_viterbi(const std::string& seq, py::object pa, 
                int min_hairpin, int max_internal, int max_helix,
                const std::string& allowed_pairs,
                py::object constraint, py::object reference, 
                float pos_paired, float neg_paired, float pos_unpaired, float neg_unpaired,
                py::object paired_position_scores)
    {
        set_param(seq, pa);
        set_options(min_hairpin, max_internal, max_helix, allowed_pairs, constraint, reference, 
            pos_paired, neg_paired, pos_unpaired, neg_unpaired);
        return f_->compute_viterbi(seq_, options_);
    }

    auto traceback_viterbi()
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

    void set_param(const std::string& seq, py::object pa, int beam)
    {
        seq_ = seq;
        std::transform(seq_.begin(), seq_.end(), seq_.begin(), ::toupper);
        auto param = std::make_unique<ParamClass>(seq, pa);
        f_ = std::make_unique<LinearFold::BeamCKYParser<ParamClass>>(std::move(param), beam);
    }

    auto set_options(int min_hairpin, int max_internal, int max_helix,
            const std::string& allowed_pairs,
            py::object constraint, py::object reference, 
            float pos_paired, float neg_paired, float pos_unpaired, float neg_unpaired,
            py::object paired_position_scores)
    {   // TODO: support {pos, neg}_unpaired for LinearFold
        // TODO: support allowed_pairs for LinearFold, which is needed for modifications
        typename LinearFold::BeamCKYParser<ParamClass>::Options options;
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
        }
        if (/*!reference.is_none()*/ py::isinstance<py::list>(reference))
        {
            auto r = py::cast<py::list>(reference);
            auto r1 = convert_reference(r);
            if (r1.size() > 0)
                options.margin_terms(r1, pos_paired, neg_paired, pos_unpaired, neg_unpaired);
        }
        if (py::isinstance<py::list>(paired_position_scores))
        {
            auto sc = convert_position_scores(py::cast<py::list>(paired_position_scores));
            options.score_paired_position(sc);
        }

        std::swap(options, options_);
        return options_;
    }
    
    auto compute_viterbi(const std::string& seq, py::object pa, 
            int min_hairpin, int max_internal, int max_helix,
            const std::string& allowed_pairs,
            py::object constraint, py::object reference, 
            float pos_paired, float neg_paired, float pos_unpaired, float neg_unpaired,
            py::object paired_position_scores)
    {
        set_param(seq, pa, beam_size_);
        set_options(min_hairpin, max_internal, max_helix, allowed_pairs, 
                constraint, reference, pos_paired, neg_paired, pos_unpaired, neg_unpaired,
                paired_position_scores);
        auto r = f_->parse(seq_, options_);
        return r.score;
    }

    auto traceback_viterbi()
    {
        auto [e, p] = f_->traceback(seq_, options_);
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

private:
    std::string seq_;
    std::unique_ptr<LinearFold::BeamCKYParser<ParamClass>> f_;
    typename LinearFold::BeamCKYParser<ParamClass>::Options options_;
    int beam_size_;
};

template < class ParamClass >
class LinFoldWrapper : public FoldWrapper
{
public:
    LinFoldWrapper() {}
    
    void set_param(const std::string& seq, py::object pa)
    {
        seq_ = seq;
        auto param = std::make_unique<ParamClass>(seq, pa);
        f_ = std::make_unique<LinFold<ParamClass>>(std::move(param));
    }

    auto set_options(int min_hairpin, int max_internal, int max_helix,
            const std::string& allowed_pairs,
            py::object constraint, py::object reference, 
            float pos_paired, float neg_paired, float pos_unpaired, float neg_unpaired,
            py::object paired_position_scores)
    {
        typename LinFold<ParamClass>::Options options;
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
        }
        if (/*!reference.is_none()*/ py::isinstance<py::list>(reference))
        {
            auto r = py::cast<py::list>(reference);
            auto r1 = convert_reference(r);
            if (r1.size() > 0)
                options.margin_terms(r1, pos_paired, neg_paired, pos_unpaired, neg_unpaired);
        }
        if (py::isinstance<py::list>(paired_position_scores))
        {
            auto sc = convert_position_scores(py::cast<py::list>(paired_position_scores));
            options.score_paired_position(sc);
        }
        std::swap(options, options_);
        return options_;
    }

    auto compute_viterbi(const std::string& seq, py::object pa,
            int min_hairpin, int max_internal, int max_helix,
            const std::string& allowed_pairs,
            py::object constraint, py::object reference, 
            float pos_paired, float neg_paired, float pos_unpaired, float neg_unpaired,
            py::object paired_position_scores)
    {
        set_param(seq, pa);
        set_options(min_hairpin, max_internal, max_helix, allowed_pairs, 
                constraint, reference, pos_paired, neg_paired, pos_unpaired, neg_unpaired,
                paired_position_scores);
        return f_->compute_viterbi(seq_, options_);
    }

    auto traceback_viterbi()
    {
        auto [e, p] = f_->traceback_viterbi(seq_, options_);
        auto s = LinFold<ParamClass>::make_paren(p);
        return std::make_tuple(e, s, p);
    }

#if 0
    auto compute_basepairing_probabilities(const std::string& seq, py::object pa, 
            int min_hairpin, int max_internal, int max_helix,
            const std::string& allowed_pairs,
            py::object constraint, py::object reference, 
            float pos_paired, float neg_paired, float pos_unpaired, float neg_unpaired,
            py::object paired_position_scores)
    {
        set_param(seq, pa);
        set_options(min_hairpin, max_internal, max_helix, allowed_pairs, 
                constraint, reference, pos_paired, neg_paired, pos_unpaired, neg_unpaired, 
                paired_position_scores);

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
#endif

private:
    std::string seq_;
    std::unique_ptr<LinFold<ParamClass>> f_;
    typename LinFold<ParamClass>::Options options_;
};

void set_num_threads(int n)
{
#ifdef USE_OPENMP
    omp_set_num_threads(n);
#endif
}

PYBIND11_MODULE(interface, m)
{
    using namespace std::literals::string_literals;
    using namespace pybind11::literals;
    m.doc() = "module for RNA secondary predicton with DNN";

    m.def("set_num_threads", &set_num_threads, "set number of threads for OpenMP");

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
            "loss_neg_unpaired"_a=0.0,
            "paired_position_scores"_a=py::none())
        .def("traceback_viterbi", &ZukerWrapper<TurnerNearestNeighbor>::traceback_viterbi,
            "traceback for Turner model")
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
            "loss_neg_unpaired"_a=0.0,
            "paired_position_scores"_a=py::none());

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
            "loss_neg_unpaired"_a=0.0,
            "paired_position_scores"_a=py::none())
        .def("traceback_viterbi", &ZukerWrapper<PositionalNearestNeighbor>::traceback_viterbi,
            "traceback for positional nearest neighbor model")
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
            "loss_neg_unpaired"_a=0.0,
            "paired_position_scores"_a=py::none());

    py::class_<ZukerWrapper<CONTRAfoldNearestNeighbor>>(m, "ZukerCONTRAfoldWrapper")
        .def(py::init<>())
        .def("compute_viterbi", &ZukerWrapper<CONTRAfoldNearestNeighbor>::compute_viterbi, 
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
            "loss_neg_unpaired"_a=0.0,
            "paired_position_scores"_a=py::none())
        .def("traceback_viterbi", &ZukerWrapper<CONTRAfoldNearestNeighbor>::traceback_viterbi,
            "traceback for Turner model")
        .def("compute_basepairing_probabilities", &ZukerWrapper<CONTRAfoldNearestNeighbor>::compute_basepairing_probabilities,
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
            "loss_neg_unpaired"_a=0.0,
            "paired_position_scores"_a=py::none());

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
            "loss_neg_unpaired"_a=0.0,
            "paired_position_scores"_a=py::none())
        .def("traceback_viterbi", &ZukerWrapper<MixedNearestNeighbor>::traceback_viterbi,
            "traceback for mixed nearest neighbor model")
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
            "loss_neg_unpaired"_a=0.0,
            "paired_position_scores"_a=py::none());

    py::class_<ZukerWrapper<MixedNearestNeighbor2>>(m, "ZukerMixedWrapper2")
        .def(py::init<>())
        .def("compute_viterbi", &ZukerWrapper<MixedNearestNeighbor2>::compute_viterbi, 
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
            "loss_neg_unpaired"_a=0.0,
            "paired_position_scores"_a=py::none())
        .def("traceback_viterbi", &ZukerWrapper<MixedNearestNeighbor2>::traceback_viterbi,
            "traceback for mixed nearest neighbor model")
        .def("compute_basepairing_probabilities", &ZukerWrapper<MixedNearestNeighbor2>::compute_basepairing_probabilities,
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
            "loss_neg_unpaired"_a=0.0,
            "paired_position_scores"_a=py::none());

    py::class_<ZukerWrapper<CFMixedNearestNeighbor>>(m, "CONTRAfoldMixedWrapper")
        .def(py::init<>())
        .def("compute_viterbi", &ZukerWrapper<CFMixedNearestNeighbor>::compute_viterbi, 
            "predict RNA secondary structure with CONTRAfold-mixed nearest neighbor model", 
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
            "loss_neg_unpaired"_a=0.0,
            "paired_position_scores"_a=py::none())
        .def("traceback_viterbi", &ZukerWrapper<CFMixedNearestNeighbor>::traceback_viterbi,
            "traceback for CONTRAfold-mixed nearest neighbor model")
        .def("compute_basepairing_probabilities", &ZukerWrapper<CFMixedNearestNeighbor>::compute_basepairing_probabilities,
            "Partition function with CONTRAfold-mixed nearest neighbor model", 
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
            "loss_neg_unpaired"_a=0.0,
            "paired_position_scores"_a=py::none());

    py::class_<ZukerWrapper<CFMixedNearestNeighbor2>>(m, "CONTRAfoldMixedWrapper2")
        .def(py::init<>())
        .def("compute_viterbi", &ZukerWrapper<CFMixedNearestNeighbor2>::compute_viterbi, 
            "predict RNA secondary structure with CONTRAfold-mixed nearest neighbor model", 
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
            "loss_neg_unpaired"_a=0.0,
            "paired_position_scores"_a=py::none())
        .def("traceback_viterbi", &ZukerWrapper<CFMixedNearestNeighbor2>::traceback_viterbi,
            "traceback for CONTRAfold-mixed nearest neighbor model")
        .def("compute_basepairing_probabilities", &ZukerWrapper<CFMixedNearestNeighbor2>::compute_basepairing_probabilities,
            "Partition function with CONTRAfold-mixed nearest neighbor model", 
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
            "loss_neg_unpaired"_a=0.0,
            "paired_position_scores"_a=py::none());

#if 0
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
            "loss_neg_unpaired"_a=0.0,
            "paired_position_scores"_a=py::none);
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
            "loss_neg_unpaired"_a=0.0,
            "paired_position_scores"_a=py::none);

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
            "loss_neg_unpaired"_a=0.0,
            "paired_position_scores"_a=py::none);
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
            "loss_neg_unpaired"_a=0.0,
            "paired_position_scores"_a=py::none);
#endif
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
            "loss_neg_unpaired"_a=0.0,
            "paired_position_scores"_a=py::none())
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
            "loss_neg_unpaired"_a=0.0,
            "paired_position_scores"_a=py::none());

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
            "loss_neg_unpaired"_a=0.0,
            "paired_position_scores"_a=py::none())
        .def("traceback_viterbi", &NussinovWrapper<PositionalBasePairScore>::traceback_viterbi,
            "traceback for positional Nussinov model");

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
            "loss_neg_unpaired"_a=0.0,
            "paired_position_scores"_a=py::none())
        .def("traceback_viterbi", &LinearFoldWrapper<TurnerNearestNeighbor>::traceback_viterbi, 
            "traceback for LinearFold-V");

    py::class_<LinearFoldWrapper<CONTRAfoldNearestNeighbor>>(m, "LinearFoldCONTRAWrapper")
        .def(py::init<int>(), "constructor", "beam_size"_a=100)
        .def("compute_viterbi", &LinearFoldWrapper<CONTRAfoldNearestNeighbor>::compute_viterbi, 
            "predict RNA secondary structure with LinearFold-C Model", 
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
            "loss_neg_unpaired"_a=0.0,
            "paired_position_scores"_a=py::none())
        .def("traceback_viterbi", &LinearFoldWrapper<CONTRAfoldNearestNeighbor>::traceback_viterbi, 
            "traceback for LinearFold-C");
#if 0
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
            "traceback for LinearFold model");

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
            "traceback for Mixed LinearFold model");
#endif
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
            "loss_neg_unpaired"_a=0.0,
            "paired_position_scores"_a=py::none())
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
            "loss_neg_unpaired"_a=0.0,
            "paired_position_scores"_a=py::none())
        .def("traceback_viterbi", &LinearFoldWrapper<MixedNearestNeighbor>::traceback_viterbi,
            "traceback for LinearFold model");

    py::class_<LinearFoldWrapper<MixedNearestNeighbor2>>(m, "MixedLinearFoldPositional2DWrapper2")
        .def(py::init<int>(), "constructor", "beam_size"_a=100)
        .def("compute_viterbi", &LinearFoldWrapper<MixedNearestNeighbor2>::compute_viterbi, 
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
            "loss_neg_unpaired"_a=0.0,
            "paired_position_scores"_a=py::none())
        .def("traceback_viterbi", &LinearFoldWrapper<MixedNearestNeighbor2>::traceback_viterbi,
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
            "loss_neg_unpaired"_a=0.0,
            "paired_position_scores"_a=py::none())
        .def("traceback_viterbi", &LinearFoldWrapper<MixedNearestNeighbor1D>::traceback_viterbi,
            "traceback for LinearFold model");

    py::class_<LinFoldWrapper<TurnerNearestNeighbor>>(m, "LinFoldTurnerWrapper")
        //.def(py::init<int>(), "constructor", "beam_size"_a=100)
        .def(py::init())
        .def("compute_viterbi", &LinFoldWrapper<TurnerNearestNeighbor>::compute_viterbi, 
            "predict RNA secondary structure with LinFold-V Model", 
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
            "loss_neg_unpaired"_a=0.0,
            "paired_position_scores"_a=py::none())
        .def("traceback_viterbi", &LinFoldWrapper<TurnerNearestNeighbor>::traceback_viterbi, 
            "traceback for LinearFold-V");
}