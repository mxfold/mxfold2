#pragma once

#include <string>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

// #define USE_PYSCORE // VERY SLOW!
namespace py = pybind11;
class PositionalNearestNeighborBL
{
public:
    using ScoreType = float;

private:
    template<int D> using ParamType = pybind11::detail::unchecked_reference<float, D> ;
    template<int D> using CountType = pybind11::detail::unchecked_mutable_reference<float, D> ;

public:
    PositionalNearestNeighborBL(const std::string& seq, pybind11::object obj);
    ~PositionalNearestNeighborBL() {};

    ScoreType score_hairpin(size_t i, size_t j) const;
    ScoreType score_single_loop(size_t i, size_t j, size_t k, size_t l) const;
    ScoreType score_helix(size_t i, size_t j, size_t m) const;
    ScoreType score_multi_loop(size_t i, size_t j) const;
    ScoreType score_multi_paired(size_t i, size_t j) const;
    ScoreType score_multi_unpaired(size_t i, size_t j) const;
    ScoreType score_external_zero() const { return 0.0; }
    ScoreType score_external_paired(size_t i, size_t j) const;
    ScoreType score_external_unpaired(size_t i, size_t j) const;

    void count_hairpin(size_t i, size_t j, ScoreType v);
    void count_single_loop(size_t i, size_t j, size_t k, size_t l, ScoreType v);
    void count_helix(size_t i, size_t j, size_t m, ScoreType v);
    void count_multi_loop(size_t i, size_t j, ScoreType v);
    void count_multi_paired(size_t i, size_t j, ScoreType v);
    void count_multi_unpaired(size_t i, size_t j, ScoreType v);
    void count_external_zero(ScoreType v) { }
    void count_external_paired(size_t i, size_t j, ScoreType v);
    void count_external_unpaired(size_t i, size_t j, ScoreType v);

private:
    ParamType<2> embedding_;

    ParamType<2> bl_w_helix_stacking_;
    ParamType<1> bl_b_helix_stacking_;
    ParamType<2> bl_w_mismatch_external_;
    ParamType<1> bl_b_mismatch_external_;
    ParamType<2> bl_w_mismatch_hairpin_;
    ParamType<1> bl_b_mismatch_hairpin_;
    ParamType<2> bl_w_mismatch_internal_;
    ParamType<1> bl_b_mismatch_internal_;
    ParamType<2> bl_w_mismatch_multi_;
    ParamType<1> bl_b_mismatch_multi_;
    ParamType<2> bl_w_base_hairpin_;
    ParamType<1> bl_b_base_hairpin_;
    ParamType<2> bl_w_base_internal_;
    ParamType<1> bl_b_base_internal_;
    ParamType<2> bl_w_base_multi_;
    ParamType<1> bl_b_base_multi_;
    ParamType<2> bl_w_base_external_;
    ParamType<1> bl_b_base_external_;

    ParamType<1> score_hairpin_length_;
    ParamType<1> score_bulge_length_;
    ParamType<1> score_internal_length_;
    ParamType<2> score_internal_explicit_;
    ParamType<1> score_internal_symmetry_;
    ParamType<1> score_internal_asymmetry_;
    ParamType<1> score_helix_length_;

    pybind11::object cnt_;

    auto score_basepair_(size_t i, size_t j) const -> ScoreType;
    auto score_helix_stacking_(size_t i, size_t j) const -> ScoreType;
    auto score_mismatch_external_(size_t i, size_t j) const -> ScoreType;
    auto score_mismatch_hairpin_(size_t i, size_t j) const -> ScoreType;
    auto score_mismatch_internal_(size_t i, size_t j) const -> ScoreType;
    auto score_mismatch_multi_(size_t i, size_t j) const -> ScoreType;
    auto score_base_hairpin_(size_t i, size_t j) const -> ScoreType;
    auto score_base_internal_(size_t i, size_t j) const -> ScoreType;
    auto score_base_multi_(size_t i, size_t j) const -> ScoreType;
    auto score_base_external_(size_t i, size_t j) const -> ScoreType;

#ifdef USE_PYSCORE
    py::object py_score_hairpin;
    py::object py_score_single_loop;
    py::object py_score_helix;
    py::object py_score_multi_loop;
    py::object py_score_multi_paired;
    py::object py_score_multi_unpaired;
    py::object py_score_external_paired;
    py::object py_score_external_unpaired;
#endif

public:
    const u_int32_t MAX_HAIRPIN_LENGTH;
    const u_int32_t MAX_BULGE_LENGTH;
    const u_int32_t MAX_INTERNAL_LENGTH;
    const u_int32_t MAX_SINGLE_LENGTH;
    const u_int32_t MAX_INTERNAL_SYMMETRIC_LENGTH;
    const u_int32_t MAX_INTERNAL_ASYMMETRY;
    const u_int32_t MAX_INTERNAL_EXPLICIT_LENGTH;
    const u_int32_t MAX_HELIX_LENGTH;
};