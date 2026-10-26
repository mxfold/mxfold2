#!/usr/bin/env python3
"""Apply narrowly scoped, audited adapters to the pinned EternaFold source."""
from pathlib import Path
import argparse
import difflib
import shutil


def replace(path, old, new, count=1):
    s = path.read_text()
    if s.count(old) != count:
        raise ValueError(f"Unexpected upstream source: {path.name}: {old[:60]!r}")
    path.write_text(s.replace(old, new))


def patch(upstream, target):
    shutil.copytree(upstream, target)
    for p in target.glob('*.o'):
        p.unlink()
    for name in ('contrafold', 'api_test', 'score_prediction'):
        (target / name).unlink(missing_ok=True)
    for p in (Path(__file__).parent / 'native').glob('*.hpp'):
        shutil.copyfile(p, target / p.name)
    replace(target / 'SStruct.hpp', '    // assignment operator', '''    void LoadRecord(const std::string &name, const std::string &sequence,
                    const std::vector<int> &pairs, int sources, bool structural,
                    int channel, const std::vector<double> &evidence);

    // assignment operator''')
    with (target / 'SStruct.cpp').open('a') as f:
        f.write('''
void SStruct::LoadRecord(const std::string &name, const std::string &sequence,
                         const std::vector<int> &pairs, int sources, bool structural,
                         int channel, const std::vector<double> &evidence) {
    names.assign(1, name); sequences.assign(1, FilterSequence("@" + sequence));
    mapping = pairs; mapping2.clear(); mapping3.clear(); kd_data.clear();
    has_struct = structural; has_kd = false; num_data_sources = sources;
    which_evidence.assign(sources, false); has_evidence = channel >= 0;
    unpaired_potentials.assign(sources, std::vector<double>(sequence.size(), UNKNOWN_POTENTIAL));
    if (channel >= 0) { unpaired_potentials[channel] = evidence; which_evidence[channel] = true; }
    ValidateMapping(mapping);
}
''')
    replace(target / 'FileDescription.hpp', '    FileDescription(const FileDescription &rhs);', '''    FileDescription(const SStruct &structure, const std::string &name, double example_weight);
    FileDescription(const FileDescription &rhs);''')
    with (target / 'FileDescription.cpp').open('a') as f:
        f.write('''
FileDescription::FileDescription(const SStruct &structure, const std::string &name, double example_weight) :
    sstruct(structure), input_filename(name), size(int(Pow(double(structure.GetLength()), 3.0))), weight(example_weight) {}
''')
    replace(target / 'Contrafold.cpp', '#include "OptimizationWrapper.hpp"', '#include "OptimizationWrapper.hpp"\n#include "ExperimentSupport.hpp"')
    replace(target / 'Contrafold.cpp', '        descriptions.push_back(FileDescription(filenames[i],', '''        if (filenames[i].substr(0, 8) == "@EFPACK:") {
            EFLoadPack(filenames[i].substr(8), options.GetIntValue("num_data_sources"), descriptions);
            continue;
        }
        descriptions.push_back(FileDescription(filenames[i],''')
    old = '    std::vector<int> units = computation_wrapper.FilterNonparsable(computation_wrapper.GetAllUnits());'
    new = old + '''
    if (std::getenv("EF_AUDIT_FILE")) {
        std::ofstream audit(std::getenv("EF_AUDIT_FILE"));
        std::vector<bool> accepted(descriptions.size(), false);
        for (int i : units) accepted[i] = true;
        for (size_t i = 0; i < descriptions.size(); ++i)
            audit << descriptions[i].input_filename << "\\t" << accepted[i] << "\\n";
        audit.close(); computation_engine.StopComputeNodes(); return;
    }
    if (units.size() != descriptions.size()) Error("Unannounced exclusion from frozen training inputs.");
    if (std::getenv("EF_OBJECTIVE_OUT")) {
        const char *out = std::getenv("EF_OBJECTIVE_OUT");
        const RealT value = computation_wrapper.ComputeFunction(units, w, false, true, 1., hyperparam_data, kd_hyperparam_data, lig_hyperparam_data);
        const std::vector<RealT> gradient = computation_wrapper.ComputeGradient(units, w, false, true, 1., hyperparam_data, kd_hyperparam_data, lig_hyperparam_data);
        std::ofstream file(out); file << std::setprecision(17) << value << "\\n"; file.close();
        parameter_manager.WriteToFile(std::string(out) + ".gradient", gradient);
        computation_engine.StopComputeNodes(); return;
    }
'''
    replace(target / 'Contrafold.cpp', old, new)
    # A continuous probability density may exceed one, hence chemical NLL may
    # legitimately be negative. The upstream guard silently zeroed those terms.
    p = target / 'ComputationEngine.ipp'
    old = '    if (function_value_nonevidence < 0)\n'
    replace(p, old, '    if (function_value_nonevidence < 0 && sstruct.HasStruct())\n')
    # Rebuild chemical likelihoods for the current example, after LoadSequence
    # has cleared its evidence arrays. The upstream order reused stale storage.
    s = p.read_text()
    start = s.index('void ComputationEngine<RealT>::ComputeFunctionAndGradientSE(')
    end = s.index('// ComputationEngine::ComputeHessianVectorProduct()', start)
    section = s[start:end]
    section = section.replace('    inference_engine.UpdateEvidenceStructures();\n', '', 1)
    old = '    inference_engine.LoadSequence(sstruct);'
    if section.count(old) != 1:
        raise ValueError('Unexpected chemical example loading')
    section = section.replace(old, old + '\n    inference_engine.UpdateEvidenceStructures();')
    p.write_text(s[:start] + section + s[end:])
    # Standard and evidence ensembles share helix caches in upstream code.
    # Invalidate on each inside pass so changing ensemble rebuilds the cache.
    p = target / 'InferenceEngine.ipp'
    replace(p, 'void InferenceEngine<RealT>::ComputeInside()\n{',
            'void InferenceEngine<RealT>::ComputeInside()\n{\n    cache_initialized = false;')
    replace(p, 'void InferenceEngine<RealT>::ComputeInsideESS() \n{',
            'void InferenceEngine<RealT>::ComputeInsideESS() \n{\n    cache_initialized = false;')
    # Only structure-labelled examples need a constrained parse check. This
    # also supports mixed structure/chemical lists without N chemical checks.
    p = target / 'ComputationWrapper.ipp'
    needle = '''    std::vector<RealT> parsable;

    shared_info.command = CHECK_PARSABILITY;
'''
    replace(p, needle, '''    std::vector<int> structural_units;
    for (int unit : units) if (GetDescriptions()[unit].sstruct.HasStruct()) structural_units.push_back(unit);
    if (structural_units.empty()) return units;
    std::vector<RealT> parsable;

    shared_info.command = CHECK_PARSABILITY;
''')
    s = p.read_text()
    start = s.index('std::vector<int> ComputationWrapper<RealT>::FilterNonparsable(')
    end = s.index('// ComputationWrapper::ComputeLoss()', start)
    section = s[start:end]
    section = section.replace('nonshared_info.resize(units.size());', 'nonshared_info.resize(structural_units.size());')
    section = section.replace('i < units.size(); i++)', 'i < structural_units.size(); i++)', 1)
    section = section.replace('nonshared_info[i].index = units[i];', 'nonshared_info[i].index = structural_units[i];')
    section = section.replace('if (parsable[units[i]])', 'if (!GetDescriptions()[units[i]].sstruct.HasStruct() || parsable[units[i]])')
    p.write_text(s[:start] + section + s[end:])
    # Full optimizer history, including curvature vectors and accepted iterate.
    p = target / 'LBFGS.ipp'
    s = p.read_text()
    s = '#include "LBFGSCheckpoint.hpp"\n' + s
    start = s.index('    // check for termination criteria at beginning')
    end = s.index('    while (true)', start)
    init = s[start:end]
    declarations = '''    bool progress_made = false;
    int num_consecutive_small_steps = 0;
    int k = 0;'''
    if declarations not in init:
        raise ValueError('Unexpected L-BFGS state declarations')
    init = init.replace(declarations, '')
    init = init.replace('        return f[0];', '        checkpoint.done(); return f[0];')
    state_names = ['f', 'gamma', 'x', 'g', 's', 'y', 'rho', 'gradient_ratio', 'f0', 'progress_made', 'num_consecutive_small_steps', 'k']
    load = '\n'.join(f'            EFLoad(in, {v});' for v in state_names)
    save = '\n'.join(f'        EFSave(out, {v});' for v in state_names)
    support = declarations + '''
    EFCheckpoint checkpoint;
    bool resumed = false;
    if (!checkpoint.path.empty()) {
        std::ifstream in(checkpoint.path.c_str(), std::ios::binary);
        if (in) {
            checkpoint.read_signature(in);
''' + load + '''
            if (!in) Error("Truncated L-BFGS checkpoint.");
            resumed = true; checkpoint.initial_iteration = k;
            Report(SPrintF("Resuming exact L-BFGS state at iteration %d", k));
        }
    }
    if (!resumed) {
''' + init + '''
    }
    auto save_state = [&]() {
        if (checkpoint.path.empty()) return;
        std::ofstream out((checkpoint.path + ".tmp").c_str(), std::ios::binary);
        out << "EFLBFGS01:" << checkpoint.signature << "\\n";
''' + save + '''
        out.close(); if (!out) Error("Cannot write L-BFGS checkpoint.");
        if (std::rename((checkpoint.path + ".tmp").c_str(), checkpoint.path.c_str())) Error("Cannot commit L-BFGS checkpoint.");
    };
'''
    s = s[:start] + support + s[end:]
    s = s.replace('    while (true)\n    {', '''    while (true)
    {
        save_state();
        if (checkpoint.expired(k)) {
            Report("Yielding at accepted iterate; complete L-BFGS state saved.");
            x0 = x[k%2]; return f[k%2];
        }''', 1)
    s = s.replace('    x0 = x[(k+1)%2];', '    checkpoint.done();\n    x0 = x[(k+1)%2];')
    p.write_text(s)
    # Round-trip full precision for both frozen initial parameters and outputs.
    p = target / 'ParameterManager.ipp'
    replace(p, 'std::setprecision(10)', 'std::setprecision(17)')
    differences = []
    for p in sorted(target.iterdir()):
        if p.suffix not in ('.hpp', '.cpp', '.ipp'):
            continue
        orig = upstream / p.name
        a = orig.read_text().splitlines(True) if orig.exists() else []
        b = p.read_text().splitlines(True)
        differences.extend(difflib.unified_diff(a, b, fromfile='upstream/' + p.name, tofile='adapted/' + p.name))
    (target.parent / 'adapter.patch').write_text(''.join(differences))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('upstream', type=Path)
    p.add_argument('target', type=Path)
    a = p.parse_args()
    patch(a.upstream, a.target)
