#ifndef EF_EXPERIMENT_SUPPORT_HPP
#define EF_EXPERIMENT_SUPPORT_HPP
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>

template<class T> T EFRead(std::istream &in) {
    T v; in.read(reinterpret_cast<char *>(&v), sizeof(v));
    if (!in) Error("Truncated EternaFold packed input.");
    return v;
}

// EFPACK01 is little endian, with one sparse evidence channel per observation.
// It changes loading and example weights, not the inference/gradient equations.
inline void EFLoadPack(const std::string &path, int sources,
                       std::vector<FileDescription> &descriptions) {
    std::ifstream in(path.c_str(), std::ios::binary);
    char magic[8]; in.read(magic, 8);
    if (!in || std::memcmp(magic, "EFPACK01", 8)) Error("Invalid EternaFold pack.");
    if (EFRead<uint32_t>(in) != uint32_t(sources)) Error("Pack source count mismatch.");
    const uint64_t count = EFRead<uint64_t>(in);
    descriptions.reserve(descriptions.size() + count);
    for (uint64_t i = 0; i < count; ++i) {
        const uint32_t length = EFRead<uint32_t>(in), name_length = EFRead<uint32_t>(in);
        const double weight = EFRead<double>(in);
        const int32_t kind = EFRead<int32_t>(in), channel = EFRead<int32_t>(in);
        if (!length || length > 100000 || !name_length || name_length > 100000 ||
            !std::isfinite(weight) || weight <= 0 || kind < 0 || kind > 2)
            Error("Invalid EternaFold packed record.");
        std::string name(name_length, '\0'), sequence(length, '\0');
        in.read(&name[0], name_length); in.read(&sequence[0], length);
        std::vector<int> mapping(length + 1, SStruct::UNKNOWN);
        std::vector<double> evidence(length, SStruct::UNKNOWN_POTENTIAL);
        if (kind == 1) for (uint32_t j = 1; j <= length; ++j) mapping[j] = EFRead<int32_t>(in);
        if (kind == 2) {
            if (channel < 0 || channel >= sources) Error("Invalid evidence channel.");
            for (uint32_t j = 0; j < length; ++j) {
                evidence[j] = EFRead<float>(in);
                if (!std::isfinite(evidence[j]) || (evidence[j] != -1 && evidence[j] <= 1e-5))
                    Error("Invalid transformed evidence value.");
            }
        }
        if (!in) Error("Truncated packed sequence.");
        SStruct structure;
        structure.LoadRecord(name, sequence, mapping, sources,
                             kind == 1, kind == 2 ? channel : -1, evidence);
        descriptions.push_back(FileDescription(structure, name, weight));
    }
    if (in.peek() != std::char_traits<char>::eof()) Error("Trailing bytes in EternaFold pack.");
}
#endif
