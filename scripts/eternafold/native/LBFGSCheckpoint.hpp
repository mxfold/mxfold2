#ifndef EF_LBFGS_CHECKPOINT_HPP
#define EF_LBFGS_CHECKPOINT_HPP
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <type_traits>
#include <vector>

template<class T> typename std::enable_if<std::is_arithmetic<T>::value>::type
EFSave(std::ostream &out, const T &v) { out.write(reinterpret_cast<const char *>(&v), sizeof(v)); }
template<class T> typename std::enable_if<std::is_arithmetic<T>::value>::type
EFLoad(std::istream &in, T &v) { in.read(reinterpret_cast<char *>(&v), sizeof(v)); }
template<class T> void EFSave(std::ostream &out, const std::vector<T> &v) {
    uint64_t n = v.size(); EFSave(out, n); for (const auto &x : v) EFSave(out, x);
}
template<class T> void EFLoad(std::istream &in, std::vector<T> &v) {
    uint64_t n; EFLoad(in, n);
    if (!in || n != v.size()) Error("L-BFGS checkpoint shape mismatch.");
    for (auto &x : v) EFLoad(in, x);
}

class EFCheckpoint {
    std::chrono::steady_clock::time_point started;
public:
    std::string path, signature;
    double seconds;
    int slice, initial_iteration;
    EFCheckpoint() : started(std::chrono::steady_clock::now()), seconds(0), slice(0), initial_iteration(0) {
        const char *p = std::getenv("EF_LBFGS_STATE"), *s = std::getenv("EF_SIGNATURE");
        path = p ? p : ""; signature = s ? s : "";
        if (std::getenv("EF_SLICE_SECONDS")) seconds = std::atof(std::getenv("EF_SLICE_SECONDS"));
        if (std::getenv("EF_SLICE_ITERATIONS")) slice = std::atoi(std::getenv("EF_SLICE_ITERATIONS"));
    }
    bool expired(int k) const {
        return (slice > 0 && k - initial_iteration >= slice) ||
            (seconds > 0 && std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() >= seconds);
    }
    bool read_signature(std::istream &in) {
        std::string saved; std::getline(in, saved);
        if (saved != "EFLBFGS01:" + signature) Error("Checkpoint belongs to a different frozen run.");
        return true;
    }
    void done() const {
        if (path.empty()) return;
        std::ofstream out((path + ".done.tmp").c_str()); out << signature << '\n'; out.close();
        if (std::rename((path + ".done.tmp").c_str(), (path + ".done").c_str())) Error("Cannot finalize checkpoint marker.");
    }
};
#endif
