#pragma once

#include <vector>

template <typename T>
class RangedVector
{
    public:
        RangedVector() : data_(), start_(0), end_(0) {}

        RangedVector(int start, int end, T v = T()) 
            : data_(end-start, v), start_(start), end_(end) {}

        void resize(int start, int end , T v = T())
        {
            data_.resize(end-start, v);
            start_ = start;
            end_ = end;
        }

        void clear()
        {
            data_.clear();
        }

        T& operator[](int idx)
        {
            return data_[idx-start_];
        }

        const T& operator[](int idx) const
        {
            return data_[idx-start_];
        }

    private:
        std::vector<T> data_;
        int start_;
        int end_;
};

template <typename T>
class TriMatrix
{
    public:
        TriMatrix() : data_() {}

        TriMatrix(int sz, T v = T(), int diag=0) : data_(sz)
        {
            for (auto i=0; i!=sz; ++i)
                data_[i] = std::move(RangedVector<T>(i+diag, sz, v));
        }

        void resize(int sz, T v = T(), int diag=0)
        {
            data_.resize(sz);
            for (auto i=0; i!=sz; ++i)
                data_[i].resize(i+diag, sz, v);
        }

        void clear()
        {
            data_.clear();
        }

        size_t size() const { return data_.size(); }

        RangedVector<T>& operator[](size_t idx) { return data_[idx]; }
        const RangedVector<T>& operator[](size_t idx) const { return data_[idx]; }

    private:
        std::vector< RangedVector<T> > data_;
};
