#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>

#endif
#include <stx/btree_map.h>

#include "optimal.hpp"

namespace Optimal {

template <typename KeyType>
struct Segment {
    Segment(const long double &slope, const long double &intercept, const KeyType &first_x)
        : slope(slope), intercept(intercept), first_x(first_x) {}
    Segment() = default;

    long double slope;
    long double intercept;
    KeyType first_x;
    KeyType last_x;
};

template <typename KeyType>
struct LeafNode {
    std::vector<Segment<KeyType>> segments;

    size_t bytes_used() const {
        return sizeof(*this) + segments.size() * sizeof(Segment<KeyType>);
    }
};

template <typename KeyType>
class FittingTree {
public:
    explicit FittingTree(double error_bound = 16.0) : error_bound_(error_bound) {
        if (error_bound <= 0) {
            throw std::invalid_argument("Error bound must be positive");
        }
    }

    ~FittingTree() = default;

    void build(std::vector<KeyType> data) {
        using Seg = typename Optimal::internal::OptimalPiecewiseLinearModel<KeyType, size_t>::CanonicalSegment;

        // Build the tree index using greedy segmentation.
        auto in = [&](int i) { return data[i]; };
        std::vector<Seg> canonical_segments;
        auto out = [&](const auto &cs) { canonical_segments.push_back(cs); };
        auto n = data.size();
        int parallelism = omp_get_num_procs();
        Optimal::internal::make_segmentation_par(n, error_bound_, in, out, parallelism);

        for (const auto &seg : canonical_segments) {
            auto leaf = std::make_shared<LeafNode<KeyType>>();
            KeyType first_x = seg.get_first_x();
            auto [slope, intercept] = seg.get_floating_point_segment(first_x);

            Segment<KeyType> new_segment(slope, intercept, first_x);
            leaf->segments.push_back(new_segment);
            tree_index_.insert({seg.get_first_x(), std::move(leaf)});
        }
    }

    int search(KeyType key) const {
        auto it = tree_index_.upper_bound(key);
        if (it == tree_index_.begin()) {
            return 0;
        }
        --it;
        const Segment<KeyType> &seg = it->second->segments[0];
        auto slope = seg.slope;
        auto first_x = seg.first_x;
        auto intercept = seg.intercept;
        size_t idx = slope * (key - first_x) + intercept;

        return static_cast<int>(idx);
    }

    void clear() { tree_index_.clear(); }

    size_t size_in_bytes() const {
        size_t total_size = sizeof(*this);
        for (const auto &entry : tree_index_) {
            total_size += sizeof(entry.first);
            if (entry.second) {
                total_size += entry.second->bytes_used();
            }
        }
        return total_size;
    }

    size_t get_segment_count() const { return tree_index_.size(); }

    size_t get_segment_size_bytes() const {
        size_t total_size = 0;
        total_size += sizeof(LeafNode<KeyType>) * tree_index_.size();
        return total_size;
    }

private:
    using TreeMap = stx::btree_map<KeyType, std::shared_ptr<LeafNode<KeyType>>>;

    TreeMap tree_index_;
    double error_bound_;
};

} // namespace Optimal
