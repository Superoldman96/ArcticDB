/* Copyright 2023 Man Group Operations Limited
 *
 * Use of this software is governed by the Business Source License 1.1 included in the file licenses/BSL.txt.
 *
 * As of the Change Date specified in that file, in accordance with the Business Source License, use of this software will be governed by the Apache License, version 2.0.
 */

#pragma once

#include <memory>
#include <optional>
#include <string>
#include <unordered_set>

#include <folly/futures/FutureSplitter.h>

#include <arcticdb/entity/stream_descriptor.hpp>
#include <arcticdb/pipeline/frame_slice.hpp>
#include <arcticdb/processing/component_manager.hpp>
#include <arcticdb/processing/processing_unit.hpp>
#include <arcticdb/processing/sorted_aggregation.hpp>

namespace arcticdb {

using RangesAndKey = pipelines::RangesAndKey;
using SliceAndKey = pipelines::SliceAndKey;

enum class ProcessingStructure {
    ROW_SLICE,
    TIME_BUCKETED,
    HASH_BUCKETED,
    ALL,
    MULTI_SYMBOL
};

struct KeepCurrentIndex{};
struct KeepCurrentTopLevelIndex{};
using NewIndex = std::string;

// Contains constant data about the clause identifiable at construction time
struct ClauseInfo {
    // The arrangement of segments this clause needs in order for processing to be done correctly
    ProcessingStructure input_structure_{ProcessingStructure::ROW_SLICE};
    // The arrangement of segment this clause produces
    ProcessingStructure output_structure_{ProcessingStructure::ROW_SLICE};
    // Whether it makes sense to combine this clause with specifying which columns to view in a call to read or similar
    bool can_combine_with_column_selection_{true};
    // The names of the columns that are needed for this clause to make sense
    // Could either be on disk, or columns created by earlier clauses in the processing pipeline
    std::optional<std::unordered_set<std::string>> input_columns_{std::nullopt};
    // KeepCurrentIndex if this clause does not modify the index in any way
    // KeepCurrentTopLevelIndex if this clause requires multi-index levels>0 to be dropped, but otherwise does not modify it
    // NewIndex if this clause has changed the index to a new (supplied) name
    std::variant<KeepCurrentIndex, KeepCurrentTopLevelIndex, NewIndex> index_{KeepCurrentIndex()};
    // Whether this clause operates on one or multiple symbols
    bool multi_symbol_{false};
};

// Changes how the clause behaves based on information only available after it is constructed
struct ProcessingConfig {
    bool dynamic_schema_{false};
    uint64_t total_rows_{0};
    IndexDescriptor::Type index_type_{IndexDescriptor::Type::UNKNOWN};
};

// Used when restructuring segments inbetween clauses with differing ProcessingStructures
struct RangesAndEntity {
    explicit RangesAndEntity(EntityId id, std::shared_ptr<RowRange> row_range, std::shared_ptr<ColRange> col_range, std::optional<TimestampRange>&& timestamp_range=std::nullopt):
            id_(id),
            row_range_(std::move(row_range)),
            col_range_(std::move(col_range)),
            timestamp_range_(std::move(timestamp_range)) {
    }
    ARCTICDB_MOVE_COPY_DEFAULT(RangesAndEntity)

    [[nodiscard]] const RowRange& row_range() const {
        return *row_range_;
    }

    [[nodiscard]] const ColRange& col_range() const {
        return *col_range_;
    }

    [[nodiscard]] timestamp start_time() const {
        return timestamp_range_->first;
    }

    [[nodiscard]] timestamp end_time() const {
        return timestamp_range_->second;
    }

    friend bool operator==(const RangesAndEntity& left, const RangesAndEntity& right) {
        return *left.row_range_ == *right.row_range_ && *left.col_range_ == *right.col_range_ && left.timestamp_range_ == right.timestamp_range_;
    }

    bool operator!=(const RangesAndEntity& right) const {
        return !(*this == right);
    }

    EntityId id_;
    std::shared_ptr<RowRange> row_range_;
    std::shared_ptr<ColRange> col_range_;
    std::optional<TimestampRange> timestamp_range_;
};

template<typename T>
requires std::is_same_v<T, RangesAndKey> || std::is_same_v<T, RangesAndEntity>
std::vector<std::vector<size_t>> structure_by_row_slice(
        std::vector<T>& ranges) {
    std::sort(std::begin(ranges), std::end(ranges), [] (const T& left, const T& right) {
        return std::tie(left.row_range().first, left.col_range().first) < std::tie(right.row_range().first, right.col_range().first);
    });

    std::vector<std::vector<size_t>> res;
    RowRange previous_row_range{-1, -1};
    for (const auto& [idx, ranges_and_key]: folly::enumerate(ranges)) {
        RowRange current_row_range{ranges_and_key.row_range()};
        if (current_row_range != previous_row_range) {
            res.emplace_back();
        }
        res.back().emplace_back(idx);
        previous_row_range = current_row_range;
    }
    return res;
}

template<ResampleBoundary closed_boundary>
bool index_range_outside_bucket_range(timestamp start_index, timestamp end_index, const std::vector<timestamp>& bucket_boundaries) {
    if constexpr (closed_boundary == ResampleBoundary::LEFT) {
        return start_index >= bucket_boundaries.back() || end_index < bucket_boundaries.front();
    } else {
        // closed_boundary == ResampleBoundary::RIGHT
        return start_index > bucket_boundaries.back() || end_index <= bucket_boundaries.front();
    }
}

// Advances the bucket boundary iterator to the end of the last bucket that includes a value from a row slice with the given last index value
template<ResampleBoundary closed_boundary>
void advance_boundary_past_value(const std::vector<timestamp>& bucket_boundaries,
                                 std::vector<timestamp>::const_iterator& bucket_boundaries_it,
                                 timestamp value) {
    // These loops are equivalent to bucket_boundaries_it = std::upper_bound(bucket_boundaries_it, bucket_boundaries.end(), value, std::less[_equal]{})
    // but optimised for the case where most buckets are non-empty.
    // Mathematically, this will be faster when b / log_2(b) < n, where b is the number of buckets and n is the number of index values
    // Even if n is only 1000, this corresponds to 7/8 buckets being empty, rising to 19/20 for n=100,000
    // Experimentally, this implementation is around 10x faster when every bucket contains values, and 3x slower when 99.9% of buckets are empty
    // If we wanted to speed this up when most buckets are empty, we could make this method adaptive to the number of buckets and rows
    if constexpr(closed_boundary == ResampleBoundary::LEFT) {
        while(bucket_boundaries_it != bucket_boundaries.end() && *bucket_boundaries_it <= value) {
            ++bucket_boundaries_it;
        }
    } else {
        // closed_boundary == ResampleBoundary::RIGHT
        while(bucket_boundaries_it != bucket_boundaries.end() && *bucket_boundaries_it < value) {
            ++bucket_boundaries_it;
        }
    }
}

template<ResampleBoundary closed_boundary, typename T>
requires std::is_same_v<T, RangesAndKey> || std::is_same_v<T, RangesAndEntity>
std::vector<std::vector<size_t>> structure_by_time_bucket(
    std::vector<T>& ranges,
    const std::vector<timestamp>& bucket_boundaries);

std::vector<std::vector<EntityId>> structure_by_row_slice(ComponentManager& component_manager, std::vector<std::vector<EntityId>>&& entity_ids_vec);

std::vector<std::vector<EntityId>> offsets_to_entity_ids(const std::vector<std::vector<size_t>>& offsets,
                                                         const std::vector<RangesAndEntity>& ranges_and_entities);

/*
 * On entry to a clause, construct ProcessingUnits from the input entity IDs. These will be provided by the
 * structure_for_processing method for the first clause in the pipeline and for clauses whose expected input structure
 * differs from the previous clause's output structure. Otherwise, these come directly from the previous clause.
 */
template<class... Args>
ProcessingUnit gather_entities(ComponentManager& component_manager, const std::vector<EntityId>& entity_ids) {
    ProcessingUnit res;
    auto components = component_manager.get_entities_and_decrement_refcount<Args...>(entity_ids);
    ([&]{
        auto component = std::move(std::get<std::vector<Args>>(components));
        if constexpr (std::is_same_v<Args, std::shared_ptr<SegmentInMemory>>) {
            res.set_segments(std::move(component));
        } else if constexpr (std::is_same_v<Args, std::shared_ptr<RowRange>>) {
            res.set_row_ranges(std::move(component));
        } else if constexpr (std::is_same_v<Args, std::shared_ptr<ColRange>>) {
            res.set_col_ranges(std::move(component));
        } else if constexpr (std::is_same_v<Args, std::shared_ptr<AtomKey>>) {
            res.set_atom_keys(std::move(component));
        } else if constexpr (std::is_same_v<Args, EntityFetchCount>) {
            res.set_entity_fetch_count(std::move(component));
        } else {
            static_assert(sizeof(Args) == 0, "Unexpected component type provided in gather_entities");
        }
    }(), ...);
    return res;
}
std::vector<EntityId> flatten_entities(std::vector<std::vector<EntityId>>&& entity_ids_vec);

using FutureOrSplitter = std::variant<folly::Future<pipelines::SegmentAndSlice>, folly::FutureSplitter<pipelines::SegmentAndSlice>>;

std::vector<FutureOrSplitter> split_futures(
    std::vector<folly::Future<pipelines::SegmentAndSlice>>&& segment_and_slice_futures,
    std::vector<EntityFetchCount>& segment_fetch_counts);

std::vector<EntityId> push_entities(
    ComponentManager& component_manager,
    ProcessingUnit&& proc,
    EntityFetchCount entity_fetch_count=1);

std::shared_ptr<std::vector<EntityFetchCount>> generate_segment_fetch_counts(
    const std::vector<std::vector<size_t>>& processing_unit_indexes,
    size_t num_segments);

// Multi-symbol join utilities
enum class JoinType: uint8_t {
    OUTER,
    INNER
};

std::pair<StreamDescriptor, proto::descriptors::NormalizationMetadata> join_indexes(std::vector<OutputSchema>& input_schemas);

IndexDescriptorImpl generate_index_descriptor(const std::vector<OutputSchema>& input_schemas);

std::unordered_set<size_t> add_index_fields(StreamDescriptor& stream_desc, std::vector<OutputSchema>& input_schemas);

proto::descriptors::NormalizationMetadata generate_norm_meta(const std::vector<OutputSchema>& input_schemas,
                                                              std::unordered_set<size_t>&& non_matching_name_indices);

void inner_join(StreamDescriptor& stream_desc, std::vector<OutputSchema>& input_schemas);

void outer_join(StreamDescriptor& stream_desc, std::vector<OutputSchema>& input_schemas);

} //namespace arcticdb
