#pragma once

#ifndef LUX_BVH_CPU_LBVH_BUILDER_H
#define LUX_BVH_CPU_LBVH_BUILDER_H

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include "bvh_build_input.h"
#include "cpu_bvh.h"
#include "core/types.h"

namespace lbvh_detail {

constexpr uint32_t kLeafMask = 0x80000000u;
constexpr uint32_t kIndexMask = 0x7fffffffu;

struct MortonPrimitive {
    BvhBuildPrimitive primitive;
    uint32_t morton_code = 0;
};

struct BuildNode {
    uint32_t parent = 0;
    uint32_t left = 0;
    uint32_t right = 0;
    uint32_t fence = 0;
};

struct IndexRange {
    int first = 0;
    int last = 0;
};

LuxInline uint32_t expand_bits_10(uint32_t value) {
    value &= 0x000003ffu;
    value = (value * 0x00010001u) & 0xff0000ffu;
    value = (value * 0x00000101u) & 0x0f00f00fu;
    value = (value * 0x00000011u) & 0xc30c30c3u;
    value = (value * 0x00000005u) & 0x49249249u;
    return value;
}

LuxInline uint32_t morton_encode_3d(uint32_t x, uint32_t y, uint32_t z) {
    return (expand_bits_10(x) << 2) | (expand_bits_10(y) << 1) | expand_bits_10(z);
}

LuxInline uint32_t quantize_morton_coord(Float value) {
    value = fmaxf(Float(0), fminf(Float(1023), value * Float(1024)));
    return static_cast<uint32_t>(value);
}

LuxInline uint32_t centroid_to_morton(const AABB& scene_bounds, const vec3& centroid) {
    vec3 offset = scene_bounds.normalized_offset(centroid);
    return morton_encode_3d(
        quantize_morton_coord(offset.x),
        quantize_morton_coord(offset.y),
        quantize_morton_coord(offset.z));
}

LuxInline uint64_t merged_key(uint32_t morton_code, int sorted_index) {
    return (uint64_t(morton_code) << 32u) | uint64_t(uint32_t(sorted_index));
}

LuxInline int count_leading_zeros64(uint64_t value) {
    if (value == 0) return 64;
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_clzll(value);
#else
    int count = 0;
    for (int bit = 63; bit >= 0; --bit) {
        if ((value >> bit) & 1u) break;
        ++count;
    }
    return count;
#endif
}

LuxInline int common_prefix(const std::vector<MortonPrimitive>& primitives, int lhs, int rhs) {
    uint64_t lhs_key = merged_key(primitives[lhs].morton_code, lhs);
    uint64_t rhs_key = merged_key(primitives[rhs].morton_code, rhs);
    return count_leading_zeros64(lhs_key ^ rhs_key);
}

LuxInline IndexRange determine_range(const std::vector<MortonPrimitive>& primitives, int idx) {
    int num_primitives = static_cast<int>(primitives.size());
    if (idx == 0) return IndexRange{0, num_primitives - 1};

    int left_delta = common_prefix(primitives, idx, idx - 1);
    int right_delta = common_prefix(primitives, idx, idx + 1);
    int direction = (right_delta > left_delta) ? 1 : -1;
    int min_delta = std::min(left_delta, right_delta);

    int max_length = 2;
    while (true) {
        int candidate = idx + direction * max_length;
        if (candidate < 0 || candidate >= num_primitives) break;
        if (common_prefix(primitives, idx, candidate) <= min_delta) break;
        max_length <<= 1;
    }

    int length = 0;
    for (int step = max_length >> 1; step > 0; step >>= 1) {
        int candidate = idx + (length + step) * direction;
        if (candidate < 0 || candidate >= num_primitives) continue;
        if (common_prefix(primitives, idx, candidate) > min_delta) {
            length += step;
        }
    }

    int other = idx + length * direction;
    return direction < 0 ? IndexRange{other, idx} : IndexRange{idx, other};
}

LuxInline int find_split(const std::vector<MortonPrimitive>& primitives, int first, int last) {
    int node_delta = common_prefix(primitives, first, last);
    int split = first;
    int stride = last - first;

    do {
        stride = (stride + 1) >> 1;
        int middle = split + stride;
        if (middle < last && common_prefix(primitives, first, middle) > node_delta) {
            split = middle;
        }
    } while (stride > 1);

    return split;
}

LuxInline int final_node_index(uint32_t child, int num_internal_nodes) {
    uint32_t index = child & kIndexMask;
    return (child & kLeafMask) ? num_internal_nodes + static_cast<int>(index)
                               : static_cast<int>(index);
}

LuxInline AABB fill_internal_bounds(CpuBvhTree& bvh,
                                     const std::vector<BuildNode>& build_nodes,
                                     int build_node_index,
                                     int num_internal_nodes) {
    const BuildNode& build_node = build_nodes[build_node_index];
    int left = final_node_index(build_node.left, num_internal_nodes);
    int right = final_node_index(build_node.right, num_internal_nodes);

    bvh.nodes[build_node_index].left = left;
    bvh.nodes[build_node_index].right = right;

    AABB left_bounds = (build_node.left & kLeafMask)
        ? bvh.nodes[left].bounds
        : fill_internal_bounds(bvh, build_nodes, left, num_internal_nodes);
    AABB right_bounds = (build_node.right & kLeafMask)
        ? bvh.nodes[right].bounds
        : fill_internal_bounds(bvh, build_nodes, right, num_internal_nodes);

    bvh.nodes[build_node_index].bounds = union_aabb(left_bounds, right_bounds);
    return bvh.nodes[build_node_index].bounds;
}

} // namespace lbvh_detail

LuxInline CpuBvhTree build_cpu_lbvh_tree(std::vector<BvhBuildPrimitive> primitives) {
    CpuBvhTree bvh;
    if (primitives.empty()) return bvh;

    int max_primitive_id = -1;
    for (const BvhBuildPrimitive& primitive : primitives) {
        max_primitive_id = std::max(max_primitive_id, primitive.primitive_id);
    }
    if (max_primitive_id >= 0) {
        bvh.primitive_refs.assign(
            static_cast<size_t>(max_primitive_id) + 1, invalid_primitive_ref());
        for (const BvhBuildPrimitive& primitive : primitives) {
            if (primitive.primitive_id < 0) continue;
            bvh.primitive_refs[primitive.primitive_id] = PrimitiveRef{
                primitive.instance_id,
                primitive.mesh_id,
                primitive.local_primitive_id,
                primitive.primitive_kind
            };
        }
    }

    if (primitives.size() == 1) {
        bvh.root_node = 0;
        bvh.primitive_ids.push_back(primitives[0].primitive_id);
        bvh.nodes.push_back(CpuBvhNode{
            primitives[0].bounds,
            -1,
            -1,
            0,
            1
        });
        return bvh;
    }

    AABB scene_bounds;
    for (const BvhBuildPrimitive& primitive : primitives) {
        scene_bounds.expand(primitive.bounds);
    }

    std::vector<lbvh_detail::MortonPrimitive> sorted_primitives;
    sorted_primitives.reserve(primitives.size());
    for (const BvhBuildPrimitive& primitive : primitives) {
        sorted_primitives.push_back(lbvh_detail::MortonPrimitive{
            primitive,
            lbvh_detail::centroid_to_morton(scene_bounds, primitive.centroid)
        });
    }

    std::stable_sort(sorted_primitives.begin(), sorted_primitives.end(),
                     [](const lbvh_detail::MortonPrimitive& a,
                        const lbvh_detail::MortonPrimitive& b) {
                         return a.morton_code < b.morton_code;
                     });

    int num_primitives = static_cast<int>(sorted_primitives.size());
    int num_internal_nodes = num_primitives - 1;
    std::vector<lbvh_detail::BuildNode> build_nodes(num_internal_nodes);

    for (int node_index = 0; node_index < num_internal_nodes; ++node_index) {
        lbvh_detail::IndexRange range = lbvh_detail::determine_range(sorted_primitives, node_index);
        build_nodes[node_index].fence = (node_index == range.first)
            ? static_cast<uint32_t>(range.last)
            : static_cast<uint32_t>(range.first);

        int split = lbvh_detail::find_split(sorted_primitives, range.first, range.last);

        if (range.first == split) {
            build_nodes[node_index].left = lbvh_detail::kLeafMask | static_cast<uint32_t>(split);
        } else {
            build_nodes[node_index].left = static_cast<uint32_t>(split);
            build_nodes[split].parent = static_cast<uint32_t>(node_index);
        }

        if (range.last == split + 1) {
            build_nodes[node_index].right = lbvh_detail::kLeafMask | static_cast<uint32_t>(split + 1);
        } else {
            build_nodes[node_index].right = static_cast<uint32_t>(split + 1);
            build_nodes[split + 1].parent = static_cast<uint32_t>(node_index);
        }
    }

    bvh.root_node = 0;
    bvh.nodes.resize(num_primitives * 2 - 1);
    bvh.primitive_ids.reserve(num_primitives);

    for (int leaf_index = 0; leaf_index < num_primitives; ++leaf_index) {
        int node_index = num_internal_nodes + leaf_index;
        bvh.primitive_ids.push_back(sorted_primitives[leaf_index].primitive.primitive_id);
        bvh.nodes[node_index].bounds = sorted_primitives[leaf_index].primitive.bounds;
        bvh.nodes[node_index].first_primitive = leaf_index;
        bvh.nodes[node_index].primitive_count = 1;
    }

    lbvh_detail::fill_internal_bounds(bvh, build_nodes, 0, num_internal_nodes);
    return bvh;
}

#endif // LUX_BVH_CPU_LBVH_BUILDER_H
