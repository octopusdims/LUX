#pragma once

#ifndef LUX_BVH_CPU_H
#define LUX_BVH_CPU_H

#include <cstddef>
#include <vector>

#include "core/aabb.cuh"
#include "core/types.h"
#include "scene/primitive_ref.h"

// Binary CPU BVH node. Interior nodes use left/right child indices; leaf nodes
// reference a range in CpuBvhTree::primitive_ids.
struct CpuBvhNode {
    AABB bounds;
    int left = -1;
    int right = -1;
    int first_primitive = 0;
    int primitive_count = 0;

    LuxHDInline bool is_leaf() const { return primitive_count > 0; }
};

struct CpuBvhTree {
    std::vector<CpuBvhNode> nodes;
    std::vector<int> primitive_ids;
    std::vector<PrimitiveRef> primitive_refs;
    int root_node = -1;

    LuxInline bool empty() const { return root_node < 0 || nodes.empty(); }
};

struct CpuBvh {
    std::vector<CpuBvhTree> blas;
    CpuBvhTree tlas;
    bool has_two_level = false;
};

struct CpuBvhStats {
    std::size_t tlas_nodes = 0;
    std::size_t tlas_primitives = 0;
    std::size_t blas_count = 0;
    std::size_t blas_nodes = 0;
    std::size_t blas_primitives = 0;
};

LuxInline CpuBvhStats cpu_bvh_stats(const CpuBvh& bvh) {
    CpuBvhStats stats;
    stats.tlas_nodes = bvh.tlas.nodes.size();
    stats.tlas_primitives = bvh.tlas.primitive_ids.size();
    stats.blas_count = bvh.blas.size();
    for (const CpuBvhTree& tree : bvh.blas) {
        stats.blas_nodes += tree.nodes.size();
        stats.blas_primitives += tree.primitive_ids.size();
    }
    return stats;
}

#endif // LUX_BVH_CPU_H
