#pragma once

#ifndef LUX_BVH_OVERLAP_CUH
#define LUX_BVH_OVERLAP_CUH

#include <vector>

#include "bvh/gpu_bvh.cuh"

struct PrimitivePair {
    int a = -1;
    int b = -1;
};

struct OverlapQueryResult {
    std::vector<PrimitivePair> pairs;
    int found_count = 0;
    bool truncated = false;
};

OverlapQueryResult query_aabb_overlaps_self(const GpuBvh& bvh, int max_pairs = 1 << 20);
OverlapQueryResult query_aabb_overlaps_cross(const GpuBvh& left, const GpuBvh& right,
                                             int max_pairs = 1 << 20);

#endif // LUX_BVH_OVERLAP_CUH
