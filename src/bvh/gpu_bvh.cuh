#pragma once

#ifndef LUX_BVH_GPU_CUH
#define LUX_BVH_GPU_CUH

#include <memory>
#include <string>
#include <vector>

#include "bvh/cpu_bvh.h"
#include "bvh/bvh_device.cuh"
#include "bvh/bvh_build_input.h"
#include "core/aabb.cuh"
#include "core/ray.cuh"
#include "core/types.h"
#include "geometry/triangle.h"

struct GpuBvhStorage;

struct GpuRayHit {
    int primitive_id = -1;
    Float t = INFINITY;
};

struct GpuBvh {
    GpuBvh();
    ~GpuBvh();

    GpuBvh(GpuBvh&& other) noexcept;
    GpuBvh& operator=(GpuBvh&& other) noexcept;

    GpuBvh(const GpuBvh&) = delete;
    GpuBvh& operator=(const GpuBvh&) = delete;

    bool is_valid() const;

    // POD descriptor for passing into device kernels (wavefront path tracing).
    GpuBvhView device_view() const;

    // Raw device-pointer accessors for broadphase/overlap internal use only.
    const AABB* overlap_bounds_device_ptr() const;
    const int* overlap_primitive_ids_device_ptr() const;
    const uint32_t* overlap_sorted_indices_device_ptr() const;

    std::unique_ptr<GpuBvhStorage> storage;
    AABB root_bounds;
    int primitive_count = 0;
};

GpuBvh build_gpu_lbvh(const std::vector<BvhBuildPrimitive>& primitives);
bool refit_gpu_lbvh(GpuBvh& bvh, const std::vector<BvhBuildPrimitive>& primitives);
CpuBvhTree download_bvh_for_validation(const GpuBvh& bvh);
bool validate_gpu_bvh_structure(const GpuBvh& bvh, std::string* error_message = nullptr);
bool validate_gpu_bvh_matches_primitives(
    const GpuBvh& bvh,
    const std::vector<BvhBuildPrimitive>& primitives,
    std::string* error_message = nullptr);

std::vector<GpuRayHit> intersect_gpu_bvh_stackless(const GpuBvh& bvh,
                                                   const std::vector<Triangle>& triangles,
                                                   const std::vector<Ray>& rays);

#endif // LUX_BVH_GPU_CUH
