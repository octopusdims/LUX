#pragma once

#ifndef LUX_BVH_VALIDATION_H
#define LUX_BVH_VALIDATION_H

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "bvh/cpu_bvh.h"
#include "bvh/bvh_overlap.cuh"
#include "bvh/gpu_bvh.cuh"
#include "camera/camera.h"
#include "core/constants.h"
#include "core/ray.cuh"
#include "core/types.h"
#include "scene/scene_intersection.h"

namespace bvh_validation_detail {

LuxInline Float rand_float(std::mt19937& rng) {
    return std::generate_canonical<Float, 24>(rng);
}

LuxInline vec3 safe_normalize(const vec3& value) {
    Float len = length(value);
    return len > 0 ? value / len : value;
}

LuxInline Ray validation_ray(int index, int ray_count, std::mt19937& rng,
                              const Camera& camera) {
    if (index < ray_count / 2) {
        return generate_ray(camera, rand_float(rng), rand_float(rng));
    }

    vec3 origin(rand_float(rng) * 2.8f - 1.4f,
                rand_float(rng) * 2.8f - 0.4f,
                rand_float(rng) * 3.2f - 0.4f);
    vec3 direction;
    do {
        direction = vec3(rand_float(rng) * 2.0f - 1.0f,
                         rand_float(rng) * 2.0f - 1.0f,
                         rand_float(rng) * 2.0f - 1.0f);
    } while (length2(direction) <= 1e-8f);
    return Ray(origin, safe_normalize(direction), kRayEpsilon);
}

LuxInline std::vector<Triangle> collect_scene_triangles_for_gpu(const Scene& scene) {
    std::vector<Triangle> triangles;
    triangles.reserve(scene_triangle_count(scene));
    for (int triangle_id = 0; triangle_id < scene_triangle_count(scene); ++triangle_id) {
        triangles.push_back(scene_triangle_geometry(scene, triangle_id));
    }
    return triangles;
}

LuxInline bool pair_less(const PrimitivePair& a, const PrimitivePair& b) {
    return a.a == b.a ? a.b < b.b : a.a < b.a;
}

LuxInline bool gpu_hit_is_valid_tie(const Scene& scene, const Ray& ray,
                                     int brute_triangle_id, int gpu_triangle_id,
                                     Float reference_t, Float tolerance) {
    if (gpu_triangle_id < 0
        || gpu_triangle_id >= scene_triangle_count(scene)) {
        return false;
    }

    Float brute_t, brute_u, brute_v;
    Float gpu_t, gpu_u, gpu_v;
    if (!intersect_triangle(
            ray, scene_triangle_geometry(scene, brute_triangle_id), brute_t, brute_u, brute_v)) {
        return false;
    }
    if (!intersect_triangle(
            ray, scene_triangle_geometry(scene, gpu_triangle_id), gpu_t, gpu_u, gpu_v)) {
        return false;
    }

    return fabsf(brute_t - reference_t) <= tolerance
        && fabsf(gpu_t - reference_t) <= tolerance
        && fabsf(brute_t - gpu_t) <= tolerance;
}

} // namespace bvh_validation_detail

LuxInline bool validate_bvh_intersections(const Scene& scene, const CpuBvh& bvh,
                                       const Camera& camera) {
    constexpr int kRayCount = 2048;
    constexpr Float kTolerance = 1e-4f;

    std::mt19937 rng(12345);
    for (int i = 0; i < kRayCount; ++i) {
        Ray ray = bvh_validation_detail::validation_ray(i, kRayCount, rng, camera);

        SurfaceHit brute_hit;
        SurfaceHit bvh_hit;
        bool brute_found = intersect_scene_bruteforce(scene, ray, brute_hit);
        bool stack_overflow = false;
        bool bvh_found = intersect_scene_bvh(scene, bvh, ray, bvh_hit, &stack_overflow);

        if (stack_overflow) {
            std::fprintf(stderr, "BVH validation failed: traversal stack overflow at ray %d\n", i);
            return false;
        }

        if (brute_found != bvh_found) {
            std::fprintf(stderr,
                         "BVH validation failed: hit/miss mismatch at ray %d "
                         "(brute=%d, bvh=%d)\n",
                         i, brute_found ? 1 : 0, bvh_found ? 1 : 0);
            return false;
        }

        if (!brute_found) continue;

        if (fabsf(brute_hit.t - bvh_hit.t) > kTolerance) {
            std::fprintf(stderr,
                         "BVH validation failed: hit mismatch at ray %d "
                         "(brute tri=%d t=%g, bvh tri=%d t=%g)\n",
                         i, brute_hit.triangle_id, double(brute_hit.t),
                         bvh_hit.triangle_id, double(bvh_hit.t));
            return false;
        }
    }

    return true;
}

LuxInline bool validate_gpu_bvh_intersections(const Scene& scene, const GpuBvh& bvh,
                                           const Camera& camera) {
    constexpr int kRayCount = 2048;
    constexpr Float kTolerance = 1e-4f;

    std::vector<Ray> rays;
    rays.reserve(kRayCount);
    std::mt19937 rng(12345);
    for (int i = 0; i < kRayCount; ++i) {
        rays.push_back(bvh_validation_detail::validation_ray(i, kRayCount, rng, camera));
    }

    std::vector<Triangle> triangles = bvh_validation_detail::collect_scene_triangles_for_gpu(scene);
    std::vector<GpuRayHit> gpu_hits = intersect_gpu_bvh_stackless(bvh, triangles, rays);

    for (int i = 0; i < kRayCount; ++i) {
        SurfaceHit brute_hit;
        bool brute_found = intersect_scene_bruteforce(scene, rays[i], brute_hit);
        bool gpu_found = gpu_hits[i].primitive_id >= 0;

        if (brute_found != gpu_found) {
            std::fprintf(stderr,
                         "GPU BVH validation failed: hit/miss mismatch at ray %d "
                         "(brute=%d, gpu=%d)\n",
                         i, brute_found ? 1 : 0, gpu_found ? 1 : 0);
            return false;
        }

        if (!brute_found) continue;

        if (fabsf(brute_hit.t - gpu_hits[i].t) > kTolerance) {
            std::fprintf(stderr,
                         "GPU BVH validation failed: hit mismatch at ray %d "
                         "(brute tri=%d t=%g, gpu tri=%d t=%g)\n",
                         i, brute_hit.triangle_id, double(brute_hit.t),
                         gpu_hits[i].primitive_id, double(gpu_hits[i].t));
            return false;
        }

        if (brute_hit.triangle_id != gpu_hits[i].primitive_id) {
            if (bvh_validation_detail::gpu_hit_is_valid_tie(
                    scene, rays[i], brute_hit.triangle_id, gpu_hits[i].primitive_id,
                    brute_hit.t, kTolerance)) {
                std::fprintf(stderr,
                             "GPU BVH validation tie at ray %d "
                             "(brute tri=%d, gpu tri=%d, t=%g)\n",
                             i, brute_hit.triangle_id, gpu_hits[i].primitive_id,
                             double(brute_hit.t));
                continue;
            }

            std::fprintf(stderr,
                         "GPU BVH validation failed: primitive mismatch at ray %d "
                         "(brute tri=%d t=%g, gpu tri=%d t=%g)\n",
                         i, brute_hit.triangle_id, double(brute_hit.t),
                         gpu_hits[i].primitive_id, double(gpu_hits[i].t));
            return false;
        }
    }

    return true;
}

LuxInline bool validate_gpu_bvh_self_overlaps(const Scene& scene, const GpuBvh& bvh) {
    std::vector<PrimitivePair> expected;
    for (int i = 0; i < scene_triangle_count(scene); ++i) {
        AABB a = triangle_bounds(scene_triangle_geometry(scene, i));
        for (int j = i + 1; j < scene_triangle_count(scene); ++j) {
            AABB b = triangle_bounds(scene_triangle_geometry(scene, j));
            if (a.overlaps(b)) expected.push_back(PrimitivePair{i, j});
        }
    }

    OverlapQueryResult gpu_result = query_aabb_overlaps_self(
        bvh, std::max(1, static_cast<int>(expected.size()) + 16));
    if (gpu_result.truncated) {
        std::fprintf(stderr, "GPU BVH overlap validation failed: result buffer truncated\n");
        return false;
    }

    std::sort(expected.begin(), expected.end(), bvh_validation_detail::pair_less);
    std::sort(gpu_result.pairs.begin(), gpu_result.pairs.end(), bvh_validation_detail::pair_less);

    if (expected.size() != gpu_result.pairs.size()) {
        std::fprintf(stderr,
                     "GPU BVH overlap validation failed: count mismatch "
                     "(cpu=%zu, gpu=%zu, gpu_found=%d)\n",
                     expected.size(), gpu_result.pairs.size(), gpu_result.found_count);
        return false;
    }

    for (int i = 0; i < static_cast<int>(expected.size()); ++i) {
        if (expected[i].a == gpu_result.pairs[i].a && expected[i].b == gpu_result.pairs[i].b) {
            continue;
        }
        std::fprintf(stderr,
                     "GPU BVH overlap validation failed: pair mismatch at %d "
                     "(cpu=%d,%d gpu=%d,%d)\n",
                     i, expected[i].a, expected[i].b,
                     gpu_result.pairs[i].a, gpu_result.pairs[i].b);
        return false;
    }

    return true;
}

#endif // LUX_BVH_VALIDATION_H
