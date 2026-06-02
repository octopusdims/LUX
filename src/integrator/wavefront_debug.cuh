#pragma once

#ifndef LUX_INTEGRATOR_WAVEFRONT_DEBUG_CUH
#define LUX_INTEGRATOR_WAVEFRONT_DEBUG_CUH

#include "core/constants.h"
#include "core/ray.cuh"
#include "geometry/triangle_mesh_view.h"
#include "integrator/pathstate.cuh"
#include "material/material.h"
#include "scene/gpu_scene_view.h"

struct WavefrontDebugViews {
    vec3* shadow = nullptr;
    vec3* geometric_normal = nullptr;
    vec3* shading_normal = nullptr;
};

namespace lux_wavefront_debug {

constexpr Float kBoundaryBarycentricEpsilon = Float(0.02);

LuxDeviceInline bool is_same_source_material(GpuScene scene,
                                             const ShadowRayWorkItem& item,
                                             PrimitiveRef occluder_ref,
                                             int occluder_id) {
    int material_id = gpu_scene_material_id(scene, occluder_ref);
    return item.source_material_id >= 0
        && occluder_id >= 0
        && material_id == item.source_material_id;
}

LuxDeviceInline bool is_backface_shadow_hit(const Ray& shadow_ray,
                                            const GpuSceneTriangle& occluder) {
    if (!occluder.valid) return false;
    vec3 occluder_normal = triangle_normal(occluder.triangle);
    return dot(occluder_normal, -shadow_ray.direction) <= 0;
}

LuxDeviceInline bool is_near_boundary_edge(const GpuSceneTriangle& triangle,
                                           Float u,
                                           Float v) {
    unsigned char boundary_edges = triangle.boundary_edges;
    if (boundary_edges == 0) return false;

    Float w = Float(1) - u - v;
    bool near_v0_v1 = (boundary_edges & 1u) != 0 && v <= kBoundaryBarycentricEpsilon;
    bool near_v1_v2 = (boundary_edges & 2u) != 0 && w <= kBoundaryBarycentricEpsilon;
    bool near_v2_v0 = (boundary_edges & 4u) != 0 && u <= kBoundaryBarycentricEpsilon;
    return near_v0_v1 || near_v1_v2 || near_v2_v0;
}

LuxDeviceInline vec3 shadow_debug_classification_color(
        const ShadowRayWorkItem& item,
        const Ray& shadow_ray,
        GpuScene scene,
        int occluder_id,
        PrimitiveRef occluder_ref,
        Float hit_t,
        Float hit_u,
        Float hit_v,
        int passthrough_hits) {
    Float shading_cos = dot(item.source_shading_normal, shadow_ray.direction);
    Float geometric_cos = dot(item.source_geometric_normal, shadow_ray.direction);
    if (shading_cos > 0 && geometric_cos <= 0) {
        return vec3(1, 0, 0);
    }

    if (occluder_id < 0) {
        return passthrough_hits > 0 ? vec3(0, 0.35f, 0.8f) : vec3(0.03f);
    }

    bool same_material = is_same_source_material(scene, item, occluder_ref, occluder_id);
    if (same_material && hit_t < Float(16) * kRayEpsilon) {
        return vec3(1, 1, 0);
    }

    if (same_material) {
        GpuSceneTriangle occluder = gpu_scene_triangle(scene, occluder_ref);
        bool backface_hit = is_backface_shadow_hit(shadow_ray, occluder);
        if (backface_hit) {
            return is_near_boundary_edge(occluder, hit_u, hit_v)
                ? vec3(1, 0, 1)
                : vec3(0, 0, 1);
        }
        return vec3(0, 1, 1);
    }

    return vec3(0, 1, 0);
}

LuxDeviceInline vec3 encode_normal_color(const vec3& normal) {
    return Float(0.5) * (normalize(normal) + vec3(1));
}

LuxDeviceInline vec3 primary_normal_debug_color(
        GpuScene scene,
        PrimitiveRef primitive_ref,
        int primitive_id,
        Float hit_u,
        Float hit_v,
        bool shading_normal) {
    if (primitive_id < 0) {
        return vec3(0);
    }

    GpuSceneTriangle triangle = gpu_scene_triangle(scene, primitive_ref);
    if (!triangle.valid) return vec3(0);
    vec3 ng = triangle_normal(triangle.triangle);
    if (!shading_normal) {
        return encode_normal_color(ng);
    }

    vec3 ns = gpu_scene_interpolate_triangle_normal(triangle, hit_u, hit_v);
    if (dot(ng, ns) < 0) {
        ns = -ns;
    }
    return encode_normal_color(ns);
}

} // namespace lux_wavefront_debug

#endif // LUX_INTEGRATOR_WAVEFRONT_DEBUG_CUH
