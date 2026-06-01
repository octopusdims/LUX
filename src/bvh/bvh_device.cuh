#pragma once

#ifndef LUX_BVH_DEVICE_CUH
#define LUX_BVH_DEVICE_CUH

#include "core/aabb.cuh"
#include "core/ray.cuh"
#include "core/types.h"
#include "geometry/triangle.h"
#include "geometry/triangle_mesh_view.h"

struct alignas(32) GpuBvhNode {
    AABB bounds;
    int primitive_id = -1;   // -1 = internal node, >=0 = leaf primitive index
    int escape = -1;         // next node on miss / after finishing subtree
};
static_assert(sizeof(GpuBvhNode) == 32, "GpuBvhNode must stay one 32-byte record");

struct GpuBvhView {
    const GpuBvhNode* nodes = nullptr;
    int node_count = 0;
    AABB root_bounds;
};

namespace lux_bvh_device_detail {

struct TriangleArrayProvider {
    const Triangle* triangles = nullptr;

    LuxDeviceInline Triangle operator()(int primitive_id) const {
        return triangles[primitive_id];
    }
};

struct TriangleMeshProvider {
    TriangleMeshView mesh;

    LuxDeviceInline Triangle operator()(int primitive_id) const {
        return get_triangle(mesh, primitive_id);
    }
};

template <typename TriangleProvider>
LuxDeviceInline bool intersect_bvh_device_impl(GpuBvhView bvh,
                                               TriangleProvider triangle_provider,
                                               Ray ray,
                                               int& hit_primitive_id,
                                               Float& hit_t,
                                               Float& hit_u,
                                               Float& hit_v,
                                               int ignored_primitive_id) {
    int node_index = 0;
    bool found = false;
    hit_primitive_id = -1;
    hit_t = ray.t_max;
    hit_u = 0;
    hit_v = 0;

    while (node_index >= 0 && node_index < bvh.node_count) {
        const GpuBvhNode& node = bvh.nodes[node_index];
        Float t_near, t_far;
        if (!node.bounds.intersect(ray, t_near, t_far)) {
            node_index = node.escape;
            continue;
        }

        if (node.primitive_id >= 0) {
            if (node.primitive_id == ignored_primitive_id) {
                node_index = node.escape;
                continue;
            }

            Float t, u, v;
            Triangle triangle = triangle_provider(node.primitive_id);
            if (intersect_triangle(ray, triangle, t, u, v) && t < hit_t) {
                found = true;
                hit_t = t;
                hit_primitive_id = node.primitive_id;
                hit_u = u;
                hit_v = v;
                ray.t_max = t;
            }
            node_index = node.escape;
        } else {
            node_index += 1;
        }
    }

    return found;
}

template <typename IntersectPrimitive>
LuxDeviceInline bool intersect_bvh_device_custom(GpuBvhView bvh,
                                                 Ray ray,
                                                 IntersectPrimitive intersect_primitive,
                                                 int& hit_primitive_id,
                                                 Float& hit_t,
                                                 Float& hit_u,
                                                 Float& hit_v) {
    int node_index = 0;
    bool found = false;
    hit_primitive_id = -1;
    hit_t = ray.t_max;
    hit_u = 0;
    hit_v = 0;

    while (node_index >= 0 && node_index < bvh.node_count) {
        const GpuBvhNode& node = bvh.nodes[node_index];
        Float t_near, t_far;
        if (!node.bounds.intersect(ray, t_near, t_far)) {
            node_index = node.escape;
            continue;
        }

        if (node.primitive_id >= 0) {
            Float t = ray.t_max;
            Float u = 0;
            Float v = 0;
            if (intersect_primitive(node.primitive_id, ray, t, u, v) && t < hit_t) {
                found = true;
                hit_t = t;
                hit_primitive_id = node.primitive_id;
                hit_u = u;
                hit_v = v;
                ray.t_max = t;
            }
            node_index = node.escape;
        } else {
            node_index += 1;
        }
    }

    return found;
}

} // namespace lux_bvh_device_detail

// Device-inline: any kernel can call this.
LuxDeviceInline bool intersect_bvh_device(const GpuBvhNode* nodes,
                                           int node_count,
                                           const Triangle* triangles,
                                           Ray ray,
                                           int& hit_primitive_id,
                                           Float& hit_t,
                                           Float& hit_u,
                                           Float& hit_v,
                                           int ignored_primitive_id = -1) {
    return lux_bvh_device_detail::intersect_bvh_device_impl(
        GpuBvhView{nodes, node_count, AABB{}},
        lux_bvh_device_detail::TriangleArrayProvider{triangles},
        ray, hit_primitive_id, hit_t, hit_u, hit_v, ignored_primitive_id);
}

// Convenience wrapper that unpacks GpuBvhView.
LuxDeviceInline bool intersect_bvh_device(GpuBvhView bvh,
                                           const Triangle* triangles,
                                           Ray ray,
                                           int& hit_primitive_id,
                                           Float& hit_t,
                                           Float& hit_u,
                                           Float& hit_v,
                                           int ignored_primitive_id = -1) {
    return lux_bvh_device_detail::intersect_bvh_device_impl(
        bvh, lux_bvh_device_detail::TriangleArrayProvider{triangles},
        ray, hit_primitive_id, hit_t, hit_u, hit_v, ignored_primitive_id);
}

LuxDeviceInline bool intersect_bvh_device(GpuBvhView bvh,
                                           TriangleMeshView mesh,
                                           Ray ray,
                                           int& hit_primitive_id,
                                           Float& hit_t,
                                           Float& hit_u,
                                           Float& hit_v,
                                           int ignored_primitive_id = -1) {
    return lux_bvh_device_detail::intersect_bvh_device_impl(
        bvh, lux_bvh_device_detail::TriangleMeshProvider{mesh},
        ray, hit_primitive_id, hit_t, hit_u, hit_v, ignored_primitive_id);
}

#endif // LUX_BVH_DEVICE_CUH
