#pragma once

#ifndef LUX_SCENE_GPU_BVH_CUH
#define LUX_SCENE_GPU_BVH_CUH

#include "bvh/bvh_device.cuh"
#include "core/mat4.cuh"
#include "scene/gpu_scene_view.h"

LuxDeviceInline bool gpu_should_skip_primitive(const PrimitiveRef& candidate,
                                               const PrimitiveRef& ignored) {
    return primitive_ref_valid(ignored) && same_primitive_ref(candidate, ignored);
}

struct GpuMeshPrimitiveIntersector {
    GpuSceneMesh mesh;
    int instance_id = -1;
    int mesh_id = -1;
    PrimitiveRef ignored_primitive;

    LuxDeviceInline bool operator()(int local_primitive_id, const Ray& ray,
                                    Float& t, Float& u, Float& v) const {
        PrimitiveRef ref{
            instance_id, mesh_id, local_primitive_id, PrimitiveKind::Triangle};
        if (gpu_should_skip_primitive(ref, ignored_primitive)) return false;
        switch (ref.kind) {
            case PrimitiveKind::Triangle:
                if (ref.local_primitive_id < 0
                    || ref.local_primitive_id >= mesh.triangle_count) {
                    return false;
                }
                return intersect_triangle(
                    ray, get_triangle(mesh.mesh, ref.local_primitive_id), t, u, v);
            case PrimitiveKind::Sphere:
            case PrimitiveKind::Curve:
            case PrimitiveKind::Volume:
            case PrimitiveKind::None:
                return false;
        }
        return false;
    }
};

LuxDeviceInline bool intersect_scene_bvh_device(GpuBvhView bvh,
                                                GpuScene scene,
                                                Ray ray,
                                                int& hit_primitive_id,
                                                PrimitiveRef& hit_primitive_ref,
                                                Float& hit_t,
                                                Float& hit_u,
                                                Float& hit_v,
                                                PrimitiveRef ignored_primitive =
                                                    invalid_primitive_ref()) {
    if (!scene.has_two_level) {
        (void)bvh;
        (void)scene;
        (void)ray;
        (void)ignored_primitive;
        hit_primitive_id = -1;
        hit_primitive_ref = invalid_primitive_ref();
        hit_t = ray.t_max;
        hit_u = 0;
        hit_v = 0;
        return false;
    }

    int node_index = 0;
    bool found = false;
    hit_primitive_id = -1;
    hit_primitive_ref = invalid_primitive_ref();
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
            int instance_id = node.primitive_id;
            if (instance_id < 0 || instance_id >= scene.instance_count) {
                node_index = node.escape;
                continue;
            }

            const GpuSceneInstance& instance = scene.instances[instance_id];
            if (instance.mesh_id < 0 || instance.mesh_id >= scene.mesh_count
                || instance.blas_id < 0 || instance.blas_id >= scene.blas_count) {
                node_index = node.escape;
                continue;
            }

            const GpuSceneMesh& mesh = scene.meshes[instance.mesh_id];
            GpuBvhView blas = scene.blas_views[instance.blas_id];
            Ray object_ray = inverse_transform_ray(instance.object_to_world, ray);
            object_ray.t_max = hit_t;

            int local_primitive_id = -1;
            Float local_hit_t = object_ray.t_max;
            Float local_hit_u = 0;
            Float local_hit_v = 0;
            bool local_found =
                lux_bvh_device_detail::intersect_bvh_device_custom(
                    blas, object_ray,
                    GpuMeshPrimitiveIntersector{
                        mesh, instance_id, instance.mesh_id, ignored_primitive},
                    local_primitive_id, local_hit_t, local_hit_u, local_hit_v);
            if (local_found && local_hit_t < hit_t) {
                PrimitiveRef local_ref = gpu_scene_ref_from_instance_local(
                    scene, instance_id, local_primitive_id);
                int scene_primitive_id = gpu_scene_primitive_index(scene, local_ref);
                if (scene_primitive_id < 0) {
                    node_index = node.escape;
                    continue;
                }
                found = true;
                hit_t = local_hit_t;
                hit_u = local_hit_u;
                hit_v = local_hit_v;
                hit_primitive_id = scene_primitive_id;
                hit_primitive_ref = local_ref;
                ray.t_max = local_hit_t;
            }
            node_index = node.escape;
        } else {
            node_index += 1;
        }
    }

    return found;
}

LuxDeviceInline bool intersect_scene_bvh_device(GpuBvhView bvh,
                                                GpuScene scene,
                                                Ray ray,
                                                int& hit_primitive_id,
                                                PrimitiveRef& hit_primitive_ref,
                                                Float& hit_t,
                                                Float& hit_u,
                                                Float& hit_v,
                                                int ignored_primitive_id) {
    PrimitiveRef ignored_primitive = gpu_scene_primitive_ref(scene, ignored_primitive_id);
    return intersect_scene_bvh_device(
        bvh, scene, ray, hit_primitive_id, hit_primitive_ref,
        hit_t, hit_u, hit_v, ignored_primitive);
}

LuxDeviceInline bool intersect_scene_bvh_device(GpuBvhView bvh,
                                                GpuScene scene,
                                                Ray ray,
                                                int& hit_primitive_id,
                                                Float& hit_t,
                                                Float& hit_u,
                                                Float& hit_v,
                                                PrimitiveRef ignored_primitive =
                                                    invalid_primitive_ref()) {
    PrimitiveRef hit_primitive_ref;
    return intersect_scene_bvh_device(
        bvh, scene, ray, hit_primitive_id, hit_primitive_ref,
        hit_t, hit_u, hit_v, ignored_primitive);
}

LuxDeviceInline bool intersect_scene_bvh_device(GpuBvhView bvh,
                                                GpuScene scene,
                                                Ray ray,
                                                int& hit_primitive_id,
                                                Float& hit_t,
                                                Float& hit_u,
                                                Float& hit_v,
                                                int ignored_primitive_id) {
    PrimitiveRef hit_primitive_ref;
    PrimitiveRef ignored_primitive = gpu_scene_primitive_ref(scene, ignored_primitive_id);
    return intersect_scene_bvh_device(
        bvh, scene, ray, hit_primitive_id, hit_primitive_ref,
        hit_t, hit_u, hit_v, ignored_primitive);
}

#endif // LUX_SCENE_GPU_BVH_CUH
