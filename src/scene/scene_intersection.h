#pragma once

#ifndef LUX_SCENE_INTERSECTION_H
#define LUX_SCENE_INTERSECTION_H

#include <limits>

#include "bvh/cpu_bvh_traversal.h"
#include "core/ray.cuh"
#include "core/types.h"
#include "geometry/triangle.h"
#include "scene.h"

struct SurfaceHit {
    Float t = std::numeric_limits<Float>::infinity();
    int triangle_id = -1;
    PrimitiveRef primitive_ref;
    PrimitiveKind primitive_kind = PrimitiveKind::None;
    vec3 position;
    vec3 ng;
    vec3 ns;
    Float u = 0;  // barycentric
    Float v = 0;  // barycentric
};

LuxHDInline vec3 face_forward(const vec3& normal, const vec3& reference) {
    return dot(normal, reference) < 0 ? -normal : normal;
}

LuxInline bool intersect_scene_bruteforce(const Scene& scene, const Ray& ray, SurfaceHit& hit) {
    bool found = false;
    for (int i = 0; i < scene_triangle_count(scene); ++i) {
        Float t, u, v;
        Triangle triangle = scene_triangle_geometry(scene, i);
        if (!intersect_triangle(ray, triangle, t, u, v)) continue;
        if (t >= hit.t) continue;

        found = true;
        hit.t = t;
        hit.triangle_id = i;
        hit.primitive_ref = scene_primitive_ref(scene, i);
        hit.primitive_kind = hit.primitive_ref.kind;
        hit.u = u;
        hit.v = v;
        hit.position = ray.at(t);
        SceneTriangle scene_triangle = scene_triangle_view(scene, i);
        hit.ng = triangle_normal(scene_triangle.triangle);
        hit.ns = interpolate_triangle_normal(scene_triangle, u, v);
        if (dot(hit.ng, hit.ns) < 0) hit.ns = -hit.ns;
    }
    return found;
}

LuxInline bool finish_surface_hit_from_triangle(
        const Scene& scene, const Ray& ray, int primitive_id,
        Float hit_t, Float hit_u, Float hit_v, SurfaceHit& hit) {
    if (primitive_id < 0 || primitive_id >= scene_triangle_count(scene)) {
        return false;
    }

    PrimitiveRef ref = scene_primitive_ref(scene, primitive_id);
    if (!scene_primitive_ref_exists(scene, ref)) {
        return false;
    }

    hit.t = hit_t;
    hit.triangle_id = primitive_id;
    hit.primitive_ref = ref;
    hit.primitive_kind = hit.primitive_ref.kind;
    hit.u = hit_u;
    hit.v = hit_v;
    hit.position = ray.at(hit_t);
    SceneTriangle scene_triangle = scene_triangle_view(scene, primitive_id);
    hit.ng = triangle_normal(scene_triangle.triangle);
    hit.ns = interpolate_triangle_normal(scene_triangle, hit_u, hit_v);
    if (dot(hit.ng, hit.ns) < 0) hit.ns = -hit.ns;
    return true;
}

LuxInline bool finish_surface_hit_from_ref(
        const Scene& scene, const Ray& ray, const PrimitiveRef& ref,
        Float hit_t, Float hit_u, Float hit_v, SurfaceHit& hit) {
    if (!scene_primitive_ref_exists(scene, ref)) {
        return false;
    }

    hit.t = hit_t;
    hit.triangle_id = scene_primitive_index(scene, ref);
    hit.primitive_ref = ref;
    hit.primitive_kind = ref.kind;
    hit.u = hit_u;
    hit.v = hit_v;
    hit.position = ray.at(hit_t);
    SceneTriangle scene_triangle = scene_triangle_view(scene, ref);
    hit.ng = triangle_normal(scene_triangle.triangle);
    hit.ns = interpolate_triangle_normal(scene_triangle, hit_u, hit_v);
    if (dot(hit.ng, hit.ns) < 0) hit.ns = -hit.ns;
    return true;
}

LuxInline bool intersect_scene_triangle_primitive(
        const Triangle& triangle, const Ray& ray, CpuBvhPrimitiveHit& primitive_hit) {
    Float u = 0;
    Float v = 0;
    bool ok = intersect_triangle(ray, triangle, primitive_hit.t, u, v);
    if (ok) {
        primitive_hit.u = u;
        primitive_hit.v = v;
    }
    return ok;
}

LuxInline bool intersect_scene_mesh_primitive(
        const SceneMesh& mesh, const PrimitiveRef& ref,
        const Ray& ray, CpuBvhPrimitiveHit& primitive_hit) {
    switch (ref.kind) {
        case PrimitiveKind::Triangle:
            if (ref.local_primitive_id < 0
                || ref.local_primitive_id >= scene_mesh_triangle_count(mesh)) {
                return false;
            }
            primitive_hit.primitive_id = ref.local_primitive_id;
            return intersect_scene_triangle_primitive(
                scene_mesh_triangle_geometry(mesh, ref.local_primitive_id),
                ray, primitive_hit);
        case PrimitiveKind::Sphere:
        case PrimitiveKind::Curve:
        case PrimitiveKind::Volume:
        case PrimitiveKind::None:
            return false;
    }
    return false;
}

LuxInline bool scene_bvh_has_two_level(const Scene& scene, const CpuBvh& bvh) {
    return bvh.has_two_level
        && !bvh.tlas.empty()
        && bvh.blas.size() >= scene.mesh_assets.size()
        && !scene.instances.empty();
}

LuxInline bool should_skip_primitive(const PrimitiveRef& candidate,
                                     const PrimitiveRef& ignored) {
    return primitive_ref_valid(ignored) && same_primitive_ref(candidate, ignored);
}

LuxInline bool intersect_scene_two_level_bvh(
        const Scene& scene, const CpuBvh& bvh, const Ray& ray,
        PrimitiveRef ignored_primitive, SurfaceHit& hit, bool* stack_overflow) {
    Ray bounded_ray = ray;
    bounded_ray.t_max = fminf(bounded_ray.t_max, hit.t);
    int primitive_id = -1;
    Float hit_t = bounded_ray.t_max;
    Float hit_u = 0, hit_v = 0;
    bool any_stack_overflow = false;
    bool tlas_stack_overflow = false;

    bool found = intersect_cpu_bvh_tree(
        bvh.tlas, bounded_ray,
        [&scene, &bvh, ignored_primitive, &any_stack_overflow](
            int instance_id, const Ray& candidate_ray, CpuBvhPrimitiveHit& primitive_hit) {
            if (any_stack_overflow) return false;
            if (instance_id < 0 || instance_id >= static_cast<int>(scene.instances.size())) {
                return false;
            }

            const SceneInstance& instance = scene.instances[instance_id];
            if (instance.mesh_id < 0
                || instance.mesh_id >= static_cast<int>(scene.mesh_assets.size())
                || instance.mesh_id >= static_cast<int>(bvh.blas.size())) {
                return false;
            }

            const CpuBvhTree& mesh_bvh = bvh.blas[instance.mesh_id];
            if (mesh_bvh.empty()) return false;
            const SceneMesh& mesh = scene.mesh_assets[instance.mesh_id];
            Ray object_ray = inverse_transform_ray(instance.object_to_world, candidate_ray);

            int local_primitive_id = -1;
            Float local_hit_t = object_ray.t_max;
            Float local_hit_u = 0;
            Float local_hit_v = 0;
            bool blas_stack_overflow = false;
            bool blas_found = intersect_cpu_bvh_tree(
                mesh_bvh, object_ray,
                [&mesh, &mesh_bvh, &instance, instance_id, ignored_primitive](
                    int local_id,
                    const Ray& local_ray,
                    CpuBvhPrimitiveHit& local_hit) {
                    if (local_id < 0) {
                        return false;
                    }
                    PrimitiveRef local_ref = PrimitiveRef{
                        -1, instance.mesh_id, local_id, PrimitiveKind::Triangle};
                    if (local_id < static_cast<int>(mesh_bvh.primitive_refs.size())
                        && mesh_bvh.primitive_refs[local_id].kind != PrimitiveKind::None
                        && mesh_bvh.primitive_refs[local_id].local_primitive_id >= 0) {
                        local_ref = mesh_bvh.primitive_refs[local_id];
                    }
                    PrimitiveRef candidate_ref{
                        instance_id,
                        instance.mesh_id,
                        local_ref.local_primitive_id >= 0
                            ? local_ref.local_primitive_id
                            : local_id,
                        local_ref.kind};
                    if (should_skip_primitive(candidate_ref, ignored_primitive)) return false;
                    return intersect_scene_mesh_primitive(
                        mesh, candidate_ref, local_ray, local_hit);
                },
                local_primitive_id, local_hit_t, local_hit_u, local_hit_v,
                &blas_stack_overflow);
            if (blas_stack_overflow) {
                any_stack_overflow = true;
                return false;
            }
            if (!blas_found) return false;

            PrimitiveRef candidate_ref{
                instance_id, instance.mesh_id, local_primitive_id,
                PrimitiveKind::Triangle};
            int primitive_id = scene_primitive_index(scene, candidate_ref);
            if (primitive_id < 0) {
                return false;
            }

            primitive_hit.t = local_hit_t;
            primitive_hit.u = local_hit_u;
            primitive_hit.v = local_hit_v;
            primitive_hit.primitive_id = primitive_id;
            return true;
        },
        primitive_id, hit_t, hit_u, hit_v, &tlas_stack_overflow);

    any_stack_overflow = any_stack_overflow || tlas_stack_overflow;
    if (stack_overflow) *stack_overflow = any_stack_overflow;
    if (any_stack_overflow || !found) return false;

    return finish_surface_hit_from_triangle(scene, ray, primitive_id, hit_t, hit_u, hit_v, hit);
}

LuxInline bool intersect_scene_bvh(const Scene& scene, const CpuBvh& bvh, const Ray& ray,
                                SurfaceHit& hit, bool* stack_overflow = nullptr) {
    if (scene_bvh_has_two_level(scene, bvh)) {
        return intersect_scene_two_level_bvh(
            scene, bvh, ray, invalid_primitive_ref(), hit, stack_overflow);
    }
    if (stack_overflow) *stack_overflow = false;
    return false;
}

LuxInline bool intersect_scene_bvh_skip_ref(const Scene& scene, const CpuBvh& bvh,
                                            const Ray& ray, PrimitiveRef ignored_primitive,
                                            SurfaceHit& hit,
                                            bool* stack_overflow = nullptr) {
    if (scene_bvh_has_two_level(scene, bvh)) {
        return intersect_scene_two_level_bvh(
            scene, bvh, ray, ignored_primitive, hit, stack_overflow);
    }
    if (stack_overflow) *stack_overflow = false;
    return false;
}

LuxInline bool intersect_scene_bvh_skip(const Scene& scene, const CpuBvh& bvh,
                                        const Ray& ray, int ignored_triangle,
                                        SurfaceHit& hit,
                                        bool* stack_overflow = nullptr) {
    return intersect_scene_bvh_skip_ref(
        scene, bvh, ray, scene_primitive_ref(scene, ignored_triangle), hit, stack_overflow);
}

#endif // LUX_SCENE_INTERSECTION_H
