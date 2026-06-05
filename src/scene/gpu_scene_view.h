#pragma once

#ifndef LUX_SCENE_GPU_VIEW_H
#define LUX_SCENE_GPU_VIEW_H

#include "core/mat4.cuh"
#include "core/types.h"
#include "camera/camera.h"
#include "geometry/triangle_mesh_view.h"
#include "light/light_distribution.h"
#include "light/light.h"
#include "scene/primitive_ref.h"

struct Material;
struct GpuBvhView;

struct GpuSceneMesh {
    TriangleMeshView mesh;
    const int* material_ids = nullptr;
    const unsigned char* boundary_edges = nullptr;
    int triangle_count = 0;
};

struct GpuSceneInstance {
    int mesh_id = -1;
    int blas_id = -1;
    int material_override = -1;
    int first_primitive = -1;
    Transform object_to_world;
};

struct GpuImageAsset {
    const vec3* pixels = nullptr;
    int width = 0;
    int height = 0;
};

struct GpuSceneTriangle {
    Triangle triangle;
    int material_id = -1;
    vec3 n0 = vec3(0);
    vec3 n1 = vec3(0);
    vec3 n2 = vec3(0);
    vec2 uv0 = vec2(0);
    vec2 uv1 = vec2(0);
    vec2 uv2 = vec2(0);
    unsigned char boundary_edges = 0;
    PrimitiveRef primitive_ref;
    bool valid = false;
};

// POD descriptor for device kernels. All pointers are device-side.
struct GpuScene {
    Camera camera;
    const Material* materials = nullptr;
    int material_count = 0;
    const PrimitiveRef* light_primitive_refs = nullptr;
    const Float* light_primitive_areas = nullptr;
    const Float* light_area_cdf = nullptr;
    const AliasTableEntry* light_area_alias_table = nullptr;
    int light_count = 0;
    Float total_light_area = 0;
    const Light* lights = nullptr;
    const Float* light_power_cdf = nullptr;
    const AliasTableEntry* light_power_alias_table = nullptr;
    int generic_light_count = 0;
    Float total_light_power = 0;
    const LightBvhNode* light_bvh_nodes = nullptr;
    const int* light_bvh_leaf_nodes = nullptr;
    int light_bvh_node_count = 0;
    int light_bvh_root = -1;
    const int* non_bvh_light_indices = nullptr;
    const Float* non_bvh_light_cdf = nullptr;
    const AliasTableEntry* non_bvh_light_alias_table = nullptr;
    int non_bvh_light_count = 0;
    Float non_bvh_light_power = 0;
    const int* primitive_light_indices = nullptr;
    int primitive_light_index_count = 0;
    const GpuSceneMesh* meshes = nullptr;
    int mesh_count = 0;
    const GpuSceneInstance* instances = nullptr;
    int instance_count = 0;
    const PrimitiveRef* primitive_refs = nullptr;
    int primitive_count = 0;
    const GpuBvhView* blas_views = nullptr;
    int blas_count = 0;
    bool has_two_level = false;
    const GpuImageAsset* image_assets = nullptr;
    int image_asset_count = 0;
};

LuxHDInline PrimitiveRef gpu_scene_primitive_ref(GpuScene scene, int triangle_id) {
    if (!scene.primitive_refs || triangle_id < 0 || triangle_id >= scene.primitive_count) {
        return invalid_primitive_ref();
    }
    return scene.primitive_refs[triangle_id];
}

LuxHDInline bool gpu_scene_ref_valid(GpuScene scene, const PrimitiveRef& ref) {
    if (ref.kind != PrimitiveKind::Triangle
        || ref.instance_id < 0 || ref.instance_id >= scene.instance_count
        || ref.mesh_id < 0 || ref.mesh_id >= scene.mesh_count
        || ref.local_primitive_id < 0) {
        return false;
    }
    const GpuSceneInstance& instance = scene.instances[ref.instance_id];
    if (instance.mesh_id != ref.mesh_id) return false;
    const GpuSceneMesh& mesh = scene.meshes[ref.mesh_id];
    return ref.local_primitive_id < mesh.triangle_count;
}

LuxHDInline int gpu_scene_primitive_index(GpuScene scene, const PrimitiveRef& ref) {
    if (!gpu_scene_ref_valid(scene, ref)) return -1;
    const GpuSceneInstance& instance = scene.instances[ref.instance_id];
    if (instance.first_primitive < 0) return -1;
    int primitive_id = instance.first_primitive + ref.local_primitive_id;
    if (primitive_id < 0 || primitive_id >= scene.primitive_count) return -1;
    return primitive_id;
}

LuxHDInline PrimitiveRef gpu_scene_ref_from_instance_local(GpuScene scene,
                                                          int instance_id,
                                                          int local_primitive_id) {
    if (instance_id < 0 || instance_id >= scene.instance_count) {
        return invalid_primitive_ref();
    }
    const GpuSceneInstance& instance = scene.instances[instance_id];
    if (instance.first_primitive < 0 || local_primitive_id < 0) {
        return invalid_primitive_ref();
    }
    return gpu_scene_primitive_ref(scene, instance.first_primitive + local_primitive_id);
}

LuxHDInline int gpu_scene_triangle_count(GpuScene scene) {
    return scene.primitive_count;
}

LuxHDInline vec3 gpu_transform_instance_normal(const Transform& object_to_world,
                                               const vec3& normal) {
    if (length2(normal) == 0) return vec3(0);
    vec3 transformed = transform_normal(object_to_world, normal);
    if (length2(transformed) == 0) return vec3(0);
    return normalize(transformed);
}

LuxHDInline GpuSceneTriangle gpu_scene_triangle_from_ref(GpuScene scene,
                                                        PrimitiveRef ref) {
    GpuSceneTriangle result;
    if (!gpu_scene_ref_valid(scene, ref)) return result;

    const GpuSceneInstance& instance = scene.instances[ref.instance_id];
    if (instance.mesh_id != ref.mesh_id
        || instance.mesh_id < 0 || instance.mesh_id >= scene.mesh_count) {
        return result;
    }

    const GpuSceneMesh& mesh = scene.meshes[instance.mesh_id];
    int local_id = ref.local_primitive_id;
    if (local_id >= mesh.triangle_count || !mesh.mesh.indices || !mesh.mesh.vertices) {
        return result;
    }

    Triangle local_triangle = get_triangle(mesh.mesh, local_id);
    result.triangle = Triangle{
        transform_point(instance.object_to_world, local_triangle.v0),
        transform_point(instance.object_to_world, local_triangle.v1),
        transform_point(instance.object_to_world, local_triangle.v2)};
    result.material_id = instance.material_override >= 0
        ? instance.material_override
        : (mesh.material_ids ? mesh.material_ids[local_id] : -1);

    const TriangleIndices& idx = mesh.mesh.indices[local_id];
    if (mesh.mesh.normals) {
        result.n0 = gpu_transform_instance_normal(
            instance.object_to_world, mesh.mesh.normals[idx.v0]);
        result.n1 = gpu_transform_instance_normal(
            instance.object_to_world, mesh.mesh.normals[idx.v1]);
        result.n2 = gpu_transform_instance_normal(
            instance.object_to_world, mesh.mesh.normals[idx.v2]);
    }
    if (mesh.mesh.uvs) {
        result.uv0 = mesh.mesh.uvs[idx.v0];
        result.uv1 = mesh.mesh.uvs[idx.v1];
        result.uv2 = mesh.mesh.uvs[idx.v2];
    }
    result.boundary_edges = mesh.boundary_edges ? mesh.boundary_edges[local_id] : 0;
    result.primitive_ref = ref;
    result.valid = result.material_id >= 0;
    return result;
}

LuxHDInline GpuSceneTriangle gpu_scene_triangle(GpuScene scene, PrimitiveRef ref) {
    return gpu_scene_triangle_from_ref(scene, ref);
}

LuxHDInline GpuSceneTriangle gpu_scene_triangle_by_id(GpuScene scene, int primitive_id) {
    return gpu_scene_triangle(scene, gpu_scene_primitive_ref(scene, primitive_id));
}

LuxHDInline int gpu_scene_material_id(GpuScene scene, PrimitiveRef ref) {
    return gpu_scene_triangle(scene, ref).material_id;
}

LuxHDInline int gpu_scene_material_id_by_id(GpuScene scene, int primitive_id) {
    return gpu_scene_triangle_by_id(scene, primitive_id).material_id;
}

LuxHDInline bool gpu_scene_has_vertex_normals(const GpuSceneTriangle& triangle) {
    return length2(triangle.n0) > 0
        && length2(triangle.n1) > 0
        && length2(triangle.n2) > 0;
}

LuxHDInline vec3 gpu_scene_interpolate_triangle_normal(
        const GpuSceneTriangle& triangle, Float u, Float v) {
    vec3 face_normal = triangle_normal(triangle.triangle);
    if (!gpu_scene_has_vertex_normals(triangle)) return face_normal;
    Float w = Float(1) - u - v;
    vec3 normal = w * triangle.n0 + u * triangle.n1 + v * triangle.n2;
    if (length2(normal) == 0) return face_normal;
    return normalize(normal);
}

LuxHDInline vec2 gpu_scene_interpolate_triangle_uv(
        const GpuSceneTriangle& triangle, Float u, Float v) {
    Float w = Float(1) - u - v;
    return w * triangle.uv0 + u * triangle.uv1 + v * triangle.uv2;
}

LuxHDInline vec3 gpu_image_asset_lookup_nearest(GpuScene scene,
                                                int image_id,
                                                const vec2& uv,
                                                const vec3& scale = vec3(1)) {
    if (image_id < 0 || image_id >= scene.image_asset_count || !scene.image_assets) {
        return vec3(0);
    }
    GpuImageAsset asset = scene.image_assets[image_id];
    if (!asset.pixels || asset.width <= 0 || asset.height <= 0) return vec3(0);
    Float u = uv.x - floorf(uv.x);
    Float v = uv.y - floorf(uv.y);
    int x = static_cast<int>(fminf(Float(asset.width - 1), floorf(u * asset.width)));
    int y = static_cast<int>(fminf(Float(asset.height - 1), floorf(v * asset.height)));
    return asset.pixels[y * asset.width + x] * scale;
}

LuxHDInline vec3 gpu_material_emission_value(GpuScene scene,
                                             const Material& material,
                                             const vec2& uv) {
    if (material.emission_source == EmissionSourceKind::Image) {
        return gpu_image_asset_lookup_nearest(
            scene, material.emission_image_id, uv, material.emission_scale);
    }
    return material.emission;
}

LuxHDInline vec3 gpu_eval_emission(GpuScene scene,
                                   const Material& material,
                                   const vec2& uv,
                                   const vec3& ng,
                                   const vec3& wo) {
    return eval_emission(
        gpu_material_emission_value(scene, material, uv),
        material.emission_sidedness, ng, wo);
}

#endif // LUX_SCENE_GPU_VIEW_H
