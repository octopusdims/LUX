#pragma once

#ifndef LUX_SCENE_DATA_H
#define LUX_SCENE_DATA_H

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "camera/camera.h"
#include "core/constants.h"
#include "core/mat4.cuh"
#include "core/vec2.cuh"
#include "core/types.h"
#include "geometry/triangle.h"
#include "geometry/triangle_mesh_view.h"
#include "light/light.h"
#include "material/material.h"
#include "scene/primitive_ref.h"

struct SceneTriangle {
    Triangle triangle;
    int material_id;
    vec3 n0, n1, n2;
    vec2 uv0, uv1, uv2;
    unsigned char boundary_edges = 0;
};

struct SceneMesh {
    std::vector<vec3> vertices;
    std::vector<vec3> normals;
    std::vector<vec2> uvs;
    std::vector<TriangleIndices> indices;
    std::vector<int> material_ids;
    std::vector<unsigned char> boundary_edges;
    AABB bounds;
};

struct SceneInstance {
    int mesh_id = -1;
    Transform object_to_world;
    int material_override = -1;
    AABB world_bounds;
};

struct SceneImageLightAsset {
    std::vector<vec3> pixels;
    std::vector<Float> cdf;
    int width = 0;
    int height = 0;
    Float total_weight = 0;
};

struct SceneMeshAppendRange {
    int mesh_id = -1;
    int first_vertex = 0;
    int vertex_count = 0;
    int first_triangle = 0;
    int triangle_count = 0;
};

struct ScenePrimitive {
    PrimitiveRef ref;
    int ordinal = -1;
};

struct Scene {
    std::vector<Material> materials;
    std::vector<SceneMesh> mesh_assets;
    std::vector<SceneInstance> instances;
    int default_mesh_id = -1;
    int default_instance_id = -1;
    std::vector<Light> lights;
    std::vector<SceneImageLightAsset> image_light_assets;
    Camera camera;

    AABB bounds;
    std::uint64_t revision = 0;
};

LuxInline void mark_scene_modified(Scene& scene) {
    ++scene.revision;
}

LuxInline vec3 image_asset_lookup_nearest(const SceneImageLightAsset& asset,
                                          const vec2& uv,
                                          const vec3& scale = vec3(1)) {
    if (asset.pixels.empty() || asset.width <= 0 || asset.height <= 0) return vec3(0);
    Float u = uv.x - floorf(uv.x);
    Float v = uv.y - floorf(uv.y);
    int x = static_cast<int>(fminf(Float(asset.width - 1), floorf(u * asset.width)));
    int y = static_cast<int>(fminf(Float(asset.height - 1), floorf(v * asset.height)));
    return asset.pixels[y * asset.width + x] * scale;
}

LuxInline vec3 scene_material_emission_value(const Scene& scene,
                                             const Material& material,
                                             const vec2& uv) {
    if (material.emission_source == EmissionSourceKind::Image) {
        int image_id = material.emission_image_id;
        if (image_id < 0 || image_id >= static_cast<int>(scene.image_light_assets.size())) {
            return vec3(0);
        }
        return image_asset_lookup_nearest(
            scene.image_light_assets[image_id], uv, material.emission_scale);
    }
    return material.emission;
}

LuxInline Float scene_material_emission_power_hint(const Scene& scene,
                                                   const Material& material) {
    if (material.emission_source == EmissionSourceKind::Image) {
        int image_id = material.emission_image_id;
        if (image_id < 0 || image_id >= static_cast<int>(scene.image_light_assets.size())) {
            return 0;
        }
        const SceneImageLightAsset& asset = scene.image_light_assets[image_id];
        if (asset.pixels.empty()) return 0;
        Float sum = 0;
        for (const vec3& pixel : asset.pixels) {
            sum += luminance(pixel * material.emission_scale);
        }
        return sum / static_cast<Float>(asset.pixels.size());
    }
    return luminance(material.emission);
}

LuxInline vec3 scene_eval_emission(const Scene& scene,
                                   const Material& material,
                                   const vec2& uv,
                                   const vec3& ng,
                                   const vec3& wo) {
    return eval_emission(
        scene_material_emission_value(scene, material, uv),
        material.emission_sidedness, ng, wo);
}

LuxHDInline bool has_vertex_normals(const SceneTriangle& triangle) {
    return length2(triangle.n0) > 0 && length2(triangle.n1) > 0 && length2(triangle.n2) > 0;
}

LuxHDInline vec3 interpolate_triangle_normal(const SceneTriangle& triangle, Float u, Float v) {
    if (!has_vertex_normals(triangle)) return triangle_normal(triangle.triangle);

    Float w = Float(1) - u - v;
    vec3 normal = w * triangle.n0 + u * triangle.n1 + v * triangle.n2;
    if (length2(normal) == 0) return triangle_normal(triangle.triangle);
    return normalize(normal);
}

LuxHDInline vec2 interpolate_triangle_uv(const SceneTriangle& triangle, Float u, Float v) {
    Float w = Float(1) - u - v;
    return w * triangle.uv0 + u * triangle.uv1 + v * triangle.uv2;
}

LuxInline int scene_mesh_triangle_count(const SceneMesh& mesh) {
    return static_cast<int>(mesh.indices.size());
}

LuxInline Triangle scene_mesh_triangle_geometry(const SceneMesh& mesh, int triangle_id) {
    const TriangleIndices& idx = mesh.indices[triangle_id];
    return Triangle{mesh.vertices[idx.v0], mesh.vertices[idx.v1], mesh.vertices[idx.v2]};
}

LuxInline SceneTriangle scene_mesh_triangle_view(const SceneMesh& mesh, int triangle_id) {
    const TriangleIndices& idx = mesh.indices[triangle_id];
    return SceneTriangle{
        Triangle{mesh.vertices[idx.v0], mesh.vertices[idx.v1], mesh.vertices[idx.v2]},
        mesh.material_ids[triangle_id],
        mesh.normals[idx.v0], mesh.normals[idx.v1], mesh.normals[idx.v2],
        mesh.uvs[idx.v0], mesh.uvs[idx.v1], mesh.uvs[idx.v2],
        mesh.boundary_edges[triangle_id]
    };
}

LuxInline void finalize_scene_mesh(SceneMesh& mesh) {
    mesh.bounds = AABB{};
    for (int triangle_id = 0; triangle_id < scene_mesh_triangle_count(mesh); ++triangle_id) {
        mesh.bounds.expand(triangle_bounds(scene_mesh_triangle_geometry(mesh, triangle_id)));
    }
}

LuxInline vec3 transform_instance_normal(const Transform& object_to_world,
                                         const vec3& normal) {
    if (length2(normal) == 0) return vec3(0);
    vec3 transformed = transform_normal(object_to_world, normal);
    if (length2(transformed) == 0) return vec3(0);
    return normalize(transformed);
}

LuxInline int scene_triangle_count(const Scene& scene) {
    int count = 0;
    for (const SceneInstance& instance : scene.instances) {
        if (instance.mesh_id < 0
            || instance.mesh_id >= static_cast<int>(scene.mesh_assets.size())) {
            continue;
        }
        count += scene_mesh_triangle_count(scene.mesh_assets[instance.mesh_id]);
    }
    return count;
}

LuxInline bool scene_primitive_ref_exists(const Scene& scene, const PrimitiveRef& ref) {
    return ref.kind == PrimitiveKind::Triangle
        && ref.instance_id >= 0 && ref.instance_id < static_cast<int>(scene.instances.size())
        && ref.mesh_id >= 0 && ref.mesh_id < static_cast<int>(scene.mesh_assets.size())
        && scene.instances[ref.instance_id].mesh_id == ref.mesh_id
        && ref.local_primitive_id >= 0
        && ref.local_primitive_id < scene_mesh_triangle_count(scene.mesh_assets[ref.mesh_id]);
}

LuxInline PrimitiveRef scene_primitive_ref_from_index(const Scene& scene, int ordinal) {
    if (ordinal < 0) return invalid_primitive_ref();
    int cursor = 0;
    for (int instance_id = 0; instance_id < static_cast<int>(scene.instances.size());
         ++instance_id) {
        const SceneInstance& instance = scene.instances[instance_id];
        if (instance.mesh_id < 0
            || instance.mesh_id >= static_cast<int>(scene.mesh_assets.size())) {
            continue;
        }
        const SceneMesh& mesh = scene.mesh_assets[instance.mesh_id];
        int mesh_triangle_count = scene_mesh_triangle_count(mesh);
        if (ordinal < cursor + mesh_triangle_count) {
            return PrimitiveRef{
                instance_id, instance.mesh_id, ordinal - cursor, PrimitiveKind::Triangle};
        }
        cursor += mesh_triangle_count;
    }
    return invalid_primitive_ref();
}

LuxInline int scene_primitive_index(const Scene& scene, const PrimitiveRef& ref) {
    if (!scene_primitive_ref_exists(scene, ref)) return -1;
    int cursor = 0;
    for (int instance_id = 0; instance_id < static_cast<int>(scene.instances.size());
         ++instance_id) {
        const SceneInstance& instance = scene.instances[instance_id];
        if (instance.mesh_id < 0
            || instance.mesh_id >= static_cast<int>(scene.mesh_assets.size())) {
            continue;
        }
        int mesh_triangle_count = scene_mesh_triangle_count(scene.mesh_assets[instance.mesh_id]);
        if (instance_id == ref.instance_id) {
            return cursor + ref.local_primitive_id;
        }
        cursor += mesh_triangle_count;
    }
    return -1;
}

LuxInline PrimitiveRef scene_primitive_ref(const Scene& scene, int ordinal) {
    return scene_primitive_ref_from_index(scene, ordinal);
}

LuxInline Triangle scene_triangle_geometry(const Scene& scene, const PrimitiveRef& ref) {
    if (!scene_primitive_ref_exists(scene, ref)) return Triangle{};
    const SceneInstance& instance = scene.instances[ref.instance_id];
    Triangle local = scene_mesh_triangle_geometry(
        scene.mesh_assets[ref.mesh_id], ref.local_primitive_id);
    return Triangle{
        transform_point(instance.object_to_world, local.v0),
        transform_point(instance.object_to_world, local.v1),
        transform_point(instance.object_to_world, local.v2)};
}

LuxInline Triangle scene_triangle_geometry(const Scene& scene, int ordinal) {
    return scene_triangle_geometry(scene, scene_primitive_ref(scene, ordinal));
}

LuxInline SceneTriangle scene_triangle_view(const Scene& scene, const PrimitiveRef& ref) {
    if (!scene_primitive_ref_exists(scene, ref)) {
        return SceneTriangle{Triangle{}, -1, vec3(0), vec3(0), vec3(0),
                             vec2(0), vec2(0), vec2(0), 0};
    }

    const SceneInstance& instance = scene.instances[ref.instance_id];
    const SceneMesh& mesh = scene.mesh_assets[ref.mesh_id];
    SceneTriangle local = scene_mesh_triangle_view(mesh, ref.local_primitive_id);
    return SceneTriangle{
        scene_triangle_geometry(scene, ref),
        instance.material_override >= 0 ? instance.material_override : local.material_id,
        transform_instance_normal(instance.object_to_world, local.n0),
        transform_instance_normal(instance.object_to_world, local.n1),
        transform_instance_normal(instance.object_to_world, local.n2),
        local.uv0, local.uv1, local.uv2,
        local.boundary_edges};
}

LuxInline SceneTriangle scene_triangle_view(const Scene& scene, int ordinal) {
    return scene_triangle_view(scene, scene_primitive_ref(scene, ordinal));
}

LuxInline SceneTriangle scene_triangle(const Scene& scene, int ordinal) {
    return scene_triangle_view(scene, ordinal);
}

#endif // LUX_SCENE_DATA_H
