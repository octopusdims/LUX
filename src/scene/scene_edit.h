#pragma once

#ifndef LUX_SCENE_EDIT_H
#define LUX_SCENE_EDIT_H

#include "scene/scene_data.h"

LuxInline int ensure_default_scene_mesh(Scene& scene) {
    if (scene.default_mesh_id >= 0
        && scene.default_mesh_id < static_cast<int>(scene.mesh_assets.size())
        && scene.default_instance_id >= 0
        && scene.default_instance_id < static_cast<int>(scene.instances.size())
        && scene.instances[scene.default_instance_id].mesh_id == scene.default_mesh_id) {
        return scene.default_mesh_id;
    }

    SceneMesh mesh;
    scene.mesh_assets.push_back(std::move(mesh));
    scene.default_mesh_id = static_cast<int>(scene.mesh_assets.size()) - 1;

    SceneInstance instance;
    instance.mesh_id = scene.default_mesh_id;
    instance.object_to_world = Transform{};
    scene.instances.push_back(instance);
    scene.default_instance_id = static_cast<int>(scene.instances.size()) - 1;
    return scene.default_mesh_id;
}

LuxInline void refresh_scene_geometry_cache(Scene& scene) {
    scene.bounds = AABB{};
    for (SceneMesh& mesh : scene.mesh_assets) {
        finalize_scene_mesh(mesh);
    }
    for (SceneInstance& instance : scene.instances) {
        if (instance.mesh_id < 0
            || instance.mesh_id >= static_cast<int>(scene.mesh_assets.size())) {
            continue;
        }
        const SceneMesh& mesh = scene.mesh_assets[instance.mesh_id];
        instance.world_bounds = transform_bounds(instance.object_to_world, mesh.bounds);
        scene.bounds.expand(instance.world_bounds);
    }
}

LuxInline void commit_scene_geometry_edit(Scene& scene) {
    refresh_scene_geometry_cache(scene);
    mark_scene_modified(scene);
}

LuxInline int scene_mesh_instance_use_count(const Scene& scene, int mesh_id) {
    int count = 0;
    for (const SceneInstance& instance : scene.instances) {
        if (instance.mesh_id == mesh_id) ++count;
    }
    return count;
}

LuxInline void require_unique_mesh_edit(const Scene& scene, int mesh_id,
                                        const char* operation) {
    if (scene_mesh_instance_use_count(scene, mesh_id) <= 1) return;
    throw std::runtime_error(
        std::string(operation)
        + ": world-space edit would modify a mesh shared by multiple instances");
}

LuxInline int add_triangle(Scene& scene, const Triangle& triangle, int material_id,
                           const vec3& n0 = vec3(0), const vec3& n1 = vec3(0),
                           const vec3& n2 = vec3(0), const vec2& uv0 = vec2(0),
                           const vec2& uv1 = vec2(0), const vec2& uv2 = vec2(0),
                           unsigned char boundary_edges = 0) {
    int mesh_id = ensure_default_scene_mesh(scene);
    SceneMesh& mesh = scene.mesh_assets[mesh_id];

    uint32_t first_vertex = static_cast<uint32_t>(mesh.vertices.size());
    int local_triangle_id = static_cast<int>(mesh.indices.size());
    mesh.vertices.push_back(triangle.v0);
    mesh.vertices.push_back(triangle.v1);
    mesh.vertices.push_back(triangle.v2);
    mesh.normals.push_back(n0);
    mesh.normals.push_back(n1);
    mesh.normals.push_back(n2);
    mesh.uvs.push_back(uv0);
    mesh.uvs.push_back(uv1);
    mesh.uvs.push_back(uv2);
    mesh.indices.push_back(TriangleIndices{first_vertex, first_vertex + 1, first_vertex + 2});
    mesh.material_ids.push_back(material_id);
    mesh.boundary_edges.push_back(boundary_edges);

    commit_scene_geometry_edit(scene);
    return scene_primitive_index(
        scene, PrimitiveRef{
            scene.default_instance_id, mesh_id, local_triangle_id, PrimitiveKind::Triangle});
}

LuxInline SceneMeshAppendRange append_indexed_mesh(
        Scene& scene,
        const std::vector<vec3>& vertices,
        const std::vector<vec3>& normals,
        const std::vector<vec2>& uvs,
        const std::vector<TriangleIndices>& indices,
        const std::vector<int>& material_ids,
        const std::vector<unsigned char>& boundary_edges = {}) {
    if (!normals.empty() && normals.size() != vertices.size()) {
        throw std::runtime_error("append_indexed_mesh: normals size must match vertices");
    }
    if (!uvs.empty() && uvs.size() != vertices.size()) {
        throw std::runtime_error("append_indexed_mesh: uvs size must match vertices");
    }
    if (material_ids.size() != indices.size()) {
        throw std::runtime_error("append_indexed_mesh: material id count must match triangles");
    }
    if (!boundary_edges.empty() && boundary_edges.size() != indices.size()) {
        throw std::runtime_error("append_indexed_mesh: boundary edge count must match triangles");
    }
    for (const TriangleIndices& tri : indices) {
        if (tri.v0 >= vertices.size() || tri.v1 >= vertices.size() || tri.v2 >= vertices.size()) {
            throw std::runtime_error("append_indexed_mesh: triangle index out of range");
        }
    }

    int mesh_id = ensure_default_scene_mesh(scene);
    SceneMesh& mesh = scene.mesh_assets[mesh_id];

    int local_first_vertex = static_cast<int>(mesh.vertices.size());
    int local_first_triangle = static_cast<int>(mesh.indices.size());

    SceneMeshAppendRange range;
    range.mesh_id = mesh_id;
    range.first_vertex = local_first_vertex;
    range.vertex_count = static_cast<int>(vertices.size());
    range.first_triangle = local_first_triangle;
    range.triangle_count = static_cast<int>(indices.size());

    mesh.vertices.insert(mesh.vertices.end(), vertices.begin(), vertices.end());
    if (normals.empty()) {
        mesh.normals.insert(mesh.normals.end(), vertices.size(), vec3(0));
    } else {
        mesh.normals.insert(mesh.normals.end(), normals.begin(), normals.end());
    }
    if (uvs.empty()) {
        mesh.uvs.insert(mesh.uvs.end(), vertices.size(), vec2(0));
    } else {
        mesh.uvs.insert(mesh.uvs.end(), uvs.begin(), uvs.end());
    }

    uint32_t vertex_offset = static_cast<uint32_t>(local_first_vertex);
    mesh.indices.reserve(mesh.indices.size() + indices.size());
    mesh.material_ids.reserve(mesh.material_ids.size() + indices.size());
    mesh.boundary_edges.reserve(mesh.boundary_edges.size() + indices.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        TriangleIndices tri = indices[i];
        mesh.indices.push_back(TriangleIndices{
            tri.v0 + vertex_offset,
            tri.v1 + vertex_offset,
            tri.v2 + vertex_offset
        });
        mesh.material_ids.push_back(material_ids[i]);
        mesh.boundary_edges.push_back(boundary_edges.empty() ? 0 : boundary_edges[i]);
    }

    commit_scene_geometry_edit(scene);
    return range;
}

LuxInline void set_triangle_geometry(Scene& scene, const PrimitiveRef& ref,
                                     const Triangle& triangle) {
    if (!scene_primitive_ref_exists(scene, ref)) {
        throw std::runtime_error("set_triangle_geometry: primitive ref out of range");
    }
    require_unique_mesh_edit(scene, ref.mesh_id, "set_triangle_geometry");
    const SceneInstance& instance = scene.instances[ref.instance_id];
    SceneMesh& mesh = scene.mesh_assets[ref.mesh_id];
    const TriangleIndices& idx = mesh.indices[ref.local_primitive_id];
    mesh.vertices[idx.v0] = inverse_transform_point(instance.object_to_world, triangle.v0);
    mesh.vertices[idx.v1] = inverse_transform_point(instance.object_to_world, triangle.v1);
    mesh.vertices[idx.v2] = inverse_transform_point(instance.object_to_world, triangle.v2);
    commit_scene_geometry_edit(scene);
}

LuxInline void set_triangle_geometry(Scene& scene, int triangle_id,
                                     const Triangle& triangle) {
    set_triangle_geometry(scene, scene_primitive_ref(scene, triangle_id), triangle);
}

LuxInline void offset_scene_vertices(Scene& scene, const SceneMeshAppendRange& range,
                                     const vec3& offset) {
    if (range.mesh_id < 0 || range.mesh_id >= static_cast<int>(scene.mesh_assets.size())) {
        throw std::runtime_error("offset_scene_vertices: mesh id out of range");
    }
    SceneMesh& mesh = scene.mesh_assets[range.mesh_id];
    if (range.first_vertex < 0 || range.vertex_count < 0
        || range.first_vertex + range.vertex_count > static_cast<int>(mesh.vertices.size())) {
        throw std::runtime_error("offset_scene_vertices: vertex range out of bounds");
    }
    require_unique_mesh_edit(scene, range.mesh_id, "offset_scene_vertices");
    for (int i = 0; i < range.vertex_count; ++i) {
        mesh.vertices[range.first_vertex + i] += offset;
    }
    commit_scene_geometry_edit(scene);
}

LuxInline void set_triangle_normals(Scene& scene, const PrimitiveRef& ref,
                                    const vec3& n0, const vec3& n1, const vec3& n2) {
    if (!scene_primitive_ref_exists(scene, ref)) {
        throw std::runtime_error("set_triangle_normals: primitive ref out of range");
    }
    require_unique_mesh_edit(scene, ref.mesh_id, "set_triangle_normals");
    const SceneInstance& instance = scene.instances[ref.instance_id];
    SceneMesh& mesh = scene.mesh_assets[ref.mesh_id];
    const TriangleIndices& idx = mesh.indices[ref.local_primitive_id];
    mesh.normals[idx.v0] = inverse_transform_normal(instance.object_to_world, n0);
    mesh.normals[idx.v1] = inverse_transform_normal(instance.object_to_world, n1);
    mesh.normals[idx.v2] = inverse_transform_normal(instance.object_to_world, n2);
    commit_scene_geometry_edit(scene);
}

LuxInline void set_triangle_normals(Scene& scene, int triangle_id,
                                    const vec3& n0, const vec3& n1, const vec3& n2) {
    set_triangle_normals(scene, scene_primitive_ref(scene, triangle_id), n0, n1, n2);
}

LuxInline void set_triangle_uvs(Scene& scene, const PrimitiveRef& ref,
                                const vec2& uv0, const vec2& uv1, const vec2& uv2) {
    if (!scene_primitive_ref_exists(scene, ref)) {
        throw std::runtime_error("set_triangle_uvs: primitive ref out of range");
    }
    require_unique_mesh_edit(scene, ref.mesh_id, "set_triangle_uvs");
    SceneMesh& mesh = scene.mesh_assets[ref.mesh_id];
    const TriangleIndices& idx = mesh.indices[ref.local_primitive_id];
    mesh.uvs[idx.v0] = uv0;
    mesh.uvs[idx.v1] = uv1;
    mesh.uvs[idx.v2] = uv2;
    commit_scene_geometry_edit(scene);
}

LuxInline void set_triangle_uvs(Scene& scene, int triangle_id,
                                const vec2& uv0, const vec2& uv1, const vec2& uv2) {
    set_triangle_uvs(scene, scene_primitive_ref(scene, triangle_id), uv0, uv1, uv2);
}

LuxInline void set_triangle_boundary_edges(Scene& scene, const PrimitiveRef& ref,
                                           unsigned char edges) {
    if (!scene_primitive_ref_exists(scene, ref)) {
        throw std::runtime_error("set_triangle_boundary_edges: primitive ref out of range");
    }
    require_unique_mesh_edit(scene, ref.mesh_id, "set_triangle_boundary_edges");
    scene.mesh_assets[ref.mesh_id].boundary_edges[ref.local_primitive_id] = edges;
    commit_scene_geometry_edit(scene);
}

LuxInline void set_triangle_boundary_edges(Scene& scene, int triangle_id,
                                           unsigned char edges) {
    set_triangle_boundary_edges(scene, scene_primitive_ref(scene, triangle_id), edges);
}

LuxInline void set_scene_material(Scene& scene, int material_id, const Material& material) {
    scene.materials[material_id] = material;
    mark_scene_modified(scene);
}

LuxInline void add_quad(Scene& scene, const vec3& p0, const vec3& p1, const vec3& p2,
                        const vec3& p3, int material_id) {
    add_triangle(scene, Triangle{p0, p1, p2}, material_id);
    add_triangle(scene, Triangle{p0, p2, p3}, material_id);
}

LuxInline void clear_scene_vertex_normals(Scene& scene) {
    for (SceneMesh& mesh : scene.mesh_assets) {
        for (vec3& normal : mesh.normals) normal = vec3(0);
    }
    commit_scene_geometry_edit(scene);
}

LuxInline void ensure_scene_instance_assets(Scene& scene) {
    if (!scene.mesh_assets.empty()) return;
    ensure_default_scene_mesh(scene);
}

LuxInline int add_scene_mesh(Scene& scene, SceneMesh mesh) {
    finalize_scene_mesh(mesh);
    scene.mesh_assets.push_back(std::move(mesh));
    mark_scene_modified(scene);
    return static_cast<int>(scene.mesh_assets.size()) - 1;
}

LuxInline int add_scene_instance(Scene& scene, int mesh_id, const Transform& object_to_world,
                                 int material_override = -1) {
    if (mesh_id < 0 || mesh_id >= static_cast<int>(scene.mesh_assets.size())) {
        throw std::runtime_error("add_scene_instance: mesh id out of range");
    }
    SceneInstance instance;
    instance.mesh_id = mesh_id;
    instance.object_to_world = object_to_world;
    instance.material_override = material_override;
    instance.world_bounds = transform_bounds(object_to_world, scene.mesh_assets[mesh_id].bounds);
    scene.instances.push_back(instance);
    int instance_id = static_cast<int>(scene.instances.size()) - 1;
    commit_scene_geometry_edit(scene);
    return instance_id;
}

#endif // LUX_SCENE_EDIT_H
