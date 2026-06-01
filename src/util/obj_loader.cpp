#include "obj_loader.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "geometry/triangle.h"
#include "material/material.h"

bool load_obj_into_scene(const std::string& obj_path, Scene& scene,
                         const ObjLoadOptions& options,
                         SceneMeshAppendRange* appended_range) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    std::filesystem::path path(obj_path);
    std::string base_dir = path.parent_path().string();
    if (!base_dir.empty() && base_dir.back() != '/' && base_dir.back() != '\\') {
        base_dir += '/';
    }

    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err,
                                obj_path.c_str(), base_dir.c_str(), true);

    if (!warn.empty()) {
        std::fprintf(stderr, "OBJ warning: %s\n", warn.c_str());
    }
    if (!err.empty()) {
        std::fprintf(stderr, "OBJ error: %s\n", err.c_str());
    }
    if (!ret) {
        return false;
    }

    // Convert MTL materials to LUX Materials.
    int material_offset = static_cast<int>(scene.materials.size());
    if (materials.empty()) {
        scene.materials.push_back(Material::lambert(vec3(0.8f)));
    } else {
        for (const auto& mtl : materials) {
            vec3 albedo(mtl.diffuse[0], mtl.diffuse[1], mtl.diffuse[2]);
            if (max_component(albedo) == 0) {
                albedo = vec3(mtl.ambient[0], mtl.ambient[1], mtl.ambient[2]);
            }
            vec3 emission(mtl.emission[0], mtl.emission[1], mtl.emission[2]);

            Material mat;
            if (max_component(emission) > 0) {
                mat = Material::emissive(emission, albedo);
            } else if (mtl.shininess > 100.0f && mtl.illum == 2) {
                Float roughness = std::sqrt(2.0f / (mtl.shininess + 2.0f));
                mat = Material::ggx(albedo, roughness);
            } else {
                mat = Material::lambert(albedo);
            }
            scene.materials.push_back(mat);
        }
    }

    struct LoadedFace {
        tinyobj::index_t idx[3];
        int material_id;
    };

    struct BoundaryEdge {
        int from;
        int to;
        int material_id;
    };

    struct ObjVertexKey {
        int vertex_index;
        int normal_index;
        int texcoord_index;

        bool operator==(const ObjVertexKey& other) const {
            return vertex_index == other.vertex_index
                && normal_index == other.normal_index
                && texcoord_index == other.texcoord_index;
        }
    };

    struct ObjVertexKeyHash {
        size_t operator()(const ObjVertexKey& key) const {
            uint64_t h = static_cast<uint32_t>(key.vertex_index);
            h = h * 1099511628211ULL ^ static_cast<uint32_t>(key.normal_index);
            h = h * 1099511628211ULL ^ static_cast<uint32_t>(key.texcoord_index);
            return static_cast<size_t>(h);
        }
    };

    std::vector<vec3> generated_normal_sums(attrib.vertices.size() / 3, vec3(0));
    std::vector<LoadedFace> loaded_faces;
    std::unordered_map<uint64_t, int> edge_counts;

    auto edge_key = [](int a, int b) -> uint64_t {
        uint32_t lo = static_cast<uint32_t>(std::min(a, b));
        uint32_t hi = static_cast<uint32_t>(std::max(a, b));
        return (static_cast<uint64_t>(lo) << 32) | hi;
    };

    auto vertex_position = [&](int vertex_index) -> vec3 {
        return vec3(attrib.vertices[3 * vertex_index + 0],
                    attrib.vertices[3 * vertex_index + 1],
                    attrib.vertices[3 * vertex_index + 2]);
    };

    // Convert geometry: each face is a triangle (triangulate=true).
    for (const auto& shape : shapes) {
        size_t index_offset = 0;
        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); ++f) {
            int fv = shape.mesh.num_face_vertices[f];
            if (fv != 3) {
                index_offset += fv;
                continue;
            }

            int mat_id = shape.mesh.material_ids[f] + material_offset;
            if (mat_id < material_offset || mat_id >= static_cast<int>(scene.materials.size())) {
                mat_id = material_offset;
            }

            tinyobj::index_t idx0 = shape.mesh.indices[index_offset + 0];
            tinyobj::index_t idx1 = shape.mesh.indices[index_offset + 1];
            tinyobj::index_t idx2 = shape.mesh.indices[index_offset + 2];
            if (idx0.vertex_index < 0 || idx1.vertex_index < 0 || idx2.vertex_index < 0) {
                index_offset += fv;
                continue;
            }

            vec3 v0 = vertex_position(idx0.vertex_index);
            vec3 v1 = vertex_position(idx1.vertex_index);
            vec3 v2 = vertex_position(idx2.vertex_index);

            Triangle tri{v0, v1, v2};
            if (triangle_area(tri) <= Float(1e-12)) {
                index_offset += fv;
                continue;
            }
            loaded_faces.push_back(LoadedFace{{idx0, idx1, idx2}, mat_id});
            ++edge_counts[edge_key(idx0.vertex_index, idx1.vertex_index)];
            ++edge_counts[edge_key(idx1.vertex_index, idx2.vertex_index)];
            ++edge_counts[edge_key(idx2.vertex_index, idx0.vertex_index)];

            if (options.generate_missing_normals) {
                vec3 area_weighted_normal = cross(v1 - v0, v2 - v0);
                generated_normal_sums[idx0.vertex_index] += area_weighted_normal;
                generated_normal_sums[idx1.vertex_index] += area_weighted_normal;
                generated_normal_sums[idx2.vertex_index] += area_weighted_normal;
            }

            index_offset += fv;
        }
    }

    std::vector<BoundaryEdge> boundary_edges_to_cap;
    std::vector<vec3> mesh_vertices;
    std::vector<vec3> mesh_normals;
    std::vector<vec2> mesh_uvs;
    std::vector<TriangleIndices> mesh_indices;
    std::vector<int> mesh_material_ids;
    std::vector<unsigned char> mesh_boundary_edges;
    std::unordered_map<ObjVertexKey, uint32_t, ObjVertexKeyHash> vertex_map;

    mesh_vertices.reserve(loaded_faces.size() * 3);
    mesh_normals.reserve(loaded_faces.size() * 3);
    mesh_uvs.reserve(loaded_faces.size() * 3);
    mesh_indices.reserve(loaded_faces.size());
    mesh_material_ids.reserve(loaded_faces.size());
    mesh_boundary_edges.reserve(loaded_faces.size());
    vertex_map.reserve(loaded_faces.size() * 3);

    auto get_normal = [&](const tinyobj::index_t& idx, const vec3& fallback) -> vec3 {
        int ni = idx.normal_index;
        if (ni >= 0 && !attrib.normals.empty()) {
            return normalize(vec3(attrib.normals[3 * ni + 0],
                                  attrib.normals[3 * ni + 1],
                                  attrib.normals[3 * ni + 2]));
        }
        if (options.generate_missing_normals && idx.vertex_index >= 0) {
            vec3 normal = generated_normal_sums[idx.vertex_index];
            if (length2(normal) > 0) return normalize(normal);
        }
        return fallback;
    };

    auto get_texcoord = [&](const tinyobj::index_t& idx) -> vec2 {
        int ti = idx.texcoord_index;
        if (ti < 0 || attrib.texcoords.empty()) return vec2(0);
        return vec2(attrib.texcoords[2 * ti + 0],
                    attrib.texcoords[2 * ti + 1]);
    };

    auto get_or_add_vertex = [&](const tinyobj::index_t& idx,
                                 const vec3& fallback_normal) -> uint32_t {
        ObjVertexKey key{idx.vertex_index, idx.normal_index, idx.texcoord_index};
        auto found = vertex_map.find(key);
        if (found != vertex_map.end()) return found->second;

        uint32_t local_index = static_cast<uint32_t>(mesh_vertices.size());
        vertex_map.emplace(key, local_index);
        mesh_vertices.push_back(vertex_position(idx.vertex_index));
        mesh_normals.push_back(get_normal(idx, fallback_normal));
        mesh_uvs.push_back(get_texcoord(idx));
        return local_index;
    };

    for (const LoadedFace& loaded : loaded_faces) {
        int vi[3] = {
            loaded.idx[0].vertex_index,
            loaded.idx[1].vertex_index,
            loaded.idx[2].vertex_index
        };
        Triangle triangle{
            vertex_position(vi[0]),
            vertex_position(vi[1]),
            vertex_position(vi[2])
        };
        const vec3 face_normal = triangle_normal(triangle);

        auto is_boundary_edge = [&](int a, int b) -> bool {
            if (a < 0 || b < 0) return false;
            auto found = edge_counts.find(edge_key(a, b));
            return found != edge_counts.end() && found->second == 1;
        };

        unsigned char boundary_edges = 0;
        if (is_boundary_edge(vi[0], vi[1])) {
            boundary_edges |= 1u;
            boundary_edges_to_cap.push_back(BoundaryEdge{
                vi[0], vi[1], loaded.material_id});
        }
        if (is_boundary_edge(vi[1], vi[2])) {
            boundary_edges |= 2u;
            boundary_edges_to_cap.push_back(BoundaryEdge{
                vi[1], vi[2], loaded.material_id});
        }
        if (is_boundary_edge(vi[2], vi[0])) {
            boundary_edges |= 4u;
            boundary_edges_to_cap.push_back(BoundaryEdge{
                vi[2], vi[0], loaded.material_id});
        }

        uint32_t v0 = get_or_add_vertex(loaded.idx[0], face_normal);
        uint32_t v1 = get_or_add_vertex(loaded.idx[1], face_normal);
        uint32_t v2 = get_or_add_vertex(loaded.idx[2], face_normal);
        mesh_indices.push_back(TriangleIndices{v0, v1, v2});
        mesh_material_ids.push_back(loaded.material_id);
        mesh_boundary_edges.push_back(options.cap_boundary_holes ? 0 : boundary_edges);
    }

    SceneMeshAppendRange mesh_range = append_indexed_mesh(
        scene, mesh_vertices, mesh_normals, mesh_uvs, mesh_indices,
        mesh_material_ids, mesh_boundary_edges);
    if (appended_range) {
        *appended_range = mesh_range;
    }

    if (options.cap_boundary_holes && !boundary_edges_to_cap.empty()) {
        vec3 mesh_center(0);
        int vertex_count = static_cast<int>(attrib.vertices.size() / 3);
        for (int i = 0; i < vertex_count; ++i) {
            mesh_center += vertex_position(i);
        }
        mesh_center /= Float(vertex_count);

        std::unordered_map<int, BoundaryEdge> next_edge;
        next_edge.reserve(boundary_edges_to_cap.size());
        for (const BoundaryEdge& edge : boundary_edges_to_cap) {
            next_edge[edge.from] = edge;
        }

        std::unordered_set<int> visited_starts;
        visited_starts.reserve(boundary_edges_to_cap.size());
        int cap_count = 0;
        for (const BoundaryEdge& first_edge : boundary_edges_to_cap) {
            if (visited_starts.find(first_edge.from) != visited_starts.end()) continue;

            std::vector<int> loop;
            int material_id = first_edge.material_id;
            int start = first_edge.from;
            int current = start;
            while (visited_starts.find(current) == visited_starts.end()) {
                auto found = next_edge.find(current);
                if (found == next_edge.end()) break;
                visited_starts.insert(current);
                loop.push_back(current);
                material_id = found->second.material_id;
                current = found->second.to;
                if (current == start) break;
            }

            if (current != start || loop.size() < 3) continue;

            vec3 cap_center(0);
            for (int vertex_index : loop) {
                cap_center += vertex_position(vertex_index);
            }
            cap_center /= Float(loop.size());

            for (size_t i = 0; i < loop.size(); ++i) {
                vec3 v0 = cap_center;
                vec3 v1 = vertex_position(loop[i]);
                vec3 v2 = vertex_position(loop[(i + 1) % loop.size()]);
                Triangle cap_tri{v0, v1, v2};
                vec3 cap_normal = triangle_normal(cap_tri);
                if (dot(cap_normal, cap_center - mesh_center) < 0) {
                    cap_tri = Triangle{v0, v2, v1};
                    cap_normal = -cap_normal;
                }

                add_triangle(scene, cap_tri, material_id, cap_normal, cap_normal, cap_normal);
                ++cap_count;
            }
        }

        if (cap_count > 0) {
            std::fprintf(stderr, "OBJ capped %d boundary triangles\n", cap_count);
        }
    }

    return true;
}
