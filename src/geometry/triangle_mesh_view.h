#pragma once

#ifndef LUX_GEOMETRY_TRIANGLE_MESH_VIEW_H
#define LUX_GEOMETRY_TRIANGLE_MESH_VIEW_H

#include <cstdint>

#include "core/vec2.cuh"
#include "core/vec3.cuh"
#include "core/types.h"
#include "geometry/triangle.h"

struct TriangleIndices {
    uint32_t v0 = 0;
    uint32_t v1 = 0;
    uint32_t v2 = 0;
};

struct TriangleMeshView {
    const vec3* vertices = nullptr;
    const vec3* normals = nullptr;
    const vec2* uvs = nullptr;
    const TriangleIndices* indices = nullptr;
    int triangle_count = 0;
};

LuxHDInline Triangle get_triangle(TriangleMeshView mesh, int triangle_id) {
    const TriangleIndices& idx = mesh.indices[triangle_id];
    return Triangle{mesh.vertices[idx.v0], mesh.vertices[idx.v1], mesh.vertices[idx.v2]};
}

LuxHDInline bool has_vertex_normals(TriangleMeshView mesh, int triangle_id) {
    const TriangleIndices& idx = mesh.indices[triangle_id];
    return length2(mesh.normals[idx.v0]) > 0
        && length2(mesh.normals[idx.v1]) > 0
        && length2(mesh.normals[idx.v2]) > 0;
}

LuxHDInline vec3 interpolate_triangle_normal(TriangleMeshView mesh, int triangle_id,
                                             Float u, Float v) {
    Triangle triangle = get_triangle(mesh, triangle_id);
    if (!has_vertex_normals(mesh, triangle_id)) return triangle_normal(triangle);

    const TriangleIndices& idx = mesh.indices[triangle_id];
    Float w = Float(1) - u - v;
    vec3 normal = w * mesh.normals[idx.v0] + u * mesh.normals[idx.v1] + v * mesh.normals[idx.v2];
    if (length2(normal) == 0) return triangle_normal(triangle);
    return normalize(normal);
}

LuxHDInline vec2 interpolate_triangle_uv(TriangleMeshView mesh, int triangle_id,
                                         Float u, Float v) {
    const TriangleIndices& idx = mesh.indices[triangle_id];
    Float w = Float(1) - u - v;
    return w * mesh.uvs[idx.v0] + u * mesh.uvs[idx.v1] + v * mesh.uvs[idx.v2];
}

#endif // LUX_GEOMETRY_TRIANGLE_MESH_VIEW_H
