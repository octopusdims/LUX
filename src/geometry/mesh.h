#pragma once

#ifndef LUX_GEOMETRY_MESH_H
#define LUX_GEOMETRY_MESH_H

#include <vector>
#include <cstdint>
#include "triangle.h"
#include "core/vec2.cuh"

struct Mesh {
    std::vector<vec3> vertices;
    std::vector<vec3> normals;
    std::vector<vec2> texcoords;
    std::vector<uint32_t> indices;   // 3 per triangle
    int material_id = -1;

    int num_triangles() const { return static_cast<int>(indices.size() / 3); }

    Triangle get_triangle(int tri_idx) const {
        uint32_t i0 = indices[tri_idx * 3 + 0];
        uint32_t i1 = indices[tri_idx * 3 + 1];
        uint32_t i2 = indices[tri_idx * 3 + 2];
        return Triangle{vertices[i0], vertices[i1], vertices[i2]};
    }
};
#endif // LUX_GEOMETRY_MESH_H
