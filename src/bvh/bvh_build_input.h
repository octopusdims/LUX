#pragma once

#ifndef LUX_BVH_BUILD_INPUT_H
#define LUX_BVH_BUILD_INPUT_H

#include <vector>

#include "core/aabb.cuh"
#include "core/types.h"
#include "geometry/mesh.h"
#include "geometry/primitive_kind.h"

// Minimal host-side primitive record used as input to BVH construction.
// primitive_id is the triangle index inside the source mesh.
struct BvhBuildPrimitive {
    AABB bounds;
    vec3 centroid;
    int primitive_id;
    int instance_id = -1;
    int mesh_id = -1;
    int local_primitive_id = -1;
    PrimitiveKind primitive_kind = PrimitiveKind::Triangle;
};

LuxInline std::vector<BvhBuildPrimitive> collect_bvh_build_primitives(const Mesh& mesh) {
    std::vector<BvhBuildPrimitive> primitives;
    int triangle_count = mesh.num_triangles();
    primitives.reserve(triangle_count);

    for (int triangle_id = 0; triangle_id < triangle_count; ++triangle_id) {
        Triangle triangle = mesh.get_triangle(triangle_id);
        AABB bounds = triangle_bounds(triangle);
        primitives.push_back(BvhBuildPrimitive{
            bounds,
            bounds.centroid(),
            triangle_id,
            -1,
            -1,
            triangle_id,
            PrimitiveKind::Triangle
        });
    }

    return primitives;
}
#endif // LUX_BVH_BUILD_INPUT_H
