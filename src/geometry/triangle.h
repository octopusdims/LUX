#pragma once

#ifndef LUX_GEOMETRY_TRIANGLE_H
#define LUX_GEOMETRY_TRIANGLE_H

#include "core/vec3.cuh"
#include "core/ray.cuh"
#include "core/aabb.cuh"

struct Triangle {
    vec3 v0, v1, v2;
};

LuxHDInline vec3 triangle_normal(const Triangle& tri) {
    return normalize(cross(tri.v1 - tri.v0, tri.v2 - tri.v0));
}

LuxHDInline Float triangle_area(const Triangle& tri) {
    return Float(0.5) * length(cross(tri.v1 - tri.v0, tri.v2 - tri.v0));
}

LuxHDInline AABB triangle_bounds(const Triangle& tri) {
    AABB bounds;
    bounds.expand(tri.v0);
    bounds.expand(tri.v1);
    bounds.expand(tri.v2);
    return bounds;
}

// Möller-Trumbore ray-triangle intersection.
// Returns true if hit within ray [t_min, t_max].
// Outputs: t (ray parameter), u, v (barycentric coordinates).
LuxHDInline bool intersect_triangle(const Ray& ray, const Triangle& tri,
                                     Float& t, Float& u, Float& v) {
    vec3 e1 = tri.v1 - tri.v0;
    vec3 e2 = tri.v2 - tri.v0;
    vec3 h = cross(ray.direction, e2);
    Float a = dot(e1, h);
    if (fabs(a) < Float(1e-7)) return false;

    Float f = 1.0f / a;
    vec3 s = ray.origin - tri.v0;
    u = f * dot(s, h);
    if (u < 0.0f || u > 1.0f) return false;

    vec3 q = cross(s, e1);
    v = f * dot(ray.direction, q);
    if (v < 0.0f || u + v > 1.0f) return false;

    t = f * dot(e2, q);
    return t > ray.t_min && t < ray.t_max;
}
#endif // LUX_GEOMETRY_TRIANGLE_H
