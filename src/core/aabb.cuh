#pragma once

#ifndef LUX_CORE_AABB_CUH
#define LUX_CORE_AABB_CUH

#include "ray.cuh"

// Axis-aligned bounding box. The default constructor creates an empty box so
// callers can build bounds incrementally with expand().
struct AABB {
    vec3 p_min;
    vec3 p_max;

    LuxHDInline AABB() : p_min(INFINITY), p_max(-INFINITY) {}
    LuxHDInline AABB(const vec3& p_min_, const vec3& p_max_) : p_min(p_min_), p_max(p_max_) {}
    LuxHDInline explicit AABB(const vec3& point) : p_min(point), p_max(point) {}

    LuxHDInline vec3 center() const { return (p_min + p_max) * Float(0.5); }

    LuxHDInline vec3 centroid() const { return center(); }

    LuxHDInline vec3 diagonal() const { return p_max - p_min; }

    LuxHDInline vec3 extent() const { return diagonal(); }

    LuxHDInline Float surface_area() const {
        vec3 d = diagonal();
        return 2 * (d.x * d.y + d.y * d.z + d.z * d.x);
    }

    LuxHDInline Float volume() const {
        vec3 d = diagonal();
        return d.x * d.y * d.z;
    }

    LuxHDInline int max_extent_axis() const {
        vec3 d = diagonal();
        return (d.x > d.y && d.x > d.z) ? 0
             : (d.y > d.z) ? 1
             : 2;
    }

    LuxHDInline bool is_empty() const {
        return p_min.x > p_max.x || p_min.y > p_max.y || p_min.z > p_max.z;
    }

    LuxHDInline void expand(const vec3& point) {
        p_min = min(p_min, point);
        p_max = max(p_max, point);
    }

    LuxHDInline void expand(const AABB& b) {
        p_min = min(p_min, b.p_min);
        p_max = max(p_max, b.p_max);
    }

    LuxHDInline void pad(Float padding) {
        p_min -= vec3(padding);
        p_max += vec3(padding);
    }

    LuxHDInline bool contains(const vec3& point) const {
        return point.x >= p_min.x && point.x <= p_max.x
            && point.y >= p_min.y && point.y <= p_max.y
            && point.z >= p_min.z && point.z <= p_max.z;
    }

    LuxHDInline bool overlaps(const AABB& b) const {
        return !(b.p_min.x > p_max.x || b.p_max.x < p_min.x
              || b.p_min.y > p_max.y || b.p_max.y < p_min.y
              || b.p_min.z > p_max.z || b.p_max.z < p_min.z);
    }

    LuxHDInline vec3 normalized_offset(const vec3& point) const {
        vec3 d = diagonal();
        return vec3(
            d.x > 0 ? (point.x - p_min.x) / d.x : Float(0.5),
            d.y > 0 ? (point.y - p_min.y) / d.y : Float(0.5),
            d.z > 0 ? (point.z - p_min.z) / d.z : Float(0.5)
        );
    }

    // Slab method. Returns true if ray hits the box within [t_min, t_max].
    LuxHDInline bool intersect(const Ray& ray, Float& t_near, Float& t_far) const {
        t_near = ray.t_min;
        t_far  = ray.t_max;

        for (int i = 0; i < 3; ++i) {
            // Parallel rays do not restrict the interval if the origin lies
            // inside this axis slab; otherwise they miss immediately.
            if (fabs(ray.direction[i]) < Float(1e-12)) {
                if (ray.origin[i] < p_min[i] || ray.origin[i] > p_max[i]) return false;
                continue;
            }

            Float inv_d = 1 / ray.direction[i];
            Float axis_t_min = (p_min[i] - ray.origin[i]) * inv_d;
            Float axis_t_max = (p_max[i] - ray.origin[i]) * inv_d;
            if (inv_d < 0) {
                Float tmp = axis_t_min;
                axis_t_min = axis_t_max;
                axis_t_max = tmp;
            }
            t_near = fmaxf(t_near, axis_t_min);
            t_far  = fminf(t_far,  axis_t_max);
            if (t_near > t_far) return false;
        }
        return true;
    }

    // Convenience overload when you only need a bool.
    LuxHDInline bool intersect(const Ray& ray) const {
        Float t_near, t_far;
        return intersect(ray, t_near, t_far);
    }
};

LuxHDInline AABB union_aabb(const AABB& a, const AABB& b) {
    return AABB(min(a.p_min, b.p_min), max(a.p_max, b.p_max));
}
#endif // LUX_CORE_AABB_CUH
