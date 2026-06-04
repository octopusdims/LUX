#pragma once

#ifndef LUX_CORE_MAT4_CUH
#define LUX_CORE_MAT4_CUH

#include "vec3.cuh"
#include "ray.cuh"
#include "aabb.cuh"

// Row-major storage, column-vector convention: p' = M * p.
// Points use an implicit w=1, vectors w=0.
// translate() stores offsets in m[0][3], m[1][3], m[2][3] to match transform_point().
struct mat4 {
    Float m[4][4];

    LuxHDInline mat4() {
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                m[i][j] = (i == j) ? 1 : 0;
    }

    LuxHDInline Float* operator[](int i) { return m[i]; }
    LuxHDInline const Float* operator[](int i) const { return m[i]; }

    LuxHDInline mat4 operator*(const mat4& b) const {
        mat4 result;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                result.m[i][j] = m[i][0] * b.m[0][j] + m[i][1] * b.m[1][j]
                               + m[i][2] * b.m[2][j] + m[i][3] * b.m[3][j];
        return result;
    }
};

struct Transform {
    mat4 m;
    mat4 inv;

    LuxHDInline Transform() = default;
    LuxHDInline Transform(const mat4& m_, const mat4& inv_) : m(m_), inv(inv_) {}
};

LuxHDInline mat4 translate(const vec3& offset) {
    mat4 result;
    result.m[0][3] = offset.x;
    result.m[1][3] = offset.y;
    result.m[2][3] = offset.z;
    return result;
}

LuxHDInline mat4 scale(const vec3& scale_factors) {
    mat4 result;
    result.m[0][0] = scale_factors.x;
    result.m[1][1] = scale_factors.y;
    result.m[2][2] = scale_factors.z;
    return result;
}

LuxHDInline mat4 rotate_x(Float theta) {
    mat4 result;
    Float cos_theta = cosf(theta);
    Float sin_theta = sinf(theta);
    result.m[1][1] = cos_theta;  result.m[1][2] = -sin_theta;
    result.m[2][1] = sin_theta;  result.m[2][2] =  cos_theta;
    return result;
}

LuxHDInline mat4 rotate_y(Float theta) {
    mat4 result;
    Float cos_theta = cosf(theta);
    Float sin_theta = sinf(theta);
    result.m[0][0] =  cos_theta; result.m[0][2] = sin_theta;
    result.m[2][0] = -sin_theta; result.m[2][2] = cos_theta;
    return result;
}

LuxHDInline mat4 rotate_z(Float theta) {
    mat4 result;
    Float cos_theta = cosf(theta);
    Float sin_theta = sinf(theta);
    result.m[0][0] = cos_theta;  result.m[0][1] = -sin_theta;
    result.m[1][0] = sin_theta;  result.m[1][1] =  cos_theta;
    return result;
}

LuxHDInline Float affine_determinant3x3(const mat4& transform) {
    Float a00 = transform.m[0][0], a01 = transform.m[0][1], a02 = transform.m[0][2];
    Float a10 = transform.m[1][0], a11 = transform.m[1][1], a12 = transform.m[1][2];
    Float a20 = transform.m[2][0], a21 = transform.m[2][1], a22 = transform.m[2][2];
    return a00 * (a11 * a22 - a12 * a21)
         - a01 * (a10 * a22 - a12 * a20)
         + a02 * (a10 * a21 - a11 * a20);
}

LuxHDInline bool is_affine_matrix(const mat4& transform) {
    return fabsf(transform.m[3][0]) <= Float(1e-6)
        && fabsf(transform.m[3][1]) <= Float(1e-6)
        && fabsf(transform.m[3][2]) <= Float(1e-6)
        && fabsf(transform.m[3][3] - Float(1)) <= Float(1e-6);
}

LuxHDInline bool is_affine_invertible(const mat4& transform) {
    return is_affine_matrix(transform)
        && fabsf(affine_determinant3x3(transform)) > Float(1e-12);
}

LuxHDInline bool mat4_near_identity(const mat4& transform,
                                    Float epsilon = Float(1e-4)) {
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            Float expected = (row == col) ? Float(1) : Float(0);
            if (!(fabsf(transform.m[row][col] - expected) <= epsilon)) {
                return false;
            }
        }
    }
    return true;
}

LuxHDInline bool transform_inverse_pair_consistent(const Transform& transform,
                                                   Float epsilon = Float(1e-4)) {
    return mat4_near_identity(transform.m * transform.inv, epsilon)
        && mat4_near_identity(transform.inv * transform.m, epsilon);
}

LuxHDInline mat4 inverse_affine(const mat4& transform) {
    mat4 result;
    Float a00 = transform.m[0][0], a01 = transform.m[0][1], a02 = transform.m[0][2];
    Float a10 = transform.m[1][0], a11 = transform.m[1][1], a12 = transform.m[1][2];
    Float a20 = transform.m[2][0], a21 = transform.m[2][1], a22 = transform.m[2][2];

    Float det = affine_determinant3x3(transform);
    if (fabsf(det) <= Float(1e-12)) {
        return result;
    }
    Float inv_det = Float(1) / det;

    result.m[0][0] =  (a11 * a22 - a12 * a21) * inv_det;
    result.m[0][1] = -(a01 * a22 - a02 * a21) * inv_det;
    result.m[0][2] =  (a01 * a12 - a02 * a11) * inv_det;
    result.m[1][0] = -(a10 * a22 - a12 * a20) * inv_det;
    result.m[1][1] =  (a00 * a22 - a02 * a20) * inv_det;
    result.m[1][2] = -(a00 * a12 - a02 * a10) * inv_det;
    result.m[2][0] =  (a10 * a21 - a11 * a20) * inv_det;
    result.m[2][1] = -(a00 * a21 - a01 * a20) * inv_det;
    result.m[2][2] =  (a00 * a11 - a01 * a10) * inv_det;

    vec3 t(transform.m[0][3], transform.m[1][3], transform.m[2][3]);
    vec3 inv_t(
        -(result.m[0][0] * t.x + result.m[0][1] * t.y + result.m[0][2] * t.z),
        -(result.m[1][0] * t.x + result.m[1][1] * t.y + result.m[1][2] * t.z),
        -(result.m[2][0] * t.x + result.m[2][1] * t.y + result.m[2][2] * t.z));
    result.m[0][3] = inv_t.x;
    result.m[1][3] = inv_t.y;
    result.m[2][3] = inv_t.z;
    result.m[3][0] = result.m[3][1] = result.m[3][2] = 0;
    result.m[3][3] = 1;
    return result;
}

LuxHDInline Transform make_transform(const mat4& m) {
    return Transform(m, inverse_affine(m));
}

LuxHDInline Transform inverse_transform(const Transform& transform) {
    return Transform(transform.inv, transform.m);
}

LuxHDInline Transform operator*(const Transform& a, const Transform& b) {
    return Transform(a.m * b.m, b.inv * a.inv);
}

LuxHDInline vec3 transform_point(const mat4& transform, const vec3& point) {
    Float x = transform.m[0][0] * point.x + transform.m[0][1] * point.y
            + transform.m[0][2] * point.z + transform.m[0][3];
    Float y = transform.m[1][0] * point.x + transform.m[1][1] * point.y
            + transform.m[1][2] * point.z + transform.m[1][3];
    Float z = transform.m[2][0] * point.x + transform.m[2][1] * point.y
            + transform.m[2][2] * point.z + transform.m[2][3];
    Float w = transform.m[3][0] * point.x + transform.m[3][1] * point.y
            + transform.m[3][2] * point.z + transform.m[3][3];
    return (w == 1) ? vec3(x, y, z) : vec3(x / w, y / w, z / w);
}

LuxHDInline vec3 transform_point(const Transform& transform, const vec3& point) {
    return transform_point(transform.m, point);
}

LuxHDInline vec3 inverse_transform_point(const Transform& transform, const vec3& point) {
    return transform_point(transform.inv, point);
}

LuxHDInline vec3 transform_vector(const mat4& transform, const vec3& vector) {
    return vec3(
        transform.m[0][0] * vector.x + transform.m[0][1] * vector.y + transform.m[0][2] * vector.z,
        transform.m[1][0] * vector.x + transform.m[1][1] * vector.y + transform.m[1][2] * vector.z,
        transform.m[2][0] * vector.x + transform.m[2][1] * vector.y + transform.m[2][2] * vector.z
    );
}

LuxHDInline vec3 transform_vector(const Transform& transform, const vec3& vector) {
    return transform_vector(transform.m, vector);
}

LuxHDInline vec3 inverse_transform_vector(const Transform& transform, const vec3& vector) {
    return transform_vector(transform.inv, vector);
}

LuxHDInline vec3 transform_normal(const mat4& transform, const vec3& normal) {
    // Use adj(M)^T * n = det(M) * M^{-T} * n. After normalization this matches
    // inverse-transpose for positive determinants and preserves the handedness
    // flip needed for mirrored instance transforms.
    Float a00 = transform.m[1][1] * transform.m[2][2] - transform.m[1][2] * transform.m[2][1];
    Float a01 = transform.m[1][2] * transform.m[2][0] - transform.m[1][0] * transform.m[2][2];
    Float a02 = transform.m[1][0] * transform.m[2][1] - transform.m[1][1] * transform.m[2][0];
    Float a10 = transform.m[0][2] * transform.m[2][1] - transform.m[0][1] * transform.m[2][2];
    Float a11 = transform.m[0][0] * transform.m[2][2] - transform.m[0][2] * transform.m[2][0];
    Float a12 = transform.m[0][1] * transform.m[2][0] - transform.m[0][0] * transform.m[2][1];
    Float a20 = transform.m[0][1] * transform.m[1][2] - transform.m[0][2] * transform.m[1][1];
    Float a21 = transform.m[0][2] * transform.m[1][0] - transform.m[0][0] * transform.m[1][2];
    Float a22 = transform.m[0][0] * transform.m[1][1] - transform.m[0][1] * transform.m[1][0];

    // adj(M)[i][j] = C_{ji}(M).  We need adj(M)^T * n = det * M^{-T} * n.
    vec3 transformed_normal(
        a00 * normal.x + a01 * normal.y + a02 * normal.z,
        a10 * normal.x + a11 * normal.y + a12 * normal.z,
        a20 * normal.x + a21 * normal.y + a22 * normal.z
    );
    return normalize(transformed_normal);
}

LuxHDInline vec3 transform_normal(const Transform& transform, const vec3& normal) {
    return transform_normal(transform.m, normal);
}

LuxHDInline vec3 inverse_transform_normal(const Transform& transform, const vec3& normal) {
    return transform_normal(transform.inv, normal);
}

LuxHDInline Ray transform_ray(const mat4& transform, const Ray& ray) {
    return Ray(transform_point(transform, ray.origin),
               transform_vector(transform, ray.direction),
               ray.t_min, ray.t_max, ray.time);
}

LuxHDInline Ray transform_ray(const Transform& transform, const Ray& ray) {
    return transform_ray(transform.m, ray);
}

LuxHDInline Ray inverse_transform_ray(const Transform& transform, const Ray& ray) {
    return transform_ray(transform.inv, ray);
}

LuxHDInline AABB transform_bounds(const mat4& transform, const AABB& bounds) {
    if (bounds.is_empty()) return AABB{};

    AABB result;
    for (int z = 0; z <= 1; ++z) {
        for (int y = 0; y <= 1; ++y) {
            for (int x = 0; x <= 1; ++x) {
                vec3 corner(
                    x ? bounds.p_max.x : bounds.p_min.x,
                    y ? bounds.p_max.y : bounds.p_min.y,
                    z ? bounds.p_max.z : bounds.p_min.z);
                result.expand(transform_point(transform, corner));
            }
        }
    }
    return result;
}

LuxHDInline AABB transform_bounds(const Transform& transform, const AABB& bounds) {
    return transform_bounds(transform.m, bounds);
}
#endif // LUX_CORE_MAT4_CUH
