#pragma once

#ifndef LUX_CORE_VALIDATION_CUH
#define LUX_CORE_VALIDATION_CUH

#include <cmath>

#include "core/mat4.cuh"
#include "core/vec2.cuh"
#include "core/vec3.cuh"

LuxInline bool finite_vec2(const vec2& value) {
    return std::isfinite(static_cast<double>(value.x))
        && std::isfinite(static_cast<double>(value.y));
}

LuxInline bool finite_vec3(const vec3& value) {
    return std::isfinite(static_cast<double>(value.x))
        && std::isfinite(static_cast<double>(value.y))
        && std::isfinite(static_cast<double>(value.z));
}

LuxInline bool finite_mat4(const mat4& value) {
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            if (!std::isfinite(static_cast<double>(value.m[row][col]))) {
                return false;
            }
        }
    }
    return true;
}

LuxInline bool instance_transform_matrices_finite(const Transform& transform) {
    return finite_mat4(transform.m) && finite_mat4(transform.inv);
}

LuxInline bool instance_transform_matrices_affine_invertible(
        const Transform& transform) {
    return is_affine_invertible(transform.m)
        && is_affine_invertible(transform.inv);
}

LuxInline bool instance_transform_valid(const Transform& transform) {
    return instance_transform_matrices_finite(transform)
        && instance_transform_matrices_affine_invertible(transform)
        && transform_inverse_pair_consistent(transform);
}

#endif // LUX_CORE_VALIDATION_CUH
