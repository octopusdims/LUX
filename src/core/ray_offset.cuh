#pragma once

#ifndef LUX_CORE_RAY_OFFSET_CUH
#define LUX_CORE_RAY_OFFSET_CUH

#include "core/constants.h"
#include "core/vec3.cuh"

LuxHDInline vec3 offset_ray_origin_along_normal(const vec3& position,
                                                const vec3& geometric_normal,
                                                const vec3& direction) {
    vec3 offset_normal = dot(direction, geometric_normal) >= 0
        ? geometric_normal
        : -geometric_normal;
    return position + offset_normal * kRayEpsilon;
}

LuxHDInline vec3 offset_ray_origin_along_direction(const vec3& position,
                                                   const vec3& direction) {
    return position + direction * kRayEpsilon;
}

#endif // LUX_CORE_RAY_OFFSET_CUH
