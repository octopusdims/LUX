#pragma once

#ifndef LUX_CORE_RAY_CUH
#define LUX_CORE_RAY_CUH

#include "vec3.cuh"

// Ray interval is [t_min, t_max] by convention for acceleration queries.
// t_min is where self-intersection epsilons should be applied by callers.
struct Ray {
    vec3 origin;
    vec3 direction;
    Float t_min;
    Float t_max;
    Float time;   // motion blur (M6), default 0

    LuxHDInline Ray()
        : t_min(0), t_max(INFINITY), time(0) {}

    LuxHDInline Ray(const vec3& origin_, const vec3& direction_, Float t_min_ = 0,
                     Float t_max_ = INFINITY, Float time_ = 0)
        : origin(origin_), direction(direction_), t_min(t_min_), t_max(t_max_), time(time_) {}

    LuxHDInline vec3 at(Float t) const { return origin + direction * t; }
};
#endif // LUX_CORE_RAY_CUH
