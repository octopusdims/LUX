#pragma once

#ifndef LUX_INTEGRATOR_CPU_PATH_TRACER_DIAGNOSTICS_H
#define LUX_INTEGRATOR_CPU_PATH_TRACER_DIAGNOSTICS_H

#include "bvh/cpu_bvh.h"
#include "core/vec3.cuh"
#include "scene/primitive_ref.h"
#include "scene/scene_intersection.h"

namespace cpu_path_tracer_detail {

struct ShadowTraceResult {
    vec3 transmittance = vec3(1);
    int passthrough_hits = 0;
    bool opaque_hit_found = false;
    SurfaceHit opaque_hit;
};

ShadowTraceResult trace_shadow(const Scene& scene, const CpuBvh& bvh,
                               const vec3& from, const vec3& to,
                               PrimitiveRef skip_primitive);

vec3 shadow_transmittance(const Scene& scene, const CpuBvh& bvh,
                          const vec3& from, const vec3& to,
                          int skip_triangle_id);

} // namespace cpu_path_tracer_detail

#endif // LUX_INTEGRATOR_CPU_PATH_TRACER_DIAGNOSTICS_H
