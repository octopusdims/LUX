#pragma once

#ifndef LUX_DIAGNOSTICS_CPU_H
#define LUX_DIAGNOSTICS_CPU_H

#include "diagnostics/diagnostic_config.h"

struct CpuBvh;
struct PreparedLightSampling;
struct Camera;
struct RenderSettings;
struct Scene;

bool run_cpu_diagnostic_request(const Scene& scene, const CpuBvh& bvh,
                                const PreparedLightSampling& light_sampling,
                                const Camera& camera,
                                int width, int height,
                                const RenderSettings& settings,
                                const DebugRequest& request);

#endif // LUX_DIAGNOSTICS_CPU_H
