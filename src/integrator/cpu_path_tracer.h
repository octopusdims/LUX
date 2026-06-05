#pragma once

#ifndef LUX_INTEGRATOR_CPU_PATH_TRACER_H
#define LUX_INTEGRATOR_CPU_PATH_TRACER_H

#include "bvh/cpu_bvh.h"
#include "core/ray.cuh"
#include "core/vec3.cuh"
#include "film/film.h"
#include "integrator/integrator.h"
#include "integrator/render_context.h"
#include "light/light_distribution.h"
#include "sampler/sampler.h"
#include "scene/prepared_scene.h"
#include "scene/scene_light_sampler.h"

vec3 trace_path(const Scene& scene, const CpuBvh& bvh,
                const SceneLightSampler& lights, Ray ray, SamplerState& sampler,
                int max_depth);

vec3 trace_path(const Scene& scene, const CpuBvh& bvh,
                const PreparedLightSampling& light_sampling, Ray ray, SamplerState& sampler,
                int max_depth);

void render_cpu_path_tracer(const Scene& scene, const CpuBvh& bvh,
                            const PreparedLightSampling& light_sampling, Film& film,
                            const RenderSettings& settings,
                            RenderOutputs* outputs = nullptr);

class CpuPathTracer final : public Integrator {
public:
    bool requires_host_bvh() const override;
    bool requires_gpu_scene() const override;

    void render(const PreparedScene& scene, Film& film, const RenderSettings& settings,
                RenderOutputs* outputs = nullptr) override;
};

#endif // LUX_INTEGRATOR_CPU_PATH_TRACER_H
