#pragma once

#ifndef LUX_INTEGRATOR_H
#define LUX_INTEGRATOR_H

#include "film/film.h"
#include "integrator/render_context.h"
#include "scene/prepared_scene.h"

enum class IntegratorKind {
    CpuPathTracer,
    WavefrontPathTracer,
};

class Integrator {
public:
    virtual ~Integrator() = default;
    virtual bool requires_host_bvh() const { return false; }
    virtual bool requires_gpu_scene() const { return false; }
    virtual void render(const PreparedScene& scene, Film& film, const RenderSettings& settings,
                        RenderOutputs* outputs = nullptr) = 0;
};

#endif // LUX_INTEGRATOR_H
