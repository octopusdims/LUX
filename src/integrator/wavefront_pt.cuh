#pragma once

#ifndef LUX_INTEGRATOR_WAVEFRONT_PT_CUH
#define LUX_INTEGRATOR_WAVEFRONT_PT_CUH

#include <memory>

#include "film/film.h"
#include "integrator.h"
#include "integrator/render_context.h"
#include "scene/prepared_scene.h"

struct WavefrontRuntime;

class WavefrontPathTracer : public Integrator {
public:
    WavefrontPathTracer();
    ~WavefrontPathTracer() override;

    bool requires_host_bvh() const override { return false; }
    bool requires_gpu_scene() const override;

    void render(const PreparedScene& scene, Film& film, const RenderSettings& settings,
                RenderOutputs* outputs = nullptr) override;

private:
    std::unique_ptr<WavefrontRuntime> runtime_;
};

#endif // LUX_INTEGRATOR_WAVEFRONT_PT_CUH
