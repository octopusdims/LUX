#pragma once

#ifndef LUX_INTEGRATOR_RENDER_CONTEXT_H
#define LUX_INTEGRATOR_RENDER_CONTEXT_H

#include <array>
#include <cstdint>
#include <vector>

#include "film/film.h"
#include "sampler/sampler.h"

struct PathLogRecord;

struct RenderSettings {
    int max_depth = 1;
    int batch_size = 65536;
    uint64_t seed = 0;
    int frame_index = 0;
    int sample_offset = 0;
    SamplerKind sampler_kind = SamplerKind::Independent;
};

LuxHDInline SamplerConfig make_render_sampler_config(const RenderSettings& settings) {
    uint64_t frame = static_cast<uint64_t>(settings.frame_index);
    return SamplerConfig{
        settings.sampler_kind,
        settings.seed ^ (frame * 0x9e3779b97f4a7c15ULL)
    };
}

LuxHDInline SamplerState make_render_sampler(const RenderSettings& settings,
                                             int pixel_index,
                                             int sample_index,
                                             int start_dimension = 0) {
    return make_sampler_state(
        make_render_sampler_config(settings),
        pixel_index,
        sample_index + settings.sample_offset,
        start_dimension);
}

enum class RenderAov {
    ShadowDebug = 0,
    GeometricNormal,
    ShadingNormal,
    Count
};

struct RenderOutputs {
    std::array<Film*, static_cast<size_t>(RenderAov::Count)> aovs{};
    std::vector<PathLogRecord>* path_log_records = nullptr;

    Film* get(RenderAov aov) const {
        return aovs[static_cast<size_t>(aov)];
    }

    void set(RenderAov aov, Film* film) {
        aovs[static_cast<size_t>(aov)] = film;
    }

    bool wants(RenderAov aov) const {
        return get(aov) != nullptr;
    }

    void set_path_log(std::vector<PathLogRecord>* records) {
        path_log_records = records;
    }

    bool wants_path_log() const {
        return path_log_records != nullptr;
    }
};

#endif // LUX_INTEGRATOR_RENDER_CONTEXT_H
