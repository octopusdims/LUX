#pragma once

#ifndef LUX_INTEGRATOR_WAVEFRONT_RUNTIME_CUH
#define LUX_INTEGRATOR_WAVEFRONT_RUNTIME_CUH

#include <thrust/device_vector.h>

#include "core/vec3.cuh"
#include "film/film.h"
#include "integrator/pathstate.cuh"
#include "integrator/render_context.h"
#include "integrator/wavefront_debug.cuh"
#include "integrator/wavefront_material_queues.cuh"
#include "integrator/workqueue.cuh"

class PreparedScene;

struct WavefrontRuntimeOptions {
    int image_size = 0;
    int batch_size = 0;
    bool enable_shadow_debug = false;
    bool enable_primary_debug = false;
    bool enable_normal_debug = false;
    bool enable_shading_normal_debug = false;
    bool enable_path_log = false;
};

struct WavefrontRuntime {
    static constexpr int kBlockSize = 256;
    int batch_size = 0;
    WavefrontRuntimeOptions allocated_options;
    int allocated_max_depth = -1;
    bool allocated = false;

    PathStateStorage paths;

    WorkQueueStorage<RayWorkItem> ray_queue;
    WorkQueueStorage<RayWorkItem> next_ray_queue;
    MaterialHitQueues hit_queues;
    WorkQueueStorage<ShadowRayWorkItem> shadow_queue;
    WorkQueueStorage<TerminatedPathItem> terminate_queue;

    thrust::device_vector<vec3> output_pixels;
    thrust::device_vector<vec3> shadow_debug_pixels;
    thrust::device_vector<vec3> primary_debug_pixels;
    thrust::device_vector<vec3> normal_debug_pixels;
    thrust::device_vector<vec3> shading_normal_debug_pixels;
    PathLogStorage path_log;

    ~WavefrontRuntime();

    void ensure_allocated(const WavefrontRuntimeOptions& options, int max_depth);
    bool has_compatible_allocation(const WavefrontRuntimeOptions& options,
                                   int max_depth) const;
    void free_queues();
    void check_queue_overflows() const;
    void clear_batch_queues();
    void clear_frame_outputs(const WavefrontRuntimeOptions& options);

    PathStateView path_state();
    WavefrontDebugViews debug_views();

    vec3* output_pixels_ptr();
    vec3* shadow_debug_pixels_ptr();
    vec3* primary_debug_pixels_ptr();
    vec3* normal_debug_pixels_ptr();
    vec3* shading_normal_debug_pixels_ptr();

    void append_path_log_to(RenderOutputs* outputs) const;
    void download_outputs(Film& film, RenderOutputs* outputs) const;

    void render_baseline_path_tracing(const PreparedScene& scene,
                                      Film& film,
                                      const RenderSettings& settings,
                                      RenderOutputs* outputs,
                                      const WavefrontRuntimeOptions& options);
};

#endif // LUX_INTEGRATOR_WAVEFRONT_RUNTIME_CUH
