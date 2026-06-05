#pragma once

#ifndef LUX_INTEGRATOR_WAVEFRONT_KERNELS_CUH
#define LUX_INTEGRATOR_WAVEFRONT_KERNELS_CUH

#include "bvh/bvh_device.cuh"
#include "camera/camera.h"
#include "integrator/render_context.h"
#include "integrator/wavefront_runtime.cuh"
#include "scene/gpu_scene_view.h"

void launch_generate_camera_rays(WavefrontRuntime& runtime,
                                 PathStateView paths,
                                 const Camera& camera,
                                 const RenderSettings& settings,
                                 int width,
                                 int height,
                                 int spp,
                                 int sample_index,
                                 int batch_pixel_offset,
                                 int batch_size);

void launch_intersect_bvh(WavefrontRuntime& runtime,
                          PathStateView paths,
                          GpuBvhView bvh,
                          GpuScene scene,
                          GpuRenderParams render_params,
                          WavefrontDebugViews debug,
                          int width,
                          int height,
                          int batch_size);

void launch_evaluate_hits(WavefrontRuntime& runtime,
                          PathStateView paths,
                          GpuScene scene,
                          GpuRenderParams render_params,
                          WavefrontDebugViews debug,
                          PathLogView path_log,
                          bool enable_path_log,
                          int width,
                          int height,
                          int wavefront_depth,
                          int max_depth,
                          int batch_size);

void launch_trace_shadow_rays(WavefrontRuntime& runtime,
                              PathStateView paths,
                              GpuBvhView bvh,
                              GpuScene scene,
                              WavefrontDebugViews debug,
                              int batch_size);

void launch_flush_batch_paths(WavefrontRuntime& runtime,
                              PathStateView paths,
                              int batch_size);

void launch_clear_bounce_queues(WavefrontRuntime& runtime);

#endif // LUX_INTEGRATOR_WAVEFRONT_KERNELS_CUH
