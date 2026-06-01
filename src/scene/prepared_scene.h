#pragma once

#ifndef LUX_SCENE_PREPARED_H
#define LUX_SCENE_PREPARED_H

#include <memory>
#include <vector>

#include "bvh/bvh_device.cuh"
#include "bvh/cpu_bvh.h"
#include "light/light_distribution.h"
#include "scene/gpu_scene_view.h"
#include "scene/scene.h"

struct GpuBvh;
struct PreparedSceneGpuStorage;

class PreparedScene {
public:
    explicit PreparedScene(Scene scene);
    ~PreparedScene();

    PreparedScene(const PreparedScene&) = delete;
    PreparedScene& operator=(const PreparedScene&) = delete;
    PreparedScene(PreparedScene&&) = delete;
    PreparedScene& operator=(PreparedScene&&) = delete;

    const Scene& host_scene() const;
    const LightDistribution& light_distribution() const;

    void prepare_host_bvh() const;
    bool has_host_bvh() const;
    const CpuBvh& host_bvh() const;

    void prepare_gpu();
    bool has_gpu() const;

    GpuScene device_scene() const;
    GpuBvhView device_bvh() const;

private:
    void refresh_gpu_views();

    Scene scene_;
    LightDistribution light_distribution_;
    mutable CpuBvh cpu_bvh_;
    std::unique_ptr<PreparedSceneGpuStorage> gpu_;
    GpuScene gpu_scene_view_;
    GpuBvhView gpu_bvh_view_;
    mutable bool host_bvh_ready_ = false;
    bool gpu_ready_ = false;
};

#endif // LUX_SCENE_PREPARED_H
