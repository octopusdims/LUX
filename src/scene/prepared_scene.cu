#include "scene/prepared_scene.h"

#include <stdexcept>
#include <utility>

#include <thrust/device_ptr.h>

#include "bvh/cpu_lbvh_builder.h"
#include "bvh/gpu_bvh.cuh"
#include "scene/gpu_scene.h"
#include "scene/scene_prepare.h"

struct PreparedSceneGpuStorage {
    GpuSceneData scene_data;
    std::vector<GpuBvh> blas_bvhs;
    GpuBvh tlas_bvh;
};

PreparedScene::PreparedScene(Scene scene)
    : scene_(std::move(scene)), gpu_(std::make_unique<PreparedSceneGpuStorage>()) {
    finalize_scene(scene_);
    light_distribution_ = build_scene_light_distribution(scene_);
}

PreparedScene::~PreparedScene() = default;

const Scene& PreparedScene::host_scene() const {
    return scene_;
}

const LightDistribution& PreparedScene::light_distribution() const {
    return light_distribution_;
}

void PreparedScene::prepare_host_bvh() const {
    if (host_bvh_ready_) return;

    cpu_bvh_ = CpuBvh{};
    cpu_bvh_.blas.clear();
    cpu_bvh_.blas.reserve(scene_.mesh_assets.size());
    for (int mesh_id = 0; mesh_id < static_cast<int>(scene_.mesh_assets.size()); ++mesh_id) {
        cpu_bvh_.blas.push_back(build_cpu_lbvh_tree(
            collect_scene_mesh_bvh_build_primitives(scene_.mesh_assets[mesh_id], mesh_id)));
    }
    cpu_bvh_.tlas = build_cpu_lbvh_tree(collect_scene_tlas_build_primitives(scene_));
    cpu_bvh_.has_two_level = !cpu_bvh_.tlas.empty()
        && cpu_bvh_.blas.size() >= scene_.mesh_assets.size()
        && !scene_.instances.empty();
    host_bvh_ready_ = true;
}

bool PreparedScene::has_host_bvh() const {
    return host_bvh_ready_;
}

const CpuBvh& PreparedScene::host_bvh() const {
    prepare_host_bvh();
    return cpu_bvh_;
}

void PreparedScene::prepare_gpu() {
    gpu_->scene_data.upload(scene_, light_distribution_);
    gpu_->blas_bvhs.clear();
    gpu_->blas_bvhs.reserve(scene_.mesh_assets.size());
    for (int mesh_id = 0; mesh_id < static_cast<int>(scene_.mesh_assets.size()); ++mesh_id) {
        gpu_->blas_bvhs.push_back(build_gpu_lbvh(
            collect_scene_mesh_bvh_build_primitives(scene_.mesh_assets[mesh_id], mesh_id)));
    }
    gpu_->tlas_bvh = build_gpu_lbvh(collect_scene_tlas_build_primitives(scene_));

    std::vector<GpuBvhView> blas_views;
    blas_views.reserve(gpu_->blas_bvhs.size());
    for (const GpuBvh& blas : gpu_->blas_bvhs) {
        blas_views.push_back(blas.device_view());
    }
    gpu_->scene_data.upload_blas_views(
        blas_views, gpu_->tlas_bvh.is_valid() && !scene_.instances.empty());
    refresh_gpu_views();
}

bool PreparedScene::has_gpu() const {
    return gpu_ready_;
}

GpuScene PreparedScene::device_scene() const {
    if (!gpu_ready_) {
        throw std::runtime_error("PreparedScene::prepare_gpu() must be called before device_scene()");
    }
    return gpu_scene_view_;
}

GpuBvhView PreparedScene::device_bvh() const {
    if (!gpu_ready_) {
        throw std::runtime_error("PreparedScene::prepare_gpu() must be called before device_bvh()");
    }
    return gpu_bvh_view_;
}

void PreparedScene::refresh_gpu_views() {
    gpu_scene_view_ = gpu_->scene_data.view();
    gpu_bvh_view_ = gpu_->tlas_bvh.device_view();
    gpu_ready_ = true;
}
