#include "scene/prepared_scene.h"

#include <stdexcept>
#include <utility>

#include "bvh/cpu_lbvh_builder.h"
#include "scene/scene_prepare.h"

struct PreparedSceneGpuStorage {};

PreparedScene::PreparedScene(Scene scene)
    : scene_(std::move(scene)), gpu_(std::make_unique<PreparedSceneGpuStorage>()) {
    finalize_scene(scene_);
    light_sampling_ = prepare_scene_light_sampling(scene_);
}

PreparedScene::~PreparedScene() = default;

const Scene& PreparedScene::host_scene() const {
    return scene_;
}

const PreparedLightSampling& PreparedScene::light_sampling() const {
    return light_sampling_;
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
    throw std::runtime_error("PreparedScene::prepare_gpu() requires LUX_ENABLE_CUDA=ON");
}

bool PreparedScene::has_gpu() const {
    return false;
}

GpuScene PreparedScene::device_scene() const {
    throw std::runtime_error("PreparedScene::device_scene() requires LUX_ENABLE_CUDA=ON");
}

GpuBvhView PreparedScene::device_bvh() const {
    throw std::runtime_error("PreparedScene::device_bvh() requires LUX_ENABLE_CUDA=ON");
}

void PreparedScene::refresh_gpu_views() {
    gpu_ready_ = false;
}
