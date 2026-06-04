#include "app/render_runner.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <memory>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "bvh/cpu_bvh.h"
#include "camera/pinhole_camera.h"
#include "core/constants.h"
#include "film/film.h"
#include "film/pfm.h"
#include "film/ppm.h"
#include "integrator/integrator_factory.h"
#include "integrator/render_context.h"
#include "scene/cornell_box.h"
#include "scene/direct_light_test.h"
#include "scene/prepared_scene.h"
#include "scene/scene_intersection.h"
#include "scene/stanford_bunny.h"
#include "scene/stanford_dragon.h"
#include "scene/transparent_filter.h"
#include "scene/lux_cover.h"
#include "diagnostics/cpu/cpu_diagnostics.h"

namespace {

using Clock = std::chrono::high_resolution_clock;

std::filesystem::path executable_adjacent_assets(const RenderConfig& config) {
    if (config.executable_path.empty()) {
        return {};
    }

    std::error_code error;
    std::filesystem::path executable =
        std::filesystem::absolute(config.executable_path, error);
    if (error) {
        executable = config.executable_path;
    }

    std::filesystem::path directory = executable.parent_path();
    if (directory.empty()) {
        directory = std::filesystem::current_path();
    }
    return directory / "assets";
}

bool directory_exists(const std::filesystem::path& path) {
    if (path.empty()) {
        return false;
    }

    std::error_code error;
    return std::filesystem::is_directory(path, error) && !error;
}

std::filesystem::path resolve_demo_asset_root(const RenderConfig& config) {
    std::vector<std::filesystem::path> candidates;
    std::filesystem::path executable_assets = executable_adjacent_assets(config);
    if (!executable_assets.empty()) {
        candidates.push_back(executable_assets);
    }
    candidates.push_back(std::filesystem::current_path() / "assets");

    for (const std::filesystem::path& candidate : candidates) {
        if (directory_exists(candidate)) {
            return candidate;
        }
    }

    return candidates.empty() ? std::filesystem::path("assets") : candidates.front();
}

Float render_aspect(const RenderConfig& config) {
    return Float(config.width) / Float(config.height);
}

Scene make_cornell_scene(const RenderConfig& config) {
    Scene scene = make_cornell_box();
    scene.camera = make_pinhole_camera(
        vec3(0, 1, 3.35f), vec3(0, 1, 0.9f), 40.0f, render_aspect(config));
    return scene;
}

Scene make_direct_light_scene(const RenderConfig& config) {
    Scene scene = make_direct_light_test_scene();
    scene.camera = make_pinhole_camera(
        vec3(0, 1.2f, 3.0f), vec3(0, 0, 0), 35.0f, render_aspect(config));
    return scene;
}

Scene make_probe_base_scene(const RenderConfig& config) {
    Scene scene;
    int receiver = static_cast<int>(scene.materials.size());
    scene.materials.push_back(Material::lambert(vec3(0.75f)));
    add_quad(scene,
             vec3(-1.2f, 0.0f, 1.2f), vec3(1.2f, 0.0f, 1.2f),
             vec3(1.2f, 0.0f, -1.2f), vec3(-1.2f, 0.0f, -1.2f),
             receiver);
    scene.camera = make_pinhole_camera(
        vec3(0, 0.8f, 2.8f), vec3(0, 0, 0), 35.0f, render_aspect(config));
    return scene;
}

Scene make_point_light_scene(const RenderConfig& config) {
    Scene scene = make_probe_base_scene(config);
    add_point_light(scene, vec3(0.0f, 1.6f, 0.2f), vec3(25.0f));
    return scene;
}

Scene make_distant_light_scene(const RenderConfig& config) {
    Scene scene = make_probe_base_scene(config);
    add_distant_light(scene, normalize(vec3(-0.4f, 1.0f, 0.3f)), vec3(1.2f));
    return scene;
}

Scene make_uniform_env_scene(const RenderConfig& config) {
    Scene scene = make_probe_base_scene(config);
    add_uniform_infinite_light(scene, vec3(0.15f, 0.18f, 0.25f));
    return scene;
}

Scene make_image_env_scene(const RenderConfig& config) {
    Scene scene = make_probe_base_scene(config);
    std::vector<vec3> pixels{
        vec3(1.4f, 1.0f, 0.7f), vec3(0.2f, 0.3f, 0.8f),
        vec3(0.1f, 0.2f, 0.5f), vec3(0.8f, 1.1f, 1.5f)
    };
    add_image_infinite_light(scene, pixels, 2, 2, vec3(0.35f));
    return scene;
}

Scene make_transparent_filter_scene_from_config(const RenderConfig& config) {
    Scene scene = make_transparent_filter_scene();
    scene.camera = make_pinhole_camera(
        vec3(0, 0.85f, 3.2f), vec3(0, 0.65f, -0.7f), 38.0f,
        render_aspect(config));
    return scene;
}

Scene make_bunny_scene_from_config(const RenderConfig& config) {
    Scene scene = make_stanford_bunny_scene(resolve_demo_asset_root(config));
    scene.camera = make_pinhole_camera(
        vec3(0.0f, 0.3f, 0.5f), vec3(-0.02f, 0.11f, 0.0f), 40.0f,
        render_aspect(config));
    return scene;
}

Scene make_dragon_scene_from_config(const RenderConfig& config) {
    Scene scene = make_stanford_dragon_scene(resolve_demo_asset_root(config));
    scene.camera = make_pinhole_camera(
        vec3(0.0f, 0.35f, 0.8f), vec3(0.0f, 0.14f, 0.0f), 42.0f,
        render_aspect(config));
    return scene;
}

Scene make_lux_cover_scene_from_config(const RenderConfig& config) {
    Scene scene = make_lux_cover_scene(resolve_demo_asset_root(config));
    scene.camera = make_pinhole_camera(
        vec3(0.0f, 0.52f, -3.0f), vec3(0.0f, 0.36f, 0.0f), 32.0f,
        render_aspect(config));
    return scene;
}

struct SceneDescriptor {
    const char* name;
    Scene (*make_scene)(const RenderConfig&);
};

constexpr SceneDescriptor kSceneRegistry[] = {
    {"cornell", make_cornell_scene},
    {"direct-light", make_direct_light_scene},
    {"point-light", make_point_light_scene},
    {"distant-light", make_distant_light_scene},
    {"uniform-env", make_uniform_env_scene},
    {"image-env", make_image_env_scene},
    {"transparent-filter", make_transparent_filter_scene_from_config},
    {"bunny", make_bunny_scene_from_config},
    {"dragon", make_dragon_scene_from_config},
    {"lux-cover", make_lux_cover_scene_from_config},
};

const SceneDescriptor* find_scene_descriptor(const std::string& name) {
    for (const SceneDescriptor& descriptor : kSceneRegistry) {
        if (name == descriptor.name) return &descriptor;
    }
    return nullptr;
}

std::string available_scene_names() {
    std::string names;
    for (const SceneDescriptor& descriptor : kSceneRegistry) {
        if (!names.empty()) names += ", ";
        names += descriptor.name;
    }
    return names;
}

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

bool is_pfm_path(const std::string& path) {
    return path.size() >= 4 && path.substr(path.size() - 4) == ".pfm";
}

struct OutputPaths {
    std::string beauty;
    std::string debug_prefix;
};

std::string run_directory_output_path(const std::string& run_name) {
    std::filesystem::path run_path(run_name);
    std::string file_stem = run_path.filename().string();
    if (file_stem.empty()) file_stem = "render";
    return (std::filesystem::path("out") / run_path / (file_stem + ".ppm")).string();
}

OutputPaths resolve_output_paths(const RenderConfig& config) {
    OutputPaths paths;
    if (config.output_mode == OutputMode::ExplicitPath) {
        paths.beauty = config.output;
    } else if (config.output_mode == OutputMode::RunDirectory) {
        paths.beauty = run_directory_output_path(config.output);
    } else {
        paths.beauty = run_directory_output_path(config.scene_name);
    }

    paths.debug_prefix = config.debug.output_path.empty()
        ? paths.beauty
        : config.debug.output_path;
    return paths;
}

void write_pixels(const std::string& path, const Film& film) {
    if (is_pfm_path(path)) {
        write_pfm(path, film);
    } else {
        write_ppm(path, film);
    }
}

void write_debug_pixels(const std::string& path, const Film& film) {
    if (is_pfm_path(path)) {
        write_pfm(path, film);
    } else {
        write_debug_ppm_linear(path, film);
    }
}

void allocate_if_enabled(bool enabled, Film& film, const RenderConfig& config) {
    if (enabled) {
        film.resize(config.width, config.height, config.samples_per_pixel);
    }
}

struct StageTimings {
    double scene_setup_ms = 0.0;
    double prepared_scene_ms = 0.0;
    double host_bvh_ms = 0.0;
    double cpu_diagnostic_ms = 0.0;
    double gpu_prepare_ms = 0.0;
    double render_ms = 0.0;
    double write_ms = 0.0;
};

void print_timing_breakdown(const StageTimings& timings,
                            Clock::time_point total_start,
                            Clock::time_point total_end) {
    std::printf("Timing breakdown:\n");
    std::printf("  scene setup: %.3f ms\n", timings.scene_setup_ms);
    std::printf("  prepared scene construction: %.3f ms\n", timings.prepared_scene_ms);
    std::printf("  host TLAS/BLAS LBVH build: %.3f ms\n", timings.host_bvh_ms);
    std::printf("  CPU diagnostic: %.3f ms\n", timings.cpu_diagnostic_ms);
    std::printf("  GPU scene preparation: %.3f ms\n", timings.gpu_prepare_ms);
    std::printf("  render: %.3f ms\n", timings.render_ms);
    std::printf("  output write: %.3f ms\n", timings.write_ms);
    std::printf("  total: %.3f ms\n", elapsed_ms(total_start, total_end));
}

} // namespace

std::string available_render_scene_names() {
    return available_scene_names();
}

int run_render(const RenderConfig& config) {
    auto total_start = Clock::now();
    StageTimings timings;
    auto finish = [&](int code) {
        if (config.show_timing) {
            print_timing_breakdown(timings, total_start, Clock::now());
        }
        return code;
    };

    const SceneDescriptor* scene_descriptor = find_scene_descriptor(config.scene_name);
    if (!scene_descriptor) {
        std::string names = available_render_scene_names();
        std::fprintf(stderr, "Unknown scene '%s'. Available scenes: %s\n",
                     config.scene_name.c_str(), names.c_str());
        return finish(1);
    }

    auto scene_setup_start = Clock::now();
    Scene scene;
    try {
        scene = scene_descriptor->make_scene(config);
    } catch (const std::exception& error) {
        timings.scene_setup_ms = elapsed_ms(scene_setup_start, Clock::now());
        std::fprintf(stderr, "Scene setup failed: %s\n", error.what());
        return finish(1);
    }
    OutputPaths output_paths = resolve_output_paths(config);
    DebugRequest debug_request = config.debug;
    if (debug_request.has_aov() && debug_request.output_path.empty()) {
        debug_request.output_path = output_paths.debug_prefix;
    }
    RenderSettings settings;
    settings.max_depth = config.max_depth;
    IntegratorKind integrator_kind = config.use_gpu
        ? IntegratorKind::WavefrontPathTracer
        : IntegratorKind::CpuPathTracer;
    std::unique_ptr<Integrator> integrator = make_integrator(integrator_kind);
    auto scene_setup_end = Clock::now();
    timings.scene_setup_ms = elapsed_ms(scene_setup_start, scene_setup_end);

    auto prepared_scene_start = Clock::now();
    std::unique_ptr<PreparedScene> prepared_scene;
    try {
        prepared_scene = std::make_unique<PreparedScene>(std::move(scene));
    } catch (const std::exception& error) {
        timings.prepared_scene_ms = elapsed_ms(prepared_scene_start, Clock::now());
        std::fprintf(stderr, "Prepared scene construction failed: %s\n", error.what());
        return finish(1);
    }
    auto prepared_scene_end = Clock::now();
    timings.prepared_scene_ms = elapsed_ms(prepared_scene_start, prepared_scene_end);

    const Scene& host_scene = prepared_scene->host_scene();
    const Camera& camera = host_scene.camera;
    const LightDistribution& scene_lights = prepared_scene->light_distribution();
    unsigned render_aov_mask = debug_request.has_aov()
        ? (debug_request.aov_mask & ~DebugAovOrientation)
        : 0;

    bool needs_host_bvh = integrator->requires_host_bvh() || debug_request.has_cpu_request();
    const CpuBvh* host_bvh = nullptr;
    if (needs_host_bvh) {
        auto host_bvh_start = Clock::now();
        prepared_scene->prepare_host_bvh();
        auto host_bvh_end = Clock::now();
        timings.host_bvh_ms += elapsed_ms(host_bvh_start, host_bvh_end);

        host_bvh = &prepared_scene->host_bvh();
        CpuBvhStats stats = cpu_bvh_stats(*host_bvh);
        std::printf("Built host TLAS/BLAS LBVH: %zu TLAS nodes, %zu instances, "
                    "%zu BLAS nodes, %zu mesh primitives\n",
                    stats.tlas_nodes, stats.tlas_primitives,
                    stats.blas_nodes, stats.blas_primitives);
    }

    if (debug_request.has_probe() || debug_request.has_mesh()) {
        auto diagnostic_start = Clock::now();
        bool handled = false;
        try {
            handled = run_cpu_diagnostic_request(
                host_scene, *host_bvh, scene_lights, camera,
                config.width, config.height, settings, debug_request);
        } catch (const std::exception& error) {
            auto diagnostic_end = Clock::now();
            timings.cpu_diagnostic_ms += elapsed_ms(diagnostic_start, diagnostic_end);
            std::fprintf(stderr, "CPU diagnostic failed: %s\n", error.what());
            return finish(1);
        }
        auto diagnostic_end = Clock::now();
        timings.cpu_diagnostic_ms += elapsed_ms(diagnostic_start, diagnostic_end);
        if (handled) {
            return finish(0);
        }
    }

    if (debug_request.has_aov()) {
        if (debug_request.wants_aov(DebugAovOrientation)) {
            DebugRequest orientation_request = debug_request;
            orientation_request.aov_mask = DebugAovOrientation;
            auto diagnostic_start = Clock::now();
            try {
                run_cpu_diagnostic_request(
                    host_scene, *host_bvh, scene_lights, camera,
                    config.width, config.height, settings, orientation_request);
            } catch (const std::exception& error) {
                auto diagnostic_end = Clock::now();
                timings.cpu_diagnostic_ms += elapsed_ms(diagnostic_start, diagnostic_end);
                std::fprintf(stderr, "CPU diagnostic failed: %s\n", error.what());
                return finish(1);
            }
            auto diagnostic_end = Clock::now();
            timings.cpu_diagnostic_ms += elapsed_ms(diagnostic_start, diagnostic_end);
        }
        if (render_aov_mask == 0) {
            return finish(0);
        }
    }

    if (integrator->requires_gpu_scene()) {
        auto gpu_prepare_start = Clock::now();
        try {
            prepared_scene->prepare_gpu();
        } catch (const std::exception& error) {
            auto gpu_prepare_end = Clock::now();
            timings.gpu_prepare_ms += elapsed_ms(gpu_prepare_start, gpu_prepare_end);
            std::fprintf(stderr, "GPU scene preparation failed: %s\n", error.what());
            return finish(1);
        }
        auto gpu_prepare_end = Clock::now();
        timings.gpu_prepare_ms += elapsed_ms(gpu_prepare_start, gpu_prepare_end);

        GpuBvhView gpu_bvh = prepared_scene->device_bvh();
        std::printf("Built GPU TLAS/BLAS: %d TLAS nodes, %zu instances, %zu meshes\n",
                    gpu_bvh.node_count, host_scene.instances.size(),
                    host_scene.mesh_assets.size());
    }

    Film beauty(config.width, config.height, config.samples_per_pixel);
    Film shadow_debug_film;
    Film normal_debug_film;
    Film shading_normal_debug_film;
    RenderOutputs render_outputs;
    bool enable_shadow_aov = (render_aov_mask & DebugAovShadow) != 0;
    bool enable_ng_aov = (render_aov_mask & DebugAovNg) != 0;
    bool enable_ns_aov = (render_aov_mask & DebugAovNs) != 0;
    bool enable_render_aov = enable_shadow_aov || enable_ng_aov || enable_ns_aov;
    allocate_if_enabled(enable_shadow_aov, shadow_debug_film, config);
    allocate_if_enabled(enable_ng_aov, normal_debug_film, config);
    allocate_if_enabled(enable_ns_aov, shading_normal_debug_film, config);
    render_outputs.set(RenderAov::ShadowDebug,
                       enable_shadow_aov ? &shadow_debug_film : nullptr);
    render_outputs.set(RenderAov::GeometricNormal,
                       enable_ng_aov ? &normal_debug_film : nullptr);
    render_outputs.set(RenderAov::ShadingNormal,
                       enable_ns_aov ? &shading_normal_debug_film : nullptr);

    if (config.use_gpu) {
        try {
            auto render_start = Clock::now();
            integrator->render(*prepared_scene, beauty, settings,
                               enable_render_aov ? &render_outputs : nullptr);
            auto render_end = Clock::now();
            timings.render_ms += elapsed_ms(render_start, render_end);
            std::printf("GPU render-only time: %.3f ms\n",
                        timings.render_ms);
        } catch (const std::exception& error) {
            std::fprintf(stderr, "GPU render failed: %s\n", error.what());
            return finish(1);
        }
    } else {
        auto render_start = Clock::now();
        integrator->render(*prepared_scene, beauty, settings,
                           enable_render_aov ? &render_outputs : nullptr);
        auto render_end = Clock::now();
        timings.render_ms += elapsed_ms(render_start, render_end);
        std::printf("CPU render-only time: %.3f ms\n",
                    timings.render_ms);
    }

    auto write_start = Clock::now();
    write_pixels(output_paths.beauty, beauty);
    std::printf("Wrote %s (%dx%d, %d spp, depth %d)\n",
                output_paths.beauty.c_str(), config.width, config.height, config.samples_per_pixel,
                config.max_depth);
    if (enable_render_aov) {
        const char* backend_label = config.use_gpu ? "GPU" : "CPU";
        if (enable_shadow_aov) {
            std::string path = debug_aov_output_path(debug_request.output_path, DebugAovShadow);
            write_debug_pixels(path, shadow_debug_film);
            std::printf("Wrote %s (%s shadow classification AOV)\n",
                        path.c_str(), backend_label);
        }
        if (enable_ng_aov) {
            std::string path = debug_aov_output_path(debug_request.output_path, DebugAovNg);
            write_debug_pixels(path, normal_debug_film);
            std::printf("Wrote %s (%s geometric normal AOV)\n",
                        path.c_str(), backend_label);
        }
        if (enable_ns_aov) {
            std::string path = debug_aov_output_path(debug_request.output_path, DebugAovNs);
            write_debug_pixels(path, shading_normal_debug_film);
            std::printf("Wrote %s (%s shading normal AOV)\n",
                        path.c_str(), backend_label);
        }
    }
    auto write_end = Clock::now();
    timings.write_ms += elapsed_ms(write_start, write_end);
    return finish(0);
}
