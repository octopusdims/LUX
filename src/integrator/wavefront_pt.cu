#include "integrator/wavefront_pt.cuh"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

#include "integrator/wavefront_runtime.cuh"

namespace {

int checked_image_size(const Film& film) {
    if (film.width <= 0 || film.height <= 0 || film.samples_per_pixel <= 0) {
        throw std::runtime_error("WavefrontPathTracer requires a non-empty Film");
    }

    size_t image_size = static_cast<size_t>(film.width)
        * static_cast<size_t>(film.height);
    if (image_size > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("WavefrontPathTracer image is too large");
    }
    if (film.pixels.size() != image_size) {
        throw std::runtime_error("WavefrontPathTracer Film pixel storage does not match dimensions");
    }
    return static_cast<int>(image_size);
}

void validate_aov_film(const Film* aov, const Film& beauty, const char* label) {
    if (!aov) return;

    checked_image_size(*aov);
    if (aov->width != beauty.width || aov->height != beauty.height
        || aov->samples_per_pixel != beauty.samples_per_pixel) {
        std::string message = "WavefrontPathTracer AOV Film mismatch: ";
        message += label;
        throw std::runtime_error(message);
    }
}

} // namespace

WavefrontPathTracer::WavefrontPathTracer()
    : runtime_(std::make_unique<WavefrontRuntime>()) {}

WavefrontPathTracer::~WavefrontPathTracer() = default;

bool WavefrontPathTracer::requires_gpu_scene() const {
    return true;
}

void WavefrontPathTracer::render(const PreparedScene& scene, Film& film,
                                 const RenderSettings& settings,
                                 RenderOutputs* outputs) {
    if (!scene.has_gpu()) {
        throw std::runtime_error("WavefrontPathTracer requires PreparedScene::prepare_gpu() first");
    }
    if (settings.max_depth < 0) {
        throw std::runtime_error("WavefrontPathTracer requires max_depth >= 0");
    }
    if (settings.batch_size <= 0) {
        throw std::runtime_error("WavefrontPathTracer requires batch_size > 0");
    }

    int image_size = checked_image_size(film);
    int batch_size = std::min(image_size, settings.batch_size);

    Film* shadow_debug = outputs ? outputs->get(RenderAov::ShadowDebug) : nullptr;
    Film* normal_debug = outputs ? outputs->get(RenderAov::GeometricNormal) : nullptr;
    Film* shading_normal_debug = outputs ? outputs->get(RenderAov::ShadingNormal) : nullptr;
    validate_aov_film(shadow_debug, film, "shadow debug");
    validate_aov_film(normal_debug, film, "geometric normal");
    validate_aov_film(shading_normal_debug, film, "shading normal");

    WavefrontRuntimeOptions runtime_options;
    runtime_options.image_size = image_size;
    runtime_options.batch_size = batch_size;
    runtime_options.enable_shadow_debug = shadow_debug != nullptr;
    runtime_options.enable_normal_debug = normal_debug != nullptr;
    runtime_options.enable_shading_normal_debug = shading_normal_debug != nullptr;
    runtime_options.enable_path_log = outputs && outputs->wants_path_log();

    runtime_->render_baseline_path_tracing(
        scene, film, settings, outputs, runtime_options);
}
