#pragma once

#ifndef LUX_APP_RENDER_CONFIG_H
#define LUX_APP_RENDER_CONFIG_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>

#include "core/types.h"
#include "diagnostics/diagnostic_config.h"
#include "light/light_distribution.h"

enum class OutputMode {
    DefaultRunDirectory,
    ExplicitPath,
    RunDirectory,
};

struct RenderConfig {
    int width = 256;
    int height = 256;
    int samples_per_pixel = 64;
    int max_depth = 4;
    std::string scene_name = "cornell";
    std::string output;
    OutputMode output_mode = OutputMode::DefaultRunDirectory;
    DebugRequest debug;
    bool use_gpu = false;
    bool show_timing = false;
    bool show_help = false;
    LightSamplerKind light_sampler_kind = LightSamplerKind::Power;
    std::string executable_path;
};

namespace render_config_detail {

LuxInline bool is_flag(const char* value) {
    return value[0] == '-';
}

LuxInline bool is_option(const char* value, const char* short_name, const char* long_name) {
    return std::strcmp(value, short_name) == 0 || std::strcmp(value, long_name) == 0;
}

LuxInline bool parse_positive_int(const char* text, int& value) {
    char* end = nullptr;
    long parsed = std::strtol(text, &end, 10);
    if (end == text || *end != '\0') return false;
    if (parsed <= 0 || parsed > std::numeric_limits<int>::max()) return false;
    value = static_cast<int>(parsed);
    return true;
}

LuxInline bool parse_nonnegative_int(const char* text, int& value) {
    char* end = nullptr;
    long parsed = std::strtol(text, &end, 10);
    if (end == text || *end != '\0') return false;
    if (parsed < 0 || parsed > std::numeric_limits<int>::max()) return false;
    value = static_cast<int>(parsed);
    return true;
}

LuxInline bool parse_int_value(int argc, char** argv, int& index, int& value) {
    if (index + 1 >= argc) return false;
    ++index;
    return parse_positive_int(argv[index], value);
}

LuxInline bool parse_nonnegative_int_value(int argc, char** argv, int& index, int& value) {
    if (index + 1 >= argc) return false;
    ++index;
    return parse_nonnegative_int(argv[index], value);
}

LuxInline bool parse_string_value(int argc, char** argv, int& index, std::string& value) {
    if (index + 1 >= argc) return false;
    ++index;
    value = argv[index];
    return true;
}

LuxInline bool parse_debug_probe_mode(const char* text, DebugProbeMode& mode) {
    if (std::strcmp(text, "camera-hits") == 0) {
        mode = DebugProbeMode::CameraHits;
        return true;
    }
    if (std::strcmp(text, "path") == 0) {
        mode = DebugProbeMode::Path;
        return true;
    }
    if (std::strcmp(text, "nee") == 0) {
        mode = DebugProbeMode::Nee;
        return true;
    }
    return false;
}

LuxInline bool add_debug_aov_mode(const char* text, unsigned& mask) {
    if (std::strcmp(text, "orientation") == 0) {
        mask |= DebugAovOrientation;
        return true;
    }
    if (std::strcmp(text, "shadow") == 0) {
        mask |= DebugAovShadow;
        return true;
    }
    if (std::strcmp(text, "ng") == 0) {
        mask |= DebugAovNg;
        return true;
    }
    if (std::strcmp(text, "ns") == 0) {
        mask |= DebugAovNs;
        return true;
    }
    return false;
}

LuxInline bool parse_debug_aov_list(const char* text, unsigned& mask) {
    std::string list(text);
    size_t begin = 0;
    bool parsed_any = false;
    while (begin <= list.size()) {
        size_t end = list.find(',', begin);
        std::string token = list.substr(
            begin, end == std::string::npos ? std::string::npos : end - begin);
        if (token.empty() || !add_debug_aov_mode(token.c_str(), mask)) {
            return false;
        }
        parsed_any = true;
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    return parsed_any;
}

LuxInline bool parse_debug_mesh_mode(const char* text, DebugMeshMode& mode) {
    if (std::strcmp(text, "orientation-audit") == 0) {
        mode = DebugMeshMode::OrientationAudit;
        return true;
    }
    if (std::strcmp(text, "winding") == 0) {
        mode = DebugMeshMode::Winding;
        return true;
    }
    return false;
}

LuxInline bool parse_light_sampler_kind(const char* text, LightSamplerKind& kind) {
    if (std::strcmp(text, "power") == 0) {
        kind = LightSamplerKind::Power;
        return true;
    }
    if (std::strcmp(text, "uniform") == 0) {
        kind = LightSamplerKind::Uniform;
        return true;
    }
    if (std::strcmp(text, "bvh") == 0) {
        kind = LightSamplerKind::Bvh;
        return true;
    }
    return false;
}

LuxInline bool validate_debug_request(const RenderConfig& config) {
    const DebugRequest& debug = config.debug;
    int categories = (debug.has_probe() ? 1 : 0)
        + (debug.has_mesh() ? 1 : 0)
        + (debug.has_aov() ? 1 : 0);
    if (categories > 1) return false;
    if (debug.has_probe() && !debug.has_probe_pixel) return false;
    return true;
}

} // namespace render_config_detail

LuxInline bool parse_render_config(int argc, char** argv, RenderConfig& config) {
    if (argc > 0 && argv[0]) {
        config.executable_path = argv[0];
    }
    bool saw_positional = false;

    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];

        if (render_config_detail::is_option(arg, "-help", "--help")) {
            config.show_help = true;
            continue;
        }

        if (render_config_detail::is_option(arg, "-width", "--width")) {
            if (!render_config_detail::parse_int_value(argc, argv, i, config.width)) return false;
            continue;
        }

        if (render_config_detail::is_option(arg, "-height", "--height")) {
            if (!render_config_detail::parse_int_value(argc, argv, i, config.height)) return false;
            continue;
        }

        if (render_config_detail::is_option(arg, "-size", "--size")) {
            if (!render_config_detail::parse_int_value(argc, argv, i, config.width)) return false;
            if (!render_config_detail::parse_int_value(argc, argv, i, config.height)) return false;
            continue;
        }

        if (render_config_detail::is_option(arg, "-spp", "--spp")) {
            if (!render_config_detail::parse_int_value(
                    argc, argv, i, config.samples_per_pixel)) {
                return false;
            }
            continue;
        }

        if (render_config_detail::is_option(arg, "-depth", "--depth")) {
            if (!render_config_detail::parse_nonnegative_int_value(argc, argv, i, config.max_depth)) {
                return false;
            }
            continue;
        }

        if (render_config_detail::is_option(arg, "-scene", "--scene")) {
            if (!render_config_detail::parse_string_value(argc, argv, i, config.scene_name)) {
                return false;
            }
            continue;
        }

        if (render_config_detail::is_option(arg, "-output", "--output")) {
            if (!render_config_detail::parse_string_value(argc, argv, i, config.output)) {
                return false;
            }
            config.output_mode = OutputMode::ExplicitPath;
            continue;
        }

        if (render_config_detail::is_option(arg, "-out", "--out")) {
            if (!render_config_detail::parse_string_value(argc, argv, i, config.output)) {
                return false;
            }
            config.output_mode = OutputMode::RunDirectory;
            continue;
        }

        if (render_config_detail::is_option(arg, "-gpu", "--gpu")) {
            config.use_gpu = true;
            continue;
        }

        if (render_config_detail::is_option(arg, "-light-sampler", "--light-sampler")) {
            if (i + 1 >= argc) return false;
            ++i;
            if (!render_config_detail::parse_light_sampler_kind(
                    argv[i], config.light_sampler_kind)) {
                return false;
            }
            continue;
        }

        if (render_config_detail::is_option(arg, "-time", "--time")) {
            config.show_timing = true;
            continue;
        }

        if (render_config_detail::is_option(arg, "-debug-probe", "--debug-probe")) {
            if (!render_config_detail::parse_nonnegative_int_value(
                    argc, argv, i, config.debug.pixel_x)) {
                return false;
            }
            if (!render_config_detail::parse_nonnegative_int_value(
                    argc, argv, i, config.debug.pixel_y)) {
                return false;
            }
            config.debug.has_probe_pixel = true;
            if (config.debug.probe_mode == DebugProbeMode::None) {
                config.debug.probe_mode = DebugProbeMode::CameraHits;
            }
            continue;
        }

        if (render_config_detail::is_option(arg, "-debug-mode", "--debug-mode")) {
            if (i + 1 >= argc) return false;
            ++i;
            if (!render_config_detail::parse_debug_probe_mode(
                    argv[i], config.debug.probe_mode)) {
                return false;
            }
            continue;
        }

        if (render_config_detail::is_option(arg, "-debug-aov", "--debug-aov")) {
            if (i + 1 >= argc) return false;
            ++i;
            if (!render_config_detail::parse_debug_aov_list(
                    argv[i], config.debug.aov_mask)) {
                return false;
            }
            continue;
        }

        if (render_config_detail::is_option(arg, "-debug-output", "--debug-output")) {
            if (!render_config_detail::parse_string_value(
                    argc, argv, i, config.debug.output_path)) {
                return false;
            }
            continue;
        }

        if (render_config_detail::is_option(arg, "-debug-samples", "--debug-samples")) {
            if (!render_config_detail::parse_int_value(
                    argc, argv, i, config.debug.sample_count)) {
                return false;
            }
            continue;
        }

        if (render_config_detail::is_option(arg, "-debug-max-hits",
                                            "--debug-max-hits")) {
            if (!render_config_detail::parse_int_value(
                    argc, argv, i, config.debug.max_hits)) {
                return false;
            }
            continue;
        }

        if (render_config_detail::is_option(arg, "-debug-mesh", "--debug-mesh")) {
            if (i + 1 >= argc) return false;
            ++i;
            if (!render_config_detail::parse_debug_mesh_mode(
                    argv[i], config.debug.mesh_mode)) {
                return false;
            }
            continue;
        }

        if (render_config_detail::is_option(arg, "-debug-triangle",
                                            "--debug-triangle")) {
            if (!render_config_detail::parse_nonnegative_int_value(
                    argc, argv, i, config.debug.triangle_id)) {
                return false;
            }
            if (config.debug.mesh_mode == DebugMeshMode::None) {
                config.debug.mesh_mode = DebugMeshMode::Winding;
            }
            continue;
        }

        if (render_config_detail::is_flag(arg)) return false;
        if (saw_positional || argc != 2) return false;

        config.output = arg;
        config.output_mode = OutputMode::ExplicitPath;
        saw_positional = true;
    }

    return render_config_detail::validate_debug_request(config);
}

LuxInline void print_render_usage(const char* executable, const char* scene_names) {
    std::printf("Usage:\n");
    std::printf("  %s [options]\n", executable);
    std::printf("  %s <output.ppm>\n", executable);
    std::printf("\nOptions:\n");
    std::printf("  -size W H           Set image width and height\n");
    std::printf("  -width N            Set image width\n");
    std::printf("  -height N           Set image height\n");
    std::printf("  -spp N              Set samples per pixel\n");
    std::printf("  -depth N            Set maximum path depth\n");
    std::printf("  -scene name         Set scene: %s\n", scene_names);
    std::printf("  -out name           Write to out/name/name.ppm\n");
    std::printf("  -output path        Set exact PPM/PFM output path\n");
    std::printf("  -gpu                Use GPU wavefront path tracer\n");
    std::printf("  -light-sampler kind Select light sampler: power, uniform, or bvh\n");
    std::printf("  -time               Print stage timing breakdown\n");
    std::printf("  -debug-probe X Y    Select a pixel for probe modes\n");
    std::printf("  -debug-mode mode    Probe mode: camera-hits, path, or nee\n");
    std::printf("  -debug-samples N    Set sample count for probe commands\n");
    std::printf("  -debug-max-hits N   Set max printed hits for camera-hits probe\n");
    std::printf("  -debug-aov list     AOV list: orientation, shadow, ng, ns\n");
    std::printf("                       Example: -debug-aov ng,ns,shadow\n");
    std::printf("  -debug-output path  Override debug AOV output prefix\n");
    std::printf("  -debug-mesh mode    Mesh mode: orientation-audit or winding\n");
    std::printf("  -debug-triangle N   Triangle id for winding mesh debug\n");
    std::printf("  -help               Show this help\n");
    std::printf("\nDefaults: -scene cornell -size 256 256 -spp 64 -depth 4 -out cornell\n");
}

#endif // LUX_APP_RENDER_CONFIG_H
