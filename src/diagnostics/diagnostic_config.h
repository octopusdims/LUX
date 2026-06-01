#pragma once

#ifndef LUX_DIAGNOSTICS_H
#define LUX_DIAGNOSTICS_H

#include <string>

enum class DebugProbeMode {
    None,
    Primary,
    Peel,
    Path,
    Nee,
};

enum DebugAovMask : unsigned {
    DebugAovOrientation = 1u << 0,
    DebugAovShadow = 1u << 1,
    DebugAovPrimary = 1u << 2,
    DebugAovNg = 1u << 3,
    DebugAovNs = 1u << 4,
};

inline const char* debug_aov_name(DebugAovMask mask) {
    switch (mask) {
        case DebugAovOrientation: return "orientation";
        case DebugAovShadow: return "shadow";
        case DebugAovPrimary: return "primary";
        case DebugAovNg: return "ng";
        case DebugAovNs: return "ns";
    }
    return "unknown";
}

inline std::string debug_aov_output_path(const std::string& prefix, DebugAovMask mask) {
    std::string base = prefix;
    std::string extension = ".ppm";
    if (base.size() >= 4) {
        std::string suffix = base.substr(base.size() - 4);
        if (suffix == ".ppm" || suffix == ".pfm") {
            extension = suffix;
            base.resize(base.size() - 4);
        }
    }
    return base + "_" + debug_aov_name(mask) + extension;
}

enum class DebugMeshMode {
    None,
    OrientationAudit,
    Winding,
};

struct DebugRequest {
    DebugProbeMode probe_mode = DebugProbeMode::None;
    unsigned aov_mask = 0;
    DebugMeshMode mesh_mode = DebugMeshMode::None;
    std::string output_path;
    bool has_probe_pixel = false;
    int pixel_x = 0;
    int pixel_y = 0;
    int sample_count = 8;
    int max_hits = 32;
    int triangle_id = -1;

    bool has_probe() const { return probe_mode != DebugProbeMode::None; }
    bool has_aov() const { return aov_mask != 0; }
    bool has_mesh() const { return mesh_mode != DebugMeshMode::None; }
    bool wants_aov(DebugAovMask mask) const { return (aov_mask & mask) != 0; }
    bool has_cpu_request() const {
        return has_probe() || has_mesh() || wants_aov(DebugAovOrientation);
    }
};

#endif // LUX_DIAGNOSTICS_H
