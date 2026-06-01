#include "cpu_diagnostics.h"

#include <cstdio>
#include <stdexcept>

#include "cpu_diagnostics_internal.h"

bool run_cpu_diagnostic_request(const Scene& scene, const CpuBvh& bvh,
                                const LightDistribution& lights,
                                const Camera& camera,
                                int width, int height,
                                const RenderSettings& settings,
                                const DebugRequest& request) {
    if (request.probe_mode == DebugProbeMode::Primary
        || request.probe_mode == DebugProbeMode::Peel) {
        lux_cpu_diagnostics_detail::print_primary_or_peel_probe(
            scene, bvh, camera, width, height, settings, request);
        return true;
    }
    if (request.probe_mode == DebugProbeMode::Path) {
        SceneLightSampler light_sampler = make_scene_light_sampler(scene, lights);
        lux_cpu_diagnostics_detail::print_path_probe(
            scene, bvh, light_sampler, camera, width, height, settings, request);
        return true;
    }
    if (request.probe_mode == DebugProbeMode::Nee) {
        SceneLightSampler light_sampler = make_scene_light_sampler(scene, lights);
        lux_cpu_diagnostics_detail::print_nee_probe(
            scene, bvh, light_sampler, camera, width, height, settings, request);
        return true;
    }

    if (request.mesh_mode == DebugMeshMode::OrientationAudit) {
        lux_cpu_diagnostics_detail::print_orientation_audit(scene);
        return true;
    }
    if (request.mesh_mode == DebugMeshMode::Winding) {
        lux_cpu_diagnostics_detail::print_winding_probe(scene, request.triangle_id);
        return true;
    }

    if (request.wants_aov(DebugAovOrientation)) {
        if (request.output_path.empty()) {
            throw std::runtime_error("-debug-aov requires -debug-output prefix");
        }
        lux_cpu_diagnostics_detail::write_cpu_orientation_aov_image(
            scene, bvh, camera, width, height, settings, request);
        return true;
    }

    return false;
}
