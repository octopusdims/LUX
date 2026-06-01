#include "light/light_distribution.h"

LightDistribution build_area_light_distribution(
        const std::vector<AreaLightPrimitive>& lights) {
    LightDistribution distribution;
    distribution.primitive_refs.reserve(lights.size());
    distribution.primitive_areas.reserve(lights.size());
    distribution.area_cdf.reserve(lights.size());

    Float accumulated_area = 0;
    for (const AreaLightPrimitive& light : lights) {
        if (!primitive_ref_valid(light.primitive_ref) || light.area <= 0) {
            continue;
        }

        accumulated_area += light.area;
        distribution.primitive_refs.push_back(light.primitive_ref);
        distribution.primitive_areas.push_back(light.area);
        distribution.area_cdf.push_back(accumulated_area);
    }

    distribution.total_area = accumulated_area;
    return distribution;
}

void append_light_distribution_entry(LightDistribution& distribution,
                                     const Light& light,
                                     Float power) {
    if (power <= 0) return;
    distribution.total_light_power += power;
    distribution.lights.push_back(light);
    distribution.light_power_cdf.push_back(distribution.total_light_power);
}
