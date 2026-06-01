#pragma once

#ifndef LUX_LIGHT_DISTRIBUTION_H
#define LUX_LIGHT_DISTRIBUTION_H

#include <vector>

#include "core/types.h"
#include "light/light.h"

struct AreaLightPrimitive {
    PrimitiveRef primitive_ref;
    Float area = 0;
};

struct LightDistribution {
    std::vector<PrimitiveRef> primitive_refs;
    std::vector<Float> primitive_areas;
    std::vector<Float> area_cdf;
    Float total_area = 0;
    std::vector<Light> lights;
    std::vector<Float> light_power_cdf;
    Float total_light_power = 0;

    bool empty() const {
        return (primitive_refs.empty() || total_area <= 0)
            && (lights.empty() || total_light_power <= 0);
    }
};

LightDistribution build_area_light_distribution(
    const std::vector<AreaLightPrimitive>& lights);
void append_light_distribution_entry(LightDistribution& distribution,
                                     const Light& light,
                                     Float power);

#endif // LUX_LIGHT_DISTRIBUTION_H
