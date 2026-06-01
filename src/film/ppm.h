#pragma once

#ifndef LUX_FILM_PPM_H
#define LUX_FILM_PPM_H

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "core/types.h"
#include "core/vec3.cuh"
#include "film/film.h"

namespace ppm_detail {

LuxInline void create_parent_directories(const std::string& filename) {
    std::filesystem::path path(filename);
    std::filesystem::path parent = path.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
}

LuxInline Float clamp01(Float value) {
    return std::max(Float(0), std::min(Float(1), value));
}

LuxInline vec3 clamp01(const vec3& value) {
    return vec3(clamp01(value.x), clamp01(value.y), clamp01(value.z));
}

} // namespace ppm_detail

LuxInline void write_ppm(const std::string& filename, const std::vector<vec3>& pixels,
                      int width, int height, int samples_per_pixel) {
    ppm_detail::create_parent_directories(filename);
    std::ofstream out(filename, std::ios::binary);
    out << "P6\n" << width << " " << height << "\n255\n";

    for (const vec3& pixel : pixels) {
        vec3 color = pixel * film_inv_sample_count(samples_per_pixel);
        color = vec3(color.x / (1.0f + color.x),
                     color.y / (1.0f + color.y),
                     color.z / (1.0f + color.z));
        color = ppm_detail::clamp01(color);
        unsigned char r = static_cast<unsigned char>(255.999f * powf(color.x, 1.0f / 2.2f));
        unsigned char g = static_cast<unsigned char>(255.999f * powf(color.y, 1.0f / 2.2f));
        unsigned char b = static_cast<unsigned char>(255.999f * powf(color.z, 1.0f / 2.2f));
        out.write(reinterpret_cast<const char*>(&r), 1);
        out.write(reinterpret_cast<const char*>(&g), 1);
        out.write(reinterpret_cast<const char*>(&b), 1);
    }
}

LuxInline void write_debug_ppm_linear(const std::string& filename,
                                      const std::vector<vec3>& pixels,
                                      int width, int height,
                                      int samples_per_pixel) {
    ppm_detail::create_parent_directories(filename);
    std::ofstream out(filename, std::ios::binary);
    out << "P6\n" << width << " " << height << "\n255\n";

    for (const vec3& pixel : pixels) {
        vec3 color = ppm_detail::clamp01(pixel * film_inv_sample_count(samples_per_pixel));
        unsigned char r = static_cast<unsigned char>(255.999f * color.x);
        unsigned char g = static_cast<unsigned char>(255.999f * color.y);
        unsigned char b = static_cast<unsigned char>(255.999f * color.z);
        out.write(reinterpret_cast<const char*>(&r), 1);
        out.write(reinterpret_cast<const char*>(&g), 1);
        out.write(reinterpret_cast<const char*>(&b), 1);
    }
}

LuxInline void write_ppm(const std::string& filename, const Film& film) {
    write_ppm(filename, film.pixels, film.width, film.height, film.samples_per_pixel);
}

LuxInline void write_debug_ppm_linear(const std::string& filename, const Film& film) {
    write_debug_ppm_linear(
        filename, film.pixels, film.width, film.height, film.samples_per_pixel);
}

#endif // LUX_FILM_PPM_H
