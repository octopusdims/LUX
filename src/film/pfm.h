#pragma once

#ifndef LUX_FILM_PFM_H
#define LUX_FILM_PFM_H

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "core/types.h"
#include "core/vec3.cuh"
#include "film/film.h"

namespace pfm_detail {

LuxInline void create_parent_directories(const std::string& filename) {
    std::filesystem::path path(filename);
    std::filesystem::path parent = path.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
}

} // namespace pfm_detail

LuxInline void write_pfm(const std::string& filename, const std::vector<vec3>& pixels,
                          int width, int height, int samples_per_pixel) {
    pfm_detail::create_parent_directories(filename);
    std::ofstream out(filename, std::ios::binary);
    out << "PF\n" << width << " " << height << "\n-1.0\n";

    for (int y = height - 1; y >= 0; --y) {
        for (int x = 0; x < width; ++x) {
            vec3 color = pixels[y * width + x] * film_inv_sample_count(samples_per_pixel);
            float rgb[3] = {color.x, color.y, color.z};
            out.write(reinterpret_cast<const char*>(rgb), sizeof(rgb));
        }
    }
}

LuxInline void write_pfm(const std::string& filename, const Film& film) {
    write_pfm(filename, film.pixels, film.width, film.height, film.samples_per_pixel);
}

#endif // LUX_FILM_PFM_H
