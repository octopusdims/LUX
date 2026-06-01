#pragma once

#ifndef LUX_FILM_H
#define LUX_FILM_H

#include <vector>

#include "core/types.h"
#include "core/vec3.cuh"

LuxInline Float film_inv_sample_count(int samples_per_pixel) {
    return samples_per_pixel > 0 ? Float(1) / Float(samples_per_pixel) : Float(0);
}

struct Film {
    int width = 0;
    int height = 0;
    int samples_per_pixel = 1;
    // Accumulated linear radiance. Writers and comparison helpers divide by
    // samples_per_pixel when producing averaged pixel values.
    std::vector<vec3> pixels;

    Film() = default;

    Film(int width_, int height_, int samples_per_pixel_)
        : width(width_), height(height_), samples_per_pixel(samples_per_pixel_),
          pixels(static_cast<size_t>(width_) * static_cast<size_t>(height_), vec3(0)) {}

    void resize(int width_, int height_, int samples_per_pixel_) {
        width = width_;
        height = height_;
        samples_per_pixel = samples_per_pixel_;
        pixels.assign(static_cast<size_t>(width) * static_cast<size_t>(height), vec3(0));
    }

    int size() const {
        return width * height;
    }

    bool empty() const {
        return pixels.empty();
    }

    Float inv_sample_count() const {
        return film_inv_sample_count(samples_per_pixel);
    }

    vec3 average_pixel(int index) const {
        return pixels[index] * inv_sample_count();
    }

    vec3* data() {
        return pixels.data();
    }

    const vec3* data() const {
        return pixels.data();
    }

    vec3& operator[](int index) {
        return pixels[index];
    }

    const vec3& operator[](int index) const {
        return pixels[index];
    }
};

#endif // LUX_FILM_H
