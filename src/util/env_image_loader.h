#pragma once

#ifndef LUX_UTIL_ENV_IMAGE_LOADER_H
#define LUX_UTIL_ENV_IMAGE_LOADER_H

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/vec3.cuh"
#include "scene/scene.h"

struct LoadedEnvironmentImage {
    std::vector<vec3> pixels;
    int width = 0;
    int height = 0;
};

namespace env_image_loader_detail {

LuxInline std::string lower_extension(const std::string& path) {
    std::string::size_type dot = path.find_last_of('.');
    std::string ext = dot == std::string::npos ? std::string() : path.substr(dot);
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return ext;
}

LuxInline float read_f32(std::istream& in, bool file_little_endian) {
    unsigned char bytes[4] = {};
    in.read(reinterpret_cast<char*>(bytes), 4);
    if (!in) throw std::runtime_error("failed to read PFM float");

    constexpr uint16_t endian_test = 1;
    bool host_little_endian =
        *reinterpret_cast<const unsigned char*>(&endian_test) == 1;
    if (host_little_endian != file_little_endian) {
        std::swap(bytes[0], bytes[3]);
        std::swap(bytes[1], bytes[2]);
    }

    float value = 0;
    std::memcpy(&value, bytes, sizeof(float));
    return value;
}

LuxInline std::string trim_trailing_cr(std::string line) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    return line;
}

LuxInline std::string read_token_skipping_comments(std::istream& in) {
    while (true) {
        in >> std::ws;
        if (!in) throw std::runtime_error("unexpected EOF while reading image header");
        if (in.peek() != '#') break;
        std::string comment;
        std::getline(in, comment);
    }
    std::string token;
    in >> token;
    if (!in) throw std::runtime_error("unexpected EOF while reading image header");
    return token;
}

LuxInline vec3 rgbe_to_rgb(unsigned char r, unsigned char g,
                           unsigned char b, unsigned char e) {
    if (e == 0) return vec3(0);
    float scale = std::ldexp(1.0f, static_cast<int>(e) - (128 + 8));
    return vec3(Float(r) * scale, Float(g) * scale, Float(b) * scale);
}

LuxInline void read_hdr_scanline_flat(std::istream& in,
                                      unsigned char first[4],
                                      std::vector<vec3>& pixels,
                                      int row, int width) {
    pixels[row * width] = rgbe_to_rgb(first[0], first[1], first[2], first[3]);
    for (int x = 1; x < width; ++x) {
        unsigned char rgbe[4] = {};
        in.read(reinterpret_cast<char*>(rgbe), 4);
        if (!in) throw std::runtime_error("truncated HDR scanline");
        pixels[row * width + x] = rgbe_to_rgb(rgbe[0], rgbe[1], rgbe[2], rgbe[3]);
    }
}

LuxInline void read_hdr_scanline_rle(std::istream& in,
                                     std::vector<vec3>& pixels,
                                     int row, int width) {
    std::vector<unsigned char> scanline(4 * width);
    for (int channel = 0; channel < 4; ++channel) {
        int x = 0;
        while (x < width) {
            unsigned char code = 0;
            in.read(reinterpret_cast<char*>(&code), 1);
            if (!in) throw std::runtime_error("truncated HDR RLE packet");

            if (code > 128) {
                int count = static_cast<int>(code) - 128;
                unsigned char value = 0;
                in.read(reinterpret_cast<char*>(&value), 1);
                if (!in || count <= 0 || x + count > width) {
                    throw std::runtime_error("invalid HDR RLE run");
                }
                for (int i = 0; i < count; ++i) {
                    scanline[channel * width + x++] = value;
                }
            } else {
                int count = static_cast<int>(code);
                if (count <= 0 || x + count > width) {
                    throw std::runtime_error("invalid HDR RLE literal");
                }
                in.read(reinterpret_cast<char*>(&scanline[channel * width + x]), count);
                if (!in) throw std::runtime_error("truncated HDR RLE literal");
                x += count;
            }
        }
    }

    for (int x = 0; x < width; ++x) {
        pixels[row * width + x] = rgbe_to_rgb(
            scanline[x],
            scanline[width + x],
            scanline[2 * width + x],
            scanline[3 * width + x]);
    }
}

} // namespace env_image_loader_detail

LuxInline LoadedEnvironmentImage load_pfm_environment_image(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("failed to open PFM environment image");

    std::string magic = env_image_loader_detail::read_token_skipping_comments(in);
    int width = std::stoi(env_image_loader_detail::read_token_skipping_comments(in));
    int height = std::stoi(env_image_loader_detail::read_token_skipping_comments(in));
    float scale = std::stof(env_image_loader_detail::read_token_skipping_comments(in));
    if (magic != "PF" || width <= 0 || height <= 0 || scale == 0) {
        throw std::runtime_error("invalid PFM environment image");
    }
    in.get();

    bool file_little_endian = scale < 0;
    float value_scale = std::fabs(scale);
    LoadedEnvironmentImage image;
    image.width = width;
    image.height = height;
    image.pixels.assign(width * height, vec3(0));
    for (int y = height - 1; y >= 0; --y) {
        for (int x = 0; x < width; ++x) {
            float r = env_image_loader_detail::read_f32(in, file_little_endian);
            float g = env_image_loader_detail::read_f32(in, file_little_endian);
            float b = env_image_loader_detail::read_f32(in, file_little_endian);
            image.pixels[y * width + x] =
                vec3(r * value_scale, g * value_scale, b * value_scale);
        }
    }
    return image;
}

LuxInline LoadedEnvironmentImage load_radiance_hdr_environment_image(
        const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("failed to open HDR environment image");

    bool found_format = false;
    std::string line;
    while (std::getline(in, line)) {
        line = env_image_loader_detail::trim_trailing_cr(line);
        if (line.find("FORMAT=32-bit_rle_rgbe") != std::string::npos) {
            found_format = true;
        }
        if (line.empty()) break;
    }
    if (!found_format) {
        throw std::runtime_error("HDR image must use FORMAT=32-bit_rle_rgbe");
    }

    std::string resolution;
    std::getline(in, resolution);
    std::istringstream res(resolution);
    char y_sign = 0;
    char x_sign = 0;
    char y_axis = 0;
    char x_axis = 0;
    int height = 0;
    int width = 0;
    res >> y_sign >> y_axis >> height >> x_sign >> x_axis >> width;
    if ((y_axis != 'Y' && y_axis != 'y') || (x_axis != 'X' && x_axis != 'x')
        || width <= 0 || height <= 0) {
        throw std::runtime_error("unsupported HDR resolution string");
    }

    LoadedEnvironmentImage image;
    image.width = width;
    image.height = height;
    image.pixels.assign(width * height, vec3(0));

    for (int scanline_y = 0; scanline_y < height; ++scanline_y) {
        int row = y_sign == '-' ? scanline_y : (height - 1 - scanline_y);
        unsigned char rgbe[4] = {};
        in.read(reinterpret_cast<char*>(rgbe), 4);
        if (!in) throw std::runtime_error("truncated HDR image");

        bool use_rle = width >= 8 && width <= 0x7fff
            && rgbe[0] == 2 && rgbe[1] == 2 && (rgbe[2] & 0x80u) == 0;
        if (use_rle) {
            int encoded_width = (static_cast<int>(rgbe[2]) << 8) | rgbe[3];
            if (encoded_width != width) {
                throw std::runtime_error("HDR RLE scanline width mismatch");
            }
            env_image_loader_detail::read_hdr_scanline_rle(
                in, image.pixels, row, width);
        } else {
            env_image_loader_detail::read_hdr_scanline_flat(
                in, rgbe, image.pixels, row, width);
        }
    }

    if (x_sign == '-') {
        for (int y = 0; y < height; ++y) {
            for (int x0 = 0, x1 = width - 1; x0 < x1; ++x0, --x1) {
                std::swap(image.pixels[y * width + x0],
                          image.pixels[y * width + x1]);
            }
        }
    }

    return image;
}

LoadedEnvironmentImage load_exr_environment_image(const std::string& path);

LuxInline LoadedEnvironmentImage load_environment_image(const std::string& path) {
    std::string ext = env_image_loader_detail::lower_extension(path);
    if (ext == ".hdr" || ext == ".rgbe") {
        return load_radiance_hdr_environment_image(path);
    }
    if (ext == ".pfm") {
        return load_pfm_environment_image(path);
    }
    if (ext == ".exr") {
        return load_exr_environment_image(path);
    }
    throw std::runtime_error("unsupported environment image extension");
}

LuxInline int add_image_infinite_light_from_file(Scene& scene,
                                                 const std::string& path,
                                                 const vec3& scale = vec3(1)) {
    LoadedEnvironmentImage image = load_environment_image(path);
    return add_image_infinite_light(scene, image.pixels, image.width, image.height, scale);
}

#endif // LUX_UTIL_ENV_IMAGE_LOADER_H
