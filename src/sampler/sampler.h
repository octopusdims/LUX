#pragma once

#ifndef LUX_SAMPLER_H
#define LUX_SAMPLER_H

#include <cstdint>

#include "core/pcg.cuh"
#include "core/types.h"
#include "core/vec2.cuh"

enum class SamplerKind : unsigned char {
    Independent,
    Sobol,
};

struct SamplerConfig {
    SamplerKind kind = SamplerKind::Independent;
    uint64_t seed = 0;
};

struct SamplerState {
    SamplerConfig config;
    PCG32 rng;
    int pixel_index = 0;
    int sample_index = 0;
    int dimension = 0;
};

LuxHDInline uint32_t reverse_bits32(uint32_t value) {
    value = (value << 16) | (value >> 16);
    value = ((value & 0x00ff00ffu) << 8) | ((value & 0xff00ff00u) >> 8);
    value = ((value & 0x0f0f0f0fu) << 4) | ((value & 0xf0f0f0f0u) >> 4);
    value = ((value & 0x33333333u) << 2) | ((value & 0xccccccccu) >> 2);
    value = ((value & 0x55555555u) << 1) | ((value & 0xaaaaaaaau) >> 1);
    return value;
}

LuxHDInline Float uint_to_unit_float(uint32_t value) {
    return (value >> 8) * (Float(1) / Float(16777216));
}

LuxHDInline uint32_t sampler_hash(uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
    uint32_t h = 2166136261u;
    h = (h ^ a) * 16777619u;
    h = (h ^ b) * 16777619u;
    h = (h ^ c) * 16777619u;
    h = (h ^ d) * 16777619u;
    h ^= h >> 16;
    h *= 0x7feb352du;
    h ^= h >> 15;
    h *= 0x846ca68bu;
    h ^= h >> 16;
    return h;
}

constexpr int kSobolMaxDimensions = 32;
constexpr int kSobolMaxDegree = 7;

LuxHDInline bool sobol_dimension_supported(int dimension) {
    return dimension >= 0 && dimension < kSobolMaxDimensions;
}

LuxHDInline void sobol_direction_seed(int dimension, int& degree,
                                      uint32_t& a, uint32_t m[kSobolMaxDegree]) {
    for (int i = 0; i < kSobolMaxDegree; ++i) m[i] = 0;

    // Joe-Kuo direction numbers for the first 32 dimensions. Dimension 0 is
    // the base-2 radical inverse; dimensions 1..31 use primitive-polynomial
    // seeds equivalent to the tabulated matrices used by pbrt-v4.
    switch (dimension) {
        case 1: degree = 1; a = 0;  m[0] = 1; return;
        case 2: degree = 2; a = 1;  m[0] = 1; m[1] = 3; return;
        case 3: degree = 3; a = 1;  m[0] = 1; m[1] = 3; m[2] = 1; return;
        case 4: degree = 3; a = 2;  m[0] = 1; m[1] = 1; m[2] = 1; return;
        case 5: degree = 4; a = 1;  m[0] = 1; m[1] = 1; m[2] = 3; m[3] = 3; return;
        case 6: degree = 4; a = 4;  m[0] = 1; m[1] = 3; m[2] = 5; m[3] = 13; return;
        case 7: degree = 5; a = 2;  m[0] = 1; m[1] = 1; m[2] = 5; m[3] = 5; m[4] = 17; return;
        case 8: degree = 5; a = 4;  m[0] = 1; m[1] = 1; m[2] = 5; m[3] = 5; m[4] = 5; return;
        case 9: degree = 5; a = 7;  m[0] = 1; m[1] = 1; m[2] = 7; m[3] = 11; m[4] = 19; return;
        case 10: degree = 5; a = 11; m[0] = 1; m[1] = 1; m[2] = 5; m[3] = 1; m[4] = 1; return;
        case 11: degree = 5; a = 13; m[0] = 1; m[1] = 1; m[2] = 1; m[3] = 3; m[4] = 11; return;
        case 12: degree = 5; a = 14; m[0] = 1; m[1] = 3; m[2] = 5; m[3] = 5; m[4] = 31; return;
        case 13: degree = 6; a = 1;  m[0] = 1; m[1] = 3; m[2] = 3; m[3] = 9; m[4] = 7; m[5] = 49; return;
        case 14: degree = 6; a = 13; m[0] = 1; m[1] = 1; m[2] = 1; m[3] = 15; m[4] = 21; m[5] = 21; return;
        case 15: degree = 6; a = 16; m[0] = 1; m[1] = 3; m[2] = 1; m[3] = 13; m[4] = 27; m[5] = 49; return;
        case 16: degree = 6; a = 19; m[0] = 1; m[1] = 1; m[2] = 1; m[3] = 15; m[4] = 7; m[5] = 5; return;
        case 17: degree = 6; a = 22; m[0] = 1; m[1] = 3; m[2] = 1; m[3] = 15; m[4] = 13; m[5] = 25; return;
        case 18: degree = 6; a = 25; m[0] = 1; m[1] = 1; m[2] = 5; m[3] = 5; m[4] = 19; m[5] = 61; return;
        case 19: degree = 7; a = 1;  m[0] = 1; m[1] = 3; m[2] = 7; m[3] = 11; m[4] = 23; m[5] = 15; m[6] = 103; return;
        case 20: degree = 7; a = 4;  m[0] = 1; m[1] = 3; m[2] = 7; m[3] = 13; m[4] = 13; m[5] = 15; m[6] = 69; return;
        case 21: degree = 7; a = 7;  m[0] = 1; m[1] = 1; m[2] = 3; m[3] = 13; m[4] = 7; m[5] = 35; m[6] = 63; return;
        case 22: degree = 7; a = 8;  m[0] = 1; m[1] = 3; m[2] = 5; m[3] = 9; m[4] = 1; m[5] = 25; m[6] = 53; return;
        case 23: degree = 7; a = 14; m[0] = 1; m[1] = 3; m[2] = 1; m[3] = 13; m[4] = 9; m[5] = 35; m[6] = 107; return;
        case 24: degree = 7; a = 19; m[0] = 1; m[1] = 3; m[2] = 1; m[3] = 5; m[4] = 27; m[5] = 61; m[6] = 31; return;
        case 25: degree = 7; a = 21; m[0] = 1; m[1] = 1; m[2] = 5; m[3] = 11; m[4] = 19; m[5] = 41; m[6] = 61; return;
        case 26: degree = 7; a = 28; m[0] = 1; m[1] = 3; m[2] = 5; m[3] = 3; m[4] = 3; m[5] = 13; m[6] = 69; return;
        case 27: degree = 7; a = 31; m[0] = 1; m[1] = 1; m[2] = 7; m[3] = 13; m[4] = 1; m[5] = 19; m[6] = 1; return;
        case 28: degree = 7; a = 32; m[0] = 1; m[1] = 3; m[2] = 7; m[3] = 5; m[4] = 13; m[5] = 19; m[6] = 59; return;
        case 29: degree = 7; a = 37; m[0] = 1; m[1] = 1; m[2] = 3; m[3] = 9; m[4] = 25; m[5] = 29; m[6] = 41; return;
        case 30: degree = 7; a = 41; m[0] = 1; m[1] = 3; m[2] = 5; m[3] = 13; m[4] = 23; m[5] = 1; m[6] = 55; return;
        case 31: degree = 7; a = 42; m[0] = 1; m[1] = 3; m[2] = 7; m[3] = 3; m[4] = 13; m[5] = 59; m[6] = 17; return;
        default: degree = 1; a = 0; m[0] = 1; return;
    }
}

LuxHDInline uint32_t sobol_direction_number(int dimension, int bit_index) {
    if (bit_index <= 0) return 0;
    if (bit_index > 32) bit_index = 32;
    if (dimension <= 0) return uint32_t(1) << (32 - bit_index);
    if (!sobol_dimension_supported(dimension)) return 0;

    int degree = 1;
    uint32_t a = 0;
    uint32_t m[kSobolMaxDegree];
    sobol_direction_seed(dimension, degree, a, m);

    uint32_t v[32];
    for (int i = 1; i <= degree; ++i) {
        v[i - 1] = m[i - 1] << (32 - i);
    }
    for (int i = degree + 1; i <= bit_index; ++i) {
        uint32_t value = v[i - degree - 1] ^ (v[i - degree - 1] >> degree);
        for (int k = 1; k < degree; ++k) {
            if ((a >> (degree - 1 - k)) & 1u) {
                value ^= v[i - k - 1];
            }
        }
        v[i - 1] = value;
    }
    return v[bit_index - 1];
}

LuxHDInline uint32_t sobol_bits(uint32_t index, int dimension) {
    uint32_t bits = 0;
    int bit_index = 1;
    while (index != 0 && bit_index <= 32) {
        if (index & 1u) {
            bits ^= sobol_direction_number(dimension, bit_index);
        }
        index >>= 1;
        ++bit_index;
    }
    return bits;
}

LuxHDInline Float sobol_sample(int pixel_index, int sample_index,
                               int dimension, uint64_t seed) {
    if (!sobol_dimension_supported(dimension)) {
        return uint_to_unit_float(sampler_hash(
            static_cast<uint32_t>(pixel_index),
            static_cast<uint32_t>(sample_index),
            static_cast<uint32_t>(dimension),
            static_cast<uint32_t>(seed ^ (seed >> 32))));
    }
    uint32_t index = static_cast<uint32_t>(sample_index);
    uint32_t scramble = sampler_hash(
        static_cast<uint32_t>(pixel_index),
        static_cast<uint32_t>(dimension),
        static_cast<uint32_t>(seed),
        static_cast<uint32_t>(seed >> 32));
    return uint_to_unit_float(sobol_bits(index, dimension) ^ scramble);
}

LuxHDInline SamplerState make_sampler_state(const SamplerConfig& config,
                                            int pixel_index,
                                            int sample_index,
                                            int start_dimension = 0) {
    uint64_t sample = static_cast<uint64_t>(sample_index);
    uint64_t base = config.seed;
    uint64_t state = base + static_cast<uint64_t>(pixel_index) * 747796405ULL + sample;
    uint64_t sequence = (base >> 1)
        ^ (sample * 2891336453ULL + static_cast<uint64_t>(pixel_index));
    return SamplerState{config, PCG32(state, sequence), pixel_index,
                        sample_index, start_dimension};
}

LuxHDInline void start_pixel_sample(SamplerState& sampler, int pixel_index,
                                    int sample_index, int dimension = 0) {
    sampler = make_sampler_state(sampler.config, pixel_index, sample_index, dimension);
}

LuxHDInline Float sampler_get_1d(SamplerState& sampler) {
    int dimension = sampler.dimension++;
    if (sampler.config.kind == SamplerKind::Sobol) {
        return sobol_sample(
            sampler.pixel_index, sampler.sample_index, dimension, sampler.config.seed);
    }
    return sampler.rng.next_float();
}

LuxHDInline vec2 sampler_get_2d(SamplerState& sampler) {
    return vec2(sampler_get_1d(sampler), sampler_get_1d(sampler));
}

#endif // LUX_SAMPLER_H
