#pragma once

#ifndef LUX_CORE_PCG_CUH
#define LUX_CORE_PCG_CUH

#include <cstdint>
#include "types.h"

// Device-side PCG-XSH-RR generator. Construct per thread from a seed and
// stream sequence; host-side seeding/upload stays outside this type.
struct PCG32 {
    uint64_t state;
    uint64_t inc;

    LuxHDInline PCG32() : state(0), inc(1) {}
    LuxHDInline PCG32(uint64_t seed, uint64_t seq = 1) {
        state = 0;
        inc = (seq << 1u) | 1u;
        next();
        state += seed;
        next();
    }

    LuxHDInline uint32_t next() {
        uint64_t old_state = state;
        state = old_state * 6364136223846793005ULL + inc;
        uint32_t xorshifted = (uint32_t)(((old_state >> 18u) ^ old_state) >> 27u);
        uint32_t rotation = (uint32_t)(old_state >> 59u);
        return (xorshifted >> rotation) | (xorshifted << ((-rotation) & 31));
    }

    LuxHDInline Float next_float() {
        // [0, 1), using upper 32 bits for precision
        return Float(next() >> 8) / Float(1 << 24);
    }

    LuxHDInline Float next_float2() {
        // Alternative: full 32-bit to [0,1)
        return Float(next()) * Float(2.3283064365386963e-10);
    }
};
#endif // LUX_CORE_PCG_CUH
