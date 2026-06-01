#pragma once

#ifndef LUX_CORE_VEC3_CUH
#define LUX_CORE_VEC3_CUH

#include "types.h"

// Minimal 3D value type for points, vectors, colors, and directions.
// Kept as a simple public-data struct so it can be copied freely in kernels.
struct vec3 {
    Float x, y, z;

    LuxHDInline vec3() : x(0), y(0), z(0) {}
    LuxHDInline vec3(Float x, Float y, Float z) : x(x), y(y), z(z) {}
    LuxHDInline explicit vec3(Float s) : x(s), y(s), z(s) {}

    LuxHDInline Float operator[](int i) const { return (&x)[i]; }
    LuxHDInline Float& operator[](int i) { return (&x)[i]; }

    LuxHDInline vec3 operator-() const { return vec3(-x, -y, -z); }

    LuxHDInline vec3 operator+(const vec3& b) const { return vec3(x + b.x, y + b.y, z + b.z); }
    LuxHDInline vec3 operator-(const vec3& b) const { return vec3(x - b.x, y - b.y, z - b.z); }
    // Component-wise product, useful for RGB throughput and albedo scaling.
    LuxHDInline vec3 operator*(const vec3& b) const { return vec3(x * b.x, y * b.y, z * b.z); }
    LuxHDInline vec3 operator*(Float s) const { return vec3(x * s, y * s, z * s); }
    LuxHDInline vec3 operator/(Float s) const { Float inv = 1 / s; return vec3(x * inv, y * inv, z * inv); }

    LuxHDInline vec3& operator+=(const vec3& b) { x += b.x; y += b.y; z += b.z; return *this; }
    LuxHDInline vec3& operator-=(const vec3& b) { x -= b.x; y -= b.y; z -= b.z; return *this; }
    LuxHDInline vec3& operator*=(Float s) { x *= s; y *= s; z *= s; return *this; }
    LuxHDInline vec3& operator/=(Float s) { Float inv = 1 / s; x *= inv; y *= inv; z *= inv; return *this; }

    LuxHDInline bool operator==(const vec3& b) const { return x == b.x && y == b.y && z == b.z; }
    LuxHDInline bool operator!=(const vec3& b) const { return !(*this == b); }
};

LuxHDInline vec3 operator*(Float s, const vec3& v) { return v * s; }

LuxHDInline Float dot(const vec3& a, const vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

LuxHDInline vec3 cross(const vec3& a, const vec3& b) {
    return vec3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

LuxHDInline Float length2(const vec3& v) { return dot(v, v); }

LuxHDInline Float length(const vec3& v) { return sqrtf(length2(v)); }

LuxHDInline vec3 normalize(const vec3& v) {
    Float len = length(v);
    // Preserve zero vectors instead of producing NaNs.
    return (len > 0) ? (v / len) : v;
}

LuxHDInline vec3 abs(const vec3& v) {
    return vec3(fabsf(v.x), fabsf(v.y), fabsf(v.z));
}

LuxHDInline vec3 min(const vec3& a, const vec3& b) {
    return vec3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}

LuxHDInline vec3 max(const vec3& a, const vec3& b) {
    return vec3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

LuxHDInline Float min_component(const vec3& v) {
    return fminf(v.x, fminf(v.y, v.z));
}

LuxHDInline Float max_component(const vec3& v) {
    return fmaxf(v.x, fmaxf(v.y, v.z));
}
#endif // LUX_CORE_VEC3_CUH
