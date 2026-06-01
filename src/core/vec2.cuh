#pragma once

#ifndef LUX_CORE_VEC2_CUH
#define LUX_CORE_VEC2_CUH

#include "types.h"

// Minimal 2D value type, primarily for UVs and screen/sample coordinates.
struct vec2 {
    Float x, y;

    LuxHDInline vec2() : x(0), y(0) {}
    LuxHDInline vec2(Float x, Float y) : x(x), y(y) {}
    LuxHDInline explicit vec2(Float s) : x(s), y(s) {}

    LuxHDInline Float operator[](int i) const { return (&x)[i]; }
    LuxHDInline Float& operator[](int i) { return (&x)[i]; }

    LuxHDInline vec2 operator+(const vec2& b) const { return vec2(x + b.x, y + b.y); }
    LuxHDInline vec2 operator-(const vec2& b) const { return vec2(x - b.x, y - b.y); }
    LuxHDInline vec2 operator*(Float s) const { return vec2(x * s, y * s); }
    LuxHDInline vec2 operator/(Float s) const { Float inv = 1 / s; return vec2(x * inv, y * inv); }

    LuxHDInline vec2& operator+=(const vec2& b) { x += b.x; y += b.y; return *this; }
    LuxHDInline vec2& operator-=(const vec2& b) { x -= b.x; y -= b.y; return *this; }
    LuxHDInline vec2& operator*=(Float s) { x *= s; y *= s; return *this; }
    LuxHDInline vec2& operator/=(Float s) { Float inv = 1 / s; x *= inv; y *= inv; return *this; }

    LuxHDInline bool operator==(const vec2& b) const { return x == b.x && y == b.y; }
    LuxHDInline bool operator!=(const vec2& b) const { return !(*this == b); }
};

LuxHDInline vec2 operator*(Float s, const vec2& v) { return v * s; }

LuxHDInline Float dot(const vec2& a, const vec2& b) { return a.x * b.x + a.y * b.y; }
LuxHDInline Float length2(const vec2& v) { return dot(v, v); }
LuxHDInline Float length(const vec2& v) { return sqrtf(length2(v)); }
#endif // LUX_CORE_VEC2_CUH
