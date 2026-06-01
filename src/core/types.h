#pragma once

#ifndef LUX_CORE_TYPES_H
#define LUX_CORE_TYPES_H

#include <cmath>

// Central scalar type used by core math. Keep this alias narrow so precision
// experiments can start here instead of touching every math header.
using Float = float;

// Functions in core are shared by host setup code and CUDA kernels.
#ifdef __CUDACC__
#define LuxHost __host__
#define LuxDevice __device__
#define LuxHD __host__ __device__
#define LuxInline inline
#define LuxForceInline __forceinline__
#define LuxHDInline __host__ __device__ __forceinline__
#define LuxDeviceInline __device__ __forceinline__
#else
#define LuxHost
#define LuxDevice
#define LuxHD
#define LuxInline inline
#define LuxForceInline inline
#define LuxHDInline inline
#define LuxDeviceInline inline
#endif

#endif // LUX_CORE_TYPES_H
