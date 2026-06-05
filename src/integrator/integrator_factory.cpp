#include "integrator/integrator_factory.h"

#include <stdexcept>

#include "integrator/cpu_path_tracer.h"
#if LUX_ENABLE_CUDA
#include "integrator/wavefront_pt.cuh"
#endif

std::unique_ptr<Integrator> make_integrator(IntegratorKind kind) {
    switch (kind) {
    case IntegratorKind::CpuPathTracer:
        return std::make_unique<CpuPathTracer>();
    case IntegratorKind::WavefrontPathTracer:
#if LUX_ENABLE_CUDA
        return std::make_unique<WavefrontPathTracer>();
#else
        throw std::runtime_error(
            "WavefrontPathTracer requires configuring with LUX_ENABLE_CUDA=ON");
#endif
    }
    throw std::runtime_error("Unknown integrator kind");
}
