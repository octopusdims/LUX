#include "integrator/integrator_factory.h"

#include <stdexcept>

#include "integrator/cpu_path_tracer.h"
#include "integrator/wavefront_pt.cuh"

std::unique_ptr<Integrator> make_integrator(IntegratorKind kind) {
    switch (kind) {
    case IntegratorKind::CpuPathTracer:
        return std::make_unique<CpuPathTracer>();
    case IntegratorKind::WavefrontPathTracer:
        return std::make_unique<WavefrontPathTracer>();
    }
    throw std::runtime_error("Unknown integrator kind");
}
