#pragma once

#ifndef LUX_INTEGRATOR_FACTORY_H
#define LUX_INTEGRATOR_FACTORY_H

#include <memory>

#include "integrator/integrator.h"

std::unique_ptr<Integrator> make_integrator(IntegratorKind kind);

#endif // LUX_INTEGRATOR_FACTORY_H
