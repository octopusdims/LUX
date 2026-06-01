#pragma once

#ifndef LUX_SAMPLER_RECORD_H
#define LUX_SAMPLER_RECORD_H

#include "core/types.h"

enum class PdfMeasure : unsigned char {
    None,
    SolidAngle,
    Area,
    Discrete,
};

enum class SampleSource : unsigned char {
    None,
    Camera,
    BSDF,
    Light,
    Guide,
};

struct PdfRecord {
    Float value = 0;
    PdfMeasure measure = PdfMeasure::None;
};

struct SampleRecord {
    SampleSource source = SampleSource::None;
    PdfRecord pdf;
    unsigned flags = 0;
};

LuxHDInline PdfRecord solid_angle_pdf(Float value) {
    return PdfRecord{value, value > 0 ? PdfMeasure::SolidAngle : PdfMeasure::None};
}

LuxHDInline PdfRecord area_pdf(Float value) {
    return PdfRecord{value, value > 0 ? PdfMeasure::Area : PdfMeasure::None};
}

LuxHDInline PdfRecord discrete_pdf(Float value) {
    return PdfRecord{value, value > 0 ? PdfMeasure::Discrete : PdfMeasure::None};
}

LuxHDInline bool is_solid_angle_pdf(const PdfRecord& pdf) {
    return pdf.measure == PdfMeasure::SolidAngle && pdf.value > 0;
}

LuxHDInline Float power_heuristic(Float sampled_pdf, Float other_pdf) {
    if (sampled_pdf <= 0) return 0;
    Float a = sampled_pdf * sampled_pdf;
    Float b = fmaxf(Float(0), other_pdf);
    b *= b;
    Float denominator = a + b;
    return denominator > 0 ? a / denominator : 0;
}

#endif // LUX_SAMPLER_RECORD_H
