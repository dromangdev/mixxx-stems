#pragma once

#include "util/types.h"
#include "util/sample.h"

#include <cmath>
#include <algorithm>

namespace mixxx {

class StemDspIsolator {
  public:
    StemDspIsolator();
    ~StemDspIsolator() = default;

    void init(double sampleRate);

    // Process a buffer of interleaved stereo samples in place.
    // numFrames is the number of stereo frames (iBufferSize / 2).
    void process(
            CSAMPLE* pInOut,
            int numFrames,
            double vocalsGain,
            double drumsGain,
            double instGain);

  private:
    // Cytomic SVF State for 2nd-order section
    struct SvfSection {
        double s1_l{0.0}, s2_l{0.0};
        double s1_r{0.0}, s2_r{0.0};

        inline void step(
                double in_l, double in_r,
                double a1, double a2, double a3, double k,
                double& lp_l, double& lp_r,
                double& hp_l, double& hp_r) {
            // Left channel
            double v3_l = in_l - s2_l;
            double v1_l = a1 * s1_l + a2 * v3_l;
            double v2_l = s2_l + a2 * s1_l + a3 * v3_l;
            s1_l = 2.0 * v1_l - s1_l;
            s2_l = 2.0 * v2_l - s2_l;
            lp_l = v2_l;
            hp_l = in_l - k * v1_l - v2_l;

            // Right channel
            double v3_r = in_r - s2_r;
            double v1_r = a1 * s1_r + a2 * v3_r;
            double v2_r = s2_r + a2 * s1_r + a3 * v3_r;
            s1_r = 2.0 * v1_r - s1_r;
            s2_r = 2.0 * v2_r - s2_r;
            lp_r = v2_r;
            hp_r = in_r - k * v1_r - v2_r;
        }

        void reset() {
            s1_l = s2_l = 0.0;
            s1_r = s2_r = 0.0;
        }
    };

    double m_sampleRate{44100.0};

    // Low Crossover (220 Hz) coefficients (Butterworth Q = 1/sqrt(2) -> k = sqrt(2))
    double m_a1_low{0.0}, m_a2_low{0.0}, m_a3_low{0.0}, m_k_low{1.41421356};
    SvfSection m_svfLow1;
    SvfSection m_svfLow2; // Cascaded for 4th-order Linkwitz-Riley (24 dB/oct)

    // High Crossover (3200 Hz) coefficients
    double m_a1_high{0.0}, m_a2_high{0.0}, m_a3_high{0.0}, m_k_high{1.41421356};
    SvfSection m_svfHigh1;
    SvfSection m_svfHigh2; // Cascaded for 4th-order Linkwitz-Riley (24 dB/oct)

    // Leaky RMS Power Trackers (asymmetrical attack/decay)
    double m_powerMidM{0.0};
    double m_powerMidS{0.0};
    double m_powerHighM{0.0};
    double m_powerHighS{0.0};

    // Smooth panning & extraction ratios (interpolated to prevent any clicks or buzz)
    double m_smoothCenter{0.0};
    double m_smoothHighCenter{0.0};
    double m_smoothTransient{0.0};

    // Dual-envelope peak transient follower for drum snare/percussion detection
    double m_fastMid{0.0};
    double m_slowMid{0.0};

    // Smoothing coefficients
    double m_alphaPowerAtk{0.0};
    double m_alphaPowerDec{0.0};
    double m_alphaSmoothAtk{0.0};
    double m_alphaSmoothDec{0.0};

    double m_alphaFastAtk{0.0};
    double m_alphaFastDec{0.0};
    double m_alphaSlowAtk{0.0};
    double m_alphaSlowDec{0.0};
    double m_alphaTransient{0.0};
};

} // namespace mixxx
