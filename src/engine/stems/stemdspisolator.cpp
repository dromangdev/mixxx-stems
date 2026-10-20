#include "engine/stems/stemdspisolator.h"

#include <cmath>
#include <algorithm>

namespace mixxx {

namespace {
constexpr double kLowCrossoverHz = 220.0;
constexpr double kHighCrossoverHz = 3200.0;
constexpr double kPi = 3.14159265358979323846;
} // namespace

StemDspIsolator::StemDspIsolator() {
    init(44100.0);
}

void StemDspIsolator::init(double sampleRate) {
    if (sampleRate < 8000.0) {
        sampleRate = 44100.0;
    }
    m_sampleRate = sampleRate;

    // 1. Low Crossover SVF coefficients (Butterworth Q = 1/sqrt(2) -> k = sqrt(2))
    m_k_low = std::sqrt(2.0);
    double g_low = std::tan(kPi * kLowCrossoverHz / m_sampleRate);
    m_a1_low = 1.0 / (1.0 + g_low * (g_low + m_k_low));
    m_a2_low = g_low * m_a1_low;
    m_a3_low = g_low * m_a2_low;

    // 2. High Crossover SVF coefficients
    m_k_high = std::sqrt(2.0);
    double g_high = std::tan(kPi * kHighCrossoverHz / m_sampleRate);
    m_a1_high = 1.0 / (1.0 + g_high * (g_high + m_k_high));
    m_a2_high = g_high * m_a1_high;
    m_a3_high = g_high * m_a2_high;

    // 3. Smoothing coefficients
    // Power tracker: fast attack (3 ms) to grab vocal immediately, smooth decay (25 ms) across cycles
    m_alphaPowerAtk = std::exp(-1.0 / (0.003 * m_sampleRate));
    m_alphaPowerDec = std::exp(-1.0 / (0.025 * m_sampleRate));

    // Ratio smoother: fast attack (3 ms), smooth decay (15 ms)
    m_alphaSmoothAtk = std::exp(-1.0 / (0.003 * m_sampleRate));
    m_alphaSmoothDec = std::exp(-1.0 / (0.015 * m_sampleRate));

    // Transient detector smoothing
    m_alphaFastAtk = std::exp(-1.0 / (0.0005 * m_sampleRate)); // 0.5 ms
    m_alphaFastDec = std::exp(-1.0 / (0.025 * m_sampleRate));  // 25 ms
    m_alphaSlowAtk = std::exp(-1.0 / (0.005 * m_sampleRate));  // 5 ms
    m_alphaSlowDec = std::exp(-1.0 / (0.120 * m_sampleRate));  // 120 ms
    m_alphaTransient = std::exp(-1.0 / (0.008 * m_sampleRate));// 8 ms

    // Reset filters and states
    m_svfLow1.reset();
    m_svfLow2.reset();
    m_svfHigh1.reset();
    m_svfHigh2.reset();

    m_powerMidM = 0.0;
    m_powerMidS = 0.0;
    m_powerHighM = 0.0;
    m_powerHighS = 0.0;

    m_smoothCenter = 0.0;
    m_smoothHighCenter = 0.0;
    m_smoothTransient = 0.0;

    m_fastMid = 0.0;
    m_slowMid = 0.0;
}

void StemDspIsolator::process(
        CSAMPLE* pInOut,
        int numFrames,
        double vocalsGain,
        double drumsGain,
        double instGain) {
    if (!pInOut || numFrames <= 0) {
        return;
    }

    // Fast bit-exact bypass if all gains are practically unity
    if (std::abs(vocalsGain - 1.0) < 0.0001 &&
        std::abs(drumsGain - 1.0) < 0.0001 &&
        std::abs(instGain - 1.0) < 0.0001) {
        return;
    }

    for (int i = 0; i < numFrames; ++i) {
        const double in_l = static_cast<double>(pInOut[2 * i]);
        const double in_r = static_cast<double>(pInOut[2 * i + 1]);

        // =====================================================================
        // Step 1: 4th-Order Linkwitz-Riley Low Crossover (220 Hz)
        // Two cascaded Butterworth stages give 24 dB/octave slope for clean,
        // punchy bass isolation without any bleed into the mid vocal range.
        // =====================================================================
        double lp1_l, lp1_r, hp1_l, hp1_r;
        m_svfLow1.step(in_l, in_r, m_a1_low, m_a2_low, m_a3_low, m_k_low,
                       lp1_l, lp1_r, hp1_l, hp1_r);

        double low_l, low_r, hp2_l, hp2_r;
        m_svfLow2.step(lp1_l, lp1_r, m_a1_low, m_a2_low, m_a3_low, m_k_low,
                       low_l, low_r, hp2_l, hp2_r);

        // Exact complementary flat-magnitude sum
        const double midhigh_l = in_l - low_l;
        const double midhigh_r = in_r - low_r;

        // =====================================================================
        // Step 2: 4th-Order Linkwitz-Riley High Crossover (3200 Hz)
        // Splits midhigh into mid (220 Hz - 3200 Hz) and high (> 3200 Hz).
        // =====================================================================
        double lp3_l, lp3_r, hp3_l, hp3_r;
        m_svfHigh1.step(midhigh_l, midhigh_r, m_a1_high, m_a2_high, m_a3_high, m_k_high,
                        lp3_l, lp3_r, hp3_l, hp3_r);

        double mid_l, mid_r, hp4_l, hp4_r;
        m_svfHigh2.step(lp3_l, lp3_r, m_a1_high, m_a2_high, m_a3_high, m_k_high,
                        mid_l, mid_r, hp4_l, hp4_r);

        const double high_l = midhigh_l - mid_l;
        const double high_r = midhigh_r - mid_r;

        // =====================================================================
        // Step 3: Mid-Band Spatial Vocal Extraction (Smoothed Power Follower)
        // Uses RMS leaky power averaging over 25ms to completely eliminate
        // sample-rate chatter and intermodulation buzzing.
        // =====================================================================
        const double mid_m = (mid_l + mid_r) * 0.5;
        const double mid_s = (mid_l - mid_r) * 0.5;

        const double inst_p_m = mid_m * mid_m;
        const double inst_p_s = mid_s * mid_s;
        if (inst_p_m > m_powerMidM) {
            m_powerMidM = m_alphaPowerAtk * m_powerMidM + (1.0 - m_alphaPowerAtk) * inst_p_m;
        } else {
            m_powerMidM = m_alphaPowerDec * m_powerMidM + (1.0 - m_alphaPowerDec) * inst_p_m;
        }

        if (inst_p_s > m_powerMidS) {
            m_powerMidS = m_alphaPowerAtk * m_powerMidS + (1.0 - m_alphaPowerAtk) * inst_p_s;
        } else {
            m_powerMidS = m_alphaPowerDec * m_powerMidS + (1.0 - m_alphaPowerDec) * inst_p_s;
        }

        double targetCenter = 0.0;
        const double midTotalPower = m_powerMidM + m_powerMidS;
        if (midTotalPower > 1e-8) {
            double diff = m_powerMidM - 1.3 * m_powerMidS;
            if (diff > 0.0) {
                targetCenter = std::min(1.0, diff / (m_powerMidM + 1e-8));
            }
        }
        if (targetCenter > m_smoothCenter) {
            m_smoothCenter = m_alphaSmoothAtk * m_smoothCenter + (1.0 - m_alphaSmoothAtk) * targetCenter;
        } else {
            m_smoothCenter = m_alphaSmoothDec * m_smoothCenter + (1.0 - m_alphaSmoothDec) * targetCenter;
        }

        // =====================================================================
        // Step 4: High-Band Vocal Presence / Air Extraction
        // =====================================================================
        const double high_m = (high_l + high_r) * 0.5;
        const double high_s = (high_l - high_r) * 0.5;

        const double high_p_m = high_m * high_m;
        const double high_p_s = high_s * high_s;
        if (high_p_m > m_powerHighM) {
            m_powerHighM = m_alphaPowerAtk * m_powerHighM + (1.0 - m_alphaPowerAtk) * high_p_m;
        } else {
            m_powerHighM = m_alphaPowerDec * m_powerHighM + (1.0 - m_alphaPowerDec) * high_p_m;
        }

        if (high_p_s > m_powerHighS) {
            m_powerHighS = m_alphaPowerAtk * m_powerHighS + (1.0 - m_alphaPowerAtk) * high_p_s;
        } else {
            m_powerHighS = m_alphaPowerDec * m_powerHighS + (1.0 - m_alphaPowerDec) * high_p_s;
        }

        double targetHighCenter = 0.0;
        const double highTotalPower = m_powerHighM + m_powerHighS;
        if (highTotalPower > 1e-8) {
            double diff = m_powerHighM - 1.3 * m_powerHighS;
            if (diff > 0.0) {
                targetHighCenter = std::min(1.0, diff / (m_powerHighM + 1e-8));
            }
        }
        if (targetHighCenter > m_smoothHighCenter) {
            m_smoothHighCenter = m_alphaSmoothAtk * m_smoothHighCenter + (1.0 - m_alphaSmoothAtk) * targetHighCenter;
        } else {
            m_smoothHighCenter = m_alphaSmoothDec * m_smoothHighCenter + (1.0 - m_alphaSmoothDec) * targetHighCenter;
        }

        // =====================================================================
        // Step 5: Drum Snare & Percussive Transient Follower
        // =====================================================================
        const double midEnergy = midTotalPower;
        if (midEnergy > m_fastMid) {
            m_fastMid = m_alphaFastAtk * m_fastMid + (1.0 - m_alphaFastAtk) * midEnergy;
        } else {
            m_fastMid = m_alphaFastDec * m_fastMid + (1.0 - m_alphaFastDec) * midEnergy;
        }

        if (midEnergy > m_slowMid) {
            m_slowMid = m_alphaSlowAtk * m_slowMid + (1.0 - m_alphaSlowAtk) * midEnergy;
        } else {
            m_slowMid = m_alphaSlowDec * m_slowMid + (1.0 - m_alphaSlowDec) * midEnergy;
        }

        double targetTransient = 0.0;
        if (m_fastMid > 1.35 * m_slowMid && m_fastMid > 1e-7) {
            targetTransient = std::min(1.0, (m_fastMid - 1.35 * m_slowMid) / (m_fastMid + 1e-7));
        }
        m_smoothTransient = m_alphaTransient * m_smoothTransient + (1.0 - m_alphaTransient) * targetTransient;

        // =====================================================================
        // Step 6: Stem Separation Assembly
        // =====================================================================
        // Vocals: center-panned harmonic content, ducked during heavy snare hits
        const double vocRejectDrums = 1.0 - 0.75 * m_smoothTransient;
        const double voc_mid_m = m_smoothCenter * vocRejectDrums * mid_m;
        const double voc_high_m = 0.45 * m_smoothHighCenter * vocRejectDrums * high_m;
        const double voc_l = voc_mid_m + voc_high_m;
        const double voc_r = voc_mid_m + voc_high_m;

        // Drums: full low band (< 220 Hz kick/sub) + snare mid transients + cymbal high transients
        const double drum_mid_m = m_smoothTransient * (mid_m - voc_mid_m);
        const double drum_high_m = m_smoothTransient * (high_m - voc_high_m);
        const double drum_high_s = m_smoothTransient * high_s;
        const double drum_l = low_l + drum_mid_m + drum_high_m + drum_high_s;
        const double drum_r = low_r + drum_mid_m + drum_high_m - drum_high_s;

        // Instruments: musical residual (synthesizers, guitars, pianos, stereo reverbs)
        // By subtracting drums and vocals from input, unity sum is 100% mathematically conserved!
        const double inst_l = in_l - drum_l - voc_l;
        const double inst_r = in_r - drum_r - voc_r;

        // =====================================================================
        // Step 7: Apply Gains and Sum Output
        // =====================================================================
        const double out_l = vocalsGain * voc_l + drumsGain * drum_l + instGain * inst_l;
        const double out_r = vocalsGain * voc_r + drumsGain * drum_r + instGain * inst_r;

        pInOut[2 * i] = static_cast<CSAMPLE>(out_l);
        pInOut[2 * i + 1] = static_cast<CSAMPLE>(out_r);
    }
}

} // namespace mixxx
