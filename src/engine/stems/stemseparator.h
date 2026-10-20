#pragma once

#include <QString>
#include <vector>
#include <memory>
#include <mutex>
#include <cstdint>
#include <functional>
#include <atomic>

namespace Ort {
class Env;
class Session;
class MemoryInfo;
} // namespace Ort

namespace mixxx {

class HTDemucsEngine {
  public:
    static HTDemucsEngine& instance();

    // Constant parameters matching HTDemucs v4
    static constexpr int kSampleRate = 44100;
    static constexpr int kSegmentSamples = 343980; // 7.8 seconds
    static constexpr int kOverlapSamples = kSegmentSamples / 4; // ~1.95s overlap
    static constexpr int kStrideSamples = kSegmentSamples - kOverlapSamples; // ~5.85s stride

    // Check if ONNX Runtime and HTDemucs model are available on the system
    bool isAvailable();

    // Initialize the ONNX session with the HTDemucs v4 model
    bool initialize(const QString& customModelPath = QString());

    // Check if initialized
    bool isInitialized() const;

    // Separate a single 7.8-second segment (343,980 samples)
    bool separateSegment(
            const float* in_l, const float* in_r,
            float* drums_l, float* drums_r,
            float* bass_l, float* bass_r,
            float* other_l, float* other_r,
            float* vocals_l, float* vocals_r);

    // Convenience 3-stem separation for a segment (Instruments = Bass + Other)
    bool separateSegment3Stem(
            const float* in_l, const float* in_r,
            float* voc_l, float* voc_r,
            float* drum_l, float* drum_r,
            float* inst_l, float* inst_r);

    // Full audio separation with overlap-add windowing
    bool separateTrack(
            uint32_t totalFrames,
            const float* in_l, const float* in_r,
            float* voc_l, float* voc_r,
            float* drum_l, float* drum_r,
            float* inst_l, float* inst_r,
            std::function<void(double)> progressCallback = nullptr,
            std::atomic<bool>* pCancelFlag = nullptr);

    void finalize();

  private:
    HTDemucsEngine();
    ~HTDemucsEngine();

    bool locateModel(QString* outPath) const;

    std::unique_ptr<Ort::Env> m_pEnv;
    std::unique_ptr<Ort::Session> m_pSession;
    std::unique_ptr<Ort::Session> m_pSessionVoc;
    std::unique_ptr<Ort::Session> m_pSessionDrum;
    std::unique_ptr<Ort::MemoryInfo> m_pMemoryInfo;

    // Hann-like overlap window
    std::vector<float> m_window;

    bool m_bInitialized{false};
    bool m_bFineTuned{false};
    std::recursive_mutex m_mutex;
};

} // namespace mixxx
