#pragma once

#include <vector>
#include <atomic>
#include <cstdint>
#include <algorithm>

namespace mixxx {

class StemTrackBuffer {
  public:
    StemTrackBuffer();
    ~StemTrackBuffer() = default;

    // Allocate storage for totalFrames
    void allocate(uint32_t totalFrames);

    // Clear buffer and reset frame count
    void clear();

    // Check if the given frame range has been separated and is ready
    bool isFrameReady(uint32_t frame) const;
    bool isRangeReady(uint32_t startFrame, uint32_t numFrames) const;

    uint32_t framesReady() const {
        return m_framesReady.load(std::memory_order_acquire);
    }

    uint32_t totalFrames() const {
        return m_totalFrames;
    }

    // Write a separated chunk into the buffer
    void writeChunk(
            uint32_t startFrame,
            uint32_t numFrames,
            const float* voc_l, const float* voc_r,
            const float* drum_l, const float* drum_r,
            const float* inst_l, const float* inst_r);

    // Read separated stem samples for a range of frames
    void readFrames(
            uint32_t startFrame,
            uint32_t numFrames,
            float* voc_l, float* voc_r,
            float* drum_l, float* drum_r,
            float* inst_l, float* inst_r) const;

  private:
    uint32_t m_totalFrames{0};
    std::atomic<uint32_t> m_framesReady{0};

    std::vector<float> m_vocL;
    std::vector<float> m_vocR;
    std::vector<float> m_drumL;
    std::vector<float> m_drumR;
    std::vector<float> m_instL;
    std::vector<float> m_instR;
};

} // namespace mixxx
