#include "engine/stems/stemtrackbuffer.h"

#include <cstring>

namespace mixxx {

StemTrackBuffer::StemTrackBuffer() = default;

void StemTrackBuffer::allocate(uint32_t totalFrames) {
    m_framesReady.store(0, std::memory_order_release);
    m_totalFrames = totalFrames;

    m_vocL.assign(totalFrames, 0.0f);
    m_vocR.assign(totalFrames, 0.0f);
    m_drumL.assign(totalFrames, 0.0f);
    m_drumR.assign(totalFrames, 0.0f);
    m_instL.assign(totalFrames, 0.0f);
    m_instR.assign(totalFrames, 0.0f);
}

void StemTrackBuffer::clear() {
    m_framesReady.store(0, std::memory_order_release);
    m_totalFrames = 0;
    m_vocL.clear();
    m_vocR.clear();
    m_drumL.clear();
    m_drumR.clear();
    m_instL.clear();
    m_instR.clear();
}

bool StemTrackBuffer::isFrameReady(uint32_t frame) const {
    return frame < m_framesReady.load(std::memory_order_acquire);
}

bool StemTrackBuffer::isRangeReady(uint32_t startFrame, uint32_t numFrames) const {
    const uint32_t ready = m_framesReady.load(std::memory_order_acquire);
    return (startFrame + numFrames) <= ready;
}

void StemTrackBuffer::writeChunk(
        uint32_t startFrame,
        uint32_t numFrames,
        const float* voc_l, const float* voc_r,
        const float* drum_l, const float* drum_r,
        const float* inst_l, const float* inst_r) {
    if (startFrame >= m_totalFrames || numFrames == 0) {
        return;
    }

    const uint32_t count = std::min(numFrames, m_totalFrames - startFrame);

    if (voc_l) std::memcpy(&m_vocL[startFrame], voc_l, count * sizeof(float));
    if (voc_r) std::memcpy(&m_vocR[startFrame], voc_r, count * sizeof(float));
    if (drum_l) std::memcpy(&m_drumL[startFrame], drum_l, count * sizeof(float));
    if (drum_r) std::memcpy(&m_drumR[startFrame], drum_r, count * sizeof(float));
    if (inst_l) std::memcpy(&m_instL[startFrame], inst_l, count * sizeof(float));
    if (inst_r) std::memcpy(&m_instR[startFrame], inst_r, count * sizeof(float));

    const uint32_t endFrame = startFrame + count;
    uint32_t currentReady = m_framesReady.load(std::memory_order_relaxed);
    while (endFrame > currentReady) {
        if (m_framesReady.compare_exchange_weak(
                    currentReady, endFrame,
                    std::memory_order_release,
                    std::memory_order_relaxed)) {
            break;
        }
    }
}

void StemTrackBuffer::readFrames(
        uint32_t startFrame,
        uint32_t numFrames,
        float* voc_l, float* voc_r,
        float* drum_l, float* drum_r,
        float* inst_l, float* inst_r) const {
    if (numFrames == 0) {
        return;
    }

    if (startFrame >= m_totalFrames) {
        if (voc_l) std::memset(voc_l, 0, numFrames * sizeof(float));
        if (voc_r) std::memset(voc_r, 0, numFrames * sizeof(float));
        if (drum_l) std::memset(drum_l, 0, numFrames * sizeof(float));
        if (drum_r) std::memset(drum_r, 0, numFrames * sizeof(float));
        if (inst_l) std::memset(inst_l, 0, numFrames * sizeof(float));
        if (inst_r) std::memset(inst_r, 0, numFrames * sizeof(float));
        return;
    }

    const uint32_t count = std::min(numFrames, m_totalFrames - startFrame);

    if (voc_l) std::memcpy(voc_l, &m_vocL[startFrame], count * sizeof(float));
    if (voc_r) std::memcpy(voc_r, &m_vocR[startFrame], count * sizeof(float));
    if (drum_l) std::memcpy(drum_l, &m_drumL[startFrame], count * sizeof(float));
    if (drum_r) std::memcpy(drum_r, &m_drumR[startFrame], count * sizeof(float));
    if (inst_l) std::memcpy(inst_l, &m_instL[startFrame], count * sizeof(float));
    if (inst_r) std::memcpy(inst_r, &m_instR[startFrame], count * sizeof(float));

    if (count < numFrames) {
        const uint32_t rem = numFrames - count;
        if (voc_l) std::memset(voc_l + count, 0, rem * sizeof(float));
        if (voc_r) std::memset(voc_r + count, 0, rem * sizeof(float));
        if (drum_l) std::memset(drum_l + count, 0, rem * sizeof(float));
        if (drum_r) std::memset(drum_r + count, 0, rem * sizeof(float));
        if (inst_l) std::memset(inst_l + count, 0, rem * sizeof(float));
        if (inst_r) std::memset(inst_r + count, 0, rem * sizeof(float));
    }
}

} // namespace mixxx
