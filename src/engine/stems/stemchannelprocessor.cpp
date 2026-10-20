#include "engine/stems/stemchannelprocessor.h"
#include "engine/stems/stemcache.h"
#include "engine/enginebuffer.h"

#include <cmath>
#include <algorithm>

namespace mixxx {

StemChannelProcessor::StemChannelProcessor(const QString& group, EngineBuffer* pBuffer)
        : m_group(group),
          m_pEngineBuffer(pBuffer),
          m_pStemTrackBuffer(std::make_shared<StemTrackBuffer>()) {
    // 1. Controls
    m_pStemsActive = std::make_unique<ControlPushButton>(ConfigKey(m_group, "stems_active"));
    m_pStemsActive->setButtonMode(ControlPushButton::TOGGLE);
    m_pStemsActive->set(1.0);

    m_pStemVocalsEnabled = std::make_unique<ControlPushButton>(ConfigKey(m_group, "stem_vocals_enabled"));
    m_pStemVocalsEnabled->setButtonMode(ControlPushButton::TOGGLE);
    m_pStemVocalsEnabled->set(1.0);

    m_pStemDrumsEnabled = std::make_unique<ControlPushButton>(ConfigKey(m_group, "stem_drums_enabled"));
    m_pStemDrumsEnabled->setButtonMode(ControlPushButton::TOGGLE);
    m_pStemDrumsEnabled->set(1.0);

    m_pStemInstEnabled = std::make_unique<ControlPushButton>(ConfigKey(m_group, "stem_inst_enabled"));
    m_pStemInstEnabled->setButtonMode(ControlPushButton::TOGGLE);
    m_pStemInstEnabled->set(1.0);

    m_pStemVocalsVolume = std::make_unique<ControlPotmeter>(ConfigKey(m_group, "stem_vocals_volume"), 0.0, 1.0);
    m_pStemVocalsVolume->set(1.0);
    m_pStemVocalsVolume->setDefaultValue(1.0);

    m_pStemDrumsVolume = std::make_unique<ControlPotmeter>(ConfigKey(m_group, "stem_drums_volume"), 0.0, 1.0);
    m_pStemDrumsVolume->set(1.0);
    m_pStemDrumsVolume->setDefaultValue(1.0);

    m_pStemInstVolume = std::make_unique<ControlPotmeter>(ConfigKey(m_group, "stem_inst_volume"), 0.0, 1.0);
    m_pStemInstVolume->set(1.0);
    m_pStemInstVolume->setDefaultValue(1.0);

    m_pStemVocalsSolo = std::make_unique<ControlPushButton>(ConfigKey(m_group, "stem_vocals_solo"));
    m_pStemVocalsSolo->setButtonMode(ControlPushButton::PUSH);
    connect(m_pStemVocalsSolo.get(), &ControlPushButton::valueChanged,
            this, &StemChannelProcessor::slotSoloVocals);

    m_pStemDrumsSolo = std::make_unique<ControlPushButton>(ConfigKey(m_group, "stem_drums_solo"));
    m_pStemDrumsSolo->setButtonMode(ControlPushButton::PUSH);
    connect(m_pStemDrumsSolo.get(), &ControlPushButton::valueChanged,
            this, &StemChannelProcessor::slotSoloDrums);

    m_pStemInstSolo = std::make_unique<ControlPushButton>(ConfigKey(m_group, "stem_inst_solo"));
    m_pStemInstSolo->setButtonMode(ControlPushButton::PUSH);
    connect(m_pStemInstSolo.get(), &ControlPushButton::valueChanged,
            this, &StemChannelProcessor::slotSoloInst);

    m_pStemReset = std::make_unique<ControlPushButton>(ConfigKey(m_group, "stem_reset"));
    m_pStemReset->setButtonMode(ControlPushButton::PUSH);
    connect(m_pStemReset.get(), &ControlPushButton::valueChanged,
            this, &StemChannelProcessor::slotResetStems);

    m_pStemsSource = std::make_unique<ControlPushButton>(ConfigKey(m_group, "stems_source"));
    m_pStemsSource->setButtonMode(ControlPushButton::TOGGLE);
    m_pStemsSource->set(1.0); // 1 = Real Deep Neural Network AI by default!
    connect(m_pStemsSource.get(), &ControlPushButton::valueChanged,
            this, &StemChannelProcessor::slotSourceChanged);

    m_pStemsProgress = std::make_unique<ControlObject>(ConfigKey(m_group, "stems_progress"));
    m_pStemsProgress->set(0.0);

    m_pStemsStatus = std::make_unique<ControlObject>(ConfigKey(m_group, "stems_status"));
    m_pStemsStatus->set(0.0);

    // 2. Subsystems
    m_dspIsolator.init(44100.0);

    m_pWorkerThread = std::make_unique<StemWorkerThread>(
            m_pStemTrackBuffer,
            m_pStemsProgress.get(),
            m_pStemsStatus.get());

    if (m_pEngineBuffer) {
        connect(m_pEngineBuffer, &EngineBuffer::trackLoaded,
                this, &StemChannelProcessor::slotTrackLoaded);
    }
}

StemChannelProcessor::~StemChannelProcessor() {
    if (m_pWorkerThread) {
        m_pWorkerThread->stop();
    }
}

void StemChannelProcessor::slotSoloVocals(double v) {
    if (v > 0.0) {
        m_pStemVocalsEnabled->set(1.0);
        m_pStemDrumsEnabled->set(0.0);
        m_pStemInstEnabled->set(0.0);
    }
}

void StemChannelProcessor::slotSoloDrums(double v) {
    if (v > 0.0) {
        m_pStemVocalsEnabled->set(0.0);
        m_pStemDrumsEnabled->set(1.0);
        m_pStemInstEnabled->set(0.0);
    }
}

void StemChannelProcessor::slotSoloInst(double v) {
    if (v > 0.0) {
        m_pStemVocalsEnabled->set(0.0);
        m_pStemDrumsEnabled->set(0.0);
        m_pStemInstEnabled->set(1.0);
    }
}

void StemChannelProcessor::slotResetStems(double v) {
    if (v > 0.0) {
        m_pStemVocalsEnabled->set(1.0);
        m_pStemDrumsEnabled->set(1.0);
        m_pStemInstEnabled->set(1.0);
        m_pStemVocalsVolume->set(1.0);
        m_pStemDrumsVolume->set(1.0);
        m_pStemInstVolume->set(1.0);
    }
}

void StemChannelProcessor::slotSourceChanged(double v) {
    if (v >= 0.5 && m_pEngineBuffer) {
        TrackPointer pTrack = m_pEngineBuffer->getLoadedTrack();
        if (pTrack) {
            if (StemCache::hasCachedStems(pTrack)) {
                if (StemCache::loadFromCache(pTrack, m_pStemTrackBuffer.get())) {
                    m_pStemsProgress->set(1.0);
                    m_pStemsStatus->set(2.0); // 2 = Ready!
                    return;
                }
            }
            m_pWorkerThread->loadTrack(pTrack);
        }
    }
}

void StemChannelProcessor::slotTrackLoaded(TrackPointer pNewTrack, TrackPointer pOldTrack) {
    Q_UNUSED(pOldTrack);
    m_pStemTrackBuffer->clear();
    m_pStemsProgress->set(0.0);
    m_pStemsStatus->set(0.0);

    if (pNewTrack) {
        // 1. Instant load from disk cache if this track was previously separated
        if (StemCache::hasCachedStems(pNewTrack)) {
            if (StemCache::loadFromCache(pNewTrack, m_pStemTrackBuffer.get())) {
                m_pStemsProgress->set(1.0);
                m_pStemsStatus->set(2.0); // 2 = AI Stems Ready!
                qDebug() << "StemChannelProcessor: Real AI Stems loaded instantly from cache for"
                         << pNewTrack->getTitle();
                return;
            }
        }

        // 2. Otherwise start background AI neural network separation immediately
        if (m_pStemsSource->get() >= 0.5) {
            m_pWorkerThread->loadTrack(pNewTrack);
        }
    } else {
        m_pWorkerThread->stop();
    }
}

void StemChannelProcessor::process(CSAMPLE* pInOut, int iBufferSize, audio::FramePos playPos) {
    if (!pInOut || iBufferSize <= 0) {
        return;
    }

    if (!m_pStemsActive->toBool()) {
        return; // Stems bypassed
    }

    const double vocGain = m_pStemVocalsEnabled->toBool() ? m_pStemVocalsVolume->get() : 0.0;
    const double drumGain = m_pStemDrumsEnabled->toBool() ? m_pStemDrumsVolume->get() : 0.0;
    const double instGain = m_pStemInstEnabled->toBool() ? m_pStemInstVolume->get() : 0.0;

    // Fast path: if all gains are practically unity, skip processing completely
    if (std::abs(vocGain - 1.0) < 0.0001 &&
        std::abs(drumGain - 1.0) < 0.0001 &&
        std::abs(instGain - 1.0) < 0.0001) {
        return;
    }

    const int numFrames = iBufferSize / 2;

    // Check if ML AI stems are requested and ready for the current frame
    bool usedMlStems = false;
    if (m_pStemsSource->get() >= 0.5 && playPos.isValid()) {
        const double reportedPos = playPos.value();
        if (reportedPos >= 0.0) {
            const double speed = m_pEngineBuffer ? m_pEngineBuffer->getSpeed() : 1.0;
            const double effectiveSpeed = (speed != 0.0) ? speed : 1.0;
            const double framesProcessed = numFrames * effectiveSpeed;
            // reportedPos is at the end of the audio buffer just scaled into pInOut
            const double startPos = std::max(0.0, reportedPos - framesProcessed);
            const uint32_t startFrame = static_cast<uint32_t>(startPos);

            if (std::abs(effectiveSpeed - 1.0) < 0.001) {
                // 1:1 playback speed - direct sample copy
                if (m_pStemTrackBuffer->isRangeReady(startFrame, static_cast<uint32_t>(numFrames))) {
                    if (m_scratchVocL.size() < static_cast<size_t>(numFrames)) {
                        m_scratchVocL.resize(numFrames);
                        m_scratchVocR.resize(numFrames);
                        m_scratchDrumL.resize(numFrames);
                        m_scratchDrumR.resize(numFrames);
                        m_scratchInstL.resize(numFrames);
                        m_scratchInstR.resize(numFrames);
                    }

                    m_pStemTrackBuffer->readFrames(
                            startFrame, static_cast<uint32_t>(numFrames),
                            m_scratchVocL.data(), m_scratchVocR.data(),
                            m_scratchDrumL.data(), m_scratchDrumR.data(),
                            m_scratchInstL.data(), m_scratchInstR.data());

                    for (int i = 0; i < numFrames; ++i) {
                        const double out_l = vocGain * m_scratchVocL[i] +
                                             drumGain * m_scratchDrumL[i] +
                                             instGain * m_scratchInstL[i];
                        const double out_r = vocGain * m_scratchVocR[i] +
                                             drumGain * m_scratchDrumR[i] +
                                             instGain * m_scratchInstR[i];
                        pInOut[2 * i] = static_cast<CSAMPLE>(out_l);
                        pInOut[2 * i + 1] = static_cast<CSAMPLE>(out_r);
                    }
                    usedMlStems = true;
                }
            } else if (effectiveSpeed > 0.0) {
                // Pitch/tempo altered - linear interpolation across stem frames
                const uint32_t neededFrames = static_cast<uint32_t>(std::ceil(framesProcessed)) + 2;
                if (m_pStemTrackBuffer->isRangeReady(startFrame, neededFrames)) {
                    if (m_scratchVocL.size() < neededFrames) {
                        m_scratchVocL.resize(neededFrames);
                        m_scratchVocR.resize(neededFrames);
                        m_scratchDrumL.resize(neededFrames);
                        m_scratchDrumR.resize(neededFrames);
                        m_scratchInstL.resize(neededFrames);
                        m_scratchInstR.resize(neededFrames);
                    }

                    m_pStemTrackBuffer->readFrames(
                            startFrame, neededFrames,
                            m_scratchVocL.data(), m_scratchVocR.data(),
                            m_scratchDrumL.data(), m_scratchDrumR.data(),
                            m_scratchInstL.data(), m_scratchInstR.data());

                    for (int i = 0; i < numFrames; ++i) {
                        const double posInChunk = i * effectiveSpeed;
                        const uint32_t idx0 = static_cast<uint32_t>(posInChunk);
                        const uint32_t idx1 = std::min(idx0 + 1, neededFrames - 1);
                        const float frac = static_cast<float>(posInChunk - idx0);

                        const float vl = m_scratchVocL[idx0] + frac * (m_scratchVocL[idx1] - m_scratchVocL[idx0]);
                        const float vr = m_scratchVocR[idx0] + frac * (m_scratchVocR[idx1] - m_scratchVocR[idx0]);
                        const float dl = m_scratchDrumL[idx0] + frac * (m_scratchDrumL[idx1] - m_scratchDrumL[idx0]);
                        const float dr = m_scratchDrumR[idx0] + frac * (m_scratchDrumR[idx1] - m_scratchDrumR[idx0]);
                        const float il = m_scratchInstL[idx0] + frac * (m_scratchInstL[idx1] - m_scratchInstL[idx0]);
                        const float ir = m_scratchInstR[idx0] + frac * (m_scratchInstR[idx1] - m_scratchInstR[idx0]);

                        const double out_l = vocGain * vl + drumGain * dl + instGain * il;
                        const double out_r = vocGain * vr + drumGain * dr + instGain * ir;
                        pInOut[2 * i] = static_cast<CSAMPLE>(out_l);
                        pInOut[2 * i + 1] = static_cast<CSAMPLE>(out_r);
                    }
                    usedMlStems = true;
                }
            }
        }
    }

    // High-fidelity real-time 4th-order Linkwitz-Riley stem isolator (zero-latency, transparent sum)
    if (!usedMlStems) {
        m_dspIsolator.process(pInOut, numFrames, vocGain, drumGain, instGain);
    }
}

} // namespace mixxx

#include "moc_stemchannelprocessor.cpp"
