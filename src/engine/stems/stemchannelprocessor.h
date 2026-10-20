#pragma once

#include <QObject>
#include <memory>

#include "audio/frame.h"
#include "control/controlobject.h"
#include "control/controlpushbutton.h"
#include "control/controlpotmeter.h"
#include "engine/stems/stemdspisolator.h"
#include "engine/stems/stemtrackbuffer.h"
#include "engine/stems/stemworkerthread.h"
#include "track/track.h"

class EngineBuffer;

namespace mixxx {

class StemChannelProcessor : public QObject {
    Q_OBJECT
  public:
    StemChannelProcessor(const QString& group, EngineBuffer* pBuffer);
    ~StemChannelProcessor() override;

    // Called on every audio callback in EngineDeck::process
    void process(CSAMPLE* pInOut, int iBufferSize, audio::FramePos playPos);

  public slots:
    void slotSoloVocals(double v);
    void slotSoloDrums(double v);
    void slotSoloInst(double v);
    void slotResetStems(double v);
    void slotSourceChanged(double v);
    void slotTrackLoaded(TrackPointer pNewTrack, TrackPointer pOldTrack);

  private:
    QString m_group;
    EngineBuffer* m_pEngineBuffer;

    // Stem Controls
    std::unique_ptr<ControlPushButton> m_pStemsActive;
    std::unique_ptr<ControlPushButton> m_pStemVocalsEnabled;
    std::unique_ptr<ControlPushButton> m_pStemDrumsEnabled;
    std::unique_ptr<ControlPushButton> m_pStemInstEnabled;

    std::unique_ptr<ControlPotmeter> m_pStemVocalsVolume;
    std::unique_ptr<ControlPotmeter> m_pStemDrumsVolume;
    std::unique_ptr<ControlPotmeter> m_pStemInstVolume;

    std::unique_ptr<ControlPushButton> m_pStemVocalsSolo;
    std::unique_ptr<ControlPushButton> m_pStemDrumsSolo;
    std::unique_ptr<ControlPushButton> m_pStemInstSolo;
    std::unique_ptr<ControlPushButton> m_pStemReset;

    std::unique_ptr<ControlPushButton> m_pStemsSource;
    std::unique_ptr<ControlObject> m_pStemsProgress;
    std::unique_ptr<ControlObject> m_pStemsStatus;

    // Subsystems
    StemDspIsolator m_dspIsolator;
    std::shared_ptr<StemTrackBuffer> m_pStemTrackBuffer;
    std::unique_ptr<StemWorkerThread> m_pWorkerThread;

    // Scratch buffers for ML stem mixing
    std::vector<float> m_scratchVocL;
    std::vector<float> m_scratchVocR;
    std::vector<float> m_scratchDrumL;
    std::vector<float> m_scratchDrumR;
    std::vector<float> m_scratchInstL;
    std::vector<float> m_scratchInstR;
};

} // namespace mixxx
