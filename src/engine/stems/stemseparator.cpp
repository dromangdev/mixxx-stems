#include "engine/stems/stemseparator.h"

#include <onnxruntime_cxx_api.h>
#ifdef __APPLE__
#include <coreml_provider_factory.h>
#endif

#include <QDir>
#include <QFileInfo>
#include <QDebug>
#include <cmath>
#include <cstring>

namespace mixxx {

namespace {
const char* const kInputNames[] = {"mix"};
const char* const kOutputNames[] = {"stems"};
} // namespace

HTDemucsEngine& HTDemucsEngine::instance() {
    static HTDemucsEngine* s_instance = new HTDemucsEngine();
    return *s_instance;
}

HTDemucsEngine::HTDemucsEngine() {
    // Build trapezoidal overlap-add window (flat center, linear crossfade edges)
    m_window.assign(kSegmentSamples, 1.0f);
    for (int i = 0; i < kOverlapSamples; ++i) {
        const float fade = static_cast<float>(i) / static_cast<float>(kOverlapSamples);
        m_window[i] = fade;
        m_window[kSegmentSamples - 1 - i] = fade;
    }
}

HTDemucsEngine::~HTDemucsEngine() {
    finalize();
}

bool HTDemucsEngine::locateModel(QString* outPath) const {
    const QStringList candidates = {
        QDir::currentPath() + "/models/htdemucs.onnx",
        QDir::homePath() + "/Library/Application Support/Mixxx/models/htdemucs.onnx",
        "/Users/dromang/Documents/Gemini/Gemini/Skills/replica-gemini/mixxx/models/htdemucs.onnx"
    };

    for (const auto& path : candidates) {
        if (QFileInfo::exists(path)) {
            if (outPath) {
                *outPath = path;
            }
            return true;
        }
    }
    return false;
}

bool HTDemucsEngine::isAvailable() {
    return locateModel(nullptr);
}

bool HTDemucsEngine::isInitialized() const {
    return m_bInitialized && (m_pSession != nullptr || (m_pSessionVoc != nullptr && m_pSessionDrum != nullptr));
}

bool HTDemucsEngine::initialize(const QString& customModelPath) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);

    if (isInitialized()) {
        return true;
    }

    // Check for Fine-Tuned models first (SOTA quality)
    const QString ftVocPath = "/Users/dromang/Documents/Gemini/Gemini/Skills/replica-gemini/mixxx/models/htdemucs_ft_vocals.onnx";
    const QString ftDrumPath = "/Users/dromang/Documents/Gemini/Gemini/Skills/replica-gemini/mixxx/models/htdemucs_ft_drums.onnx";

    if (customModelPath.isEmpty() && QFileInfo::exists(ftVocPath) && QFileInfo::exists(ftDrumPath)) {
        try {
            qDebug() << "HTDemucsEngine: Loading Fine-Tuned (HTDemucs-FT) ONNX specialist models...";
            m_pEnv = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "HTDemucsEngine");

            Ort::SessionOptions sessionOptions;
            sessionOptions.SetIntraOpNumThreads(8);
            sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

            m_pSessionVoc = std::make_unique<Ort::Session>(
                    *m_pEnv, ftVocPath.toUtf8().constData(), sessionOptions);
            m_pSessionDrum = std::make_unique<Ort::Session>(
                    *m_pEnv, ftDrumPath.toUtf8().constData(), sessionOptions);

            m_pMemoryInfo = std::make_unique<Ort::MemoryInfo>(
                    Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));

            m_bFineTuned = true;
            m_bInitialized = true;
            qDebug() << "HTDemucsEngine: Successfully initialized Fine-Tuned (HTDemucs-FT) models!";
            return true;
        } catch (const std::exception& e) {
            qDebug() << "HTDemucsEngine: Failed loading FT models, falling back to base model:" << e.what();
        }
    }

    QString modelPath = customModelPath;
    if (modelPath.isEmpty() || !QFileInfo::exists(modelPath)) {
        if (!locateModel(&modelPath)) {
            qDebug() << "HTDemucsEngine: No htdemucs.onnx model found!";
            return false;
        }
    }

    try {
        qDebug() << "HTDemucsEngine: Loading ONNX model from" << modelPath;
        m_pEnv = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "HTDemucsEngine");

        Ort::SessionOptions sessionOptions;
        sessionOptions.SetIntraOpNumThreads(8);
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        m_pSession = std::make_unique<Ort::Session>(
                *m_pEnv,
                modelPath.toUtf8().constData(),
                sessionOptions);

        m_pMemoryInfo = std::make_unique<Ort::MemoryInfo>(
                Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));

        m_bInitialized = true;
        m_bFineTuned = false;
        qDebug() << "HTDemucsEngine: HTDemucs v4 ONNX model successfully loaded and initialized!";
        return true;
    } catch (const Ort::Exception& e) {
        qDebug() << "HTDemucsEngine: ONNX initialization failed:" << e.what();
        m_bInitialized = false;
        return false;
    } catch (const std::exception& e) {
        qDebug() << "HTDemucsEngine: Initialization failed:" << e.what();
        m_bInitialized = false;
        return false;
    }
}

bool HTDemucsEngine::separateSegment(
        const float* in_l, const float* in_r,
        float* drums_l, float* drums_r,
        float* bass_l, float* bass_r,
        float* other_l, float* other_r,
        float* vocals_l, float* vocals_r) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);

    if (!m_bInitialized) {
        if (!initialize()) {
            return false;
        }
    }

    try {
        const int64_t N = kSegmentSamples;
        const std::vector<int64_t> inputShape = {1, 2, N};

        // Prepare planar stereo buffer: [Left channel (N), Right channel (N)]
        std::vector<float> inputData(2 * N);
        std::memcpy(&inputData[0], in_l, N * sizeof(float));
        std::memcpy(&inputData[N], in_r, N * sizeof(float));

        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
                *m_pMemoryInfo,
                inputData.data(), inputData.size(),
                inputShape.data(), inputShape.size());

        auto outputTensors = m_pSession->Run(
                Ort::RunOptions{nullptr},
                kInputNames, &inputTensor, 1,
                kOutputNames, 1);

        if (outputTensors.empty()) {
            return false;
        }

        const float* outData = outputTensors[0].GetTensorData<float>();
        // Output tensor shape is (1, 4, 2, 343980)
        // Stems order: 0 = Drums, 1 = Bass, 2 = Other, 3 = Vocals
        const size_t stemStride = 2 * N;

        // Drums (stem 0)
        if (drums_l) std::memcpy(drums_l, outData + 0 * stemStride, N * sizeof(float));
        if (drums_r) std::memcpy(drums_r, outData + 0 * stemStride + N, N * sizeof(float));

        // Bass (stem 1)
        if (bass_l) std::memcpy(bass_l, outData + 1 * stemStride, N * sizeof(float));
        if (bass_r) std::memcpy(bass_r, outData + 1 * stemStride + N, N * sizeof(float));

        // Other (stem 2)
        if (other_l) std::memcpy(other_l, outData + 2 * stemStride, N * sizeof(float));
        if (other_r) std::memcpy(other_r, outData + 2 * stemStride + N, N * sizeof(float));

        // Vocals (stem 3)
        if (vocals_l) std::memcpy(vocals_l, outData + 3 * stemStride, N * sizeof(float));
        if (vocals_r) std::memcpy(vocals_r, outData + 3 * stemStride + N, N * sizeof(float));

        return true;
    } catch (const Ort::Exception& e) {
        qDebug() << "HTDemucsEngine: separateSegment failed:" << e.what();
        return false;
    }
}

bool HTDemucsEngine::separateSegment3Stem(
        const float* in_l, const float* in_r,
        float* voc_l, float* voc_r,
        float* drum_l, float* drum_r,
        float* inst_l, float* inst_r) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);

    if (!m_bInitialized) {
        if (!initialize()) {
            return false;
        }
    }

    const int64_t N = kSegmentSamples;

    // High-Fidelity Fine-Tuned (HTDemucs-FT) Path
    if (m_bFineTuned && m_pSessionVoc && m_pSessionDrum) {
        try {
            const std::vector<int64_t> inputShape = {1, 2, N};
            std::vector<float> inputData(2 * N);
            std::memcpy(&inputData[0], in_l, N * sizeof(float));
            std::memcpy(&inputData[N], in_r, N * sizeof(float));

            // 1. Separate Vocals with Fine-Tuned specialist
            Ort::Value inTensorVoc = Ort::Value::CreateTensor<float>(
                    *m_pMemoryInfo,
                    inputData.data(), inputData.size(),
                    inputShape.data(), inputShape.size());

            auto outVoc = m_pSessionVoc->Run(
                    Ort::RunOptions{nullptr}, kInputNames, &inTensorVoc, 1, kOutputNames, 1);
            if (outVoc.empty()) {
                return false;
            }

            const float* outVocData = outVoc[0].GetTensorData<float>();
            // Row 3 is fine-tuned vocals
            if (voc_l) std::memcpy(voc_l, outVocData + 3 * (2 * N), N * sizeof(float));
            if (voc_r) std::memcpy(voc_r, outVocData + 3 * (2 * N) + N, N * sizeof(float));

            // 2. Separate Drums with Fine-Tuned specialist
            Ort::Value inTensorDrum = Ort::Value::CreateTensor<float>(
                    *m_pMemoryInfo,
                    inputData.data(), inputData.size(),
                    inputShape.data(), inputShape.size());

            auto outDrum = m_pSessionDrum->Run(
                    Ort::RunOptions{nullptr}, kInputNames, &inTensorDrum, 1, kOutputNames, 1);
            if (outDrum.empty()) {
                return false;
            }

            const float* outDrumData = outDrum[0].GetTensorData<float>();
            // Row 0 is fine-tuned drums
            if (drum_l) std::memcpy(drum_l, outDrumData + 0 * (2 * N), N * sizeof(float));
            if (drum_r) std::memcpy(drum_r, outDrumData + 0 * (2 * N) + N, N * sizeof(float));

            // 3. Perfect Transparent Residual Instruments: Inst = Mix - (Voc + Drum)
            if (inst_l) {
                for (int64_t i = 0; i < N; ++i) {
                    inst_l[i] = in_l[i] - (voc_l ? voc_l[i] : 0.0f) - (drum_l ? drum_l[i] : 0.0f);
                }
            }
            if (inst_r) {
                for (int64_t i = 0; i < N; ++i) {
                    inst_r[i] = in_r[i] - (voc_r ? voc_r[i] : 0.0f) - (drum_r ? drum_r[i] : 0.0f);
                }
            }
            return true;
        } catch (const Ort::Exception& e) {
            qDebug() << "HTDemucsEngine: Fine-tuned separateSegment3Stem failed:" << e.what();
            return false;
        }
    }

    std::vector<float> bass_l(N), bass_r(N);
    std::vector<float> other_l(N), other_r(N);

    bool ok = separateSegment(
            in_l, in_r,
            drum_l, drum_r,
            bass_l.data(), bass_r.data(),
            other_l.data(), other_r.data(),
            voc_l, voc_r);

    if (!ok) {
        return false;
    }

    // Combine Bass + Other into Instruments
    if (inst_l) {
        for (int i = 0; i < N; ++i) {
            inst_l[i] = bass_l[i] + other_l[i];
        }
    }
    if (inst_r) {
        for (int i = 0; i < N; ++i) {
            inst_r[i] = bass_r[i] + other_r[i];
        }
    }
    return true;
}

bool HTDemucsEngine::separateTrack(
        uint32_t totalFrames,
        const float* in_l, const float* in_r,
        float* voc_l, float* voc_r,
        float* drum_l, float* drum_r,
        float* inst_l, float* inst_r,
        std::function<void(double)> progressCallback,
        std::atomic<bool>* pCancelFlag) {
    if (totalFrames == 0 || !in_l || !in_r) {
        return false;
    }

    // Initialize output buffers to zero
    std::memset(voc_l, 0, totalFrames * sizeof(float));
    std::memset(voc_r, 0, totalFrames * sizeof(float));
    std::memset(drum_l, 0, totalFrames * sizeof(float));
    std::memset(drum_r, 0, totalFrames * sizeof(float));
    std::memset(inst_l, 0, totalFrames * sizeof(float));
    std::memset(inst_r, 0, totalFrames * sizeof(float));

    std::vector<float> weight(totalFrames, 0.0f);

    const uint32_t stride = kStrideSamples;
    const uint32_t nChunks = (totalFrames + stride - 1) / stride;

    std::vector<float> chunkInL(kSegmentSamples, 0.0f);
    std::vector<float> chunkInR(kSegmentSamples, 0.0f);
    std::vector<float> chunkVocL(kSegmentSamples, 0.0f);
    std::vector<float> chunkVocR(kSegmentSamples, 0.0f);
    std::vector<float> chunkDrumL(kSegmentSamples, 0.0f);
    std::vector<float> chunkDrumR(kSegmentSamples, 0.0f);
    std::vector<float> chunkInstL(kSegmentSamples, 0.0f);
    std::vector<float> chunkInstR(kSegmentSamples, 0.0f);

    for (uint32_t chunkIdx = 0; chunkIdx < nChunks; ++chunkIdx) {
        if (pCancelFlag && pCancelFlag->load(std::memory_order_relaxed)) {
            return false;
        }

        const uint32_t start = chunkIdx * stride;
        const uint32_t end = std::min(start + static_cast<uint32_t>(kSegmentSamples), totalFrames);
        const uint32_t chunkLen = end - start;

        // Copy input with zero-padding if less than kSegmentSamples
        std::fill(chunkInL.begin(), chunkInL.end(), 0.0f);
        std::fill(chunkInR.begin(), chunkInR.end(), 0.0f);
        std::memcpy(chunkInL.data(), in_l + start, chunkLen * sizeof(float));
        std::memcpy(chunkInR.data(), in_r + start, chunkLen * sizeof(float));

        bool ok = separateSegment3Stem(
                chunkInL.data(), chunkInR.data(),
                chunkVocL.data(), chunkVocR.data(),
                chunkDrumL.data(), chunkDrumR.data(),
                chunkInstL.data(), chunkInstR.data());

        if (!ok) {
            return false;
        }

        // Overlap-add into output buffers
        for (uint32_t i = 0; i < chunkLen; ++i) {
            const float w = m_window[i];
            const uint32_t dest = start + i;
            voc_l[dest] += chunkVocL[i] * w;
            voc_r[dest] += chunkVocR[i] * w;
            drum_l[dest] += chunkDrumL[i] * w;
            drum_r[dest] += chunkDrumR[i] * w;
            inst_l[dest] += chunkInstL[i] * w;
            inst_r[dest] += chunkInstR[i] * w;
            weight[dest] += w;
        }

        if (progressCallback) {
            progressCallback(static_cast<double>(chunkIdx + 1) / nChunks);
        }
    }

    // Normalize by overlap weight
    for (uint32_t i = 0; i < totalFrames; ++i) {
        const float w = std::max(weight[i], 1e-6f);
        voc_l[i] /= w;
        voc_r[i] /= w;
        drum_l[i] /= w;
        drum_r[i] /= w;
        inst_l[i] /= w;
        inst_r[i] /= w;
    }

    return true;
}

void HTDemucsEngine::finalize() {
    m_pSessionVoc.reset();
    m_pSessionDrum.reset();
    m_pSession.reset();
    m_pMemoryInfo.reset();
    m_pEnv.reset();
    m_bInitialized = false;
    m_bFineTuned = false;
}

} // namespace mixxx
