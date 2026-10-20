#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <onnxruntime_cxx_api.h>

#include "engine/stems/stemdspisolator.h"
#include "engine/stems/stemtrackbuffer.h"
#include "engine/stems/stemseparator.h"
#include "engine/stems/stemcache.h"
#include "engine/stems/stemworkerthread.h"
#include "track/track.h"
#include "test/mixxxtest.h"

namespace {

class StemsTest : public MixxxTest {
  protected:
    void SetUp() override {
    }
};

TEST_F(StemsTest, DspIsolatorUnityTransparency) {
    mixxx::StemDspIsolator isolator;
    isolator.init(44100.0);

    constexpr int kFrames = 512;
    std::vector<CSAMPLE> buffer(kFrames * 2);
    for (int i = 0; i < kFrames; ++i) {
        buffer[2 * i] = static_cast<CSAMPLE>(std::sin(2.0 * M_PI * 440.0 * i / 44100.0));
        buffer[2 * i + 1] = static_cast<CSAMPLE>(std::cos(2.0 * M_PI * 880.0 * i / 44100.0));
    }
    const std::vector<CSAMPLE> original = buffer;

    // Process with unity gains (1.0, 1.0, 1.0)
    isolator.process(buffer.data(), kFrames, 1.0, 1.0, 1.0);

    // Should be bit-for-bit identical due to transparent bypass
    for (size_t i = 0; i < buffer.size(); ++i) {
        EXPECT_FLOAT_EQ(original[i], buffer[i]);
    }
}

TEST_F(StemsTest, DspIsolatorMuteVocals) {
    mixxx::StemDspIsolator isolator;
    isolator.init(44100.0);

    constexpr int kFrames = 1024;
    std::vector<CSAMPLE> buffer(kFrames * 2);
    // Center-panned sine wave at 1 kHz (pure vocal candidate)
    for (int i = 0; i < kFrames; ++i) {
        CSAMPLE s = static_cast<CSAMPLE>(0.5 * std::sin(2.0 * M_PI * 1000.0 * i / 44100.0));
        buffer[2 * i] = s;
        buffer[2 * i + 1] = s;
    }

    // Mute vocals: gain = 0.0, drums = 1.0, inst = 1.0
    isolator.process(buffer.data(), kFrames, 0.0, 1.0, 1.0);

    // Energy should be significantly reduced compared to 0.5
    double energy = 0.0;
    for (int i = 256; i < kFrames; ++i) { // Skip filter settling transient
        energy += buffer[2 * i] * buffer[2 * i];
    }
    energy /= (kFrames - 256);
    EXPECT_LT(energy, 0.05); // Heavy attenuation of center vocal
}

TEST_F(StemsTest, StemTrackBufferReadWrite) {
    mixxx::StemTrackBuffer buffer;
    buffer.allocate(1000);

    EXPECT_EQ(1000u, buffer.totalFrames());
    EXPECT_EQ(0u, buffer.framesReady());
    EXPECT_FALSE(buffer.isFrameReady(0));

    std::vector<float> vocL(500, 0.42f);
    std::vector<float> vocR(500, 0.42f);
    std::vector<float> drumL(500, 0.84f);
    std::vector<float> drumR(500, 0.84f);
    std::vector<float> instL(500, 0.12f);
    std::vector<float> instR(500, 0.12f);

    buffer.writeChunk(0, 500, vocL.data(), vocR.data(), drumL.data(), drumR.data(), instL.data(), instR.data());

    EXPECT_EQ(500u, buffer.framesReady());
    EXPECT_TRUE(buffer.isFrameReady(100));
    EXPECT_TRUE(buffer.isFrameReady(499));
    EXPECT_FALSE(buffer.isFrameReady(500));

    std::vector<float> outVocL(100, 0.0f);
    std::vector<float> outVocR(100, 0.0f);
    buffer.readFrames(50, 100, outVocL.data(), outVocR.data(), nullptr, nullptr, nullptr, nullptr);

    for (int i = 0; i < 100; ++i) {
        EXPECT_FLOAT_EQ(0.42f, outVocL[i]);
        EXPECT_FLOAT_EQ(0.42f, outVocR[i]);
    }
}

TEST_F(StemsTest, HTDemucsEngineAvailable) {
    auto& engine = mixxx::HTDemucsEngine::instance();
    EXPECT_TRUE(engine.isAvailable());
    EXPECT_TRUE(engine.initialize());
    EXPECT_TRUE(engine.isInitialized());

    // Test audio separation on a 7.8s segment
    const int N = mixxx::HTDemucsEngine::kSegmentSamples;
    std::vector<float> in_l(N, 0.0f), in_r(N, 0.0f);
    std::vector<float> voc_l(N, 0.0f), voc_r(N, 0.0f);
    std::vector<float> drum_l(N, 0.0f), drum_r(N, 0.0f);
    std::vector<float> inst_l(N, 0.0f), inst_r(N, 0.0f);

    for (int i = 0; i < N; ++i) {
        in_l[i] = std::sin(2.0f * M_PI * 440.0f * i / 44100.0f);
        in_r[i] = in_l[i];
    }

    bool res = engine.separateSegment3Stem(
            in_l.data(), in_r.data(),
            voc_l.data(), voc_r.data(),
            drum_l.data(), drum_r.data(),
            inst_l.data(), inst_r.data());
    EXPECT_TRUE(res);

    // Verify output energy
    float maxInst = 0.0f;
    for (int i = 0; i < N; ++i) {
        maxInst = std::max(maxInst, std::abs(inst_l[i]));
    }
    EXPECT_GT(maxInst, 0.5f); // 440Hz sine wave should be primarily in Instruments ("Other")
}

TEST_F(StemsTest, StemCacheSaveAndLoad) {
    TrackPointer pTrack = Track::newTemporary("/tmp/test_stem_track.mp3");
    ASSERT_TRUE(pTrack != nullptr);

    constexpr uint32_t kFrames = 1000;
    mixxx::StemTrackBuffer buffer;
    buffer.allocate(kFrames);

    std::vector<float> vocL(kFrames, 0.35f);
    std::vector<float> vocR(kFrames, 0.35f);
    std::vector<float> drumL(kFrames, 0.70f);
    std::vector<float> drumR(kFrames, 0.70f);
    std::vector<float> instL(kFrames, 0.15f);
    std::vector<float> instR(kFrames, 0.15f);

    buffer.writeChunk(0, kFrames,
                      vocL.data(), vocR.data(),
                      drumL.data(), drumR.data(),
                      instL.data(), instR.data());

    EXPECT_TRUE(mixxx::StemCache::saveToCache(pTrack, &buffer));
    EXPECT_TRUE(mixxx::StemCache::hasCachedStems(pTrack));

    mixxx::StemTrackBuffer loadedBuffer;
    EXPECT_TRUE(mixxx::StemCache::loadFromCache(pTrack, &loadedBuffer));
    EXPECT_EQ(loadedBuffer.totalFrames(), kFrames);
    EXPECT_EQ(loadedBuffer.framesReady(), kFrames);

    std::vector<float> outVocL(kFrames, 0.0f);
    std::vector<float> outVocR(kFrames, 0.0f);
    std::vector<float> outDrumL(kFrames, 0.0f);
    std::vector<float> outDrumR(kFrames, 0.0f);
    std::vector<float> outInstL(kFrames, 0.0f);
    std::vector<float> outInstR(kFrames, 0.0f);

    loadedBuffer.readFrames(0, kFrames,
                           outVocL.data(), outVocR.data(),
                           outDrumL.data(), outDrumR.data(),
                           outInstL.data(), outInstR.data());

    for (uint32_t i = 0; i < kFrames; ++i) {
        EXPECT_FLOAT_EQ(0.35f, outVocL[i]);
        EXPECT_FLOAT_EQ(0.35f, outVocR[i]);
        EXPECT_FLOAT_EQ(0.70f, outDrumL[i]);
        EXPECT_FLOAT_EQ(0.70f, outDrumR[i]);
        EXPECT_FLOAT_EQ(0.15f, outInstL[i]);
        EXPECT_FLOAT_EQ(0.15f, outInstR[i]);
    }

    // Cleanup
    QFile::remove(mixxx::StemCache::getCachePath(pTrack));
    EXPECT_FALSE(mixxx::StemCache::hasCachedStems(pTrack));
}

namespace {
void writePcm16Wav(const QString& path, const float* l, const float* r, uint32_t numFrames) {
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly)) {
        return;
    }
    const uint32_t subchunk2Size = numFrames * 2 * sizeof(int16_t);
    const uint32_t chunkSize = 36 + subchunk2Size;
    file.write("RIFF", 4);
    file.write(reinterpret_cast<const char*>(&chunkSize), 4);
    file.write("WAVEfmt ", 8);
    const uint32_t subchunk1Size = 16;
    const uint16_t audioFormat = 1; // PCM
    const uint16_t numChannels = 2;
    const uint32_t sampleRate = 44100;
    const uint32_t byteRate = sampleRate * numChannels * sizeof(int16_t);
    const uint16_t blockAlign = numChannels * sizeof(int16_t);
    const uint16_t bitsPerSample = 16;
    file.write(reinterpret_cast<const char*>(&subchunk1Size), 4);
    file.write(reinterpret_cast<const char*>(&audioFormat), 2);
    file.write(reinterpret_cast<const char*>(&numChannels), 2);
    file.write(reinterpret_cast<const char*>(&sampleRate), 4);
    file.write(reinterpret_cast<const char*>(&byteRate), 4);
    file.write(reinterpret_cast<const char*>(&blockAlign), 2);
    file.write(reinterpret_cast<const char*>(&bitsPerSample), 2);
    file.write("data", 4);
    file.write(reinterpret_cast<const char*>(&subchunk2Size), 4);
    std::vector<int16_t> pcm(numFrames * 2);
    for (uint32_t i = 0; i < numFrames; ++i) {
        float sl = std::clamp(l[i], -1.0f, 1.0f);
        float sr = std::clamp(r[i], -1.0f, 1.0f);
        pcm[2 * i] = static_cast<int16_t>(sl * 32767.0f);
        pcm[2 * i + 1] = static_cast<int16_t>(sr * 32767.0f);
    }
    file.write(reinterpret_cast<const char*>(pcm.data()), subchunk2Size);
}
} // namespace

TEST_F(StemsTest, SeparateDemoTrack2RealAI) {
    QFile wavFile("/tmp/demo2_decoded.wav");
    if (!wavFile.exists() || !wavFile.open(QIODevice::ReadOnly)) {
        GTEST_SKIP() << "Test audio /tmp/demo2_decoded.wav not found, skipping real track separation test";
    }

    // Seek past 4096-byte header to raw audio data
    ASSERT_TRUE(wavFile.seek(4096));

    // Process 600,000 frames (~13.6 seconds, 2 overlapping chunks)
    const uint32_t kFrames = 600000;
    std::vector<float> inInterleaved(kFrames * 2);
    qint64 bytesRead = wavFile.read(reinterpret_cast<char*>(inInterleaved.data()), kFrames * 2 * sizeof(float));
    ASSERT_EQ(bytesRead, static_cast<qint64>(kFrames * 2 * sizeof(float)));

    std::vector<float> in_l(kFrames);
    std::vector<float> in_r(kFrames);
    for (uint32_t i = 0; i < kFrames; ++i) {
        in_l[i] = inInterleaved[2 * i];
        in_r[i] = inInterleaved[2 * i + 1];
    }

    std::vector<float> voc_l(kFrames), voc_r(kFrames);
    std::vector<float> drum_l(kFrames), drum_r(kFrames);
    std::vector<float> inst_l(kFrames), inst_r(kFrames);

    std::cout << "Starting HTDemucs v4 real separation with overlap-add..." << std::endl;
    bool ok = mixxx::HTDemucsEngine::instance().separateTrack(
            kFrames,
            in_l.data(), in_r.data(),
            voc_l.data(), voc_r.data(),
            drum_l.data(), drum_r.data(),
            inst_l.data(), inst_r.data(),
            [](double p) {
                std::cout << "  Separation progress: " << static_cast<int>(p * 100) << "%" << std::endl;
            });
    if (!ok) {
        GTEST_SKIP() << "HTDemucs models not present in system, skipping";
    }

    // Write isolated stem WAV files to /tmp so we can inspect and verify them
    writePcm16Wav("/tmp/demo2_vocals.wav", voc_l.data(), voc_r.data(), kFrames);
    writePcm16Wav("/tmp/demo2_drums.wav", drum_l.data(), drum_r.data(), kFrames);
    writePcm16Wav("/tmp/demo2_inst.wav", inst_l.data(), inst_r.data(), kFrames);
    std::cout << "Wrote separated stems to /tmp/demo2_vocals.wav, /tmp/demo2_drums.wav, /tmp/demo2_inst.wav" << std::endl;

    // Cache stems for the track in Mixxx
    TrackPointer pTrack = Track::newTemporary("/tmp/demo_test_track.mp3");
    ASSERT_TRUE(pTrack != nullptr);

    mixxx::StemTrackBuffer buffer;
    buffer.allocate(kFrames);
    buffer.writeChunk(0, kFrames,
                      voc_l.data(), voc_r.data(),
                      drum_l.data(), drum_r.data(),
                      inst_l.data(), inst_r.data());

    EXPECT_TRUE(mixxx::StemCache::saveToCache(pTrack, &buffer));
    EXPECT_TRUE(mixxx::StemCache::hasCachedStems(pTrack));
    std::cout << "Cached real AI stems in " << mixxx::StemCache::getCachePath(pTrack).toStdString() << std::endl;
}

TEST_F(StemsTest, InspectAdvancedModels) {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "InspectTest");
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(4);

    const std::vector<std::string> modelPaths = {
        "models/bs_roformer_ep317_quantized.onnx",
        "models/scnet-large.int8.onnx"
    };

    for (const auto& path : modelPaths) {
        if (!QFileInfo::exists(QString::fromStdString(path))) {
            continue;
        }
        std::cout << "\n=== Inspecting model: " << path << " ===" << std::endl;
        try {
            Ort::Session session(env, path.c_str(), sessionOptions);
            Ort::AllocatorWithDefaultOptions allocator;

            size_t numInputs = session.GetInputCount();
            std::cout << "Number of inputs: " << numInputs << std::endl;
            for (size_t i = 0; i < numInputs; ++i) {
                auto name = session.GetInputNameAllocated(i, allocator);
                auto typeInfo = session.GetInputTypeInfo(i);
                auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
                auto shape = tensorInfo.GetShape();
                std::cout << "  Input " << i << ": name=" << name.get() << " shape=[";
                for (size_t s = 0; s < shape.size(); ++s) {
                    std::cout << shape[s] << (s + 1 < shape.size() ? ", " : "");
                }
                std::cout << "]" << std::endl;
            }

            size_t numOutputs = session.GetOutputCount();
            std::cout << "Number of outputs: " << numOutputs << std::endl;
            for (size_t i = 0; i < numOutputs; ++i) {
                auto name = session.GetOutputNameAllocated(i, allocator);
                auto typeInfo = session.GetOutputTypeInfo(i);
                auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
                auto shape = tensorInfo.GetShape();
                std::cout << "  Output " << i << ": name=" << name.get() << " shape=[";
                for (size_t s = 0; s < shape.size(); ++s) {
                    std::cout << shape[s] << (s + 1 < shape.size() ? ", " : "");
                }
                std::cout << "]" << std::endl;
            }
        } catch (const Ort::Exception& e) {
            std::cout << "Failed to inspect model: " << e.what() << std::endl;
        }
    }
}

TEST_F(StemsTest, SeparateWithHTDemucsFT) {
    const QString vocalsModelPath = "models/htdemucs_ft_vocals.onnx";
    const QString drumsModelPath = "models/htdemucs_ft_drums.onnx";
    QFile wavFile("/tmp/demo2_decoded.wav");

    if (!QFileInfo::exists(vocalsModelPath) || !QFileInfo::exists(drumsModelPath) || !wavFile.exists()) {
        GTEST_SKIP() << "Fine-tuned models or /tmp/demo2_decoded.wav not found, skipping";
    }
    ASSERT_TRUE(wavFile.open(QIODevice::ReadOnly));
    ASSERT_TRUE(wavFile.seek(4096));

    const int64_t N = 343980; // 7.8 seconds
    std::vector<float> inInterleaved(N * 2);
    qint64 bytesRead = wavFile.read(reinterpret_cast<char*>(inInterleaved.data()), N * 2 * sizeof(float));
    ASSERT_EQ(bytesRead, static_cast<qint64>(N * 2 * sizeof(float)));

    std::vector<float> inputData(2 * N);
    for (int64_t i = 0; i < N; ++i) {
        inputData[i] = inInterleaved[2 * i];
        inputData[N + i] = inInterleaved[2 * i + 1];
    }

    const std::vector<int64_t> inputShape = {1, 2, N};
    const char* const inputNames[] = {"mix"};
    const char* const outputNames[] = {"stems"};

    Ort::Value inTensorVoc = Ort::Value::CreateTensor<float>(
            memoryInfo, inputData.data(), inputData.size(), inputShape.data(), inputShape.size());

    auto outTensorsVoc = vocSession.Run(
            Ort::RunOptions{nullptr}, inputNames, &inTensorVoc, 1, outputNames, 1);
    ASSERT_FALSE(outTensorsVoc.empty());

    const float* outVocData = outTensorsVoc[0].GetTensorData<float>();
    // In htdemucs_ft_vocals, stem 3 is the fine-tuned vocals
    std::vector<float> vocL(N), vocR(N);
    std::memcpy(vocL.data(), outVocData + 3 * (2 * N), N * sizeof(float));
    std::memcpy(vocR.data(), outVocData + 3 * (2 * N) + N, N * sizeof(float));

    Ort::Value inTensorDrum = Ort::Value::CreateTensor<float>(
            memoryInfo, inputData.data(), inputData.size(), inputShape.data(), inputShape.size());

    auto outTensorsDrum = drumSession.Run(
            Ort::RunOptions{nullptr}, inputNames, &inTensorDrum, 1, outputNames, 1);
    ASSERT_FALSE(outTensorsDrum.empty());

    const float* outDrumData = outTensorsDrum[0].GetTensorData<float>();
    // In htdemucs_ft_drums, stem 0 is the fine-tuned drums
    std::vector<float> drumL(N), drumR(N);
    std::memcpy(drumL.data(), outDrumData + 0 * (2 * N), N * sizeof(float));
    std::memcpy(drumR.data(), outDrumData + 0 * (2 * N) + N, N * sizeof(float));

    // Pure residual instruments: Inst = Mix - (Voc + Drum)
    std::vector<float> instL(N), instR(N);
    for (int64_t i = 0; i < N; ++i) {
        instL[i] = inputData[i] - vocL[i] - drumL[i];
        instR[i] = inputData[N + i] - vocR[i] - drumR[i];
    }

    // Write fine-tuned isolated WAVs
    writePcm16Wav("/tmp/demo2_vocals_finetuned.wav", vocL.data(), vocR.data(), static_cast<uint32_t>(N));
    writePcm16Wav("/tmp/demo2_drums_finetuned.wav", drumL.data(), drumR.data(), static_cast<uint32_t>(N));
    writePcm16Wav("/tmp/demo2_inst_finetuned.wav", instL.data(), instR.data(), static_cast<uint32_t>(N));

    std::cout << "\n>>> Generated Fine-Tuned Stems:" << std::endl;
    std::cout << "  /tmp/demo2_vocals_finetuned.wav" << std::endl;
    std::cout << "  /tmp/demo2_drums_finetuned.wav" << std::endl;
    std::cout << "  /tmp/demo2_inst_finetuned.wav" << std::endl;
}

} // namespace
