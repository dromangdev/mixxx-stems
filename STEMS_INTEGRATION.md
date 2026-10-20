# Mixxx Stems Integration Guide

This branch (`stems-integration`) isolates the complete **Deep Neural Network AI Stem Separation** subsystem for Mixxx. It is designed to be easily merged, cherry-picked, or patched into any Mixxx codebase, specifically including **Mixxx Android** ports and custom forks.

---

## 🏗️ Architecture Overview

```
                        Track Loaded in Deck
                                 │
                                 ▼
                     Has cached .stems file?
                            │         │
                   YES ─────┘         └───── NO
                    │                            │
                    ▼                            ▼
         Instant Read (<50ms)           Launch StemWorkerThread
         into StemTrackBuffer                    │
                    │                            ▼
                    │                   HTDemucs-FT / v4 ONNX
                    │                   (Android NNAPI / CoreML / CPU)
                    │                            │
                    │                            ▼
                    │                   Overlap-Add (25% crossfade)
                    │                            │
                    │                            ▼
                    │                   Write to .stems Cache & Buffer
                    │                            │
                    └───────────┬────────────────┘
                                │
                                ▼
                     StemChannelProcessor
             ┌──────────────────┼──────────────────┐
             ▼                  ▼                  ▼
          Vocals              Drums           Instruments
        (M/S/Fader)        (M/S/Fader)        (M/S/Fader)
             └──────────────────┼──────────────────┘
                                │
                                ▼
                        EngineDeck Output
```

---

## ⚡ Option 1: Apply via Ready-to-Use Git Patch

In your Mixxx / Android codebase, simply run:

```bash
git apply patches/stems-integration.patch
```

Or with `git am` if you want to preserve commit authorship:

```bash
git am < patches/stems-integration.patch
```

---

## 🛠️ Option 2: Manual Step-by-Step Integration

If your target codebase has diverged significantly from upstream Mixxx 2.5, follow these 4 simple steps to integrate Stems:

### Step 1: Copy Stems Subsystem Files
Copy the `src/engine/stems/` directory into your project's `src/engine/stems/`:
- `stemseparator.h` / `stemseparator.cpp`: Neural network inference engine with ONNX Runtime, NNAPI, and Overlap-Add.
- `stemchannelprocessor.h` / `stemchannelprocessor.cpp`: Real-time per-deck DSP processor handling Mute, Solo, and Volume faders.
- `stemdspisolator.h` / `stemdspisolator.cpp`: Zero-latency 4th-order Linkwitz-Riley DSP crossover fallback.
- `stemcache.h` / `stemcache.cpp`: Fast binary disk cache reader/writer (`.stems`).
- `stemtrackbuffer.h` / `stemtrackbuffer.cpp`: Ring/linear memory buffer for isolated stem audio frames.
- `stemworkerthread.h` / `stemworkerthread.cpp`: Background worker thread for non-blocking track separation.

### Step 2: Hook into `EngineDeck`
In `src/engine/channels/enginedeck.h`:
```cpp
// 1. Forward declare
namespace mixxx {
class StemChannelProcessor;
}

// 2. Add member variable inside EngineDeck class
std::unique_ptr<mixxx::StemChannelProcessor> m_pStemProcessor;
```

In `src/engine/channels/enginedeck.cpp`:
```cpp
// 1. Include header
#include "engine/stems/stemchannelprocessor.h"

// 2. In EngineDeck constructor:
m_pStemProcessor = std::make_unique<mixxx::StemChannelProcessor>(getGroup(), m_pBuffer);

// 3. In EngineDeck destructor:
m_pStemProcessor.reset();

// 4. In EngineDeck::processSamples(...) right after reading from m_pBuffer:
if (m_pStemProcessor) {
    m_pStemProcessor->process(pOut, iBufferSize, m_pBuffer->getExactPlayPos());
}
```

### Step 3: Add CMake Configuration
In your `CMakeLists.txt`, add the stems source files to `mixxx-lib`:
```cmake
target_sources(mixxx-lib PRIVATE
  src/engine/stems/stemcache.cpp
  src/engine/stems/stemchannelprocessor.cpp
  src/engine/stems/stemdspisolator.cpp
  src/engine/stems/stemseparator.cpp
  src/engine/stems/stemtrackbuffer.cpp
  src/engine/stems/stemworkerthread.cpp
)
```

And add the automated ONNX Runtime resolver (with Android AAR and NNAPI support):
```cmake
option(MIXXX_AUTO_DOWNLOAD_ONNX "Automatically download ONNX Runtime prebuilts if missing from system" ON)

find_path(ONNXRUNTIME_INCLUDE_DIR NAMES onnxruntime_cxx_api.h
  PATHS
    /opt/homebrew/include/onnxruntime
    /usr/local/include/onnxruntime
    /usr/include/onnxruntime
    ${ANDROID_NDK}/sources/third_party/onnxruntime/include
)
find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime
  PATHS
    /opt/homebrew/lib
    /usr/local/lib
    /usr/lib
    ${ANDROID_NDK}/sources/third_party/onnxruntime/libs/${ANDROID_ABI}
)

if(NOT (ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIBRARY) AND MIXXX_AUTO_DOWNLOAD_ONNX)
  set(ORT_VERSION "1.19.2")
  set(ORT_URL "")

  if(ANDROID)
    set(ORT_URL "https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/${ORT_VERSION}/onnxruntime-android-${ORT_VERSION}.aar")
  elseif(APPLE)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
      set(ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-osx-arm64-${ORT_VERSION}.tgz")
    else()
      set(ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-osx-x86_64-${ORT_VERSION}.tgz")
    endif()
  elseif(WIN32)
    set(ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-win-x64-${ORT_VERSION}.zip")
  elseif(UNIX)
    set(ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz")
  endif()

  if(ORT_URL)
    include(FetchContent)
    FetchContent_Declare(onnxruntime_prebuilt URL "${ORT_URL}" DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    FetchContent_MakeAvailable(onnxruntime_prebuilt)

    if(ANDROID)
      set(ONNXRUNTIME_INCLUDE_DIR "${onnxruntime_prebuilt_SOURCE_DIR}/headers")
      find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime
        PATHS "${onnxruntime_prebuilt_SOURCE_DIR}/jni/${ANDROID_ABI}" NO_DEFAULT_PATH)
      set_target_properties(mixxx PROPERTIES QT_ANDROID_EXTRA_LIBS "${ONNXRUNTIME_LIBRARY}")
    else()
      set(ONNXRUNTIME_INCLUDE_DIR "${onnxruntime_prebuilt_SOURCE_DIR}/include")
      find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime
        PATHS "${onnxruntime_prebuilt_SOURCE_DIR}/lib" NO_DEFAULT_PATH)
    endif()
  endif()
endif()

if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIBRARY)
  target_include_directories(mixxx-lib SYSTEM PUBLIC ${ONNXRUNTIME_INCLUDE_DIR})
  target_link_libraries(mixxx-lib PUBLIC ${ONNXRUNTIME_LIBRARY})
  target_compile_definitions(mixxx-lib PUBLIC MIXXX_HAS_ONNXRUNTIME)
endif()
```

---

## 📱 Android-Specific Notes

1. **Hardware Acceleration (NNAPI):**
   `stemseparator.cpp` automatically enables the Android NNAPI Execution Provider when running on Android. The OS delegates the mathematical tensors directly to the mobile device's NPU/DSP (Snapdragon, Tensor, MediaTek, Exynos).
2. **Thermal & Core Management:**
   Thread allocation automatically clamps between 2 and 4 worker threads on Android to avoid heating the device or triggering battery-saving CPU throttling.
3. **Bundling Models in APK:**
   You can place `.onnx` models into the Android APK's `assets/models/` directory. `findModelFile` automatically extracts them to `QStandardPaths::AppDataLocation/models/` upon first launch.
4. **Touch UI / Controls:**
   Controls for stems (`[ChannelX],stem_vocals_mute`, `[ChannelX],stem_vocals_solo`, `[ChannelX],stem_vocals_volume`, etc.) are standard Mixxx `ControlObject`s and can be directly bound in QML / Qt Quick for touchscreens.
