#pragma once

#include <QString>
#include <memory>

#include "track/track.h"
#include "engine/stems/stemtrackbuffer.h"

namespace mixxx {

class StemCache {
  public:
    static QString getCacheDir();
    static QString getCachePath(TrackPointer pTrack);

    // Check if valid cached stems exist on disk for the given track
    static bool hasCachedStems(TrackPointer pTrack);

    // Load stems from disk into buffer. Returns true on success.
    static bool loadFromCache(TrackPointer pTrack, StemTrackBuffer* pBuffer);

    // Save stems from buffer to disk cache. Returns true on success.
    static bool saveToCache(TrackPointer pTrack, const StemTrackBuffer* pBuffer);
};

} // namespace mixxx
