#include "engine/stems/stemcache.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QCryptographicHash>
#include <QDebug>

namespace mixxx {

namespace {
constexpr char kMagic[8] = {'M', 'I', 'X', 'S', 'T', 'E', 'M', '1'};
} // namespace

QString StemCache::getCacheDir() {
    QString dir = QDir::homePath() + "/Library/Application Support/Mixxx/stems_cache";
    QDir().mkpath(dir);
    return dir;
}

QString StemCache::getCachePath(TrackPointer pTrack) {
    if (!pTrack) {
        return QString();
    }
    QString key = pTrack->getLocation() + ":" + QString::number(pTrack->getFileInfo().sizeInBytes());
    QByteArray hash = QCryptographicHash::hash(key.toUtf8(), QCryptographicHash::Sha1).toHex();
    return getCacheDir() + "/" + hash + ".stems";
}

bool StemCache::hasCachedStems(TrackPointer pTrack) {
    QString path = getCachePath(pTrack);
    if (path.isEmpty()) {
        return false;
    }
    QFileInfo fi(path);
    return fi.exists() && fi.size() > static_cast<qint64>(sizeof(kMagic) + sizeof(uint32_t));
}

bool StemCache::loadFromCache(TrackPointer pTrack, StemTrackBuffer* pBuffer) {
    if (!pTrack || !pBuffer) {
        return false;
    }

    QString path = getCachePath(pTrack);
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        return false;
    }

    char magic[8];
    if (file.read(magic, sizeof(magic)) != sizeof(magic) ||
        std::memcmp(magic, kMagic, sizeof(kMagic)) != 0) {
        qDebug() << "StemCache: Invalid magic header in" << path;
        return false;
    }

    uint32_t totalFrames = 0;
    if (file.read(reinterpret_cast<char*>(&totalFrames), sizeof(uint32_t)) != sizeof(uint32_t) ||
        totalFrames == 0) {
        return false;
    }

    pBuffer->allocate(totalFrames);

    const qint64 channelBytes = static_cast<qint64>(totalFrames) * sizeof(float);
    std::vector<float> vocL(totalFrames);
    std::vector<float> vocR(totalFrames);
    std::vector<float> drumL(totalFrames);
    std::vector<float> drumR(totalFrames);
    std::vector<float> instL(totalFrames);
    std::vector<float> instR(totalFrames);

    if (file.read(reinterpret_cast<char*>(vocL.data()), channelBytes) != channelBytes ||
        file.read(reinterpret_cast<char*>(vocR.data()), channelBytes) != channelBytes ||
        file.read(reinterpret_cast<char*>(drumL.data()), channelBytes) != channelBytes ||
        file.read(reinterpret_cast<char*>(drumR.data()), channelBytes) != channelBytes ||
        file.read(reinterpret_cast<char*>(instL.data()), channelBytes) != channelBytes ||
        file.read(reinterpret_cast<char*>(instR.data()), channelBytes) != channelBytes) {
        qDebug() << "StemCache: Incomplete stem data read from" << path;
        return false;
    }

    pBuffer->writeChunk(0, totalFrames,
                        vocL.data(), vocR.data(),
                        drumL.data(), drumR.data(),
                        instL.data(), instR.data());

    qDebug() << "StemCache: Successfully loaded real AI stems from cache for" << pTrack->getTitle()
             << "(" << totalFrames << "frames)";
    return true;
}

bool StemCache::saveToCache(TrackPointer pTrack, const StemTrackBuffer* pBuffer) {
    if (!pTrack || !pBuffer || pBuffer->totalFrames() == 0) {
        return false;
    }

    const uint32_t totalFrames = pBuffer->totalFrames();
    if (pBuffer->framesReady() < totalFrames) {
        return false; // Incomplete separation, don't cache
    }

    QString path = getCachePath(pTrack);
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        qDebug() << "StemCache: Failed to create cache file" << path;
        return false;
    }

    if (file.write(kMagic, sizeof(kMagic)) != sizeof(kMagic) ||
        file.write(reinterpret_cast<const char*>(&totalFrames), sizeof(uint32_t)) != sizeof(uint32_t)) {
        return false;
    }

    const qint64 channelBytes = static_cast<qint64>(totalFrames) * sizeof(float);
    std::vector<float> vocL(totalFrames);
    std::vector<float> vocR(totalFrames);
    std::vector<float> drumL(totalFrames);
    std::vector<float> drumR(totalFrames);
    std::vector<float> instL(totalFrames);
    std::vector<float> instR(totalFrames);

    pBuffer->readFrames(0, totalFrames,
                        vocL.data(), vocR.data(),
                        drumL.data(), drumR.data(),
                        instL.data(), instR.data());

    if (file.write(reinterpret_cast<const char*>(vocL.data()), channelBytes) != channelBytes ||
        file.write(reinterpret_cast<const char*>(vocR.data()), channelBytes) != channelBytes ||
        file.write(reinterpret_cast<const char*>(drumL.data()), channelBytes) != channelBytes ||
        file.write(reinterpret_cast<const char*>(drumR.data()), channelBytes) != channelBytes ||
        file.write(reinterpret_cast<const char*>(instL.data()), channelBytes) != channelBytes ||
        file.write(reinterpret_cast<const char*>(instR.data()), channelBytes) != channelBytes) {
        qDebug() << "StemCache: Write error saving stems cache to" << path;
        return false;
    }

    qDebug() << "StemCache: Successfully saved real AI stems to cache for" << pTrack->getTitle()
             << "at" << path;
    return true;
}

} // namespace mixxx
