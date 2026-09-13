// 64-bit file positioning. std::fseek/std::ftell take a `long`, which is
// 32 bits on Windows (and on 32-bit Linux), so offsets past 2GB -- routine
// for gpusqz's inputs and payloads -- need the platform's 64-bit variants.
#pragma once
#include <cstdint>
#include <cstdio>

namespace gpusqz {

inline bool file_seek(FILE* f, uint64_t offset, int whence = SEEK_SET) {
#if defined(_WIN32)
  return _fseeki64(f, (__int64)offset, whence) == 0;
#else
  return fseeko(f, (off_t)offset, whence) == 0;
#endif
}

inline uint64_t file_tell(FILE* f) {
#if defined(_WIN32)
  return (uint64_t)_ftelli64(f);
#else
  return (uint64_t)ftello(f);
#endif
}

} // namespace gpusqz
