// 64-bit file positioning. std::fseek/std::ftell take a `long`, which is
// 32 bits on Windows (and on 32-bit Linux), so offsets past 2GB -- routine
// for gpusqz's inputs and payloads -- need the platform's 64-bit variants.
//
// Also positional writes (file_write_at), which let several threads write
// one file at once: they neither use nor move the FILE's own position.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdio>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <io.h>
#include <windows.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

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

// Whether f is a regular file on disk, so file_write_at can be used on it
// (a pipe or a device such as /dev/null gets ordinary sequential writes).
inline bool file_is_regular(FILE* f) {
#if defined(_WIN32)
  HANDLE h = (HANDLE)_get_osfhandle(_fileno(f));
  return h != INVALID_HANDLE_VALUE && GetFileType(h) == FILE_TYPE_DISK;
#else
  struct stat st;
  return fstat(fileno(f), &st) == 0 && S_ISREG(st.st_mode);
#endif
}

// Writes len bytes at absolute offset `offset`, bypassing f's stdio buffer:
// fflush(f) before the first call if f has buffered data. Safe to call from
// several threads at once on disjoint ranges.
inline bool file_write_at(FILE* f, const void* data, size_t len, uint64_t offset) {
  const char* p = static_cast<const char*>(data);
#if defined(_WIN32)
  HANDLE h = (HANDLE)_get_osfhandle(_fileno(f));
  while (len > 0) {
    DWORD part = len > (1u << 30) ? (1u << 30) : (DWORD)len, done = 0;
    OVERLAPPED ov = {};
    ov.Offset = (DWORD)offset;
    ov.OffsetHigh = (DWORD)(offset >> 32);
    if (!WriteFile(h, p, part, &done, &ov) || done == 0) return false;
    p += done;
    len -= done;
    offset += done;
  }
#else
  int fd = fileno(f);
  while (len > 0) {
    ssize_t done = pwrite(fd, p, len, (off_t)offset);
    if (done <= 0) return false;
    p += done;
    len -= (size_t)done;
    offset += (uint64_t)done;
  }
#endif
  return true;
}

} // namespace gpusqz
