#pragma once
// Portable native trace recorder. Historical instrumentation retains its include
// name, but this header does not require CUDA, NVTX, CANN or libtorch.
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <fcntl.h>
#include <memory>
#include <mutex>
#include <string>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

namespace UcmProfile {
inline uint64_t Now() {
    struct timespec value {};
    clock_gettime(CLOCK_MONOTONIC, &value);
    return static_cast<uint64_t>(value.tv_sec) * 1000000000ULL + value.tv_nsec;
}
inline long Tid() { return syscall(SYS_gettid); }
inline bool Enabled() { return std::getenv("UCM_PIPELINE_CAPTURE") != nullptr; }
inline std::string Escape(const std::string& value) {
    std::string result;
    constexpr char hex[] = "0123456789abcdef";
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') { result += '\\'; result += c; }
        else if (c < 32) { result += "\\u00"; result += hex[c >> 4]; result += hex[c & 15]; }
        else { result += c; }
    }
    return result;
}
inline void Emit(const std::string& label, uint64_t begin, uint64_t end, long tid, bool point=false) {
    const char* directory = std::getenv("BUNDLE_NATIVE_TRACE_DIR");
    if (!directory || !*directory) return;
    static std::mutex lock;
    std::lock_guard<std::mutex> guard(lock);
    // One file per worker/process. A failed write is visible in an error marker;
    // analysis must reject an incomplete trace instead of filling its gaps.
    const std::string path = std::string(directory) + "/native_" + std::to_string(getpid()) + ".jsonl";
    static int fd = -1;
    static pid_t writer_pid = 0;
    static std::string writer_path;
    if (writer_pid != getpid() || writer_path != path) {
        if (fd >= 0) close(fd);
        fd = open(path.c_str(), O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0600);
        writer_pid = getpid();
        writer_path = path;
    }
    if (fd < 0) { std::fprintf(stderr, "BUNDLE_NATIVE_TRACE_ERROR open errno=%d\n", errno); return; }
    const std::string text = "{\"schema_version\":1,\"pid\":" + std::to_string(getpid()) +
        ",\"tid\":" + std::to_string(tid) + ",\"end_tid\":" + std::to_string(Tid()) +
        ",\"start_ns\":" + std::to_string(begin) + ",\"end_ns\":" + std::to_string(end) +
        ",\"point\":" + (point ? "true" : "false") + ",\"label\":\"" + Escape(label) + "\"}\n";
    const auto count = write(fd, text.data(), text.size());
    if (count != static_cast<ssize_t>(text.size())) std::fprintf(stderr, "BUNDLE_NATIVE_TRACE_ERROR write errno=%d\n", errno);
}
inline void Mark(const char* label) { if (Enabled()) { const auto now=Now(); Emit(label, now, now, Tid(), true); } }
struct Range {
    const bool active;
    const std::string label;
    const uint64_t start;
    const long tid;
    explicit Range(const char* text) : active(Enabled()), label(active ? text : ""), start(active ? Now() : 0), tid(active ? Tid() : 0) {}
    ~Range() { if (active) Emit(label, start, Now(), tid); }
    Range(const Range&) = delete;
    Range& operator=(const Range&) = delete;
};
struct Async {
    const std::string label;
    const uint64_t start;
    const long tid;
    std::atomic<bool> ended{false};
    explicit Async(const std::string& text) : label(text), start(Now()), tid(Tid()) {}
    void End() { if (!ended.exchange(true)) Emit(label, start, Now(), tid); }
    ~Async() { End(); }
};
inline std::shared_ptr<Async> Start(const std::string& label) { return Enabled() ? std::make_shared<Async>(label) : nullptr; }
inline void End(const std::shared_ptr<Async>& span) { if (span) span->End(); }
}
