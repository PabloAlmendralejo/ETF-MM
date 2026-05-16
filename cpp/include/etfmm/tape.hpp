// Binary tape format for captured market-data frames.
//
// File layout:
//
//   [TapeHeader]                       // once at file start
//   [FrameHeader][payload bytes] ...   // repeated
//
// The TapeHeader records the wall-clock epoch at capture start; every
// FrameHeader carries an ns_offset relative to that epoch so the file
// itself is run-time-independent except for the epoch field. Replays
// of the same tape produce the same sequence of decoded frames.
//
// Payloads are opaque blobs: typically raw JSON from the Binance WS,
// but anything fits as long as `payload_len` is correct.

#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace etfmm {

// Magic bytes for the tape header. Allows future format changes via the
// `version` field.
inline constexpr char kTapeMagic[8] = {'E', 'T', 'F', 'M', 'M', 'V', '0', '1'};
inline constexpr std::uint16_t kTapeVersion = 1;

#pragma pack(push, 1)
struct TapeHeader {
    char magic[8];          // "ETFMMV01"
    std::uint64_t epoch_ns; // wall clock at capture start
    std::uint16_t version;  // kTapeVersion
    std::uint16_t reserved; // zero
};
static_assert(sizeof(TapeHeader) == 20, "TapeHeader layout drifted");

struct FrameHeader {
    std::uint64_t ns_offset;   // ns from TapeHeader.epoch_ns
    std::uint8_t  venue;       // 0 = spot, 1 = perp
    std::uint8_t  msg_type;    // 0 = depth, 1 = bookTicker, 2 = trade, 3 = snapshot
    std::uint16_t reserved;    // zero
    std::uint32_t payload_len; // bytes immediately following this header
};
static_assert(sizeof(FrameHeader) == 16, "FrameHeader layout drifted");
#pragma pack(pop)

enum class Venue : std::uint8_t {
    Spot = 0,
    Perp = 1,
};

enum class MsgType : std::uint8_t {
    Depth      = 0,
    BookTicker = 1,
    Trade      = 2,
    Snapshot   = 3,
};

// Streamed writer. Opens the tape file, writes the header on
// construction, appends frames via `write_frame`, and flushes/closes on
// destruction. Buffered I/O via a `FILE*`; not thread-safe — the caller
// should ensure a single writer at a time (e.g. one tape-writer thread).
class TapeWriter {
   public:
    // `epoch_ns` is the wall-clock instant captured frames are timed
    // against. Frames written via `write_frame` carry an `ns_offset`
    // relative to this value.
    TapeWriter(const std::string& path, std::uint64_t epoch_ns);

    // Move-only.
    TapeWriter(const TapeWriter&) = delete;
    TapeWriter& operator=(const TapeWriter&) = delete;
    TapeWriter(TapeWriter&& other) noexcept;
    TapeWriter& operator=(TapeWriter&& other) noexcept;

    ~TapeWriter();

    // Returns true on success.
    bool write_frame(std::uint64_t ns_offset,
                     Venue venue,
                     MsgType msg_type,
                     const void* payload,
                     std::uint32_t payload_len);

    // Forces a libc flush. Useful for tests that read the file back
    // before the writer goes out of scope.
    bool flush();

    bool is_open() const noexcept { return fp_ != nullptr; }
    std::uint64_t epoch_ns() const noexcept { return epoch_ns_; }

   private:
    void close_();

    std::FILE* fp_ = nullptr;
    std::uint64_t epoch_ns_ = 0;
};

// One decoded frame, payload bytes copied into a `std::vector<char>`.
struct TapeFrame {
    std::uint64_t ns_offset = 0;
    Venue         venue     = Venue::Spot;
    MsgType       msg_type  = MsgType::Depth;
    std::vector<char> payload;
};

// Streamed reader. Opens the tape file on construction; the header is
// read eagerly and exposed via `epoch_ns()`. `read_frame` advances one
// frame at a time and returns false at EOF.
class TapeReader {
   public:
    explicit TapeReader(const std::string& path);

    TapeReader(const TapeReader&) = delete;
    TapeReader& operator=(const TapeReader&) = delete;
    TapeReader(TapeReader&& other) noexcept;
    TapeReader& operator=(TapeReader&& other) noexcept;

    ~TapeReader();

    bool is_open() const noexcept { return fp_ != nullptr; }
    std::uint64_t epoch_ns() const noexcept { return epoch_ns_; }
    std::uint16_t version() const noexcept { return version_; }

    // Reads one frame. Returns false at EOF. Throws std::runtime_error
    // on partial reads or invalid headers.
    bool read_frame(TapeFrame& out);

   private:
    void close_();

    std::FILE* fp_ = nullptr;
    std::uint64_t epoch_ns_ = 0;
    std::uint16_t version_  = 0;
};

}  // namespace etfmm
