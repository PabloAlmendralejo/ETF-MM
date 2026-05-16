#include "etfmm/tape.hpp"

#include <cstring>
#include <stdexcept>
#include <string>

namespace etfmm {

namespace {

void write_or_throw(std::FILE* fp, const void* buf, std::size_t n,
                    const char* what) {
    if (std::fwrite(buf, 1, n, fp) != n) {
        throw std::runtime_error(std::string("tape write failed: ") + what);
    }
}

bool read_exact(std::FILE* fp, void* buf, std::size_t n) {
    return std::fread(buf, 1, n, fp) == n;
}

}  // namespace

// --------------------------------------------------------------------- //
// TapeWriter                                                            //
// --------------------------------------------------------------------- //

TapeWriter::TapeWriter(const std::string& path, std::uint64_t epoch_ns)
    : fp_(std::fopen(path.c_str(), "wb")), epoch_ns_(epoch_ns) {
    if (!fp_) {
        throw std::runtime_error("TapeWriter: failed to open " + path);
    }
    TapeHeader header{};
    std::memcpy(header.magic, kTapeMagic, sizeof(kTapeMagic));
    header.epoch_ns = epoch_ns;
    header.version  = kTapeVersion;
    header.reserved = 0;
    write_or_throw(fp_, &header, sizeof(header), "header");
}

TapeWriter::TapeWriter(TapeWriter&& other) noexcept
    : fp_(other.fp_), epoch_ns_(other.epoch_ns_) {
    other.fp_ = nullptr;
    other.epoch_ns_ = 0;
}

TapeWriter& TapeWriter::operator=(TapeWriter&& other) noexcept {
    if (this != &other) {
        close_();
        fp_ = other.fp_;
        epoch_ns_ = other.epoch_ns_;
        other.fp_ = nullptr;
        other.epoch_ns_ = 0;
    }
    return *this;
}

TapeWriter::~TapeWriter() { close_(); }

void TapeWriter::close_() {
    if (fp_) {
        std::fclose(fp_);
        fp_ = nullptr;
    }
}

bool TapeWriter::write_frame(std::uint64_t ns_offset, Venue venue, MsgType msg,
                             const void* payload, std::uint32_t payload_len) {
    if (!fp_) return false;
    FrameHeader hdr{};
    hdr.ns_offset   = ns_offset;
    hdr.venue       = static_cast<std::uint8_t>(venue);
    hdr.msg_type    = static_cast<std::uint8_t>(msg);
    hdr.reserved    = 0;
    hdr.payload_len = payload_len;
    if (std::fwrite(&hdr, 1, sizeof(hdr), fp_) != sizeof(hdr)) return false;
    if (payload_len > 0) {
        if (std::fwrite(payload, 1, payload_len, fp_) != payload_len) return false;
    }
    return true;
}

bool TapeWriter::flush() { return fp_ && std::fflush(fp_) == 0; }

// --------------------------------------------------------------------- //
// TapeReader                                                            //
// --------------------------------------------------------------------- //

TapeReader::TapeReader(const std::string& path)
    : fp_(std::fopen(path.c_str(), "rb")) {
    if (!fp_) {
        throw std::runtime_error("TapeReader: failed to open " + path);
    }
    TapeHeader header{};
    if (!read_exact(fp_, &header, sizeof(header))) {
        std::fclose(fp_);
        fp_ = nullptr;
        throw std::runtime_error("TapeReader: short read on header");
    }
    if (std::memcmp(header.magic, kTapeMagic, sizeof(kTapeMagic)) != 0) {
        std::fclose(fp_);
        fp_ = nullptr;
        throw std::runtime_error("TapeReader: bad magic in header");
    }
    if (header.version != kTapeVersion) {
        std::fclose(fp_);
        fp_ = nullptr;
        throw std::runtime_error("TapeReader: unsupported tape version");
    }
    epoch_ns_ = header.epoch_ns;
    version_  = header.version;
}

TapeReader::TapeReader(TapeReader&& other) noexcept
    : fp_(other.fp_), epoch_ns_(other.epoch_ns_), version_(other.version_) {
    other.fp_ = nullptr;
    other.epoch_ns_ = 0;
    other.version_ = 0;
}

TapeReader& TapeReader::operator=(TapeReader&& other) noexcept {
    if (this != &other) {
        close_();
        fp_ = other.fp_;
        epoch_ns_ = other.epoch_ns_;
        version_  = other.version_;
        other.fp_ = nullptr;
        other.epoch_ns_ = 0;
        other.version_  = 0;
    }
    return *this;
}

TapeReader::~TapeReader() { close_(); }

void TapeReader::close_() {
    if (fp_) {
        std::fclose(fp_);
        fp_ = nullptr;
    }
}

bool TapeReader::read_frame(TapeFrame& out) {
    if (!fp_) return false;
    FrameHeader hdr{};
    const auto n = std::fread(&hdr, 1, sizeof(hdr), fp_);
    if (n == 0) return false;  // clean EOF
    if (n != sizeof(hdr)) {
        throw std::runtime_error("TapeReader: short read on frame header");
    }
    out.ns_offset = hdr.ns_offset;
    out.venue     = static_cast<Venue>(hdr.venue);
    out.msg_type  = static_cast<MsgType>(hdr.msg_type);
    out.payload.resize(hdr.payload_len);
    if (hdr.payload_len > 0) {
        if (!read_exact(fp_, out.payload.data(), hdr.payload_len)) {
            throw std::runtime_error("TapeReader: short read on payload");
        }
    }
    return true;
}

}  // namespace etfmm
