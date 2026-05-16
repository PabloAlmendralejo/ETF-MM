// Tape format round-trip unit tests.
//
// Writing N synthetic frames and reading them back must yield identical
// (ns_offset, venue, msg_type, payload) tuples. This is the
// determinism contract for replay.

#include "etfmm/tape.hpp"

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace {

using etfmm::MsgType;
using etfmm::TapeFrame;
using etfmm::TapeReader;
using etfmm::TapeWriter;
using etfmm::Venue;

std::string temp_path(const char* tag) {
    auto dir = std::filesystem::temp_directory_path();
    auto fn = std::string("etfmm_tape_test_") + tag + "_" +
              std::to_string(std::random_device{}()) + ".bin";
    return (dir / fn).string();
}

TEST(Tape, RoundTripEmpty) {
    auto path = temp_path("empty");
    {
        TapeWriter w(path, /*epoch_ns=*/123456789ULL);
        ASSERT_TRUE(w.is_open());
        ASSERT_TRUE(w.flush());
    }
    {
        TapeReader r(path);
        EXPECT_EQ(r.epoch_ns(), 123456789ULL);
        EXPECT_EQ(r.version(), 1);
        TapeFrame frame;
        EXPECT_FALSE(r.read_frame(frame));  // immediate EOF
    }
    std::filesystem::remove(path);
}

TEST(Tape, RoundTripSingleFrame) {
    auto path = temp_path("single");
    const std::string payload = R"({"e":"depthUpdate","b":[["100","1"]]})";
    {
        TapeWriter w(path, 1ULL);
        ASSERT_TRUE(w.write_frame(42ULL, Venue::Spot, MsgType::Depth,
                                  payload.data(),
                                  static_cast<std::uint32_t>(payload.size())));
        ASSERT_TRUE(w.flush());
    }
    {
        TapeReader r(path);
        TapeFrame frame;
        ASSERT_TRUE(r.read_frame(frame));
        EXPECT_EQ(frame.ns_offset, 42ULL);
        EXPECT_EQ(frame.venue, Venue::Spot);
        EXPECT_EQ(frame.msg_type, MsgType::Depth);
        EXPECT_EQ(std::string(frame.payload.begin(), frame.payload.end()), payload);
        EXPECT_FALSE(r.read_frame(frame));  // EOF
    }
    std::filesystem::remove(path);
}

TEST(Tape, RoundTripManyFrames) {
    auto path = temp_path("many");
    constexpr int kN = 1000;
    std::vector<std::string> payloads;
    payloads.reserve(kN);
    for (int i = 0; i < kN; ++i) {
        payloads.push_back("payload-" + std::to_string(i));
    }
    {
        TapeWriter w(path, 1000ULL);
        for (int i = 0; i < kN; ++i) {
            const auto v = (i % 2 == 0) ? Venue::Spot : Venue::Perp;
            const auto m = (i % 4 == 0) ? MsgType::BookTicker : MsgType::Depth;
            ASSERT_TRUE(w.write_frame(static_cast<std::uint64_t>(i),
                                      v, m,
                                      payloads[i].data(),
                                      static_cast<std::uint32_t>(payloads[i].size())));
        }
        ASSERT_TRUE(w.flush());
    }
    {
        TapeReader r(path);
        TapeFrame frame;
        for (int i = 0; i < kN; ++i) {
            ASSERT_TRUE(r.read_frame(frame))
                << "expected frame #" << i;
            EXPECT_EQ(frame.ns_offset, static_cast<std::uint64_t>(i));
            const auto v = (i % 2 == 0) ? Venue::Spot : Venue::Perp;
            const auto m = (i % 4 == 0) ? MsgType::BookTicker : MsgType::Depth;
            EXPECT_EQ(frame.venue, v);
            EXPECT_EQ(frame.msg_type, m);
            EXPECT_EQ(std::string(frame.payload.begin(), frame.payload.end()),
                      payloads[i]);
        }
        EXPECT_FALSE(r.read_frame(frame));
    }
    std::filesystem::remove(path);
}

TEST(Tape, ReaderRejectsBadMagic) {
    auto path = temp_path("badmagic");
    {
        std::FILE* fp = std::fopen(path.c_str(), "wb");
        ASSERT_NE(fp, nullptr);
        const char garbage[20] = {0};
        std::fwrite(garbage, 1, sizeof(garbage), fp);
        std::fclose(fp);
    }
    EXPECT_THROW(TapeReader r(path), std::runtime_error);
    std::filesystem::remove(path);
}

}  // namespace
