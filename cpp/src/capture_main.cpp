// etfmm_capture: connect to Binance spot + futures public WS streams
// and write every received message to a binary tape file.
//
// Usage:
//
//   etfmm_capture <output.tape> [--seconds N]
//
// Default duration: runs until Ctrl-C. With --seconds N the process
// exits after N seconds.
//
// Streams (BTCUSDT):
//   spot: wss://stream.binance.com:9443/stream
//         ?streams=btcusdt@depth@100ms/btcusdt@bookTicker
//   perp: wss://fstream.binance.com:9443/stream
//         ?streams=btcusdt@depth@100ms/btcusdt@bookTicker
//
// One thread per WS, each pushing raw frames into its own SPSC ring.
// A single tape-writer thread drains both rings and serializes to disk.

#include "etfmm/ring.hpp"
#include "etfmm/tape.hpp"
#include "etfmm/ws_client.hpp"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace {

using etfmm::MsgType;
using etfmm::SpscRing;
using etfmm::TapeWriter;
using etfmm::Venue;
using etfmm::WsClient;

// Each frame on the ring carries the wall-clock ns offset and the
// venue tag plus the raw payload. We use a movable POD-ish struct.
struct CapturedFrame {
    std::uint64_t ns_offset = 0;
    Venue venue = Venue::Spot;
    MsgType msg_type = MsgType::Depth;
    std::vector<char> payload;
};

constexpr std::size_t kRingCapacity = 1u << 14;  // 16384

std::atomic<bool> g_stop{false};

void signal_handler(int) {
    g_stop.store(true, std::memory_order_release);
}

std::uint64_t now_ns_since(std::uint64_t epoch_ns) {
    using namespace std::chrono;
    const auto now = duration_cast<nanoseconds>(
                         system_clock::now().time_since_epoch())
                         .count();
    return static_cast<std::uint64_t>(now) - epoch_ns;
}

std::uint64_t wallclock_ns() {
    using namespace std::chrono;
    return static_cast<std::uint64_t>(
        duration_cast<nanoseconds>(system_clock::now().time_since_epoch())
            .count());
}

void install_handler(int sig) {
    std::signal(sig, signal_handler);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: etfmm_capture <output.tape> [--seconds N]\n";
        return 2;
    }
    const std::string tape_path = argv[1];
    int run_seconds = -1;  // -1 = until Ctrl-C
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--seconds" && i + 1 < argc) {
            run_seconds = std::atoi(argv[++i]);
        } else {
            std::cerr << "unknown argument: " << a << "\n";
            return 2;
        }
    }

    install_handler(SIGINT);
#ifdef SIGTERM
    install_handler(SIGTERM);
#endif

    const std::uint64_t epoch_ns = wallclock_ns();
    TapeWriter writer(tape_path, epoch_ns);

    // Heap-allocate the rings: at 16384 slots with cache-line-aligned
    // entries the rings are ~1 MB each, which would overflow the
    // default Windows stack if declared as locals.
    auto spot_ring = std::make_unique<SpscRing<CapturedFrame, kRingCapacity>>();
    auto perp_ring = std::make_unique<SpscRing<CapturedFrame, kRingCapacity>>();

    auto make_handler = [&](Venue venue,
                            SpscRing<CapturedFrame, kRingCapacity>& ring) {
        return [&, venue](std::string_view bytes) {
            CapturedFrame f;
            f.ns_offset = now_ns_since(epoch_ns);
            f.venue = venue;
            f.msg_type = MsgType::Depth;  // we don't disambiguate per-stream
            f.payload.assign(bytes.begin(), bytes.end());
            if (!ring.try_push(std::move(f))) {
                // Drop on overflow. We log via an atomic counter rather
                // than stderr to avoid blocking the WS thread.
            }
        };
    };

    WsClient spot_client{
        "stream.binance.com", "9443",
        "/stream?streams=btcusdt@depth@100ms/btcusdt@bookTicker"};
    spot_client.set_on_text(make_handler(Venue::Spot, *spot_ring));

    WsClient perp_client{
        "fstream.binance.com", "9443",
        "/stream?streams=btcusdt@depth@100ms/btcusdt@bookTicker"};
    perp_client.set_on_text(make_handler(Venue::Perp, *perp_ring));

    std::thread spot_thread([&] { spot_client.run(); });
    std::thread perp_thread([&] { perp_client.run(); });

    // Tape writer thread drains both rings.
    std::thread writer_thread([&] {
        std::uint64_t written = 0;
        std::uint64_t last_flush_ns = epoch_ns;
        const auto drain = [&](auto& ring) {
            CapturedFrame f;
            while (ring.try_pop(f)) {
                writer.write_frame(f.ns_offset, f.venue, f.msg_type,
                                   f.payload.data(),
                                   static_cast<std::uint32_t>(
                                       f.payload.size()));
                ++written;
            }
        };
        while (!g_stop.load(std::memory_order_acquire)) {
            drain(*spot_ring);
            drain(*perp_ring);
            const auto now = wallclock_ns();
            if (now - last_flush_ns > 1'000'000'000ULL) {  // 1 s
                writer.flush();
                last_flush_ns = now;
                std::cerr << "[capture] frames written: " << written
                          << " (spot ring: " << spot_ring->approx_size()
                          << ", perp ring: " << perp_ring->approx_size()
                          << ")\n";
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        // Final drain after stop.
        drain(*spot_ring);
        drain(*perp_ring);
        writer.flush();
        std::cerr << "[capture] total frames written: " << written << "\n";
    });

    if (run_seconds > 0) {
        const auto deadline =
            std::chrono::steady_clock::now() +
            std::chrono::seconds(run_seconds);
        while (std::chrono::steady_clock::now() < deadline &&
               !g_stop.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        g_stop.store(true, std::memory_order_release);
    } else {
        while (!g_stop.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    spot_client.stop();
    perp_client.stop();
    spot_thread.join();
    perp_thread.join();
    writer_thread.join();

    std::cerr << "[capture] spot reconnects: " << spot_client.reconnects()
              << ", perp reconnects: " << perp_client.reconnects() << "\n";
    return 0;
}
