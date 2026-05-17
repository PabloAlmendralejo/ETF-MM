// Boost.Beast WebSocket client over TLS.
//
// Connects to a single Binance public market-data WS endpoint and
// forwards every received text frame to a user-supplied callback. The
// callback runs on the WS thread; downstream consumers should push
// into an SpscRing (see ring.hpp) and process on a separate thread.
//
// The client is intentionally simple: one host, one stream URL, one
// callback. A reconnect loop with exponential backoff is built in;
// the caller signals shutdown via stop().
//
// Threading model
// ---------------
// run() blocks on its own io_context until stop() is called. Typical
// use:
//
//     WsClient client(host, port, target);
//     client.set_on_text([](std::string_view bytes) { ... });
//     std::thread t([&]{ client.run(); });
//     ...
//     client.stop();
//     t.join();

#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <string>
#include <string_view>

namespace etfmm {

class WsClient {
   public:
    using TextHandler = std::function<void(std::string_view)>;

    // host:   "stream.binance.com" or "fstream.binance.com"
    // port:   "9443"
    // target: stream path including the leading slash, e.g.
    //         "/stream?streams=btcusdt@depth@100ms/btcusdt@bookTicker"
    WsClient(std::string host, std::string port, std::string target);

    WsClient(const WsClient&) = delete;
    WsClient& operator=(const WsClient&) = delete;

    void set_on_text(TextHandler h) { on_text_ = std::move(h); }

    // Optional: called every time the client transitions to "connected"
    // after a successful WS handshake. Useful for marking RESYNC events
    // in the tape.
    void set_on_connected(std::function<void()> h) {
        on_connected_ = std::move(h);
    }

    // Blocks until stop() is called or an unrecoverable error occurs.
    // Reconnects on socket errors with capped exponential backoff.
    void run();

    // Signals the run loop to exit at the next opportunity.
    void stop();

    // Telemetry counters (atomic, lock-free reads).
    std::uint64_t messages_received() const noexcept {
        return msg_count_.load(std::memory_order_relaxed);
    }
    std::uint64_t reconnects() const noexcept {
        return reconnects_.load(std::memory_order_relaxed);
    }

   private:
    // Single connection attempt; returns when the socket closes.
    // Exceptions are caught by run() and treated as reconnect signals.
    void run_session_();

    std::string host_;
    std::string port_;
    std::string target_;

    TextHandler on_text_;
    std::function<void()> on_connected_;

    std::atomic<bool> stopping_{false};
    std::atomic<std::uint64_t> msg_count_{0};
    std::atomic<std::uint64_t> reconnects_{0};
};

}  // namespace etfmm
