// etfmm_replay: read a tape file produced by etfmm_capture, rebuild
// per-venue order books in-memory, run the spot-vs-perp basis-deviation
// strategy on every mid update, and write triggering events to a CSV.
//
// Usage:
//
//   etfmm_replay <input.tape> <output_events.csv>
//                [--threshold-bps N] [--ewma-alpha A]
//
// Defaults: threshold = 5 bps, ewma_alpha = 0.1.
//
// Binance frame shape (both spot and perp use the same wrapper):
//
//   {"stream": "btcusdt@bookTicker",
//    "data": {"u": ..., "s": "BTCUSDT", "b": "61234.5", "B": "0.5",
//             "a": "61234.6", "A": "0.3", ...}}
//   {"stream": "btcusdt@depth@100ms",
//    "data": {"e": "depthUpdate",
//             "b": [["price", "qty"], ...],
//             "a": [["price", "qty"], ...], ...}}
//
// We use the bookTicker stream for top-of-book mids (cheaper to parse,
// updated on every change, and exactly what the strategy needs). The
// depth stream is parsed into the order book for completeness even
// though it isn't strictly required for the basis strategy.

#include "etfmm/basis_strategy.hpp"
#include "etfmm/order_book.hpp"
#include "etfmm/tape.hpp"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using etfmm::BasisEvent;
using etfmm::BasisStrategy;
using etfmm::MsgType;
using etfmm::OrderBook;
using etfmm::Price;
using etfmm::Qty;
using etfmm::TapeFrame;
using etfmm::TapeReader;
using etfmm::Venue;
using json = nlohmann::json;

// Parse a single "[price, qty]" pair from Binance's depth payload.
// Both fields are sent as JSON strings.
bool parse_level(const json& pair, Price& px, Qty& qty) {
    if (!pair.is_array() || pair.size() < 2) return false;
    if (!pair[0].is_string() || !pair[1].is_string()) return false;
    try {
        px = std::stod(pair[0].get<std::string>());
        qty = std::stod(pair[1].get<std::string>());
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

void apply_book_ticker(OrderBook& book, const json& data) {
    // bookTicker carries best bid and ask only. We model that as a
    // single-level book on each side.
    Price bid_px = 0, ask_px = 0;
    Qty bid_qty = 0, ask_qty = 0;
    try {
        bid_px = std::stod(data.value("b", "0"));
        bid_qty = std::stod(data.value("B", "0"));
        ask_px = std::stod(data.value("a", "0"));
        ask_qty = std::stod(data.value("A", "0"));
    } catch (const std::exception&) {
        return;
    }
    if (bid_qty <= 0 || ask_qty <= 0) return;
    book.apply_snapshot({{bid_px, bid_qty}}, {{ask_px, ask_qty}});
}

void apply_depth_update(OrderBook& book, const json& data) {
    std::vector<std::pair<Price, Qty>> bid_updates, ask_updates;
    if (data.contains("b") && data["b"].is_array()) {
        for (const auto& lvl : data["b"]) {
            Price px;
            Qty qty;
            if (parse_level(lvl, px, qty)) bid_updates.emplace_back(px, qty);
        }
    }
    if (data.contains("a") && data["a"].is_array()) {
        for (const auto& lvl : data["a"]) {
            Price px;
            Qty qty;
            if (parse_level(lvl, px, qty)) ask_updates.emplace_back(px, qty);
        }
    }
    if (!bid_updates.empty()) book.apply_bid_diff(bid_updates);
    if (!ask_updates.empty()) book.apply_ask_diff(ask_updates);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: etfmm_replay <input.tape> <output_events.csv> "
                     "[--threshold-bps N] [--ewma-alpha A]\n";
        return 2;
    }
    const std::string tape_path = argv[1];
    const std::string csv_path = argv[2];
    double threshold_bps = 5.0;
    double ewma_alpha = 0.1;
    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--threshold-bps" && i + 1 < argc) {
            threshold_bps = std::atof(argv[++i]);
        } else if (a == "--ewma-alpha" && i + 1 < argc) {
            ewma_alpha = std::atof(argv[++i]);
        } else {
            std::cerr << "unknown argument: " << a << "\n";
            return 2;
        }
    }

    TapeReader reader(tape_path);
    OrderBook spot_book;
    OrderBook perp_book;
    BasisStrategy strategy(ewma_alpha, threshold_bps);

    std::ofstream csv(csv_path);
    if (!csv) {
        std::cerr << "failed to open output: " << csv_path << "\n";
        return 3;
    }
    csv << "ns,spot_mid,perp_mid,deviation_bps,side\n";

    TapeFrame frame;
    std::uint64_t total = 0;
    std::uint64_t events = 0;
    std::uint64_t parsed_ok = 0;
    std::uint64_t parsed_err = 0;

    while (reader.read_frame(frame)) {
        ++total;
        json envelope;
        try {
            envelope = json::parse(frame.payload.begin(), frame.payload.end());
        } catch (const std::exception&) {
            ++parsed_err;
            continue;
        }
        // Binance combined-stream frames wrap the actual payload under
        // "data"; the stream name lives in "stream".
        if (!envelope.contains("data") || !envelope.contains("stream")) {
            ++parsed_err;
            continue;
        }
        const auto& data = envelope["data"];
        const std::string stream = envelope["stream"].get<std::string>();

        OrderBook& book = (frame.venue == Venue::Spot) ? spot_book : perp_book;
        if (stream.find("@bookTicker") != std::string::npos) {
            apply_book_ticker(book, data);
        } else if (stream.find("@depth") != std::string::npos) {
            apply_depth_update(book, data);
        } else {
            ++parsed_err;
            continue;
        }
        ++parsed_ok;

        // Run the strategy when both books have a usable mid.
        const auto spot = spot_book.mid();
        const auto perp = perp_book.mid();
        if (!spot || !perp) continue;
        const auto ev = strategy.on_update(frame.ns_offset, *spot, *perp);
        if (ev) {
            csv << ev->ns << ',' << ev->spot_mid << ',' << ev->perp_mid
                << ',' << ev->deviation_bps << ',' << ev->side << '\n';
            ++events;
        }
    }

    csv.flush();
    std::cerr << "[replay] frames: " << total
              << "  parsed_ok: " << parsed_ok
              << "  parsed_err: " << parsed_err
              << "  events: " << events << "\n";
    std::cerr << "[replay] strategy updates: " << strategy.updates_seen()
              << "  events_emitted: " << strategy.events_emitted()
              << "\n";
    std::cerr << "[replay] wrote " << csv_path << "\n";
    return 0;
}
