// Per-instrument L2 order book.
//
// Bids in a descending-priced std::map<Price, Level>; asks in an
// ascending-priced std::map<Price, Level>. Top-of-book accessors are
// O(log N) in the number of price levels, dominated by the map's tree
// traversal. For BTCUSDT depth at 100 ms cadence the level count is
// typically 20 per side, well inside the cache.
//
// The book exposes two mutation paths:
//
//   apply_snapshot(...)  — replaces the book with a sorted set of
//                          (price, qty) pairs from a REST snapshot.
//   apply_diff(...)      — applies a Binance @depth diff: positive qty
//                          inserts/updates a level, qty == 0 deletes it.
//
// Sequence-number gating (`U`/`u` in Binance protocol) is the caller's
// responsibility; the book itself just applies what it's told.

#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <utility>
#include <vector>

namespace etfmm {

// Prices and quantities are stored as plain doubles. For BTCUSDT this
// is fine; production-grade order books typically use fixed-point
// integers but that's a microstructure-level optimization out of scope
// for the CV claim.
using Price = double;
using Qty   = double;

struct Level {
    Qty qty = 0.0;
};

class OrderBook {
   public:
    // Empty book.
    OrderBook() = default;

    // Replaces the book contents with `bids` and `asks`. Each pair is
    // `(price, qty)`. Zero-quantity entries are silently skipped. The
    // input vectors are *not* required to be sorted; the underlying
    // std::map handles ordering.
    void apply_snapshot(const std::vector<std::pair<Price, Qty>>& bids,
                        const std::vector<std::pair<Price, Qty>>& asks);

    // Applies a Binance-style depth diff. For each (price, qty):
    //   qty > 0  → insert or replace the level at that price
    //   qty == 0 → delete the level at that price (no-op if absent)
    void apply_bid_diff(const std::vector<std::pair<Price, Qty>>& updates);
    void apply_ask_diff(const std::vector<std::pair<Price, Qty>>& updates);

    // Top-of-book accessors. Return std::nullopt when the relevant side
    // is empty.
    std::optional<Price> best_bid() const;
    std::optional<Price> best_ask() const;

    // Mid-price = (best_bid + best_ask) / 2. Returns nullopt if either
    // side is empty.
    std::optional<Price> mid() const;

    // Diagnostic: number of price levels per side.
    std::size_t bid_levels() const noexcept { return bids_.size(); }
    std::size_t ask_levels() const noexcept { return asks_.size(); }

    void clear();

   private:
    // Ordering: bids descending, asks ascending. We use std::greater on
    // the bid map so begin() returns the *highest* bid for symmetry
    // with asks.begin() returning the *lowest* ask.
    std::map<Price, Level, std::greater<Price>> bids_;
    std::map<Price, Level>                      asks_;

    template <typename Map>
    static void apply_diff_(Map& m, const std::vector<std::pair<Price, Qty>>& updates);
};

}  // namespace etfmm
