#include "etfmm/order_book.hpp"

namespace etfmm {

template <typename Map>
void OrderBook::apply_diff_(Map& m,
                            const std::vector<std::pair<Price, Qty>>& updates) {
    for (const auto& [px, qty] : updates) {
        if (qty <= 0.0) {
            m.erase(px);
        } else {
            m[px] = Level{qty};
        }
    }
}

void OrderBook::apply_snapshot(const std::vector<std::pair<Price, Qty>>& bids,
                               const std::vector<std::pair<Price, Qty>>& asks) {
    bids_.clear();
    asks_.clear();
    for (const auto& [px, qty] : bids) {
        if (qty > 0.0) bids_[px] = Level{qty};
    }
    for (const auto& [px, qty] : asks) {
        if (qty > 0.0) asks_[px] = Level{qty};
    }
}

void OrderBook::apply_bid_diff(const std::vector<std::pair<Price, Qty>>& updates) {
    apply_diff_(bids_, updates);
}

void OrderBook::apply_ask_diff(const std::vector<std::pair<Price, Qty>>& updates) {
    apply_diff_(asks_, updates);
}

std::optional<Price> OrderBook::best_bid() const {
    if (bids_.empty()) return std::nullopt;
    return bids_.begin()->first;
}

std::optional<Price> OrderBook::best_ask() const {
    if (asks_.empty()) return std::nullopt;
    return asks_.begin()->first;
}

std::optional<Price> OrderBook::mid() const {
    auto bb = best_bid();
    auto ba = best_ask();
    if (!bb || !ba) return std::nullopt;
    return 0.5 * (*bb + *ba);
}

void OrderBook::clear() {
    bids_.clear();
    asks_.clear();
}

}  // namespace etfmm
