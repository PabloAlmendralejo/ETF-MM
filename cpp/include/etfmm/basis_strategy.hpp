// Spot-vs-perp basis-deviation strategy.
//
// Inputs: spot mid, perp mid, timestamp.
// Output: events when |perp_mid / spot_mid - 1| crosses a configurable
// threshold (in basis points). An exponential-weighted moving average
// smooths the raw deviation series so a single noisy tick can't trigger
// a spurious event; the threshold check is on the smoothed series.
//
// The strategy is structurally analogous to ETF/NAV creation-redemption
// arbitrage: when a synthetic-basket-priced instrument deviates from
// the directly-tradable instrument, an arbitrageur creates or redeems
// to capture the spread. We use spot vs perpetual futures because
// real US-equity ETF L2 data is paywalled and Binance offers free L2
// for both spot and perp on the same underlying.

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

namespace etfmm {

struct BasisEvent {
    std::uint64_t ns;     // tape ns_offset of the triggering update
    double spot_mid;
    double perp_mid;
    double deviation_bps; // smoothed, signed: perp - spot in bps of spot
    int side;             // +1 if perp rich vs spot, -1 if perp cheap
};

class BasisStrategy {
   public:
    // ewma_alpha in (0, 1]: 1.0 disables smoothing, smaller values are
    //                      more aggressive smoothing. 0.1 is a good
    //                      starting point for 100ms data.
    // threshold_bps:        emit an event when the smoothed |deviation|
    //                      crosses this value going outward (rising
    //                      above + threshold, or falling below -threshold).
    BasisStrategy(double ewma_alpha = 0.1, double threshold_bps = 5.0);

    // Update the strategy with the current spot and perp mids and the
    // timestamp of the triggering update. Returns an event when the
    // smoothed deviation crosses the threshold; otherwise std::nullopt.
    //
    // Crossing semantics: an event fires only on the *transition* from
    // |dev| <= threshold to |dev| > threshold. While |dev| stays above
    // threshold no further events are emitted. Once |dev| returns to
    // <= threshold the strategy re-arms.
    std::optional<BasisEvent> on_update(std::uint64_t ns,
                                        double spot_mid,
                                        double perp_mid);

    // Telemetry.
    std::uint64_t updates_seen() const noexcept { return updates_; }
    std::uint64_t events_emitted() const noexcept { return events_; }
    double current_smoothed_bps() const noexcept { return ewma_bps_; }

   private:
    double ewma_alpha_;
    double threshold_bps_;
    std::optional<double> ewma_bps_opt_;  // bootstrapped on first sample
    double ewma_bps_ = 0.0;
    bool armed_ = true;  // fires only on threshold-crossing transitions
    std::uint64_t updates_ = 0;
    std::uint64_t events_ = 0;
};

}  // namespace etfmm
