#include "etfmm/basis_strategy.hpp"

#include <cmath>
#include <stdexcept>

namespace etfmm {

BasisStrategy::BasisStrategy(double ewma_alpha, double threshold_bps)
    : ewma_alpha_(ewma_alpha), threshold_bps_(threshold_bps) {
    if (!(ewma_alpha_ > 0.0 && ewma_alpha_ <= 1.0)) {
        throw std::invalid_argument(
            "BasisStrategy: ewma_alpha must be in (0, 1]");
    }
    if (!(threshold_bps_ > 0.0)) {
        throw std::invalid_argument(
            "BasisStrategy: threshold_bps must be > 0");
    }
}

std::optional<BasisEvent> BasisStrategy::on_update(std::uint64_t ns,
                                                   double spot_mid,
                                                   double perp_mid) {
    ++updates_;
    if (!(spot_mid > 0.0) || !(perp_mid > 0.0)) {
        // Defensive: the order book occasionally produces a zero mid
        // during snapshot resync; skip rather than divide.
        return std::nullopt;
    }
    const double raw_bps = (perp_mid / spot_mid - 1.0) * 1e4;
    if (!ewma_bps_opt_.has_value()) {
        ewma_bps_ = raw_bps;
        ewma_bps_opt_ = raw_bps;
    } else {
        ewma_bps_ = ewma_alpha_ * raw_bps + (1.0 - ewma_alpha_) * ewma_bps_;
    }

    const bool above = std::abs(ewma_bps_) > threshold_bps_;
    if (above && armed_) {
        ++events_;
        armed_ = false;
        BasisEvent ev;
        ev.ns = ns;
        ev.spot_mid = spot_mid;
        ev.perp_mid = perp_mid;
        ev.deviation_bps = ewma_bps_;
        ev.side = ewma_bps_ > 0 ? +1 : -1;
        return ev;
    }
    if (!above) {
        // Re-arm once the smoothed deviation falls back inside the
        // threshold band. This avoids one big move generating thousands
        // of events as |dev| oscillates near the boundary.
        armed_ = true;
    }
    return std::nullopt;
}

}  // namespace etfmm
