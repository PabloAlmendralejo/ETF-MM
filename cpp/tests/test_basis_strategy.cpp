// Basis strategy unit tests.

#include "etfmm/basis_strategy.hpp"

#include <gtest/gtest.h>

namespace {

using etfmm::BasisEvent;
using etfmm::BasisStrategy;

TEST(BasisStrategy, NoEventBelowThreshold) {
    // alpha = 1.0 disables smoothing so the test reads off raw bps.
    BasisStrategy s(/*ewma_alpha=*/1.0, /*threshold_bps=*/5.0);
    auto ev = s.on_update(0, 100.0, 100.04);  // 4 bps, below threshold
    EXPECT_FALSE(ev.has_value());
    EXPECT_EQ(s.events_emitted(), 0u);
}

TEST(BasisStrategy, EventOnFirstThresholdCross) {
    BasisStrategy s(/*ewma_alpha=*/1.0, /*threshold_bps=*/5.0);
    auto ev = s.on_update(0, 100.0, 100.10);  // 10 bps
    ASSERT_TRUE(ev.has_value());
    EXPECT_EQ(ev->side, +1);
    EXPECT_NEAR(ev->deviation_bps, 10.0, 1e-9);
    EXPECT_EQ(s.events_emitted(), 1u);
}

TEST(BasisStrategy, NoSecondEventWhileAboveThreshold) {
    BasisStrategy s(/*ewma_alpha=*/1.0, /*threshold_bps=*/5.0);
    s.on_update(0, 100.0, 100.10);                  // event #1
    auto ev2 = s.on_update(1, 100.0, 100.12);       // still above
    EXPECT_FALSE(ev2.has_value());
    EXPECT_EQ(s.events_emitted(), 1u);
}

TEST(BasisStrategy, ReArmAfterReturnInsideBand) {
    BasisStrategy s(/*ewma_alpha=*/1.0, /*threshold_bps=*/5.0);
    s.on_update(0, 100.0, 100.10);     // event #1 (above +5)
    s.on_update(1, 100.0, 100.02);     // back inside
    auto ev2 = s.on_update(2, 100.0, 100.08);  // re-cross
    ASSERT_TRUE(ev2.has_value());
    EXPECT_EQ(s.events_emitted(), 2u);
}

TEST(BasisStrategy, NegativeDeviationFiresWithSideMinusOne) {
    BasisStrategy s(/*ewma_alpha=*/1.0, /*threshold_bps=*/5.0);
    auto ev = s.on_update(0, 100.0, 99.90);  // -10 bps
    ASSERT_TRUE(ev.has_value());
    EXPECT_EQ(ev->side, -1);
    EXPECT_LT(ev->deviation_bps, 0);
}

TEST(BasisStrategy, EwmaSmoothsSpike) {
    // Tight smoothing: alpha = 0.1. A single tick at 50 bps should
    // smooth to 5 bps which is at the boundary, not above it.
    BasisStrategy s(/*ewma_alpha=*/0.1, /*threshold_bps=*/10.0);
    // Seed with zero deviation.
    s.on_update(0, 100.0, 100.0);
    auto ev = s.on_update(1, 100.0, 100.50);  // raw 50 bps; smoothed 5
    EXPECT_FALSE(ev.has_value()) << "smoothed " << s.current_smoothed_bps();
}

TEST(BasisStrategy, RejectsBadConstructorArgs) {
    EXPECT_THROW(BasisStrategy(0.0, 5.0), std::invalid_argument);
    EXPECT_THROW(BasisStrategy(1.5, 5.0), std::invalid_argument);
    EXPECT_THROW(BasisStrategy(0.5, 0.0), std::invalid_argument);
    EXPECT_THROW(BasisStrategy(0.5, -1.0), std::invalid_argument);
}

TEST(BasisStrategy, NonPositiveMidIsSkipped) {
    BasisStrategy s(1.0, 5.0);
    auto ev = s.on_update(0, 0.0, 100.10);
    EXPECT_FALSE(ev.has_value());
    auto ev2 = s.on_update(1, 100.0, -1.0);
    EXPECT_FALSE(ev2.has_value());
    EXPECT_EQ(s.events_emitted(), 0u);
}

}  // namespace
