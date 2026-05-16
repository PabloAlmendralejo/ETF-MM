// OrderBook unit tests. Coverage:
//   - empty book observers return nullopt
//   - snapshot apply replaces book contents
//   - bid/ask diff: insert, update, delete
//   - bid descending / ask ascending ordering
//   - mid = 0.5 * (best_bid + best_ask)

#include "etfmm/order_book.hpp"

#include <gtest/gtest.h>

namespace {

using etfmm::OrderBook;
using etfmm::Price;
using etfmm::Qty;

TEST(OrderBook, EmptyBookHasNoTopOfBook) {
    OrderBook ob;
    EXPECT_FALSE(ob.best_bid().has_value());
    EXPECT_FALSE(ob.best_ask().has_value());
    EXPECT_FALSE(ob.mid().has_value());
    EXPECT_EQ(ob.bid_levels(), 0u);
    EXPECT_EQ(ob.ask_levels(), 0u);
}

TEST(OrderBook, SnapshotApplies) {
    OrderBook ob;
    std::vector<std::pair<Price, Qty>> bids = {{99.0, 1.0}, {98.0, 2.0}, {100.0, 3.0}};
    std::vector<std::pair<Price, Qty>> asks = {{101.0, 1.0}, {102.0, 2.0}};
    ob.apply_snapshot(bids, asks);

    ASSERT_TRUE(ob.best_bid().has_value());
    ASSERT_TRUE(ob.best_ask().has_value());
    EXPECT_DOUBLE_EQ(*ob.best_bid(), 100.0);
    EXPECT_DOUBLE_EQ(*ob.best_ask(), 101.0);
    ASSERT_TRUE(ob.mid().has_value());
    EXPECT_DOUBLE_EQ(*ob.mid(), 100.5);
    EXPECT_EQ(ob.bid_levels(), 3u);
    EXPECT_EQ(ob.ask_levels(), 2u);
}

TEST(OrderBook, SnapshotSkipsZeroQty) {
    OrderBook ob;
    ob.apply_snapshot({{99.0, 0.0}, {98.0, 2.0}}, {{101.0, 1.0}, {102.0, 0.0}});
    EXPECT_EQ(ob.bid_levels(), 1u);
    EXPECT_EQ(ob.ask_levels(), 1u);
    EXPECT_DOUBLE_EQ(*ob.best_bid(), 98.0);
    EXPECT_DOUBLE_EQ(*ob.best_ask(), 101.0);
}

TEST(OrderBook, BidDiffInsertUpdateDelete) {
    OrderBook ob;
    ob.apply_snapshot({{100.0, 1.0}}, {{101.0, 1.0}});

    // Insert a new level above the existing best bid.
    ob.apply_bid_diff({{100.5, 5.0}});
    EXPECT_DOUBLE_EQ(*ob.best_bid(), 100.5);

    // Update existing level (replaces qty).
    ob.apply_bid_diff({{100.5, 2.0}});
    EXPECT_DOUBLE_EQ(*ob.best_bid(), 100.5);

    // Delete (qty == 0). Best bid falls back to 100.0.
    ob.apply_bid_diff({{100.5, 0.0}});
    EXPECT_DOUBLE_EQ(*ob.best_bid(), 100.0);
}

TEST(OrderBook, AskDiffInsertUpdateDelete) {
    OrderBook ob;
    ob.apply_snapshot({{100.0, 1.0}}, {{101.0, 1.0}});

    ob.apply_ask_diff({{100.7, 3.0}});
    EXPECT_DOUBLE_EQ(*ob.best_ask(), 100.7);

    ob.apply_ask_diff({{100.7, 4.0}});
    EXPECT_DOUBLE_EQ(*ob.best_ask(), 100.7);

    ob.apply_ask_diff({{100.7, 0.0}});
    EXPECT_DOUBLE_EQ(*ob.best_ask(), 101.0);
}

TEST(OrderBook, MidIsAverageOfTopOfBook) {
    OrderBook ob;
    ob.apply_snapshot({{99.5, 1.0}}, {{100.5, 1.0}});
    ASSERT_TRUE(ob.mid().has_value());
    EXPECT_DOUBLE_EQ(*ob.mid(), 100.0);
}

TEST(OrderBook, ClearEmpties) {
    OrderBook ob;
    ob.apply_snapshot({{99.0, 1.0}}, {{101.0, 1.0}});
    ob.clear();
    EXPECT_EQ(ob.bid_levels(), 0u);
    EXPECT_EQ(ob.ask_levels(), 0u);
    EXPECT_FALSE(ob.mid().has_value());
}

}  // namespace
