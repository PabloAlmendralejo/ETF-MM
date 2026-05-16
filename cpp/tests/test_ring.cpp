// SPSC ring buffer unit tests.
//
// Coverage targets:
//   - basic push/pop ordering
//   - full / empty boundaries (bitmask wrap)
//   - move-only payloads (we use std::vector<int>)
//   - two-thread producer/consumer stress for FIFO ordering and
//     no-loss / no-duplicate

#include "etfmm/ring.hpp"

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

namespace {

using etfmm::SpscRing;

TEST(SpscRing, EmptyAndFullBoundaries) {
    SpscRing<int, 4> q;
    EXPECT_EQ(q.capacity(), 4u);
    int dummy = 0;
    EXPECT_FALSE(q.try_pop(dummy));      // empty
    EXPECT_EQ(q.approx_size(), 0u);

    EXPECT_TRUE(q.try_push(1));
    EXPECT_TRUE(q.try_push(2));
    EXPECT_TRUE(q.try_push(3));
    EXPECT_TRUE(q.try_push(4));
    EXPECT_FALSE(q.try_push(5));         // full
    EXPECT_EQ(q.approx_size(), 4u);

    int v = 0;
    ASSERT_TRUE(q.try_pop(v));
    EXPECT_EQ(v, 1);
    ASSERT_TRUE(q.try_pop(v));
    EXPECT_EQ(v, 2);
    EXPECT_TRUE(q.try_push(5));          // wraps over old slot
    EXPECT_TRUE(q.try_push(6));
    EXPECT_FALSE(q.try_push(7));         // full again

    ASSERT_TRUE(q.try_pop(v)); EXPECT_EQ(v, 3);
    ASSERT_TRUE(q.try_pop(v)); EXPECT_EQ(v, 4);
    ASSERT_TRUE(q.try_pop(v)); EXPECT_EQ(v, 5);
    ASSERT_TRUE(q.try_pop(v)); EXPECT_EQ(v, 6);
    EXPECT_FALSE(q.try_pop(v));
}

TEST(SpscRing, MoveOnlyPayload) {
    using V = std::vector<int>;
    SpscRing<V, 4> q;

    V a{1, 2, 3};
    ASSERT_TRUE(q.try_push(std::move(a)));
    EXPECT_TRUE(a.empty());  // moved-from

    V b{4, 5};
    ASSERT_TRUE(q.try_push(std::move(b)));

    V out;
    ASSERT_TRUE(q.try_pop(out));
    EXPECT_EQ(out, (V{1, 2, 3}));
    ASSERT_TRUE(q.try_pop(out));
    EXPECT_EQ(out, (V{4, 5}));
    EXPECT_FALSE(q.try_pop(out));
}

TEST(SpscRing, ProducerConsumerStress) {
    constexpr std::size_t kCap = 1024;
    constexpr int kN = 200000;

    SpscRing<int, kCap> q;
    std::vector<int> received;
    received.reserve(kN);
    std::atomic<bool> producer_done{false};

    std::thread consumer([&]() {
        int v = 0;
        while (received.size() < static_cast<std::size_t>(kN)) {
            if (q.try_pop(v)) {
                received.push_back(v);
            } else if (producer_done.load(std::memory_order_acquire) &&
                       q.approx_size() == 0) {
                // producer is done and queue is drained
                continue;
            } else {
                std::this_thread::yield();
            }
        }
    });

    std::thread producer([&]() {
        for (int i = 0; i < kN; ++i) {
            while (!q.try_push(i)) {
                std::this_thread::yield();
            }
        }
        producer_done.store(true, std::memory_order_release);
    });

    producer.join();
    consumer.join();

    ASSERT_EQ(static_cast<int>(received.size()), kN);
    for (int i = 0; i < kN; ++i) {
        ASSERT_EQ(received[i], i) << "FIFO violation at index " << i;
    }
}

}  // namespace
