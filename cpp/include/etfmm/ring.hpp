// SPSC lock-free ring buffer.
//
// Single-producer single-consumer bounded queue. The CV's "lock-free
// price-level data structure" claim refers to the producer ↔ consumer
// interface implemented here: the WS thread enqueues raw frames and the
// book/tape thread dequeues them with two atomics and acquire/release
// memory ordering, no mutexes on the hot path.
//
// Capacity is a compile-time power of 2 so wrap is a single bitmask.
// Head and tail counters live on separate cache lines (alignas(64)) to
// avoid false sharing between producer and consumer cores.

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <new>
#include <type_traits>
#include <utility>

namespace etfmm {

namespace detail {
template <std::size_t N>
struct is_power_of_two
    : std::integral_constant<bool, (N > 0) && ((N & (N - 1)) == 0)> {};
}  // namespace detail

// Bounded SPSC ring of capacity Capacity (must be a power of two).
//
// T must be default-constructible (slots are pre-allocated) and
// move-assignable (push moves the value into a slot; pop moves it out).
template <typename T, std::size_t Capacity>
class SpscRing {
    static_assert(detail::is_power_of_two<Capacity>::value,
                  "SpscRing capacity must be a power of two");
    static_assert(std::is_default_constructible<T>::value,
                  "SpscRing<T> requires T to be default-constructible");
    static_assert(std::is_move_assignable<T>::value,
                  "SpscRing<T> requires T to be move-assignable");

   public:
    SpscRing() : head_{0}, tail_{0} {}

    // Non-copyable, non-movable: the atomic counters identify the queue.
    SpscRing(const SpscRing&) = delete;
    SpscRing& operator=(const SpscRing&) = delete;
    SpscRing(SpscRing&&) = delete;
    SpscRing& operator=(SpscRing&&) = delete;

    // Producer side: try to enqueue. Returns false if the queue is full.
    // Move-friendly: caller may pass an rvalue to avoid a copy.
    template <typename U>
    bool try_push(U&& value) {
        const std::uint64_t head = head_.load(std::memory_order_relaxed);
        const std::uint64_t tail = tail_.load(std::memory_order_acquire);
        if (head - tail == Capacity) {
            return false;  // full
        }
        slots_[head & kMask] = std::forward<U>(value);
        // Release so the consumer sees the slot store before the head bump.
        head_.store(head + 1, std::memory_order_release);
        return true;
    }

    // Consumer side: try to dequeue. Returns false if the queue is empty.
    bool try_pop(T& out) {
        const std::uint64_t tail = tail_.load(std::memory_order_relaxed);
        const std::uint64_t head = head_.load(std::memory_order_acquire);
        if (head == tail) {
            return false;  // empty
        }
        out = std::move(slots_[tail & kMask]);
        // Release so the producer sees the slot vacated before the tail bump.
        tail_.store(tail + 1, std::memory_order_release);
        return true;
    }

    // Approximate, lock-free observers. Useful for telemetry; not a
    // synchronization primitive.
    std::size_t approx_size() const noexcept {
        const std::uint64_t head = head_.load(std::memory_order_acquire);
        const std::uint64_t tail = tail_.load(std::memory_order_acquire);
        return static_cast<std::size_t>(head - tail);
    }

    constexpr std::size_t capacity() const noexcept { return Capacity; }

   private:
    static constexpr std::uint64_t kMask = Capacity - 1;

    // Pad head and tail onto separate cache lines.
    alignas(64) std::atomic<std::uint64_t> head_;
    alignas(64) std::atomic<std::uint64_t> tail_;
    alignas(64) T slots_[Capacity]{};
};

}  // namespace etfmm
