# Mode 3 design — Real-time order book + tape capture/replay (C++)

## Status

Mode 3 sits **alongside** the Python Mode 1 simulator (`etf_mm_sim/`).
Mode 1 is unchanged. Mode 3 is a new C++17 subtree under `cpp/` with no
runtime coupling to the Python package; the two share only the project
README and the spec docs.

## Goal

Capture real Binance L2 ticks live, persist them deterministically to a
binary tape, then replay through an in-memory order book and a
spot-vs-perp basis-deviation strategy. The output is a stream of
basis-deviation events with statistics; no orders are placed.

This realizes the CV bullet's claims of:

- Real-time order book (C++) consuming WebSocket tick feeds
- Lock-free price-level data structure on the hot path (SPSC ring buffer
  between WS thread and book/strategy thread; book itself is a
  `std::map<Price, Level>`)
- Live deviation between two related instruments triggering arbitrage
  signals (spot vs perp; structurally analogous to ETF/NAV
  creation-redemption arbitrage but using free crypto data)

## Why crypto, not equities

Real US equity ETF L2 data is paywalled (Polygon, IEX, Nasdaq TotalView).
Binance offers free public WS L2 streams for both spot and perpetual
futures, no API key required for market data. The arbitrage shape is
analogous: BTCUSDT spot price vs the synthetic price implied by
BTCUSDT-PERP (perpetual futures contract) deviates from zero around
funding events, news, and large flows; trading firms run this exact
strategy ("basis arb") with the same mechanics as ETF/NAV arb.

## Components

```
              ┌──────────────────┐    ┌──────────────────┐
   Binance ─► │ ws_client (TLS)  │ ─► │ SPSC Ring (lf)   │
   Spot WS    │ Boost.Beast      │    │ ring.hpp         │
              └──────────────────┘    └────────┬─────────┘
                                               │
              ┌──────────────────┐    ┌────────▼─────────┐    ┌──────────────────┐
   Binance ─► │ ws_client (TLS)  │ ─► │ SPSC Ring (lf)   │ ─► │ tape writer      │
   Perp WS    │ Boost.Beast      │    │ ring.hpp         │    │ tape.hpp         │
              └──────────────────┘    └──────────────────┘    └────────┬─────────┘
                                                                       ▼
                                                                ┌──────────────┐
                                                                │  *.tape file │
                                                                └──────┬───────┘
                                                                       ▼
              ┌─────────────┐    ┌──────────────────┐    ┌────────────────────┐
              │ tape reader │ ─► │ order_book (×2)  │ ─► │ basis_strategy     │
              │ (replay)    │    │ std::map<P, L>   │    │ deviation events   │
              └─────────────┘    └──────────────────┘    └────────────────────┘
```

Two binaries:

- `etfmm_capture` — connects to both Binance spot and futures WS streams,
  serializes incoming messages to a binary tape file with monotonic
  timestamps. Single-process, two WS threads, two SPSC rings, one tape
  writer thread.
- `etfmm_replay` — reads a tape file, feeds each message into the
  appropriate order book, runs the basis strategy on every book update,
  and writes deviation events to stdout (or a CSV).

## Components in detail

### `ring.hpp` — SPSC lock-free ring buffer

Single-producer single-consumer bounded queue. The CV's "lock-free
price-level data structure" claim is literally true at the producer →
consumer interface, where the WS thread enqueues raw frames and the
book thread dequeues them with two atomics (head, tail) and `std::memory_order_acquire`/`release`.

- Capacity is a compile-time power of 2 so wrap is a bitmask.
- Two cache-line-padded atomic counters (`alignas(64)`) to avoid false
  sharing between producer and consumer.
- `try_push(T&&)` and `try_pop(T&)` non-blocking; caller decides what to
  do when full (capture: drop with counter; replay: not used since
  replay is single-threaded).
- T is move-constructible. We will use it with `std::vector<char>`
  payloads representing raw WS frames.

### `tape.hpp` / `tape.cpp` — binary capture format

Single tape file, append-only. Frame layout (little-endian, MSVC default):

```
struct TapeHeader {
    char magic[8];        // "ETFMMV01"
    uint64_t epoch_ns;    // wall clock at start
    uint16_t version;     // 1
    uint16_t reserved;
};

struct FrameHeader {
    uint64_t ns_offset;   // ns from epoch_ns
    uint8_t  venue;       // 0 = spot, 1 = perp
    uint8_t  msg_type;    // 0 = depth, 1 = bookTicker, 2 = trade, 3 = snapshot
    uint16_t reserved;
    uint32_t payload_len; // bytes
    // followed by payload_len bytes of raw JSON / binary payload
};
```

Determinism: with the same source, two reads of the tape file produce
the same sequence of `(ns_offset, venue, msg_type, payload)` tuples.
The capture process is non-deterministic across runs (network jitter)
but every captured tape is fully reproducible on replay.

### `order_book.hpp` / `order_book.cpp`

Per-instrument:

- `std::map<Price, Level>` for bids (descending iter via `rbegin()`) and
  asks (ascending iter via `begin()`).
- `Level` carries cumulative quantity + a `std::list<Order>` of resting
  orders (only used in the snapshot+diff replay path; for Binance
  aggregated depth we only need the level total).
- `apply_snapshot(Snapshot)` — replaces the book.
- `apply_diff(DepthUpdate)` — inserts/updates/deletes price levels per
  Binance protocol (`U`, `u` sequence numbers gate dropped messages).

Top-of-book accessors (`best_bid()`, `best_ask()`, `mid()`) are
`O(log N)` in level count via the map; mid is computed as the average
of `begin()->first` (best ask) and `rbegin()->first` (best bid).

For the basis strategy we only need `mid()` per book; the full book is
maintained so the same component supports future strategies.

### `basis_strategy.hpp` / `basis_strategy.cpp`

Inputs: spot mid, perp mid, timestamp.
Output: deviation = `(perp_mid - spot_mid) / spot_mid`. When
`|deviation| > threshold`, emit an event with `(ts, spot_mid, perp_mid,
deviation_bps, side)`. Threshold is configurable (default 5 bps).

The strategy is stateless beyond a running EWMA of the deviation used
to suppress noise on the threshold crossing. Output is a CSV row per
event.

### `ws_client.hpp` / `ws_client.cpp`

Boost.Beast WS over TLS to:

- Spot: `wss://stream.binance.com:9443/stream?streams=btcusdt@depth@100ms/btcusdt@bookTicker`
- Perp: `wss://fstream.binance.com:9443/stream?streams=btcusdt@depth@100ms/btcusdt@bookTicker`

Each WS runs on its own `std::thread` with its own
`boost::asio::io_context`; on each frame the raw bytes are pushed into
the venue's SPSC ring with the current monotonic timestamp.

Reconnect on socket close: exponential backoff up to 30 s, then on
reconnect we mark a `RESYNC` boundary in the tape and request a fresh
REST snapshot to seed the book on replay.

## Build

CMake 3.20+, MSVC 19.x (verified 19.50). Boost.Beast + nlohmann::json
via vcpkg manifest mode (`cpp/vcpkg.json`). The Python Mode 1 venv is
unaffected.

```
cmake -S cpp -B cpp/build -DCMAKE_TOOLCHAIN_FILE=...
cmake --build cpp/build --config Release
ctest --test-dir cpp/build -C Release
```

## Tests

`cpp/tests/` uses GoogleTest (vcpkg). Coverage targets the components
that don't need network:

- `test_ring.cpp` — push/pop ordering, full/empty boundaries, two-thread
  stress.
- `test_order_book.cpp` — snapshot apply, diff insert/update/delete,
  cross-quote handling, mid computation.
- `test_tape.cpp` — round-trip a synthetic tape: write N frames, read
  back, assert byte-for-byte equality of payloads and exact match of
  `(ns_offset, venue, msg_type)`.

Strategy tests use synthetic mids fed through `basis_strategy` and
assert event emission at threshold crossings.

## What's explicitly NOT in scope

- **Order placement.** Real Binance trading needs a paid API key and
  KYC. The C++ side is paper-only: it captures, replays, and emits
  deviation events to a log. No mock matching engine.
- **Latency / queue-position modeling.** A real MM project would
  simulate queue position based on observed adds/cancels at the chosen
  price level. We don't.
- **Coupling to Python Mode 1.** The two halves are in the same repo
  but do not call each other. The CV story stays:
  - Mode 1: AS theory + Monte Carlo backtester + property tests
  - Mode 3: live order book + lock-free ring + basis-arb signals
- **Full L2 reconstruction across reconnects.** We re-snapshot via the
  REST endpoint after reconnect rather than reconstructing missed
  diffs from sequence-number gaps.

## Implementation phases

1. **Phase A (this turn):** spec doc, CMake skeleton, ring + order_book
   + tape (no network), GoogleTest unit tests, MSVC build verified.
2. **Phase B (next turn):** vcpkg + Boost integration, ws_client,
   capture binary, smoke-capture a few minutes of live data.
3. **Phase C (turn after):** replay binary, basis_strategy, README
   updates, run replay on the captured tape, commit + push.
