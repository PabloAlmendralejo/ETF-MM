# C++ subtree — Mode 3

Live order book + tape capture/replay for spot-vs-perpetual basis arbitrage on Binance. See [`docs/spec/mode3_design.md`](../docs/spec/mode3_design.md).

This is **additive** to the Python Mode 1 simulator under `etf_mm_sim/`. Mode 1 is unchanged; the two halves do not call each other.

## Status

- **Phase A — done**: SPSC ring buffer, order book, binary tape format with reader/writer, GoogleTest unit tests. Builds clean with MSVC 19.x.
- **Phase B — pending**: Boost.Beast WS client + capture binary.
- **Phase C — pending**: replay binary + basis-deviation strategy.

## Build (Phase A)

From a Developer Command Prompt for VS (or after sourcing `vcvars64.bat`):

```
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

GoogleTest is fetched automatically via `FetchContent` if not provided
by vcpkg.

## Layout

```
cpp/
├── CMakeLists.txt
├── include/etfmm/
│   ├── ring.hpp          # SPSC lock-free queue (the "lock-free" claim)
│   ├── order_book.hpp    # std::map<Price, Level>
│   └── tape.hpp          # binary capture format
├── src/
│   ├── order_book.cpp
│   └── tape.cpp
└── tests/
    ├── test_ring.cpp
    ├── test_order_book.cpp
    └── test_tape.cpp
```
