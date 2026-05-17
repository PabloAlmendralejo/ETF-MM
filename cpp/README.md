# C++ subtree — Mode 3

Live order book + tape capture/replay for spot-vs-perpetual basis arbitrage on Binance. See [`docs/spec/mode3_design.md`](../docs/spec/mode3_design.md).

This is **additive** to the Python Mode 1 simulator under `etf_mm_sim/`. Mode 1 is unchanged; the two halves do not call each other.

## Status

- **Phase A — done**: SPSC ring buffer, order book, binary tape format with reader/writer, GoogleTest unit tests. Builds clean with MSVC 19.x.
- **Phase B — done**: Boost.Beast WS client over TLS (system root certs on Windows), `etfmm_capture` binary connecting to Binance spot + perp streams.
- **Phase C — done**: `etfmm_replay` binary, basis-deviation strategy with EWMA smoothing, CSV event output. 22 unit tests, all green.

## Build

From a Developer Command Prompt for VS (or after sourcing `vcvars64.bat`):

```
$env:VCPKG_ROOT = "C:\vcpkg"
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release `
      -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

The first configure compiles the dependency tree (Boost + OpenSSL + nlohmann::json + GTest) via the vcpkg manifest in `vcpkg.json`. Expect ~30 minutes on a 4-core machine on the first run; subsequent configures are seconds.

## Capture

Stream live BTCUSDT spot + perpetual L2 from Binance into a tape file:

```
build\etfmm_capture.exe data\demo.tape --seconds 300
```

Output is a binary file with a fixed header (`ETFMMV01` magic + epoch ns) and an append-only stream of length-prefixed frames. Tapes are deterministic on replay.

## Replay

Run the basis-deviation strategy on a captured tape, emit events to CSV:

```
build\etfmm_replay.exe data\demo.tape data\events.csv `
      --threshold-bps 5 --ewma-alpha 0.1
```

The strategy maintains an exponentially weighted moving average of `(perp_mid / spot_mid - 1) * 1e4` and emits an event each time the smoothed series crosses the threshold from inside the band to outside. Crossings re-arm only after the smoothed series returns inside the band, so a single sustained dislocation produces one event rather than thousands.

Sample output from a 5-minute capture (~100,000 frames):

```
ns,spot_mid,perp_mid,deviation_bps,side
68516396000,78161.7,78120.3,-5.02238,-1
80560347100,78154.9,78112.6,-5.02743,-1
85105859200,78161.5,78121.1,-5.00786,-1
107338363600,78159.5,78120.4,-5.00041,-1
...
281763162500,78122.6,78083.4,-5.00050,-1
```

24 events over 5 minutes at 5 bp threshold, all on the negative side (perp trading below spot — funding-rate-driven dislocation typical of BTC perpetuals).

## Layout

```
cpp/
├── CMakeLists.txt
├── vcpkg.json                 # boost-beast, boost-asio, openssl, nlohmann-json, gtest
├── include/etfmm/
│   ├── ring.hpp               # SPSC lock-free queue
│   ├── order_book.hpp         # std::map<Price, Level>
│   ├── tape.hpp               # binary capture format
│   ├── ws_client.hpp          # Boost.Beast WS over TLS
│   └── basis_strategy.hpp     # spot-vs-perp deviation events
├── src/
│   ├── order_book.cpp
│   ├── tape.cpp
│   ├── ws_client.cpp          # Windows root certs via wincrypt
│   ├── basis_strategy.cpp     # EWMA + threshold-crossing logic
│   ├── capture_main.cpp       # etfmm_capture
│   └── replay_main.cpp        # etfmm_replay
└── tests/                     # 22 GoogleTest unit tests
```
