cmake_minimum_required(VERSION 3.20)

# Pick up the vcpkg toolchain automatically if VCPKG_ROOT is set in the
# environment. This MUST run before `project(...)` because the vcpkg
# toolchain installs the manifest-listed dependencies during the first
# configure, and that hook needs to fire before any `find_package` is
# evaluated. Without this, `find_package(Boost ...)` below errors out
# even though VCPKG_ROOT is set.
if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING "vcpkg toolchain file")
endif()

project(etf_mm_sim_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(/W4 /permissive- /Zc:__cplusplus)
    # Suppress noise from Boost / vcpkg headers without losing real warnings on our code.
    add_compile_options(/wd4324)  # structure was padded due to alignment specifier (we want this)
    # MSVC's "_s" variants of stdio aren't portable; stick with std::fopen.
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
    # Boost.Asio / Beast require an explicit Windows version target. 0x0A00 = Windows 10.
    add_compile_definitions(_WIN32_WINNT=0x0A00)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# -------------------------------------------------------------------- #
# Phase A library: the non-network components (ring, order book, tape).
# Network components arrive in Phase B / C.
# -------------------------------------------------------------------- #

add_library(etfmm_core STATIC
    src/order_book.cpp
    src/tape.cpp
    src/ws_client.cpp
    src/basis_strategy.cpp
)
target_include_directories(etfmm_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Boost.Beast for WebSocket + TLS, OpenSSL for the TLS backend.
find_package(Boost REQUIRED CONFIG COMPONENTS asio beast)
find_package(OpenSSL REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
target_link_libraries(etfmm_core PUBLIC Boost::asio Boost::beast OpenSSL::SSL OpenSSL::Crypto nlohmann_json::nlohmann_json)
if(WIN32)
    # Boost.Asio on Windows pulls in Winsock2; ensure we link it explicitly.
    target_link_libraries(etfmm_core PUBLIC ws2_32 mswsock)
endif()

# Capture binary.
add_executable(etfmm_capture src/capture_main.cpp)
target_link_libraries(etfmm_capture PRIVATE etfmm_core)

# Replay binary.
add_executable(etfmm_replay src/replay_main.cpp)
target_link_libraries(etfmm_replay PRIVATE etfmm_core)

# -------------------------------------------------------------------- #
# Tests (GoogleTest).                                                  #
# -------------------------------------------------------------------- #

option(ETFMM_BUILD_TESTS "Build C++ unit tests" ON)

if(ETFMM_BUILD_TESTS)
    enable_testing()

    # Prefer a vcpkg-installed GTest if available, otherwise fetch it.
    find_package(GTest CONFIG QUIET)
    if(NOT GTest_FOUND)
        include(FetchContent)
        FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG        release-1.12.1
        )
        # On Windows, prevent overriding the parent project's compiler/linker settings.
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
    endif()

    add_executable(etfmm_tests
        tests/test_ring.cpp
        tests/test_order_book.cpp
        tests/test_tape.cpp
        tests/test_basis_strategy.cpp
    )
    target_link_libraries(etfmm_tests PRIVATE etfmm_core)
    if(TARGET GTest::gtest_main)
        target_link_libraries(etfmm_tests PRIVATE GTest::gtest_main)
    else()
        target_link_libraries(etfmm_tests PRIVATE gtest_main)
    endif()
    add_test(NAME etfmm_tests COMMAND etfmm_tests)
endif()
