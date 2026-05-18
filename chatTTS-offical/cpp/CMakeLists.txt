cmake_minimum_required(VERSION 3.16)
project(chattts_cpp CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Cross-compile target ──────────────────────────────────────────────────────
# When building inside Docker with aarch64-linux-gnu toolchain:
#   cmake -DCMAKE_TOOLCHAIN_FILE=../cmake/aarch64.cmake ..
# When building natively on the board:
#   cmake ..

# ── Third-party root (relative to this CMakeLists.txt) ───────────────────────
# Can be overridden via -DTHIRD_PARTY_DIR=/path at cmake invocation
if(NOT DEFINED THIRD_PARTY_DIR)
    set(THIRD_PARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../1_third_party")
endif()

# ── Sophon SDK ────────────────────────────────────────────────────────────────
set(SOPHON_INC "${THIRD_PARTY_DIR}/sophon/include")
set(SOPHON_LIB "${THIRD_PARTY_DIR}/sophon/lib")

# ── FFTW3 (aarch64 static) ───────────────────────────────────────────────────
set(FFTW_INC "${THIRD_PARTY_DIR}/fftw/include")
set(FFTW_LIB "${THIRD_PARTY_DIR}/fftw/aarch64-linux/libfftw3f.a")

# ── nlohmann/json (header only) ───────────────────────────────────────────────
# json.hpp lives at nlohmann/json.hpp, so include the parent dir
set(NLOHMANN_INC "${THIRD_PARTY_DIR}")

# ── Sources ───────────────────────────────────────────────────────────────────
set(CORE_SOURCES
    src/normalizer.cpp
    src/tokenizer.cpp
    src/istft.cpp
    src/decoder_engine.cpp
    src/vocos_engine.cpp
    src/gpt_engine.cpp
    src/chattts.cpp
)

add_executable(chattts ${CORE_SOURCES} src/main.cpp)
add_executable(chattts_bench ${CORE_SOURCES} src/benchmark.cpp)

foreach(TARGET chattts chattts_bench)
    target_include_directories(${TARGET} PRIVATE
        include
        ${SOPHON_INC}
        ${FFTW_INC}
        ${NLOHMANN_INC}
    )
    target_link_libraries(${TARGET} PRIVATE
        ${SOPHON_LIB}/libbmrt.so
        ${SOPHON_LIB}/libbmlib.so.0
        ${FFTW_LIB}
        pthread
        m
    )
    target_compile_options(${TARGET} PRIVATE
        -O3
        -ffast-math
        -march=armv8-a
    )
    set_target_properties(${TARGET} PROPERTIES
        INSTALL_RPATH "/opt/sophon/libsophon-current/lib"
        BUILD_WITH_INSTALL_RPATH ON
    )
endforeach()
