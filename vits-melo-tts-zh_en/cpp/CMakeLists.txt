cmake_minimum_required(VERSION 3.10)
project(vits_melo_tts_bm1684)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_C_COMPILER   aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

if(NOT DEFINED SOPHON_SDK)
    set(SOPHON_SDK ${CMAKE_SOURCE_DIR}/../../0_Toolkits/soc-sdk-sp4)
endif()

if(NOT DEFINED ONNXRUNTIME_DIR)
    set(ONNXRUNTIME_DIR ${CMAKE_SOURCE_DIR}/../../1_third_party/onnxruntime-linux-aarch64)
endif()

include_directories(
    src
    ${SOPHON_SDK}/include
    ${ONNXRUNTIME_DIR}/include
)

set(SOURCES
    src/main.cpp
    src/tts_inference.cpp
)

add_executable(vits_melo_tts_bm1684 ${SOURCES})

target_link_libraries(vits_melo_tts_bm1684
    ${SOPHON_SDK}/lib/libbmrt.so
    ${SOPHON_SDK}/lib/libbmlib.so
    ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so
    m
    dl
    pthread
)

set_target_properties(vits_melo_tts_bm1684 PROPERTIES
    LINK_FLAGS "-Wl,-rpath,/opt/sophon/libsophon-0.5.1/lib:/root/vits-melo-tts-zh_en/lib"
)
