cmake_minimum_required(VERSION 3.10)
project(sensevoice_bm1684)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_C_COMPILER   aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

if(NOT DEFINED SOPHON_SDK)
    set(SOPHON_SDK ${CMAKE_SOURCE_DIR}/../../0_Toolkits/soc-sdk-sp4)
endif()
if(NOT DEFINED KALDI_FBANK_DIR)
    set(KALDI_FBANK_DIR ${CMAKE_SOURCE_DIR}/../../1_third_party/kaldi_native_fbank)
endif()

include_directories(
    src
    ${SOPHON_SDK}/include
    ${KALDI_FBANK_DIR}/include
)

set(SOURCES
    src/main.cpp
    src/sensevoice_inference.cpp
    src/audio_frontend.cpp
    src/tokenizer.cpp
)

add_executable(sensevoice_bm1684 ${SOURCES})

target_link_libraries(sensevoice_bm1684
    ${SOPHON_SDK}/lib/libbmrt.so
    ${SOPHON_SDK}/lib/libbmlib.so
    ${KALDI_FBANK_DIR}/aarch64-linux/libkaldi-native-fbank-core.a
    ${KALDI_FBANK_DIR}/aarch64-linux/libkissfft-float.a
    m
    dl
    pthread
)

set_target_properties(sensevoice_bm1684 PROPERTIES
    LINK_FLAGS "-Wl,-rpath,/opt/sophon/libsophon-0.5.1/lib"
)
