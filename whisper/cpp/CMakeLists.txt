cmake_minimum_required(VERSION 3.10)
project(whisper_bm1684)

set(CMAKE_CXX_STANDARD 14)

# 交叉编译工具链（WSL → aarch64 Linux）
set(CMAKE_C_COMPILER   aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

# SDK 路径（可由 -DSOPHON_SDK=... -DFFTW_DIR=... 覆盖）
if(NOT DEFINED SOPHON_SDK)
    set(SOPHON_SDK  ${CMAKE_SOURCE_DIR}/../../0_Toolkits/soc-sdk-sp4)
endif()
if(NOT DEFINED FFTW_DIR)
    set(FFTW_DIR    ${CMAKE_SOURCE_DIR}/../../1_third_party/fftw)
endif()

# 头文件
include_directories(
    src
    src/utils
    ${SOPHON_SDK}/include
    ${FFTW_DIR}/include
)

# 源文件
set(SOURCES
    src/main.cpp
    src/whisper_inference.cpp
    src/utils/audio_utils.cpp
)

add_executable(whisper_bm1684 ${SOURCES})

# 链接库
target_link_libraries(whisper_bm1684
    ${SOPHON_SDK}/lib/libbmrt.so
    ${SOPHON_SDK}/lib/libbmlib.so
    ${FFTW_DIR}/aarch64-linux/libfftw3f.a
    m
    dl
    pthread
)

# rpath：部署到服务器后从同目录找 so
set_target_properties(whisper_bm1684 PROPERTIES
    LINK_FLAGS "-Wl,-rpath,/opt/sophon/libsophon-0.5.1/lib"
)
