cmake_minimum_required(VERSION 3.10)
project(moonshine_bm1684)

set(CMAKE_CXX_STANDARD 14)

# 交叉编译工具链（WSL → aarch64 Linux）
set(CMAKE_C_COMPILER   aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

# SDK 路径（可由 -DSOPHON_SDK=... 覆盖）
if(NOT DEFINED SOPHON_SDK)
    set(SOPHON_SDK  ${CMAKE_SOURCE_DIR}/../../0_Toolkits/soc-sdk-sp4)
endif()

# 头文件
include_directories(
    src
    src/utils
    ${SOPHON_SDK}/include
)

# 源文件（无 fftw3 依赖，moonshine 无 STFT）
set(SOURCES
    src/main.cpp
    src/moonshine_inference.cpp
    src/utils/wav_reader.cpp
)

add_executable(moonshine_bm1684 ${SOURCES})

# 链接库
target_link_libraries(moonshine_bm1684
    ${SOPHON_SDK}/lib/libbmrt.so
    ${SOPHON_SDK}/lib/libbmlib.so
    m
    dl
    pthread
)

# rpath：板卡上从 /opt/sophon/libsophon-0.5.1/lib 找 so
set_target_properties(moonshine_bm1684 PROPERTIES
    LINK_FLAGS "-Wl,-rpath,/opt/sophon/libsophon-0.5.1/lib"
)
