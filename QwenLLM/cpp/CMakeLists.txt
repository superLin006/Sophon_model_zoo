cmake_minimum_required(VERSION 3.19)
project(qwen_demo CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================
# 路径配置（相对于本 CMakeLists.txt 所在目录）
# ============================================================

# Sophon SDK（libsophon）—— 优先本地 3rdparty，回退系统路径
set(_SOPHON_3RDPARTY "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/bm1684x/libsophon")
if(EXISTS "${_SOPHON_3RDPARTY}/include/bmruntime_interface.h")
    set(SOPHON_SDK_DIR "${_SOPHON_3RDPARTY}")
    message(STATUS "[qwen_demo] Using bundled libsophon: ${SOPHON_SDK_DIR}")
else()
    set(SOPHON_SDK_DIR "/opt/sophon/libsophon-current"
        CACHE PATH "Path to Sophon SDK (libsophon)")
    message(STATUS "[qwen_demo] Using system libsophon: ${SOPHON_SDK_DIR}")
endif()

set(_SOPHON_INC "${SOPHON_SDK_DIR}/include")
set(_SOPHON_LIB "${SOPHON_SDK_DIR}/aarch64")

# tokenizers-cpp 源码目录（不含 Rust 产物，通过预编译 .a 绕过 cargo）
# 默认指向同仓库内的 llm-sdk 3rdparty，也可通过 cmake -DTOKENIZERS_CPP_DIR=... 覆盖
set(TOKENIZERS_CPP_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../../llm-sdk/3rdparty/common/tokenizers-cpp"
    CACHE PATH "Path to tokenizers-cpp source directory")

set(_TOKENIZERS_PREBUILT
    "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/bm1684x/prebuilt/libtokenizers_c.a"
    CACHE PATH "Path to prebuilt libtokenizers_c.a for aarch64")

# ============================================================
# tokenizers-cpp：绕过 cargo，直接使用预编译的 libtokenizers_c.a
# ============================================================
set(_CARGO_OUT_DIR
    "${TOKENIZERS_CPP_DIR}/rust/target/aarch64-unknown-linux-gnu/release")
set(_CARGO_OUT_LIB "${_CARGO_OUT_DIR}/libtokenizers_c.a")

if(EXISTS "${_TOKENIZERS_PREBUILT}" AND NOT EXISTS "${_CARGO_OUT_LIB}")
    message(STATUS "[qwen_demo] Copying prebuilt libtokenizers_c.a → cargo output path")
    file(MAKE_DIRECTORY "${_CARGO_OUT_DIR}")
    configure_file("${_TOKENIZERS_PREBUILT}" "${_CARGO_OUT_LIB}" COPYONLY)
endif()

add_subdirectory("${TOKENIZERS_CPP_DIR}" tokenizers-cpp EXCLUDE_FROM_ALL)

# ============================================================
# 可执行文件
# ============================================================
add_executable(qwen_demo
    src/main.cpp
    src/QwenEngine.cpp
)

target_include_directories(qwen_demo PRIVATE
    src
    "${_SOPHON_INC}"
    "${TOKENIZERS_CPP_DIR}/include"
)

# 链接 libsophon
add_library(sophon_sdk_libs INTERFACE)
target_link_directories(sophon_sdk_libs INTERFACE "${_SOPHON_LIB}")
target_link_libraries(sophon_sdk_libs INTERFACE bmrt bmlib)

target_link_libraries(qwen_demo PRIVATE
    sophon_sdk_libs
    tokenizers_cpp
)

# SoC 模式宏（板上直接运行时需要）
target_compile_definitions(qwen_demo PRIVATE SOC_TARGET)

# rpath：使二进制能在板上就地找到 .so（lib/ 子目录）
set_target_properties(qwen_demo PROPERTIES
    INSTALL_RPATH "$ORIGIN/lib"
    BUILD_WITH_INSTALL_RPATH TRUE
)

# ============================================================
# 安装：把 .so 一并打包方便 scp 到板卡
# ============================================================
install(TARGETS qwen_demo DESTINATION .)
install(FILES
    "${_SOPHON_LIB}/libbmrt.so"
    "${_SOPHON_LIB}/libbmrt.so.1.0"
    "${_SOPHON_LIB}/libbmlib.so"
    "${_SOPHON_LIB}/libbmlib.so.0"
    DESTINATION lib
)
