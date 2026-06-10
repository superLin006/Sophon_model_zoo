cmake_minimum_required(VERSION 3.10)
project(qwen_demo CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_C_COMPILER   aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

# Sophon SOC SDK（头文件 + 库）
# 从板卡 /opt/sophon/libsophon-current/ 或 sophgo 官网获取后放入 0_Toolkits/soc-sdk-sp4/
if(NOT DEFINED SOPHON_SDK)
    set(SOPHON_SDK ${CMAKE_SOURCE_DIR}/../../0_Toolkits/soc-sdk-sp4)
endif()

# tokenizers-cpp（HuggingFace tokenizer.json 解析）
# 获取方式: git clone https://github.com/mlc-ai/tokenizers-cpp 1_third_party/tokenizers-cpp
if(NOT DEFINED TOKENIZERS_CPP_DIR)
    set(TOKENIZERS_CPP_DIR ${CMAKE_SOURCE_DIR}/../../1_third_party/tokenizers-cpp)
endif()

add_subdirectory(${TOKENIZERS_CPP_DIR} tokenizers-cpp EXCLUDE_FROM_ALL)

include_directories(
    src
    ${SOPHON_SDK}/include
    ${TOKENIZERS_CPP_DIR}/include
)

add_executable(qwen_demo
    src/main.cpp
    src/QwenEngine.cpp
)

target_compile_definitions(qwen_demo PRIVATE SOC_TARGET)

target_link_libraries(qwen_demo
    ${SOPHON_SDK}/lib/libbmrt.so
    ${SOPHON_SDK}/lib/libbmlib.so
    tokenizers_cpp
    pthread
)

set_target_properties(qwen_demo PROPERTIES
    LINK_FLAGS "-Wl,-rpath,/opt/sophon/libsophon-current/lib"
)
