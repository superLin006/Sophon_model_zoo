cmake_minimum_required(VERSION 3.10)
project(qwen_demo CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_C_COMPILER   aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

if(NOT DEFINED SOPHON_SDK)
    set(SOPHON_SDK ${CMAKE_SOURCE_DIR}/../../0_Toolkits/soc-sdk-sp4)
endif()

if(NOT DEFINED TOKENIZERS_DIR)
    set(TOKENIZERS_DIR ${CMAKE_SOURCE_DIR}/../../1_third_party/tokenizers-cpp)
endif()

include_directories(
    src
    ${SOPHON_SDK}/include
    ${TOKENIZERS_DIR}/include
)

add_executable(qwen_demo
    src/main.cpp
    src/QwenEngine.cpp
)

target_compile_definitions(qwen_demo PRIVATE SOC_TARGET)

target_link_libraries(qwen_demo
    ${SOPHON_SDK}/lib/libbmrt.so
    ${SOPHON_SDK}/lib/libbmlib.so
    ${TOKENIZERS_DIR}/aarch64-linux/libtokenizers_cpp.a
    ${TOKENIZERS_DIR}/aarch64-linux/libtokenizers_c.a
    pthread dl
)

set_target_properties(qwen_demo PROPERTIES
    LINK_FLAGS "-Wl,-rpath,/opt/sophon/libsophon-current/lib"
)
