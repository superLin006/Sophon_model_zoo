cmake_minimum_required(VERSION 3.10)

set(CMAKE_C_COMPILER   aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

project(hymt_demo CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

add_executable(hymt_demo
    src/main.cpp
    src/QwenEngine.cpp
)

target_compile_definitions(hymt_demo PRIVATE SOC_TARGET)

target_link_libraries(hymt_demo
    ${SOPHON_SDK}/lib/libbmrt.so
    ${SOPHON_SDK}/lib/libbmlib.so
    ${TOKENIZERS_DIR}/aarch64-linux/libtokenizers_cpp.a
    ${TOKENIZERS_DIR}/aarch64-linux/libtokenizers_c.a
    pthread dl
)

set_target_properties(hymt_demo PROPERTIES
    LINK_FLAGS "-Wl,-rpath,/opt/sophon/libsophon-current/lib"
)
