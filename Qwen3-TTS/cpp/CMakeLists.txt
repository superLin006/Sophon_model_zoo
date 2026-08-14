cmake_minimum_required(VERSION 3.10)
project(qwen3_tts_bm1684x)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_COMPILER   aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

if(NOT DEFINED SOPHON_SDK)
    set(SOPHON_SDK ${CMAKE_SOURCE_DIR}/../../0_Toolkits/soc-sdk-sp4)
endif()
if(NOT DEFINED TOKENIZERS_DIR)
    set(TOKENIZERS_DIR ${CMAKE_SOURCE_DIR}/../../1_third_party/tokenizers-cpp)
endif()

include_directories(src ${SOPHON_SDK}/include ${TOKENIZERS_DIR}/include)

set(SOURCES src/main.cpp src/tts_engine.cpp src/tokenizer.cpp)
add_executable(qwen3_tts_bm1684x ${SOURCES})

target_link_libraries(qwen3_tts_bm1684x
    ${SOPHON_SDK}/lib/libbmrt.so
    ${SOPHON_SDK}/lib/libbmlib.so
    ${TOKENIZERS_DIR}/aarch64-linux/libtokenizers_cpp.a
    ${TOKENIZERS_DIR}/aarch64-linux/libtokenizers_c.a
    m dl pthread
)

set_target_properties(qwen3_tts_bm1684x PROPERTIES
    LINK_FLAGS "-Wl,-rpath,/opt/sophon/libsophon-0.5.1/lib"
)
