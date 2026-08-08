cmake_minimum_required(VERSION 3.10)
project(qwen3_asr_bm1684x)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_C_COMPILER   aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

if(NOT DEFINED SOPHON_SDK)
    set(SOPHON_SDK ${CMAKE_SOURCE_DIR}/../../0_Toolkits/soc-sdk-sp4)
endif()
if(NOT DEFINED NLOHMANN_DIR)
    set(NLOHMANN_DIR ${CMAKE_SOURCE_DIR}/../../1_third_party)
endif()
if(NOT DEFINED TOKENIZERS_DIR)
    set(TOKENIZERS_DIR ${CMAKE_SOURCE_DIR}/../../1_third_party/tokenizers-cpp)
endif()
if(NOT DEFINED FFTW_DIR)
    set(FFTW_DIR ${CMAKE_SOURCE_DIR}/../../1_third_party/fftw)
endif()

include_directories(
    src
    ${SOPHON_SDK}/include
    ${NLOHMANN_DIR}
    ${TOKENIZERS_DIR}/include
    ${FFTW_DIR}/include
)

set(SOURCES
    src/main.cpp
    src/asr_engine.cpp
    src/qwen_mel.cpp
    src/tokenizer.cpp
)

add_executable(qwen3_asr_bm1684x ${SOURCES})

target_link_libraries(qwen3_asr_bm1684x
    ${SOPHON_SDK}/lib/libbmrt.so
    ${SOPHON_SDK}/lib/libbmlib.so
    ${TOKENIZERS_DIR}/aarch64-linux/libtokenizers_cpp.a
    ${TOKENIZERS_DIR}/aarch64-linux/libtokenizers_c.a
    ${FFTW_DIR}/aarch64-linux/libfftw3f.a
    m dl pthread
)

set_target_properties(qwen3_asr_bm1684x PROPERTIES
    LINK_FLAGS "-Wl,-rpath,/opt/sophon/libsophon-0.5.1/lib"
)
