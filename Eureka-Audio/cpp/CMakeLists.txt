cmake_minimum_required(VERSION 3.10)
project(eureka_audio_bm1684x)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_C_COMPILER   aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

if(NOT DEFINED SOPHON_SDK)
    set(SOPHON_SDK ${CMAKE_SOURCE_DIR}/../../0_Toolkits/soc-sdk-sp4)
endif()
if(NOT DEFINED NLOHMANN_DIR)
    set(NLOHMANN_DIR ${CMAKE_SOURCE_DIR}/../../1_third_party)
endif()

include_directories(
    src
    ${SOPHON_SDK}/include
    ${NLOHMANN_DIR}
)

set(SOURCES
    src/main.cpp
    src/eureka_audio.cpp
    src/whisper_mel.cpp
    src/tokenizer.cpp
)

add_executable(eureka_audio_bm1684x ${SOURCES})

target_link_libraries(eureka_audio_bm1684x
    ${SOPHON_SDK}/lib/libbmrt.so
    ${SOPHON_SDK}/lib/libbmlib.so
    m dl pthread
)

set_target_properties(eureka_audio_bm1684x PROPERTIES
    LINK_FLAGS "-Wl,-rpath,/opt/sophon/libsophon-0.5.1/lib"
)
