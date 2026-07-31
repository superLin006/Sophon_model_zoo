cmake_minimum_required(VERSION 3.15)
project(zipformer_bm1684 LANGUAGES CXX)
option(ZIPFORMER_WITH_BM "Build real BMRuntime CLI" OFF)
option(ZIPFORMER_BUILD_TESTS "Build host-only tests" ON)
option(ZIPFORMER_BUILD_FBANK "Build strict fbank frontend and dump CLI" ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(SOPHON_SDK "" CACHE PATH "Sophon SDK root")
set(KALDI_FBANK_DIR "${CMAKE_CURRENT_LIST_DIR}/../../1_third_party/kaldi_native_fbank" CACHE PATH "kaldi-native-fbank root")
set(KALDI_FBANK_INCLUDE_DIR "${KALDI_FBANK_DIR}/include" CACHE PATH "kaldi-native-fbank headers")
set(KALDI_FBANK_LIBRARY "${KALDI_FBANK_DIR}/aarch64-linux/libkaldi-native-fbank-core.a" CACHE FILEPATH "kaldi-native-fbank core library")
set(KISSFFT_LIBRARY "${KALDI_FBANK_DIR}/aarch64-linux/libkissfft-float.a" CACHE FILEPATH "kissfft float library")
set(CORE src/manifest.cpp src/tokenizer.cpp src/zipformer.cpp
         ${CMAKE_CURRENT_LIST_DIR}/../../2_utils/wav.cpp)

# --- BMRuntime CLI (aarch64 cross-build only) ---
if(ZIPFORMER_WITH_BM)
  if(NOT SOPHON_SDK)
    message(FATAL_ERROR "ZIPFORMER_WITH_BM requires SOPHON_SDK")
  endif()
  add_executable(zipformer_cli ${CORE} src/main.cpp src/bmruntime_backend.cpp)
  target_compile_definitions(zipformer_cli PRIVATE ZIPFORMER_WITH_BM)
  target_include_directories(zipformer_cli PRIVATE src ${SOPHON_SDK}/include
    ${CMAKE_CURRENT_LIST_DIR}/../../2_utils)
  target_link_libraries(zipformer_cli PRIVATE ${SOPHON_SDK}/lib/libbmrt.so
    ${SOPHON_SDK}/lib/libbmlib.so dl pthread m)
endif()

if(ZIPFORMER_WITH_BM AND (NOT ZIPFORMER_BUILD_FBANK OR NOT EXISTS "${KALDI_FBANK_INCLUDE_DIR}"
    OR NOT EXISTS "${KALDI_FBANK_LIBRARY}" OR NOT EXISTS "${KISSFFT_LIBRARY}"))
  message(FATAL_ERROR "ZIPFORMER_WITH_BM requires kaldi-native-fbank and kissfft")
endif()

if(ZIPFORMER_BUILD_FBANK AND EXISTS "${KALDI_FBANK_INCLUDE_DIR}" AND
    EXISTS "${KALDI_FBANK_LIBRARY}" AND EXISTS "${KISSFFT_LIBRARY}")
  add_library(zipformer_fbank STATIC src/audio_frontend.cpp)
  target_include_directories(zipformer_fbank PUBLIC src ${KALDI_FBANK_INCLUDE_DIR})
  target_link_libraries(zipformer_fbank PUBLIC ${KALDI_FBANK_LIBRARY} ${KISSFFT_LIBRARY} m)
  if(TARGET zipformer_cli)
    target_link_libraries(zipformer_cli PRIVATE zipformer_fbank)
  endif()
endif()

# --- host-only tests ---
if(ZIPFORMER_BUILD_TESTS)
  enable_testing()
  add_executable(zipformer_host_tests ${CORE} tests/test_zipformer.cpp)
  target_include_directories(zipformer_host_tests PRIVATE src
    ${CMAKE_CURRENT_LIST_DIR}/../../2_utils)
  target_compile_definitions(zipformer_host_tests PRIVATE
    ZIPFORMER_TEST_MANIFEST="${CMAKE_CURRENT_LIST_DIR}/../configs/tensor_manifest.json")
  add_test(NAME zipformer_host_tests COMMAND zipformer_host_tests)
endif()
