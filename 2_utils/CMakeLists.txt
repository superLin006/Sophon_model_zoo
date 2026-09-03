cmake_minimum_required(VERSION 3.15)
project(sophon_model_zoo_utils LANGUAGES C CXX)

option(SOPHON_UTILS_BUILD_FILE "Build generic file utilities" ON)
option(SOPHON_UTILS_BUILD_WAV "Build the lightweight WAV reader" ON)
option(SOPHON_UTILS_BUILD_AUDIO "Build libsndfile audio utilities" OFF)
option(SOPHON_UTILS_BUILD_IMAGE "Build image conversion utilities" OFF)
option(SOPHON_UTILS_BUILD_DRAWING "Build image drawing utilities" OFF)

if(SOPHON_UTILS_BUILD_FILE)
    add_library(fileutils STATIC file_utils.c)
    target_include_directories(fileutils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
endif()

if(SOPHON_UTILS_BUILD_WAV)
    add_library(wavutils STATIC wav.cpp)
    target_include_directories(wavutils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
endif()

if(SOPHON_UTILS_BUILD_AUDIO)
    set(SOPHON_SNDFILE_INCLUDE_DIR "" CACHE PATH "Directory containing sndfile.h")
    set(SOPHON_SNDFILE_LIBRARY "" CACHE FILEPATH "libsndfile library")
    if(NOT SOPHON_SNDFILE_INCLUDE_DIR OR NOT SOPHON_SNDFILE_LIBRARY)
        message(FATAL_ERROR
            "SOPHON_UTILS_BUILD_AUDIO requires SOPHON_SNDFILE_INCLUDE_DIR and "
            "SOPHON_SNDFILE_LIBRARY")
    endif()

    add_library(audioutils STATIC audio_utils.c)
    target_include_directories(audioutils PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${SOPHON_SNDFILE_INCLUDE_DIR}
    )
    target_link_libraries(audioutils PUBLIC ${SOPHON_SNDFILE_LIBRARY} m)
endif()

if(SOPHON_UTILS_BUILD_DRAWING)
    add_library(imagedrawing STATIC image_drawing.c)
    target_include_directories(imagedrawing PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
endif()

if(SOPHON_UTILS_BUILD_IMAGE)
    set(SOPHON_STB_INCLUDE_DIR "" CACHE PATH "Directory containing stb headers")
    set(SOPHON_RGA_INCLUDE_DIR "" CACHE PATH "Directory containing RGA headers")
    set(SOPHON_RGA_LIBRARY "" CACHE FILEPATH "RGA library")
    set(SOPHON_JPEG_INCLUDE_DIR "" CACHE PATH "Directory containing JPEG headers")
    set(SOPHON_JPEG_LIBRARY "" CACHE FILEPATH "JPEG library")

    if(NOT SOPHON_STB_INCLUDE_DIR OR NOT SOPHON_JPEG_INCLUDE_DIR OR
       NOT SOPHON_JPEG_LIBRARY)
        message(FATAL_ERROR
            "SOPHON_UTILS_BUILD_IMAGE requires stb and JPEG include/library paths")
    endif()

    add_library(imageutils STATIC image_utils.c)
    target_include_directories(imageutils PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${SOPHON_STB_INCLUDE_DIR}
        ${SOPHON_RGA_INCLUDE_DIR}
        ${SOPHON_JPEG_INCLUDE_DIR}
    )
    if(SOPHON_RGA_LIBRARY)
        target_link_libraries(imageutils PUBLIC ${SOPHON_RGA_LIBRARY})
    endif()
    target_link_libraries(imageutils PUBLIC ${SOPHON_JPEG_LIBRARY})
endif()
