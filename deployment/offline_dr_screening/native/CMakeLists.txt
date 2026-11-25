cmake_minimum_required(VERSION 3.22)
project(offirst_jpeg LANGUAGES C)

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

include(ExternalProject)
set(jpeg_url "https://codeload.github.com/libjpeg-turbo/libjpeg-turbo/tar.gz/refs/tags/3.1.1")
if(DEFINED ENV{OFFIRST_JPEG_ARCHIVE})
    set(jpeg_url "$ENV{OFFIRST_JPEG_ARCHIVE}")
endif()

set(jpeg_options
    "-DCMAKE_BUILD_TYPE=Release"
    "-DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}"
    "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
    "-DCMAKE_POSITION_INDEPENDENT_CODE=ON"
    "-DENABLE_SHARED=OFF"
    "-DENABLE_STATIC=ON"
    "-DWITH_TURBOJPEG=OFF"
    "-DWITH_SIMD=OFF"
)
if(ANDROID)
    list(APPEND jpeg_options
        "-DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}"
        "-DANDROID_ABI=${ANDROID_ABI}"
        "-DANDROID_PLATFORM=${ANDROID_PLATFORM}"
    )
endif()

if(MSVC)
    set(jpeg_archive jpeg-static.lib)
else()
    set(jpeg_archive "${CMAKE_STATIC_LIBRARY_PREFIX}jpeg${CMAKE_STATIC_LIBRARY_SUFFIX}")
endif()

ExternalProject_Add(reference_jpeg
    URL "${jpeg_url}"
    URL_HASH SHA256=304165ae11e64ab752e9cfc07c37bfdc87abd0bfe4bc699e59f34036d9c84f72
    TLS_VERIFY ON
    SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/jpeg-src"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/jpeg-build"
    CMAKE_ARGS ${jpeg_options}
    BUILD_COMMAND "${CMAKE_COMMAND}" --build <BINARY_DIR> --config Release --target jpeg-static
    BUILD_BYPRODUCTS "${CMAKE_CURRENT_BINARY_DIR}/jpeg-build/${jpeg_archive}"
    INSTALL_COMMAND ""
)
ExternalProject_Get_Property(reference_jpeg SOURCE_DIR BINARY_DIR)

add_library(offirst_jpeg SHARED reference_jpeg.c)
add_dependencies(offirst_jpeg reference_jpeg)
target_include_directories(offirst_jpeg PRIVATE "${SOURCE_DIR}/src" "${BINARY_DIR}")
target_link_libraries(offirst_jpeg PRIVATE "${BINARY_DIR}/${jpeg_archive}")
if(ANDROID)
    target_link_options(offirst_jpeg PRIVATE "-Wl,--exclude-libs,ALL")
endif()
set_target_properties(offirst_jpeg PROPERTIES C_VISIBILITY_PRESET hidden PREFIX "${CMAKE_SHARED_LIBRARY_PREFIX}")
if(WIN32)
    set_target_properties(offirst_jpeg PROPERTIES PREFIX "")
endif()
