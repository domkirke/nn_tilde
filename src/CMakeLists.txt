cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(nn_tilde)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(SET CMP0076 NEW)

include(FetchContent)
FetchContent_Declare(
  json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.2  # Specify the version you want to use
)
FetchContent_MakeAvailable(json)


if (UNIX)
# import json
FetchContent_Declare(
  curl
  URL https://github.com/curl/curl/releases/download/curl-7_79_1/curl-7.79.1.tar.gz
)
FetchContent_MakeAvailable(curl)
endif()

if(MSVC)
  file(GLOB_RECURSE VCPKG_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/../vcpkg/installed/*windows/include")
  file(GLOB_RECURSE VCPKG_LIB_DIR "${CMAKE_SOURCE_DIR}/../vcpkg/installed/*windows/lib")
  
endif()

find_package(Torch REQUIRED)


if (APPLE)
    set(CMAKE_CXX_FLAGS "-faligned-allocation")
    if (CMAKE_OSX_ARCHITECTURES STREQUAL "")
        set(CMAKE_OSX_ARCHITECTURES ${CMAKE_HOST_SYSTEM_PROCESSOR})
    endif()
    message("CMAKE_OSX_ARCHITECTURES: ${CMAKE_OSX_ARCHITECTURES}")
    set(CMAKE_OSX_DEPLOYMENT_TARGET "10.13" CACHE STRING "Minimum OS X deployment version")
    add_compile_options(-std=c++20)
endif()


# manage CURL dependency
# file(GLOB CURL_DYLIB_PATH "../env/lib/libcurl*.dylib")
# file(GLOB CURL_INCLUDE_DIR "../env/pkgs/libcurl*/include/curl")
add_subdirectory(backend) # DEEP LEARNING BACKEND


if ((NOT ("${PUREDATA_INCLUDE_DIR}" STREQUAL "")) OR (UNIX AND NOT APPLE))
    #add_subdirectory(frontend/puredata/nn_tilde) # PURE DATA EXTERNAL
endif()

if(APPLE OR MSVC)
    add_subdirectory(frontend/maxmsp/nn.info) # MAX MSP EXTERNAL
    add_subdirectory(frontend/maxmsp/nn_tilde) # MAX MSP EXTERNAL
    add_subdirectory(frontend/maxmsp/mc.nn_tilde) # MAX MSP EXTERNAL
    add_subdirectory(frontend/maxmsp/mcs.nn_tilde) # MAX MSP EXTERNAL
endif()
