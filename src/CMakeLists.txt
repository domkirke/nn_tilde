cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(nn_tilde)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)


if (APPLE)
  set(CMAKE_OSX_DEPLOYMENT_TARGET "10.12")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(SET CMP00076)

include(FetchContent)

message("Building for architecture : ${CMAKE_OSX_ARCHITECTURES}")

# import json
FetchContent_Declare(
  json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.2  # Specify the version you want to use
  CMAKE_ARGS "-DCMAKE_OSX_ARCHITECTURES:STRING=${CMAKE_OSX_ARCHITECTURES}"  "DCMAKE_OSX_DEPLOYMENT_TARGET=10.13"
)
FetchContent_MakeAvailable(json)


if (APPLE)
  set(CMAKE_CXX_FLAGS "-faligned-allocation")
  if (CMAKE_OSX_ARCHITECTURES STREQUAL "")
      set(CMAKE_OSX_ARCHITECTURES ${CMAKE_HOST_SYSTEM_PROCESSOR})
  endif()
  message("CMAKE_OSX_ARCHITECTURES: ${CMAKE_OSX_ARCHITECTURES}")
  # set(CMAKE_OSX_DEPLOYMENT_TARGET "10.13" CACHE STRING "Minimum OS X deployment version")
  add_compile_options(-std=c++20)

    # locating homebrew
  if(DEFINED ENV{HOMEBREW_PREFIX})
    set(HOMEBREW_PREFIX $ENV{HOMEBREW_PREFIX})
    message(STATUS "Homebrew is located at: ${HOMEBREW_PREFIX}")
  else()
    if (CMAKE_OSX_ARCHITECTURES STREQUAL "arm64")
      set(HOMEBREW_PREFIX "/opt/homebrew")
      message(WARNING "HOMEBREW_PREFIX not set. Defaulting to: ${HOMEBREW_PREFIX}")
    else() 
      set(HOMEBREW_PREFIX "/usr/local")
      message(WARNING "HOMEBREW_PREFIX not set. Defaulting to: ${HOMEBREW_PREFIX}")
    endif()
  endif()
endif()


find_package(Torch REQUIRED)
add_subdirectory(backend) # DEEP LEARNING BACKEND


if ((NOT ("${PUREDATA_INCLUDE_DIR}" STREQUAL "")) OR (UNIX AND NOT APPLE))
    add_subdirectory(frontend/puredata/nn_tilde) # PURE DATA EXTERNAL
endif()

if(APPLE OR MSVC)
    add_subdirectory(frontend/maxmsp/nn.info) # MAX MSP EXTERNAL
    add_subdirectory(frontend/maxmsp/nn_tilde) # MAX MSP EXTERNAL
    add_subdirectory(frontend/maxmsp/mc.nn_tilde) # MAX MSP EXTERNAL
    add_subdirectory(frontend/maxmsp/mcs.nn_tilde) # MAX MSP EXTERNAL
endif()