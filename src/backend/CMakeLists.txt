cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(backend)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_library(backend STATIC parsing_utils.cpp backend.cpp)
target_link_libraries(backend "${TORCH_LIBRARIES}")
set_property(TARGET backend PROPERTY CXX_STANDARD 20)

if(MSVC)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
  set(CMAKE_CXX_FLAGS_MINSIZEREL  "${CMAKE_CXX_FLAGS_MINSIZEREL} /MT")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /MT")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MT")
endif()


