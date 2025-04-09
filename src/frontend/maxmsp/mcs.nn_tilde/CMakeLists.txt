# Copyright 2018 The Min-DevKit Authors. All rights reserved.
# Use of this source code is governed by the MIT License found in the License.md file.

cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

set(C74_MIN_API_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../min-api)
include(${C74_MIN_API_DIR}/script/min-pretarget.cmake)


if (APPLE)
	set(CMAKE_OSX_DEPLOYMENT_TARGET "13.0")
endif()

#############################################################
# MAX EXTERNAL
#############################################################


include_directories( 
	"${C74_INCLUDES}",
	"${CMAKE_CURRENT_SOURCE_DIR}/../shared",
	"${CMAKE_CURRENT_SOURCE_DIR}/../../shared",
	"${CMAKE_CURRENT_SOURCE_DIR}/../../../shared",
)
find_package(Torch REQUIRED)

set(
SOURCE_FILES
	mcs.nn_tilde.cpp
)

if (UNIX)
	execute_process(
			COMMAND git describe --tags
			WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
			OUTPUT_VARIABLE VERSION
			OUTPUT_STRIP_TRAILING_WHITESPACE
	)
	message("${VERSION}")
	add_definitions(-DVERSION="${VERSION}")
endif()

add_library( 
	${PROJECT_NAME} 
	MODULE
	${SOURCE_FILES}
)
target_link_libraries(${PROJECT_NAME} PRIVATE backend)


set(CONDA_ENV_PATH "${CMAKE_SOURCE_DIR}/../env")
set(CURL_INCLUDE_DIR "${CONDA_ENV_PATH}/include")
set(CURL_LIBRARY "${CONDA_ENV_PATH}/lib/libcurl.dylib")
include_directories(${CURL_INCLUDE_DIR})
message("CURL: ${CURL_INCLUDE_DIR}")
# find_package(CURL REQUIRED)

target_link_libraries(${PROJECT_NAME} PRIVATE ${CURL_LIBRARY})
target_link_libraries(${PROJECT_NAME} PRIVATE nlohmann_json::nlohmann_json)

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)


if (APPLE) # SEARCH FOR TORCH DYLIB IN THE LOADER FOLDER
set_target_properties(${PROJECT_NAME} PROPERTIES
BUILD_WITH_INSTALL_RPATH FALSE
LINK_FLAGS "-Wl,-rpath,@loader_path/"
)
endif()

include(${C74_MIN_API_DIR}/script/min-posttarget.cmake)

if (APPLE) # COPY DYLIBS IN THE LOADER FOLDER
	add_custom_command(
		TARGET ${PROJECT_NAME}
		POST_BUILD 
		COMMAND ${CMAKE_SOURCE_DIR}/../env/bin/python ${CMAKE_SOURCE_DIR}/../install/dylib_fix.py -p "$<TARGET_FILE:${PROJECT_NAME}>" -l "${CMAKE_BINARY_DIR}/_deps"  "${CMAKE_SOURCE_DIR}/../env"
		COMMENT "Fixing libraries and codesigning"
	)
endif()


if (MSVC)
	set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 20)
endif()