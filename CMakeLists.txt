cmake_minimum_required(VERSION 3.8)
project(pcs VERSION 0.9.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_PCS_TESTS "Build tests for pcs library" ON)
option(BUILD_PCS_COVERAGE "Create test coverage for pcs library" ON)
option(BUILD_SHARED_LIBS "Build shared libs" OFF)
option(BUILD_PCS_PYTHON_BINDINGS "Build python bindings" OFF)

if(NOT CMAKE_CONFIGURATION_TYPES)
    set(allowableBuildTypes Debug Release DebWithRelInfo MinSizeRel)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY
            STRINGS "${allowableBuildTypes}")
    if(NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
        message(STATUS "Build type set to Release")
    elseif(NOT CMAKE_BUILD_TYPE IN_LIST allowableBuildTypes)
        message(FATAL_ERROR "Invalid build type: ${CMAKE_BUILD_TYPE}")
    endif()
endif()

if(BUILD_PCS_COVERAGE)
    if(${CMAKE_BUILD_TYPE} STREQUAL "Debug" AND BUILD_PCS_TESTS)
        message(STATUS "Test coverage enabled")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fprofile-arcs -ftest-coverage")
    else()
        message(WARNING "Test coverage disabled (change build mode to Debug and enable tests)")
        set(BUILD_PCS_COVERAGE OFF CACHE STRING "" FORCE)
    endif()
endif()

find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)

set(LIB_SOURCE_FILES
        src/feature_estimator.cpp
        src/point_cloud.cpp
        src/down_sample.cpp)

set(LIB_HEADER_FILES
        src/kd_tree.hpp
        src/thread_pool.hpp
        src/rgb_to_hsv.hpp
        src/feature_estimator.h
        src/point_cloud.h)

add_library(pcs ${LIB_SOURCE_FILES} ${LIB_HEADER_FILES})

target_include_directories(pcs PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>)

target_link_libraries(pcs
        PUBLIC
            Eigen3::Eigen
        PRIVATE
            Threads::Threads)

target_compile_options(pcs
        PRIVATE
        -Werror
        -Wno-deprecated-declarations)

set_target_properties(pcs
        PROPERTIES
        POSITION_INDEPENDENT_CODE 1)

add_library(pcs::pcs ALIAS pcs)

if(BUILD_PCS_PYTHON_BINDINGS)
    add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/pybind11)
    pybind11_add_module(pypcs MODULE SYSTEM python/python_bindings.cpp)
    target_link_libraries(pypcs PRIVATE pcs::pcs)
endif()

if(BUILD_PCS_TESTS)
    set(TESTS_SOURCE_FILES
            tests/point_cloud_test.cpp
            tests/feature_estimator_test.cpp
            tests/down_sample_test.cpp
            tests/thread_pool_test.cpp
            tests/kd_tree_test.cpp
            tests/rgb_to_hsv_test.cpp)

    add_executable(pcs_tests
            ${TESTS_SOURCE_FILES})
    target_link_libraries(pcs_tests
            PUBLIC
                Threads::Threads
            PRIVATE
                pcs)

    enable_testing()
    add_test(NAME pcs_test
            COMMAND $<TARGET_FILE:pcs_tests> --success)

    if(BUILD_PCS_COVERAGE)
        find_program(GCOVR_EXECUTABLE gcovr)
        if(GCOVR_EXECUTABLE)
            add_custom_target(coverage
                    COMMAND
                    ${GCOVR_EXECUTABLE}
                    --root="${CMAKE_CURRENT_LIST_DIR}/src"
                    --object-directory="${CMAKE_CURRENT_BINARY_DIR}"
                    DEPENDS
                    pcs_tests)
        else()
            message(WARNING "Can not find gcovr executable. Disable test coverage")
        endif()
    endif()
endif()
