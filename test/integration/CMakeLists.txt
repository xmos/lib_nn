cmake_minimum_required(VERSION 3.21)
include($ENV{XMOS_CMAKE_PATH}/xcommon.cmake)

project(integration_test)
set(XMOS_SANDBOX_DIR ${CMAKE_CURRENT_LIST_DIR}/../../..)

set(APP_INCLUDES src src/etc)
set(APP_DEPENDENT_MODULES "lib_nn" "lib_unity(v2.7.0)")

if(NOT BUILD_NATIVE)
  message(FATAL_ERROR "Integration tests are only supported for native builds")
endif()

# Test level
if(TEST_LEVEL STREQUAL "smoke")
  set(TEST_LEVEL_VALUE 0)
elseif(TEST_LEVEL STREQUAL "default")
  set(TEST_LEVEL_VALUE 1)
elseif(TEST_LEVEL STREQUAL "extended")
  set(TEST_LEVEL_VALUE 2)
else()
  set(TEST_LEVEL_VALUE 1)  
  message(WARNING "Setting default TEST_LEVEL")
endif()

# Flags
set(APP_COMPILER_FLAGS -O3 -g -DTEST_LEVEL=${TEST_LEVEL_VALUE})

# Third-party test dependencies
include(FetchContent)
function(fetch_dep name repo tag)
  if(NOT EXISTS ${XMOS_SANDBOX_DIR}/${name})
    message(STATUS "Fetching ${name}")
    FetchContent_Declare(
      ${name}
      GIT_REPOSITORY ${repo}
      GIT_TAG ${tag}
      SOURCE_DIR ${XMOS_SANDBOX_DIR}/${name}
    )
    FetchContent_Populate(${name})
  endif()
endfunction()
fetch_dep(compute-engine               git@github.com:larq/compute-engine.git          e0683a7cb2ce1c2e15f6c59bb140f7d6f8da6a9b)
fetch_dep(flatbuffers                  git@github.com:google/flatbuffers.git           6df40a2471737b27271bdd9b900ab5f3aec746c7)
fetch_dep(gemmlowp                     git@github.com:google/gemmlowp.git              2483d846ad865dd4190fe4a1a1ba2d9cfcea78e1)
fetch_dep(ruy                          git@github.com:google/ruy.git                   baa860187fe7fc115c64210ca00bc90ccd197642)
fetch_dep(tensorflow                   git@github.com:xmos/tensorflow                  fdfd74c614bf0850c332486826f91654f2a0181a)

XMOS_REGISTER_APP()

# third party links
include(third-party.cmake)
