cmake_minimum_required(VERSION 3.21)
include($ENV{XMOS_CMAKE_PATH}/xcommon.cmake)

project(integration_test)
set(XMOS_SANDBOX_DIR ${CMAKE_CURRENT_LIST_DIR}/../../..)

set(APP_INCLUDES src src/etc)
set(APP_DEPENDENT_MODULES "lib_nn" "lib_unity(v2.7.0)")

if(NOT BUILD_NATIVE)
  message(FATAL_ERROR "Integration tests are only supported for native builds")
endif()

# Third-party test dependencies
include(FetchContent)
function(fetch_dep name repo tag)
  if(NOT EXISTS ${XMOS_SANDBOX_DIR}/${name})
    message(STATUS "Fetching ${name}")
    # GIT_SHALLOW only fetches the tips of branches/tags, so it can't be used
    # here: some pins (e.g. gemmlowp, which has no tags) are arbitrary
    # historical commits that a shallow clone would never reach.
    FetchContent_Declare(
      ${name}
      GIT_REPOSITORY ${repo}
      GIT_TAG ${tag}
      SOURCE_DIR ${XMOS_SANDBOX_DIR}/${name}
    )
    FetchContent_Populate(${name})
  endif()
endfunction()

fetch_dep(tensorflow     https://github.com/tensorflow/tensorflow.git   v2.15.0)
# gemmlowp has no tags; pinned to the exact commit tensorflow v2.15.0 itself
# vendors (see tensorflow/third_party/gemmlowp/workspace.bzl), not an arbitrary SHA.
fetch_dep(gemmlowp       https://github.com/google/gemmlowp.git        e844ffd17118c1e17d94e1ba4354c075a4577b88)

# fetch_dep(googletest     https://github.com/google/googletest.git      6c58c11d5497b6ee1df3cb400ce30deb72fc28c0)
# fetch_dep(flatbuffers    https://github.com/google/flatbuffers.git     20aad0c41e1252b04c72111c3eb221280a9c2009)
# fetch_dep(ruy            https://github.com/google/ruy.git             9c56af3fce210a8a103eda19bd6f47c08a9e3d90)
# fetch_dep(compute-engine https://github.com/larq/compute-engine.git    v0.13.0)

XMOS_REGISTER_APP()

# third party links
include(tensorflow.cmake)
