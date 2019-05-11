cmake_minimum_required(VERSION 3.12)
project(model_runner)

set(CMAKE_CXX_STANDARD 17)

include_directories(extern/tensorflow)
include_directories(extern/tensorflow/bazel-tensorflow/external/eigen_archive)
include_directories(extern/tensorflow/bazel-tensorflow/external/protobuf_archive/src)
include_directories(extern/tensorflow/bazel-tensorflow/external/com_google_absl)
include_directories(extern/tensorflow/bazel-genfiles)

include_directories(extern/json/include)

link_directories(extern/tensorflow/bazel-bin/tensorflow)

add_library(model_runner SHARED model_runner.cpp model_runner.h io_map.cpp io_map.h c_api.cpp c_api.h)
target_link_libraries(model_runner tensorflow_cc)

add_executable(test test.cpp c_api.h)
target_link_libraries(test model_runner)