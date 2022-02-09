cmake_minimum_required(VERSION 2.6)
project(gtest)
# Defines CMAKE_USE_PTHREADS_INIT and CMAKE_THREAD_LIBS_INIT.
find_package(Threads)
if (CMAKE_USE_PTHREADS_INIT)  # The pthreads library is available.
  set(cxx_base_flags "${cxx_base_flags} -DGTEST_HAS_PTHREAD=1")
endif()

include_directories(${gtest_SOURCE_DIR}/..)
add_library(gtest gtest-all.cc)
add_library(gtest_main gtest_main.cc)
target_link_libraries(gtest ${CMAKE_THREAD_LIBS_INIT})
