# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

cmake_minimum_required(VERSION 3.18...4.0)
project(vta LANGUAGES CXX)

# 查找Python解释器
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
set(PYTHON ${Python3_EXECUTABLE})
# Check for Python dependency
if(NOT PYTHON)
  message(STATUS "Python not found, VTA build is skipped..")
  return()
endif()

# 添加VTA硬件源代码目录
add_subdirectory(vta_hw)

if(USE_VTA_FSIM)
  add_library(vta_fsim SHARED
    runtime/device_api.cc
    runtime/runtime.cc
  )
  # 设置TVM根目录绝对路径
  set(TVM_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../)
  
  # 定义TVM相关包含路径变量
  set(TVM_INCLUDE_DIR ${TVM_ROOT_DIR}/include)
  # set(TVM_FFI_INCLUDE_DIR ${TVM_ROOT_DIR}/ffi/include)
  set(TVM_BUILD_INCLUDE_DIR ${TVM_ROOT_DIR}/build/include)
  target_include_directories(vta_fsim PRIVATE
      ${TVM_INCLUDE_DIR}
      ${TVM_FFI_INCLUDE_DIR}
      ${TVM_BUILD_INCLUDE_DIR}
      ${PROJECT_SOURCE_DIR}/include
  )
  target_link_libraries(vta_fsim
    PRIVATE
      vta::driver::fsim
  )
  add_library(vta::fsim ALIAS vta_fsim)
endif()

if(USE_VTA_TSIM)
  add_library(vta_tsim SHARED
    runtime/device_api.cc
    runtime/runtime.cc
  )
  target_include_directories(vta_tsim
      PUBLIC
      ${PROJECT_SOURCE_DIR}/include
  )
  target_link_libraries(vta_tsim
    PRIVATE
      vta::driver::tsim
  )
  add_library(vta::tsim ALIAS vta_tsim)
endif()

if(USE_VTA_FPGA)
  add_library(vta_fpga_${VTA_TARGET} SHARED
    runtime/device_api.cc
    runtime/runtime.cc
  )
  target_include_directories(vta_fpga_${VTA_TARGET}
      PUBLIC
      ${PROJECT_SOURCE_DIR}/include
  )
  target_link_libraries(vta_fpga_${VTA_TARGET}
    PRIVATE
      vta::driver::fpga::${VTA_TARGET}
  )
  add_library(vta::fpga::${VTA_TARGET} ALIAS vta_fpga_${VTA_TARGET})
endif()
