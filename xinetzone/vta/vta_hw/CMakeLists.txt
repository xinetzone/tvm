# 设置VTA硬件路径
set(VTA_HW_PATH "$ENV{VTA_HW_PATH}")
if(NOT DEFINED ENV{VTA_HW_PATH})
  set(VTA_HW_PATH ${CMAKE_CURRENT_SOURCE_DIR})
else()
  set(VTA_HW_PATH $ENV{VTA_HW_PATH})
endif()

# 检查VTA硬件路径是否存在
if(NOT EXISTS ${VTA_HW_PATH})
  if (USE_VTA_TSIM OR USE_VTA_FSIM OR USE_UFPGA)
    message(FATAL_ERROR "VTA路径 ${VTA_HW_PATH} 不存在")
  else()
    message(STATUS "VTA硬件路径不存在，跳过VTA构建")
    return()
  endif()
endif()
message(STATUS "VTA构建路径: ${VTA_HW_PATH}")
# 设置TVM根目录绝对路径
set(VTA_ROOT_DIR ${PROJECT_SOURCE_DIR})
set(TVM_ROOT_DIR ${VTA_ROOT_DIR}/..)

# 定义TVM相关包含路径变量
set(TVM_INCLUDE_DIR ${TVM_ROOT_DIR}/include)
set(TVM_FFI_INCLUDE_DIR ${TVM_ROOT_DIR}/ffi/include)
set(TVM_BUILD_INCLUDE_DIR ${TVM_ROOT_DIR}/build/include)

set(VTA_CONFIG ${PYTHON} ${VTA_HW_PATH}/config/vta_config.py)

# 检查是否存在预生成的配置文件
if(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/vta_config.json)
  message(STATUS "使用预生成配置: ${CMAKE_CURRENT_BINARY_DIR}/vta_config.json")
  set(VTA_CONFIG ${PYTHON} ${VTA_HW_PATH}/config/vta_config.py
    --use-cfg=${CMAKE_CURRENT_BINARY_DIR}/vta_config.json)
endif()

# 获取目标平台信息
execute_process(
  COMMAND ${VTA_CONFIG} --target
  OUTPUT_VARIABLE VTA_TARGET
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(VTA_TARGET "${VTA_TARGET}" CACHE INTERNAL "VTA target platform")
message(STATUS "VTA目标平台: ${VTA_TARGET}")
message(STATUS "TVM_ROOT_DIR : ${TVM_ROOT_DIR}")
message(STATUS "TVM_INCLUDE_DIR: ${TVM_INCLUDE_DIR}")
message(STATUS "TVM_BUILD_INCLUDE_DIR: ${TVM_BUILD_INCLUDE_DIR}")
message(STATUS "TVM_FFI_INCLUDE_DIR: ${TVM_FFI_INCLUDE_DIR}")

# 获取编译定义
execute_process(
  COMMAND ${VTA_CONFIG} --defs
  OUTPUT_VARIABLE __vta_defs
)
# 修复正则表达式语法错误（移除多余逗号）并精确匹配宏定义
string(REGEX MATCHALL "-D[A-Za-z0-9_]+=[A-Za-z0-9_]+" VTA_DEFINITIONS "${__vta_defs}")
# 移除所有-D前缀
string(REPLACE "-D" "" VTA_DEFINITIONS "${VTA_DEFINITIONS}")
# 确保定义为列表形式
separate_arguments(VTA_DEFINITIONS)

# 快速 VTA 仿真器驱动构建
if(USE_VTA_FSIM)
  file(GLOB FSIM_RUNTIME_SRCS
    # ${VTA_HW_PATH}/src/*.cc
    # vta/runtime/*.cc
    ${VTA_HW_PATH}/src/sim/sim_driver.cc
    ${VTA_HW_PATH}/src/sim/sim_tlpp.cc
    ${VTA_HW_PATH}/src/vmem/virtual_memory.cc
  )
 
  add_library(vta_driver_fsim SHARED ${FSIM_RUNTIME_SRCS})
  add_library(vta::driver::fsim ALIAS vta_driver_fsim)
  target_include_directories(vta_driver_fsim SYSTEM PUBLIC
    ${VTA_HW_PATH}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${TVM_FFI_INCLUDE_DIR}  # TVM FFI include相对路径
    ${TVM_INCLUDE_DIR}  # TVM include相对路径
    ${PROJECT_BINARY_DIR}/include  # 构建生成include路径
  )
  target_link_libraries(vta_driver_fsim PRIVATE ${LLVM_LIBS})
  target_compile_definitions(vta_driver_fsim PUBLIC
    DMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>
    USE_FSIM_TLPP
    ${VTA_DEFINITIONS}
  )

  if(APPLE)
    set_property(TARGET vta_driver_fsim APPEND PROPERTY
      LINK_FLAGS "-undefined dynamic_lookup"
    )
  endif()

  install(TARGETS vta_driver_fsim
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
  )
endif()

# cycle-accurate VTA 仿真器驱动构建
if(USE_VTA_TSIM)
  # 查找Verilator
  if(DEFINED ENV{VERILATOR_INC_DIR})
    set(VERILATOR_INC_DIR $ENV{VERILATOR_INC_DIR})
  elseif (EXISTS /usr/local/share/verilator/include)
    set(VERILATOR_INC_DIR /usr/local/share/verilator/include)
  elseif (EXISTS /usr/share/verilator/include)
    set(VERILATOR_INC_DIR /usr/share/verilator/include)
  else()
    message(FATAL_ERROR "找不到Verilator，需要设置VERILATOR_INC_DIR")
  endif()

  file(GLOB TSIM_RUNTIME_SRCS
    # vta/runtime/*.cc
    ${VTA_HW_PATH}/src/tsim/tsim_driver.cc
    ${VTA_HW_PATH}/src/dpi/module.cc
    ${VTA_HW_PATH}/src/vmem/virtual_memory.cc
  )

  add_library(vta_driver_tsim SHARED ${TSIM_RUNTIME_SRCS})
  add_library(vta::driver::tsim ALIAS vta_driver_tsim)
  target_include_directories(vta_driver_tsim SYSTEM PUBLIC
    ${VTA_HW_PATH}/include
    ${VERILATOR_INC_DIR}
    ${VERILATOR_INC_DIR}/vltstd
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${TVM_FFI_INCLUDE_DIR}  # TVM FFI include相对路径
    ${TVM_INCLUDE_DIR}  # TVM include相对路径
    ${PROJECT_BINARY_DIR}/include  # 构建生成include路径
  )
  target_compile_definitions(vta_driver_tsim PUBLIC
    DMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>
    ${VTA_DEFINITIONS}
  )

  if(APPLE)
    set_property(TARGET vta_driver_tsim APPEND PROPERTY
      LINK_FLAGS "-undefined dynamic_lookup"
    )
  endif()

  install(TARGETS vta_driver_tsim
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
  )
endif()

# FPGA VTA 驱动构建（根据目标平台）
if(USE_VTA_FPGA)
  file(GLOB FPGA_RUNTIME_SRCS
    ${VTA_HW_PATH}/src/*.cc
    # vta/runtime/*.cc
  )

  # 根据目标平台添加特定源文件
  if("${VTA_TARGET}" STREQUAL "pynq" OR "${VTA_TARGET}" STREQUAL "ultra96")
    list(APPEND FPGA_RUNTIME_SRCS ${VTA_HW_PATH}/src/pynq/pynq_driver.cc)
    find_library(__cma_lib NAMES cma PATH /usr/lib)
  elseif(${VTA_TARGET} STREQUAL "de10nano")
    file(GLOB DE10_SRCS ${VTA_HW_PATH}/src/de10nano/*.cc)
    list(APPEND FPGA_RUNTIME_SRCS ${DE10_SRCS})
  elseif(${VTA_TARGET} STREQUAL "intelfocl")
    file(GLOB FOCL_SRC ${VTA_HW_PATH}/src/oclfpga/*.cc)
    list(APPEND FPGA_RUNTIME_SRCS
      ${FOCL_SRC}
      ${VTA_HW_PATH}/src/vmem/virtual_memory.cc
    )
  else()
    file(GLOB FPGA_RUNTIME_SRCS
      ${VTA_HW_PATH}/src/sim/sim_driver.cc
      ${VTA_HW_PATH}/src/sim/sim_tlpp.cc
    )
  endif()

  if(NOT FPGA_RUNTIME_SRCS)
    message(FATAL_ERROR "No source files found for VTA_TARGET: ${VTA_HW_PATH} ${VTA_TARGET}")
    return()
  endif()
  add_library(vta_driver_fpga_${VTA_TARGET} SHARED ${FPGA_RUNTIME_SRCS})
  add_library(vta::driver::fpga::${VTA_TARGET} ALIAS vta_driver_fpga_${VTA_TARGET})
  target_include_directories(vta_driver_fpga_${VTA_TARGET} PUBLIC
    # vta/runtime
    ${VTA_HW_PATH}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${TVM_FFI_INCLUDE_DIR}  # TVM FFI include相对路径
    ${TVM_INCLUDE_DIR}  # TVM include相对路径
    ${PROJECT_BINARY_DIR}/include  # 构建生成include路径
  )
  target_compile_definitions(vta_driver_fpga_${VTA_TARGET} PUBLIC
    DMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>
    ${VTA_DEFINITIONS}
  )

  # 平台特定链接设置
  if(${VTA_TARGET} STREQUAL "pynq" OR ${VTA_TARGET} STREQUAL "ultra96")
    target_link_libraries(vta_driver_fpga_${VTA_TARGET} ${__cma_lib})
  elseif(${VTA_TARGET} STREQUAL "de10nano")
    target_include_directories(vta_driver_fpga_${VTA_TARGET} SYSTEM PUBLIC
      ${VTA_HW_PATH}/src/de10nano
      3rdparty
      /usr/local/intelFPGA_lite/18.1/embedded/ds-5/sw/gcc/arm-linux-gnueabihf/include
    )
  elseif(${VTA_TARGET} STREQUAL "intelfocl")
    target_include_directories(vta_driver_fpga_${VTA_TARGET} PUBLIC 3rdparty)
    set_property(TARGET vta_driver_fpga_${VTA_TARGET} PROPERTY CXX_STANDARD 17)
    target_link_libraries(vta_driver_fpga_${VTA_TARGET} -lOpenCL)
  endif()

  install(TARGETS vta_driver_fpga_${VTA_TARGET}
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
  )
endif()
