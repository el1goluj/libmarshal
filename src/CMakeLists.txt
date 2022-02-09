option(CUDA_MARSHAL "[libmarshal] Enable CUDA marshaling" ON)
option(LIBMARSHAL_OCL_PROFILE "[libmarshal] Turn on profiling of marshaling kernels" ON)
IF (CUDA_FOUND AND CUDA_MARSHAL)
SET (CUDA_NVCC_FLAGS -arch sm_20)
CUDA_ADD_LIBRARY(marshal marshal.cu)
ENDIF (CUDA_FOUND AND CUDA_MARSHAL)

IF(LIBMARSHAL_OCL_PROFILE)
  ADD_DEFINITIONS(-DLIBMARSHAL_OCL_PROFILE)
ENDIF(LIBMARSHAL_OCL_PROFILE)

ADD_CUSTOM_COMMAND(
    OUTPUT embd.cpp embd.hpp
    COMMAND embd ${CMAKE_CURRENT_SOURCE_DIR}/cl/cl_aos_asta.cl @ cl/cl_aos_asta.cl namespace=libmarshal
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} 
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cl/cl_aos_asta.cl
    )
INCLUDE_DIRECTORIES (${CMAKE_CURRENT_BINARY_DIR}) # for embd.hpp
ADD_LIBRARY(cl_marshal cl_marshal.cc embd.cpp)
