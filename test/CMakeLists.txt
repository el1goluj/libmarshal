IF (CUDA_FOUND AND CUDA_MARSHAL)
  #CUDA Tests
  CUDA_ADD_EXECUTABLE (clk clock.cu)
  SET (CUDA_NVCC_FLAGS -arch sm_12)
  CUDA_ADD_EXECUTABLE (unittest unittest.cu)
  TARGET_LINK_LIBRARIES (unittest marshal gtest gtest_main)
ENDIF (CUDA_FOUND AND CUDA_MARSHAL)

#OpenCL Tests
ADD_EXECUTABLE (cl_unittest cl_unittest.cc)
TARGET_LINK_LIBRARIES (cl_unittest 
    gtest gtest_main cl_marshal ${OPENCL_LIBRARIES} gsl gslcblas)
