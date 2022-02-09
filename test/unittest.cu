//===--- unittest.cu - GPU in-place marshaling library          ----------===//
// (C) Copyright 2012 The Board of Trustees of the University of Illinois.
// All rights reserved.
//
//                            libmarshal
// Developed by:
//                           IMPACT Research Group
//                  University of Illinois, Urbana-Champaign
// 
// This file is distributed under the Illinois Open Source License.
// See LICENSE.TXT for details.
//
// Author: I-Jui Sung (sung10@illinois.edu)
//
//===---------------------------------------------------------------------===//
//
//  This file defines unit testcases for libmarshal 
//
//===---------------------------------------------------------------------===//


#include <gtest/gtest.h>
#include <cstdlib>
#include "marshal.h"
//namespace {
class libmarshal_test : public ::testing::Test {
 protected:
  virtual void SetUp(void) {}
  virtual void TearDown(void) {}
  libmarshal_test() {}
//};

template<class T>
int compare_output(T *output, T *ref, int dim) {
  int pass = 1;
  int i;
  for (i = 0; i < dim; i++) {
    T diff = fabs(ref[i] - output[i]);
    if ((diff - 0.0f) > 0.00001f && diff > 0.01*fabs(ref[i])) {
      printf("line: %d ref: %f actual: %f diff: %f\n",
          i, ref[i], output[i], diff);
      pass = 0;
      break;
    }
  }
#if 0
  printf("\n");
  if (pass)
    printf("comparison passed\n");
  else
    printf("comparison failed\n");
  printf("\n");
#endif
  return pass != 1;
}

// Generate a matrix of random numbers
template <class T>
int generate_vector(T *x_vector, int dim) 
{       
  srand(5432);
  for(int i=0;i<dim;i++) {
    x_vector[i] = ((T) (rand() % 100) / 100);
  }
  return 0;
}

template <class T>
void cpu_aos_asta(T *src, T *dst, int height, int width,
    int tile_size) {
  // We only support height == multiple of tile size
  assert((height/tile_size)*tile_size == height);
  for (int i = 0; i<height/tile_size; i++) { //For all tiles
    T *src_start = src+i*tile_size*width;
    T *dst_start = dst+i*tile_size*width;
    for(int j = 0; j < tile_size; j++) {
      for (int k = 0; k < width; k++) {
        dst_start[j+k*tile_size]=src_start[j*width+k];
      }
    }
  }
}

void cpu_soa_asta(float *src, float *dst, int height, int width,
    int tile_size) {
  // We only support height == multiple of tile size
  assert((height/tile_size)*tile_size == height);
  for (int k = 0; k < width; k++) {
    for (int i = 0; i<height/tile_size; i++) { //For all tiles
      for(int j = 0; j < tile_size; j++) {
        //from src[k][i][j] to dst[i][k][j]
        dst[i*width*tile_size + k*tile_size + j] =
          src[k*height+i*tile_size + j];
      }
    }
  }
}

};

TEST_F(libmarshal_test, bug538) {
  int h = (65536+2);
  int t = 2;
  int w = 2;
  float *src = (float*)malloc(sizeof(float)*h*w);
  float *dst = (float*)malloc(sizeof(float)*h*w);
  float *dst_gpu = (float*)malloc(sizeof(float)*h*w);
  generate_vector(src, h*w);
  cpu_soa_asta(src, dst, h, w, t);
  float *d_dst;
  cudaMalloc(&d_dst, sizeof(float)*h*w);
  cudaMemcpy(d_dst, src, sizeof(float)*h*w, cudaMemcpyHostToDevice);
  bool r = gpu_soa_asta_pttwac(d_dst, h, w, t, NULL);
  ASSERT_EQ(false, r);
  cudaMemcpy(dst_gpu, d_dst, sizeof(float)*h*w, cudaMemcpyDeviceToHost);
  EXPECT_EQ(0, compare_output(dst_gpu, dst, h*w));
  free(src);
  free(dst);
  free(dst_gpu);
  cudaFree(d_dst);
}

TEST_F(libmarshal_test, bug528) {
  int h = 16*64;
  int t = 64;
  for (int w = 1; w < 100; w++) {
    float *src = (float*)malloc(sizeof(float)*h*w);
    float *dst = (float*)malloc(sizeof(float)*h*w);
    float *dst_gpu = (float*)malloc(sizeof(float)*h*w);
    generate_vector(src, h*w);
    cpu_soa_asta(src, dst, h, w, t);
    float *d_dst;
    cudaMalloc(&d_dst, sizeof(float)*h*w);
    cudaMemcpy(d_dst, src, sizeof(float)*h*w, cudaMemcpyHostToDevice);
    bool r = gpu_soa_asta_pttwac(d_dst, h, w, t, NULL);
    ASSERT_EQ(false, r);

    cudaMemcpy(dst_gpu, d_dst, sizeof(float)*h*w, cudaMemcpyDeviceToHost);
    EXPECT_EQ(0, compare_output(dst_gpu, dst, h*w));
    free(src);
    free(dst);
    free(dst_gpu);
    cudaFree(d_dst);
  }
}

TEST_F(libmarshal_test, bug525) {
  int h = 16*1024;
  int t = 16;
  for (int w = 1; w < 100; w++) {
    float *src = (float*)malloc(sizeof(float)*h*w);
    float *dst = (float*)malloc(sizeof(float)*h*w);
    float *dst_gpu = (float*)malloc(sizeof(float)*h*w);
    generate_vector(src, h*w);
    cpu_aos_asta(src, dst, h, w, t);

    float *d_dst;
    cudaMalloc(&d_dst, sizeof(float)*h*w);
    cudaMemcpy(d_dst, src, sizeof(float)*h*w, cudaMemcpyHostToDevice);
    bool r = gpu_aos_asta_pttwac(d_dst, h, w, t, NULL);
    ASSERT_EQ(false, r);

    cudaMemcpy(dst_gpu, d_dst, sizeof(float)*h*w, cudaMemcpyDeviceToHost);

    EXPECT_EQ(0, compare_output(dst_gpu, dst, h*w));
    free(src);
    free(dst);
    free(dst_gpu);
    cudaFree(d_dst);
  }
}

TEST_F(libmarshal_test, bug524) {
  int h = 16*1024;
  int w = 66;
  int t = 16;
  float *src = (float*)malloc(sizeof(float)*h*w);
  float *dst = (float*)malloc(sizeof(float)*h*w);
  float *dst_gpu = (float*)malloc(sizeof(float)*h*w);
  generate_vector(src, h*w);
  cpu_aos_asta(src, dst, h, w, t);

  float *d_dst;
  cudaMalloc(&d_dst, sizeof(float)*h*w);
  cudaMemcpy(d_dst, src, sizeof(float)*h*w, cudaMemcpyHostToDevice);
  bool r = gpu_aos_asta_pttwac(d_dst, h, w, t, NULL);
  ASSERT_EQ(false, r);
  
  cudaMemcpy(dst_gpu, d_dst, sizeof(float)*h*w, cudaMemcpyDeviceToHost);

  EXPECT_EQ(0, compare_output(dst_gpu, dst, h*w));
  free(src);
  free(dst);
  free(dst_gpu);
  cudaFree(d_dst);
}

TEST_F(libmarshal_test, bug523) {
  int h = 16*1024;
  int w = 6;
  int t = 16;
  float *src = (float*)malloc(sizeof(float)*h*w);
  float *dst = (float*)malloc(sizeof(float)*h*w);
  float *dst_gpu = (float*)malloc(sizeof(float)*h*w);
  generate_vector(src, h*w);
  cpu_aos_asta(src, dst, h, w, t);

  float *d_dst;
  cudaMalloc(&d_dst, sizeof(float)*h*w);
  cudaMemcpy(d_dst, src, sizeof(float)*h*w, cudaMemcpyHostToDevice);
  bool r = gpu_aos_asta_bs_float(d_dst, h, w, t, NULL);
  ASSERT_EQ(false, r);
  
  cudaMemcpy(dst_gpu, d_dst, sizeof(float)*h*w, cudaMemcpyDeviceToHost);

  EXPECT_EQ(0, compare_output(dst_gpu, dst, h*w));
  free(src);
  free(dst);
  free(dst_gpu);
  cudaFree(d_dst);
}

TEST_F(libmarshal_test, bug523d) {
  int h = 16*1024;
  int w = 6;
  int t = 16;
  double *src = (double*)malloc(sizeof(double)*h*w);
  double *dst = (double*)malloc(sizeof(double)*h*w);
  double *dst_gpu = (double*)malloc(sizeof(double)*h*w);
  generate_vector(src, h*w);
  cpu_aos_asta(src, dst, h, w, t);

  double *d_dst;
  cudaMalloc(&d_dst, sizeof(double)*h*w);
  cudaMemcpy(d_dst, src, sizeof(double)*h*w, cudaMemcpyHostToDevice);
  bool r = gpu_aos_asta_bs_double(d_dst, h, w, t, NULL);
  ASSERT_EQ(false, r);
  
  cudaMemcpy(dst_gpu, d_dst, sizeof(double)*h*w, cudaMemcpyDeviceToHost);

  EXPECT_EQ(0, compare_output(dst_gpu, dst, h*w));
  free(src);
  free(dst);
  free(dst_gpu);
  cudaFree(d_dst);
}
