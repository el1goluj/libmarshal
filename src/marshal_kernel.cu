//===--- marshal_kernel.cu - GPU in-place marshaling library    ----------===//
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
//  This file defines the CUDA kernels of the libmarshal 
//
//===---------------------------------------------------------------------===//


#ifndef _LIBMARSHAL_KERNEL_CU_
#define _LIBMARSHAL_KERNEL_CU_

// limitations: tile_size * width cannot exceed maximal # of threads in
// a block allowed in the system
// convert a[height/tile_size][tile_size][width] to
// a[height/tile_size][width][tile_size]
// Launch height/tile_size blocks of tile_size*width threads
template <class T, unsigned coarsen_factor>
__global__ static void BS_marshal(T *input,
    int tile_size, int width, clock_t *timer) {
//  clock_t time1 = clock();
  int tidx = threadIdx.x;
  int bid = blockIdx.x;
  input += tile_size*width*bid;
  int tidy = threadIdx.y;
  T tmp[coarsen_factor];
  for (int i = 0; i < coarsen_factor; ++i) {
    int ttidx = tidx + i * blockDim.x;
    tmp[i] = input[tidy*width+ttidx];
  }
  __syncthreads();
  for (int i = 0; i < coarsen_factor; ++i) {
    int ttidx = tidx + i * blockDim.x;
    //input[ttidx*tile_size+tidy] = tmp[i];
    input[tidy*width+ttidx] = tmp[i];
  }
}

// limitations: height must be multiple of tile_size
// convert a[height/tile_size][tile_size][width] to
// a[height/tile_size][width][tile_size]
// Launch height/tile_size blocks of NR_THREADS threads
__global__ static void PTTWAC_marshal(float *input_src, 
  int height, int tile_size, int width,
    clock_t *timer) {
  extern __shared__ unsigned finished[];
  int tidx = threadIdx.x;
  int m = tile_size*width - 1;
  for (int gid =blockIdx.x; gid < height/tile_size; gid+=gridDim.x) {
    float *input = input_src + gid*tile_size*width;
    for (int id = tidx ; id < (tile_size * width + 31) / 32;
      id += blockDim.x) {
      finished[id] = 0;
    }
    __syncthreads();
    for (;tidx < tile_size*width; tidx += blockDim.x) {
      int next = (tidx * tile_size) % m;
      if (tidx != m && next != tidx) {
        float data1 = input[tidx];
        unsigned int mask = (1 << (tidx % 32));
        unsigned int flag_id = (((unsigned int) tidx) >> 5);
        int done = atomicOr(finished+flag_id, 0);
        done = (done & mask);
        for (; done == 0; next = (next * tile_size) % m) {
          float data2 = input[next];
          mask = (1 << (next % 32));
          flag_id = (((unsigned int)next) >> 5);
          done = atomicOr(finished+flag_id, mask);
          done = (done & mask);
          if (done == 0) {
            input[next] = data1;
          }
          data1 = data2;
        }
      }
    }
  }
}

// limitations: tile_size cannot exceed # of allowed threads in the system
// convert a[width][height/tile_size][tile_size] to
// a[height/tile_size][width][tile_size]
// Launch width*height/tile_size blocks of tile_size threads
__global__ static void PTTWAC_marshal_soa(float *input, int height,
    int tile_size, int width, int *finished, clock_t *timer) {
  int m = (height*width)/tile_size-1;
  int tid = threadIdx.x;
  float data;
  __shared__ int done;
  for (int gid = blockIdx.x;gid < m; gid += gridDim.x) {

    int next_in_cycle = (gid * width)%m;
    if (next_in_cycle == gid)
      continue;

    data = input[gid*tile_size+tid];
    __syncthreads();
    if (tid == 0)
      done = atomicOr(finished+gid,
        (int)0); //make sure the read is not cached 
    __syncthreads();

    for (;done == 0; next_in_cycle = (next_in_cycle*width)%m) {
      float backup = input[next_in_cycle*tile_size+tid];
      __syncthreads();
      if (tid == 0) {
        done = atomicExch(finished+next_in_cycle, (int)1);
      }
      __syncthreads();
      if (!done) {
        input[next_in_cycle*tile_size+tid] = data;
      }
      data = backup;
    }
  }
}
#endif //_LIBMARSHAL_KERNEL_CU_
