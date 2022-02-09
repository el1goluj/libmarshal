//===--- cl_aos_astal.cl  - GPU in-place marshaling library    ----------===//
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
// Contributor: Juan Gómez Luna (el1goluj@uco.es)
//
//===---------------------------------------------------------------------===//
//
//  This file defines the OpenCL kernels of the libmarshal 
//
//===---------------------------------------------------------------------===//
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// Double check NVIDIA and SP flags have the same value as in ~/test/cl_unittest.cc and ~/src/cl_marshal.cc
#define NVIDIA 1
#define SP 1

#if SP
#define T float
#else
#define T double
#endif

#if NVIDIA
#define WARP_SIZE 32
#else
#define WARP_SIZE 64
#endif

// mymemset kernel
__kernel void mymemset (__global T *input) {
  input[get_global_id(0)] = 0.0f;
}

// limitations: tile_size * width cannot exceed maximal # of threads in
// a block allowed in the system
// convert a[height/tile_size][tile_size][width] to
// a[height/tile_size][width][tile_size]
// Launch height/tile_size blocks of tile_size*width threads
__kernel void BS_marshal (__global T *input, int tile_size, int width,
  __local T *store, int warp_size, int A) {
  int tidx = get_local_id(0);
  int m = width*tile_size-1;
  int bid = get_group_id(0);
#define ORIG 0
#if ORIG
  input += tile_size*width*bid;
  for (int i = tidx; i < tile_size*width; i+=get_local_size(0)) {
    int next = (i * tile_size)-m*(i/width);
    store[next] = input[i];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int i = tidx; i < tile_size*width; i+=get_local_size(0)) {
    input[i] = store[i];
  }
#else
  input += tile_size*width*bid;
  for (int j = bid; j < A; j += get_num_groups(0)){
    for (int i = tidx; i < tile_size*width; i+=get_local_size(0)) {
      int next = (i * tile_size)-m*(i/width);
      store[next] = input[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = tidx; i < tile_size*width; i+=get_local_size(0)) {
      input[i] = store[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    input += tile_size*width*get_num_groups(0);
  }
#endif
}

// Version with virtual-SIMD-units
__kernel void BS_marshal_vw (__global T *input, int tile_size, int width,
  volatile __local T *store, int warp_size, int A) {
  int m = width*tile_size-1;
  int group_id, tidx, warp_id, warps_group;
  // Recalculate IDs if virtual warp is used
#if !NVIDIA
  if (warp_size == 64){
    tidx = get_local_id(0) & 63;
    group_id = get_group_id(0);
    warp_id = get_local_id(0) >> 6;
    warps_group = get_local_size(0) >> 6;
  }
#endif
  if (warp_size == 32){
    tidx = get_local_id(0) & 31;
    group_id = get_group_id(0);
    warp_id = get_local_id(0) >> 5;
    warps_group = get_local_size(0) >> 5;
  }
  else if (warp_size == 16){
    tidx = get_local_id(0) & 15;
    group_id = get_group_id(0);
    warp_id = get_local_id(0) >> 4;
    warps_group = get_local_size(0) >> 4;
  }
  else if (warp_size == 8){
    tidx = get_local_id(0) & 7;
    group_id = get_group_id(0);
    warp_id = get_local_id(0) >> 3;
    warps_group = get_local_size(0) >> 3;
  }
  else if (warp_size == 4){
    tidx = get_local_id(0) & 3;
    group_id = get_group_id(0);
    warp_id = get_local_id(0) >> 2;
    warps_group = get_local_size(0) >> 2;
  }
  else if (warp_size == 2){
    tidx = get_local_id(0) & 1;
    group_id = get_group_id(0);
    warp_id = get_local_id(0) >> 1;
    warps_group = get_local_size(0) >> 1;
  }
  else if (warp_size == 1){
    tidx = get_local_id(0) & 0;
    group_id = get_group_id(0);
    warp_id = get_local_id(0) >> 0;
    warps_group = get_local_size(0) >> 0;
  }
  int bid = group_id * warps_group + warp_id;
#if ORIG
  input += tile_size*width*bid;
  store += (tile_size+1)*width*warp_id;
  for (int i = tidx; i < tile_size*width; i+=warp_size) {
    int next = (i * tile_size)-m*(i/width);
    store[next] = input[i];
  }
  for (int i = tidx; i < tile_size*width; i+=warp_size) {
    input[i] = store[i];
  }
#else
  input += tile_size*width*bid;
  store += tile_size*width*warp_id;
  for (int j = bid; j < A; j += get_num_groups(0)*warps_group){
    for (int i = tidx; i < tile_size*width; i+=warp_size) {
      int next = (i * tile_size)-m*(i/width);
      store[next] = input[i];
    }
    for (int i = tidx; i < tile_size*width; i+=warp_size) {
      input[i] = store[i];
    }
    input += tile_size*width*get_num_groups(0)*warps_group;
  }
#endif
}

// Optimized version for tile size == power of 2
// Padding shared memory is necessary to reduce conflicts
__kernel void BS_marshal_power2 (__global T *input, int lg2_tile_size, int width,
  __local T *store, int warp_size, int A) {
#define SHMT lg2_tile_size 
#define PAD(x) ((x)+((x)>>SHMT))
  int tidx = get_local_id(0);
  int m = (width<<SHMT)-1;
  int bid = get_group_id(0);
#if ORIG
  input += ((width*bid)<<SHMT); //tile_size*width*bid;
  for (int i = tidx; i < (width<<SHMT) ; i+=get_local_size(0)) {
    int next = (i << SHMT)-m*(i/width);
    store[PAD(next)] = input[i];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int i = tidx; i < (width<<SHMT); i+=get_local_size(0)) {
    input[i] = store[PAD(i)];
  }
#else
  input += ((width*bid)<<SHMT); //tile_size*width*bid;
  for (int j = bid; j < A; j += get_num_groups(0)){
    for (int i = tidx; i < (width<<SHMT) ; i+=get_local_size(0)) {
      int next = (i << SHMT)-m*(i/width);
      store[PAD(next)] = input[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = tidx; i < (width<<SHMT); i+=get_local_size(0)) {
      input[i] = store[PAD(i)];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    input += ((width*get_num_groups(0))<<SHMT); //tile_size*width*get_num_groups(0);
  }
#endif
}
#undef ORIG

// Converts input[A][a][B] to input[A][B][a]
// Launch A blocks of NR_THREADS threads
__kernel void transpose_010_PTTWAC(__global T *input, int A, 
  int a, int B, __local uint *finished, int R, int SHFT, int P) {
  int tidx = get_local_id(0);
  int m = a*B - 1;
  input += get_group_id(0)*a*B;

  int sh_sz = R * ((a * B + 31) / 32);
  sh_sz += (sh_sz >> 5) * P; // Padding each 32 locations (Number of banks)

#define PTTWAC_REMAP 1
#define P_IPT 0
#if !P_IPT
  for (int id = tidx ; id < sh_sz; id += get_local_size(0)) {
    finished[id] = 0;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
#endif
  for (;tidx < m; tidx += get_local_size(0)) {
    int next = (mul24(tidx, a))-m*(tidx/B);
#if P_IPT
    int prev = (mul24(tidx, B))-m*(tidx/a);
    if (next <= tidx || prev < tidx)
      continue;
    for (;next > tidx;next = (mul24(next, a))-m*(next/B))
      ;
    if (next !=tidx)
      continue;
    T data = input[tidx];
    for (next = (tidx * a)-m*(tidx/B);next > tidx;
        next = (next * a)-m*(next/B)) {
      T backup = input[next];
      input[next] = data;
      data = backup;
    }
    input[tidx] = data;
#else // P_IPT
    if (next != tidx) {
      T data1 = input[tidx];

      //// mask and flag_id////
#if PTTWAC_REMAP
      unsigned int mask = 1 << (tidx / sh_sz);
      unsigned int flag_id = tidx%sh_sz;
#else
      unsigned int mask = (1 << (tidx % 32));
      unsigned int flag_id = tidx >> SHFT;
      flag_id += (flag_id >> 5) * P;
#endif

      uint done = atom_or(finished+flag_id, 0);
      done = (done & mask);

      for (; done == 0; next = (next * a) - m*(next/B)) {
        T data2 = input[next];
#if PTTWAC_REMAP
        mask = 1 << (next / sh_sz);
        flag_id = next%sh_sz;
#else
        //// mask and flag_id////
        mask = (1 << (next % 32));
        flag_id = next >> SHFT;
        flag_id += (flag_id >> 5) * P;
#endif

        done = atom_or(finished+flag_id, mask);
        done = (done & mask);
        if (done == 0) {
          input[next] = data1;
        }
        data1 = data2;
      }
    }
#endif // P_IPT
#undef P_IPT
#undef PTTWAC_REMAP
  }
}

#if 0
// Version for InPar'2012 paper
// Transformation 100, or ABb to BAb
// limitations: b cannot exceed # of allowed threads in the system
// Launch A*B work-groups of b work-items
__kernel void transpose_100(__global T *input, 
    int A, int B, int b, __global int *finished) {
  int m = A*B-1;
  int tid = get_local_id(0);
  T data;
  for(int gid = get_group_id(0); gid < m; gid += get_num_groups(0)) {
    int next_in_cycle = (gid * A)-m*(gid/B);
    if (next_in_cycle == gid)
      continue;
#define P_IPT 0
#if P_IPT
    for (;next_in_cycle > gid;
      next_in_cycle = (next_in_cycle*A)-m*(next_in_cycle/B))
      ;
    if (next_in_cycle !=gid)
      continue;
    data = input[gid*b+tid];
    for (next_in_cycle = (gid * A)-m*(gid/B);
      next_in_cycle > gid;
      next_in_cycle = (next_in_cycle*A)-m*(next_in_cycle/B)) {
      T backup = input[next_in_cycle*b+tid];
      input[next_in_cycle*b+tid] = data;
      data = backup;
    }
    input[gid*b+tid] = data;
#else
    __local int done;
    data = input[gid*b+tid];
    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
    if (tid == 0)
      done = atom_or(finished+gid, (int)0); //make sure the read is not cached 
    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);

    for (;done == 0; next_in_cycle = (next_in_cycle*A)-m*(next_in_cycle/B)) {
      T backup = input[next_in_cycle*b+tid];
      barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
      if (tid == 0) {
        done = atom_xchg(finished+next_in_cycle, (int)1);
      }
      barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
      if (!done) {
        input[next_in_cycle*b+tid] = data;
      }
      data = backup;
    }
#endif
#undef P_IPT
  }
}
#endif

// Transformation 100 - Shared memory tiling
// Used by both transformation 0100 and 100 (see below)
// Assumes:
//  get_local_size(0) == wavefront size;
//  get_local_size(1) == number of warps
#define P_IPT 0
#if NVIDIA
#define LOCALMEM_TILING 0 // 1 - Local memory tiling; 0 - Register tiling
#else
#define LOCALMEM_TILING 1 
#endif
#if LOCALMEM_TILING
void _transpose_100(__global T *input,
    int A, int B, int b, __global int *finished, volatile __local T *data,
    volatile __local T *backup, volatile __local int *done, const int warp_size) {
#else
void _transpose_100(__global T *input,
    int A, int B, int b, __global int *finished, volatile __local int *done, const int warp_size) {
#endif
  int m = A*B-1;
  int tid = get_local_id(0);
  int group_id = get_group_id(0);
  int warp_id = get_local_id(1);
  int warps_group = get_local_size(1);

  // Recalculate IDs if virtual warp is used
#if !NVIDIA
  if (warp_size == 32){
    tid = get_local_id(0) & 31;
    int vwarps_in_warp = WARP_SIZE / warp_size;
    warps_group = warps_group * vwarps_in_warp;
    int vwarp_id = get_local_id(0) >> 5;
    warp_id = warp_id * vwarps_in_warp + vwarp_id;
  }
#endif
  if (warp_size == 16){
    tid = get_local_id(0) & 15;
    int vwarps_in_warp = WARP_SIZE / warp_size;
    warps_group = warps_group * vwarps_in_warp;
    int vwarp_id = get_local_id(0) >> 4;
    warp_id = warp_id * vwarps_in_warp + vwarp_id;
  }
  else if (warp_size == 8){
    tid = get_local_id(0) & 7;
    int vwarps_in_warp = WARP_SIZE / warp_size;
    warps_group = warps_group * vwarps_in_warp;
    int vwarp_id = get_local_id(0) >> 3;
    warp_id = warp_id * vwarps_in_warp + vwarp_id;
  }
  else if (warp_size == 4){
    tid = get_local_id(0) & 3;
    int vwarps_in_warp = WARP_SIZE / warp_size;
    warps_group = warps_group * vwarps_in_warp;
    int vwarp_id = get_local_id(0) >> 2;
    warp_id = warp_id * vwarps_in_warp + vwarp_id;
  }
  else if (warp_size == 2){
    tid = get_local_id(0) & 1;
    int vwarps_in_warp = WARP_SIZE / warp_size;
    warps_group = warps_group * vwarps_in_warp;
    int vwarp_id = get_local_id(0) >> 1;
    warp_id = warp_id * vwarps_in_warp + vwarp_id;
  }
  else if (warp_size == 1){
    tid = get_local_id(0) & 0;
    int vwarps_in_warp = WARP_SIZE / warp_size;
    warps_group = warps_group * vwarps_in_warp;
    int vwarp_id = get_local_id(0) >> 0;
    warp_id = warp_id * vwarps_in_warp + vwarp_id;
  }

  for(int gid = group_id * warps_group + warp_id; gid < m;
    gid += get_num_groups(0) * warps_group) {
    int next_in_cycle = (gid * A)-m*(gid/B);
    if (next_in_cycle == gid)
      continue;

#if P_IPT
    for (;next_in_cycle > gid;
      next_in_cycle = (next_in_cycle*A)-m*(next_in_cycle/B))
      ;
    if (next_in_cycle !=gid)
      continue;
    for(int i = tid; i < b; i += warp_size){
      data[warp_id*b+i] = input[gid*b+i];
    }
    for (next_in_cycle = (gid * A)-m*(gid/B);
      next_in_cycle > gid;
      next_in_cycle = (next_in_cycle*A)-m*(next_in_cycle/B)) {
      for(int i = tid; i < b; i += warp_size){
        backup[warp_id*b+i] = input[next_in_cycle*b+i];
      }
      for(int i = tid; i < b; i += warp_size){
        input[next_in_cycle*b+i] = data[warp_id*b+i];
      }
      for(int i = tid; i < b; i += warp_size){
        data[warp_id*b+i] = backup[warp_id*b+i];
      }
    }
    for(int i = tid; i < b; i += warp_size){
      input[gid*b+i] = data[warp_id*b+i];
    }

#else

#if NVIDIA
#define N 16 // Narrowing: 16 for NVIDIA, 32 for AMD
#else
#define N 32
#endif
#if LOCALMEM_TILING
    for(int i = tid; i < b; i += warp_size){
      data[warp_id*b+i] = input[gid*b+i];
    }
#else
#if SP
    float data1, data2, data3, data4, data5, data6;
    int i = tid;
    if(i < b) data1 = input[gid*b+i];
    i += warp_size;
    if(i < b) data2 = input[gid*b+i];
    i += warp_size;
    if(i < b) data3 = input[gid*b+i];
    i += warp_size;
    if(i < b) data4 = input[gid*b+i];
    i += warp_size;
    if(i < b) data5 = input[gid*b+i];
    i += warp_size;
    if(i < b) data6 = input[gid*b+i];
#else
    T data1, data2, data3;
    int i = tid;
    if(i < b) data1 = input[gid*b+i];
    i += warp_size;
    if(i < b) data2 = input[gid*b+i];
    i += warp_size;
    if(i < b) data3 = input[gid*b+i];
#endif
#endif
    if (tid == 0){
      //make sure the read is not cached 
      //done[warp_id] = atom_or(finished+gid, (int)0); 
      // Narrowing
      unsigned int flag_id = gid / N;
      unsigned int mask = 1 << (gid % N);
      unsigned int flag_read = atom_or(finished+flag_id, (int)0); //make sure the read is not cached 
      done[warp_id] = flag_read & mask;
    }

    for (;done[warp_id] == 0; 
        next_in_cycle = (next_in_cycle*A)-m*(next_in_cycle/B)) {
#if LOCALMEM_TILING
      for(int i = tid; i < b; i += warp_size){
        backup[warp_id*b+i] = input[next_in_cycle*b+i];
      }
#else
#if SP
      T backup1, backup2, backup3, backup4, backup5, backup6;
      i = tid;
      if(i < b) backup1 = input[next_in_cycle*b+i];
      i += warp_size;
      if(i < b) backup2 = input[next_in_cycle*b+i];
      i += warp_size;
      if(i < b) backup3 = input[next_in_cycle*b+i];
      i += warp_size;
      if(i < b) backup4 = input[next_in_cycle*b+i];
      i += warp_size;
      if(i < b) backup5 = input[next_in_cycle*b+i];
      i += warp_size;
      if(i < b) backup6 = input[next_in_cycle*b+i];
#else
      T backup1, backup2, backup3;
      i = tid;
      if(i < b) backup1 = input[next_in_cycle*b+i];
      i += warp_size;
      if(i < b) backup2 = input[next_in_cycle*b+i];
      i += warp_size;
      if(i < b) backup3 = input[next_in_cycle*b+i];
#endif
#endif
      if (tid == 0) {
        //done[warp_id] = atom_xchg(finished+next_in_cycle, (int)1);
        // Narrowing
        unsigned int flag_id = next_in_cycle / N;
        unsigned int mask = 1 << (next_in_cycle % N);
        unsigned int flag_read = atom_or(finished+flag_id, mask); //make sure the read is not cached 
        done[warp_id]  = flag_read & mask;
      }
      if (!done[warp_id]) {
#if LOCALMEM_TILING
        for(int i = tid; i < b; i += warp_size){
          input[next_in_cycle*b+i] = data[warp_id*b+i];
        }
#else
        i = tid;
        if(i < b) input[next_in_cycle*b+i] = data1;
        i += warp_size;
        if(i < b) input[next_in_cycle*b+i] = data2;
        i += warp_size;
        if(i < b) input[next_in_cycle*b+i] = data3;
#if SP
        i += warp_size;
        if(i < b) input[next_in_cycle*b+i] = data4;
        i += warp_size;
        if(i < b) input[next_in_cycle*b+i] = data5;
        i += warp_size;
        if(i < b) input[next_in_cycle*b+i] = data6;
#endif
#endif
      }
#if LOCALMEM_TILING
      for(int i = tid; i < b; i += warp_size){
        data[warp_id*b+i] = backup[warp_id*b+i];
      }
#else
      i = tid;
      if(i < b) data1 = backup1;
      i += warp_size;
      if(i < b) data2 = backup2;
      i += warp_size;
      if(i < b) data3 = backup3;
#if SP 
      i += warp_size;
      if(i < b) data4 = backup4;
      i += warp_size;
      if(i < b) data5 = backup5;
      i += warp_size;
      if(i < b) data6 = backup6;
#endif
#endif
    }
#endif
  }
}

// Block-centric version
#if LOCALMEM_TILING
void _transpose_100_b(__global T *input,
    int A, int B, int b, __global int *finished, volatile __local T *data,
    volatile __local T *backup, volatile __local int *done, const int warp_size) {
#else
void _transpose_100_b(__global T *input,
    int A, int B, int b, __global int *finished, volatile __local int *done, const int warp_size) {
#endif
  int m = A*B-1;
  int tid = get_local_id(0);
  int group_id = get_group_id(0);
  int bid = group_id;
  int num_groups = get_num_groups(0);
  int group_size = get_local_size(0);

  for(int gid = bid; gid < m; gid += num_groups) {
    int next_in_cycle = (gid * A)-m*(gid/B);
    if (next_in_cycle == gid)
      continue;

#if LOCALMEM_TILING
    for(int i = tid; i < b; i += group_size){
      data[i] = input[gid*b+i];
    }
#else
#if SP
    float data1, data2, data3, data4, data5, data6;
    int i = tid;
    if(i < b) data1 = input[gid*b+i];
    i += group_size;
    if(i < b) data2 = input[gid*b+i];
    i += group_size;
    if(i < b) data3 = input[gid*b+i];
    i += group_size;
    if(i < b) data4 = input[gid*b+i];
    i += group_size;
    if(i < b) data5 = input[gid*b+i];
    i += group_size;
    if(i < b) data6 = input[gid*b+i];
#else
    T data1, data2, data3;
    int i = tid;
    if(i < b) data1 = input[gid*b+i];
    i += group_size;
    if(i < b) data2 = input[gid*b+i];
    i += group_size;
    if(i < b) data3 = input[gid*b+i];
#endif
#endif
    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
    if (tid == 0){
      //make sure the read is not cached 
      //done[0] = atom_or(finished+gid, (int)0); 
      // Narrowing
      unsigned int flag_id = gid / N;
      unsigned int mask = 1 << (gid % N);
      unsigned int flag_read = atom_or(finished+flag_id, (int)0); //make sure the read is not cached 
      done[0] = flag_read & mask;
    }
    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
    for (;done[0] == 0; 
        next_in_cycle = (next_in_cycle*A)-m*(next_in_cycle/B)) {
#if LOCALMEM_TILING
      for(int i = tid; i < b; i += group_size){
        backup[i] = input[next_in_cycle*b+i];
      }
#else
#if SP
      float backup1, backup2, backup3, backup4, backup5, backup6;
      i = tid;
      if(i < b) backup1 = input[next_in_cycle*b+i];
      i += group_size;
      if(i < b) backup2 = input[next_in_cycle*b+i];
      i += group_size;
      if(i < b) backup3 = input[next_in_cycle*b+i];
      i += group_size;
      if(i < b) backup4 = input[next_in_cycle*b+i];
      i += group_size;
      if(i < b) backup5 = input[next_in_cycle*b+i];
      i += group_size;
      if(i < b) backup6 = input[next_in_cycle*b+i];
#else
      T backup1, backup2, backup3;
      i = tid;
      if(i < b) backup1 = input[next_in_cycle*b+i];
      i += group_size;
      if(i < b) backup2 = input[next_in_cycle*b+i];
      i += group_size;
      if(i < b) backup3 = input[next_in_cycle*b+i];
#endif
#endif
      barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
      if (tid == 0) {
        //done[0] = atom_xchg(finished+next_in_cycle, (int)1);
        // Narrowing
        unsigned int flag_id = next_in_cycle / N;
        unsigned int mask = 1 << (next_in_cycle % N);
        unsigned int flag_read = atom_or(finished+flag_id, mask); //make sure the read is not cached 
        done[0]  = flag_read & mask;
      }
      barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
      if (!done[0]) {
#if LOCALMEM_TILING
        for(int i = tid; i < b; i += group_size){
          input[next_in_cycle*b+i] = data[i];
        }
#else
        i = tid;
        if(i < b) input[next_in_cycle*b+i] = data1;
        i += group_size;
        if(i < b) input[next_in_cycle*b+i] = data2;
        i += group_size;
        if(i < b) input[next_in_cycle*b+i] = data3;
#if SP
        i += group_size;
        if(i < b) input[next_in_cycle*b+i] = data4;
        i += group_size;
        if(i < b) input[next_in_cycle*b+i] = data5;
        i += group_size;
        if(i < b) input[next_in_cycle*b+i] = data6;
#endif
#endif
      }
#if LOCALMEM_TILING
      for(int i = tid; i < b; i += group_size){
        data[i] = backup[i];
      }
#else
      i = tid;
      if(i < b) data1 = backup1;
      i += group_size;
      if(i < b) data2 = backup2;
      i += group_size;
      if(i < b) data3 = backup3;
#if SP
      i += group_size;
      if(i < b) data4 = backup4;
      i += group_size;
      if(i < b) data5 = backup5;
      i += group_size;
      if(i < b) data6 = backup6;
#endif
#endif
    }
  }
}

// Transformation 100 
#if LOCALMEM_TILING
__kernel void transpose_100(__global T *input,
    int A, int B, int b, __global int *finished, volatile __local T *data,
    volatile __local T *backup, int warp_size, volatile __local int *done) {
#if P_IPT
    _transpose_100(input, A, B, b, finished, data, backup, 0, WARP_SIZE);
#else
    _transpose_100(input, A, B, b, finished, data, backup, done, warp_size);
#endif
}
#else
// With register tiling
__kernel void transpose_100(__global T *input,
    int A, int B, int b, __global int *finished, int warp_size, volatile __local int *done) {
    _transpose_100(input, A, B, b, finished, done, warp_size);
}
#endif

// Transformation 0100, or AaBb to ABab
#if LOCALMEM_TILING
__kernel void transpose_0100(__global T *input,
    int A, int B, int b, __global int *finished, volatile __local T *data,
    volatile __local T *backup, int warp_size, volatile __local int *done) {
  // for supporting transformation 0100
  finished += get_group_id(2) * A * B;
  input += get_group_id(2) * A * B * b;
#if P_IPT
  _transpose_100(input, A, B, b, finished, data, backup, 0, WARP_SIZE);
#else
  _transpose_100(input, A, B, b, finished, data, backup, done, warp_size);
#endif
}
#else
// With register tiling
__kernel void transpose_0100(__global T *input,
    int A, int B, int b, __global int *finished, int warp_size, volatile __local int *done) {
  // for supporting transformation 0100
  finished += get_group_id(2) * A * B;
  input += get_group_id(2) * A * B * b;
  _transpose_100(input, A, B, b, finished, done, warp_size);
}
#endif

#undef P_IPT

// Transformation 100 - Block-centric with shared memory tiling
#if LOCALMEM_TILING
__kernel void transpose_100_b(__global T *input,
    int A, int B, int b, __global int *finished, volatile __local T *data,
    volatile __local T *backup, int warp_size, volatile __local int *done) {
    _transpose_100_b(input, A, B, b, finished, data, backup, done, warp_size);

}
#else
// With register tiling
__kernel void transpose_100_b(__global T *input,
    int A, int B, int b, __global int *finished, int warp_size, volatile __local int *done) {
    _transpose_100_b(input, A, B, b, finished, done, warp_size);

}
#endif

// Transformation 0100, or AaBb to ABab - Block-centric with shared memory tiling
#if LOCALMEM_TILING
__kernel void transpose_0100_b(__global T *input,
    int A, int B, int b, __global int *finished, volatile __local T *data,
    volatile __local T *backup, int warp_size, volatile __local int *done) {
  // for supporting transformation 0100
  finished += get_group_id(2) * A * B;
  input += get_group_id(2) * A * B * b;
  _transpose_100_b(input, A, B, b, finished, data, backup, done, warp_size);
}
#else
// With register tiling
__kernel void transpose_0100_b(__global T *input,
    int A, int B, int b, __global int *finished, int warp_size, volatile __local int *done) {
  // for supporting transformation 0100
  finished += get_group_id(2) * A * B;
  input += get_group_id(2) * A * B * b;
  _transpose_100_b(input, A, B, b, finished, done, warp_size);
}
#endif

#undef LOCALMEM_TILING

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#if NVIDIA
#define REGS 8 //16
#define ATOM 0
#else
#define REGS 64
#define ATOM 1
#endif
__kernel void padding( __global T *matrix,
    int x_size,
    int pad_size,
    int y_size,
    int rows,
    volatile __global unsigned int *flags)
{
  const int matrix_size = y_size * pad_size;
  const int matrix_size_align = (matrix_size + get_local_size(0) - 1) / get_local_size(0) * get_local_size(0);
  const int num_flags = matrix_size / (get_local_size(0) * REGS);
  // Dynamic allocation of runtime workgroup id
  __local int gid_;
  if (get_local_id(0) == 0) gid_ = atom_add(&flags[num_flags + 1], 1);
  barrier(CLK_LOCAL_MEM_FENCE);
  int my_s = gid_;

  // Declare on-chip memory
  T reg[REGS];
  int pos = matrix_size_align - 1 - (my_s * REGS * get_local_size(0) + get_local_id(0));
  int my_s_row = pos / pad_size;
  int my_x = pos % pad_size;
  int pos2 = my_s_row * x_size + my_x;
  // Load in on-chip memory
  #pragma unroll
  for (int j = 0; j < REGS; j++){
    if (pos2 >= 0 && my_x < x_size) reg[j] = matrix[pos2];
    else reg[j] = 0;
    pos -= get_local_size(0);
    my_s_row = pos / pad_size;
    my_x = pos % pad_size;
    pos2 = my_s_row * x_size + my_x;
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  // Set global synch
#if ATOM
  if (get_local_id(0) == 0){
    while (atom_or(&flags[my_s], 0) == 0){}
    atom_or(&flags[my_s + 1], 1);
  }
#else
  if (get_local_id(0) == 0){
    while (flags[my_s] == 0){}
    flags[my_s + 1] = 1;
  }
#endif

  barrier(CLK_GLOBAL_MEM_FENCE);

  pos = matrix_size_align - 1 - (my_s * REGS * get_local_size(0) + get_local_id(0));
  // Store to global memory 
  #pragma unroll
  for (int j = 0; j < REGS; j++){
    if (pos >= 0 && pos < matrix_size) matrix[pos] = reg[j];
    pos -= get_local_size(0);
  }
}

#undef REGS
#undef ATOM
#if NVIDIA
#define REGS 8 //16
#define ATOM 0
#else
#define REGS 64
#define ATOM 1
#endif
__kernel void unpadding( __global T *matrix,
    int x_size,
    int pad_size,
    int y_size,
    int rows,
    volatile __global unsigned int *flags)
{
  const int num_flags = (y_size * pad_size) / (get_local_size(0) * REGS);
  // Dynamic allocation of runtime workgroup id
  __local int gid_;
  if (get_local_id(0) == 0) gid_ = atom_add(&flags[num_flags + 1], 1);
  barrier(CLK_LOCAL_MEM_FENCE);
  const int my_s = gid_;

  // Declare on-chip memory
  T reg[REGS];
  int pos = my_s * REGS * get_local_size(0) + get_local_id(0);
  // Load in on-chip memory
  #pragma unroll
  for (int j = 0; j < REGS; j++){
    if (pos < y_size * pad_size) reg[j] = matrix[pos];
    pos += get_local_size(0);
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  // Set global synch
#if ATOM
  if (get_local_id(0) == 0){
    while (atom_or(&flags[my_s], 0) == 0){}
    atom_or(&flags[my_s + 1], 1);
  }
#else
  if (get_local_id(0) == 0){
    while (flags[my_s] == 0){}
    flags[my_s + 1] = 1;
  }
#endif

  barrier(CLK_GLOBAL_MEM_FENCE);

  pos = my_s * REGS * get_local_size(0) + get_local_id(0);
  int my_s_row = pos / pad_size;
  int my_x = pos % pad_size;
  // Store to global memory 
  //#pragma unroll
  for (int j = 0; j < REGS; j++){
    if (my_x < x_size && pos < y_size * pad_size) matrix[my_s_row * x_size + my_x] = reg[j];
    pos += get_local_size(0);
    my_s_row = pos / pad_size;
    my_x = pos % pad_size;
  }
  // Zeros at the end
  if ((my_s + 1) * REGS * get_local_size(0) >= y_size * x_size)
    for (int j = y_size * x_size + get_local_id(0); j < y_size * pad_size; j += get_local_size(0)){
      matrix[j] = 0.0f;
    }
}

