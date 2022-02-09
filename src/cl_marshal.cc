//===--- cl_marshal.cc - GPU in-place marshaling library        ----------===//
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
// Author: Juan Gómez-Luna (gomezlun@illinois.edu, el1goluj@uco.es)
//
//===---------------------------------------------------------------------===//
//
//  This file defines the interface of the libmarshal 
//
//===---------------------------------------------------------------------===//

#include <cstdlib>
#include <cassert>
#include <iostream>
#include "cl_profile.h"
#include "cl_marshal.h"
#include "local_cl.hpp"
#include "embd.hpp"
#include "singleton.hpp"
#include "plan.hpp"
#include <math.h>
#include <stdio.h>

// Double check NVIDIA and SP flags have the same value as in ~/test/cl_unittest.cc
#define NVIDIA 1 // NVIDIA or other (e.g., AMD)
// SP = 1 -> Single precision; SP = 0 -> Double precision
#define SP 1

#if SP
#define T float
#else
#define T double
#endif

#if NVIDIA
// Shared memory in Fermi and Kepler is 48 KB, i.e., 12288 SP or 6144 DP.
#if SP
#define MAX_MEM 12288 // Use 4096 for other devices.
#else
#define MAX_MEM 6144 // Use 2048 for other devices.
#endif
#else
// Local memory in AMD devices is 32 KB, i.e., 8192 SP or 4096 DP.
#if SP
#define MAX_MEM 8192 // Use 4096 for other devices.
#else
#define MAX_MEM 4096 // Use 2048 for other devices.
#endif
#endif

// Default work-group sizes
#if NVIDIA
#define NR_THREADS 1024
#else
#define NR_THREADS 256
#endif

namespace {
class MarshalProg {
 public:
  ~MarshalProg() {}
  MarshalProg(void): program(NULL), context_(NULL) {
    libmarshal::file source_file("cl/cl_aos_asta.cl");
    std::istream &in = source_file.istream();
    source_code_ = std::string(std::istreambuf_iterator<char>(in),
	(std::istreambuf_iterator<char>()));
    source_ = cl::Program::Sources(1,
      std::make_pair(source_code_.c_str(), source_code_.length()+1));
  }
  cl_uint GetCtxRef(void) const {
    cl_uint rc = 0;
    if (context_) {
      clGetContextInfo(context_,
          CL_CONTEXT_REFERENCE_COUNT, sizeof(cl_uint), &rc, NULL);
    }
    return rc;
  }
  bool Init(cl_context clcontext) {
    if (context_ != clcontext) { //Trigger recompilation
      context_ = clcontext;
      cl::Context context(clcontext);
      clRetainContext(clcontext);
      cl_uint old_ref = GetCtxRef();
      program = cl::Program(context, source_);
      if (CL_SUCCESS != program.build())
        return true;
      // On Apple and ATI, build a program has an implied clRetainContext.
      // To avoid leak, release the additional lock. Note: Not thread-safe
      if (old_ref != GetCtxRef())
        clReleaseContext(clcontext);
    }
    return false;
  }
  void Finalize(void) {
    program = cl::Program();
    context_ = NULL;
  }
  cl::Program program;
 private:
  cl::Program::Sources source_;
  std::string source_code_;
  cl_context context_;
};
typedef Singleton<MarshalProg> MarshalProgSingleton;
}
extern "C" void cl_marshal_finalize(void) {
  MarshalProg *marshalprog = MarshalProgSingleton::Instance();
  marshalprog->Finalize();
}

#define IS_POW2(x) (x && !(x &( x- 1)))
// v: 32-bit word input to count zero bits on right
static int count_zero_bits(unsigned int v) {
  unsigned int c = 32; // c will be the number of zero bits on the right
  v &= -signed(v);
  if (v) c--;
  if (v & 0x0000FFFF) c -= 16;
  if (v & 0x00FF00FF) c -= 8;
  if (v & 0x0F0F0F0F) c -= 4;
  if (v & 0x33333333) c -= 2;
  if (v & 0x55555555) c -= 1;
  return c;
}

// Transformation 010, or AaB to ABa
extern "C" bool cl_transpose_010_bs(cl_command_queue cl_queue,
    cl_mem src, int A, int a, int B, cl_ulong *elapsed_time) {
  // Standard preparation of invoking a kernel
  cl::CommandQueue queue = cl::CommandQueue(cl_queue);
  Profiling prof(queue, "AOS-ASTA BS");
  cl::Buffer buffer = cl::Buffer(src);
  clRetainMemObject(src);
  cl::Context context;
  if(buffer.getInfo(CL_MEM_CONTEXT, &context) != CL_SUCCESS)
    return true;
  MarshalProg *marshalprog = MarshalProgSingleton::Instance();
  marshalprog->Init(context());

  int sh_sz = a*B;
  cl::Kernel kernel(marshalprog->program,
#if SP
      sh_sz<256 ? "BS_marshal_vw" : "BS_marshal");
#else 
      sh_sz<128 ? "BS_marshal_vw" : "BS_marshal");
#endif
  if (CL_SUCCESS != kernel.setArg(0, buffer))
    return true;
  cl_int err = kernel.setArg(1, a);
  if (err != CL_SUCCESS)
    return true;
  err = kernel.setArg(2, B);
  if (err != CL_SUCCESS)
    return true;

  // Select work-group size and virtual-SIMD-unit size
  int nr_threads;
  int warp_size;
#if NVIDIA
#if SP
  // Block size (according to tuning tests)
  if (sh_sz <= 792) nr_threads = 128;
  else if (sh_sz > 792 && sh_sz <= 1056) nr_threads = 192;
  else if (sh_sz > 1056 && sh_sz <= 1680) nr_threads = 256;
  else if (sh_sz > 1680 && sh_sz <= 2375) nr_threads = 384;
  else if (sh_sz > 2375 && sh_sz <= 3552) nr_threads = 512;
  else nr_threads = 1024;
  // Virtual warps (according to tuning tests)
  if (sh_sz <= 1) warp_size = 1;
  else if (sh_sz > 1 && sh_sz <= 6) warp_size = 2;
  else if (sh_sz > 6 && sh_sz <= 12) warp_size = 4;
  else if (sh_sz > 12 && sh_sz <= 24) warp_size = 8;
  else if (sh_sz > 24 && sh_sz <= 48) warp_size = 16;
  else warp_size = 32;
#else 
  // Block size (according to tuning tests)
  if (sh_sz <= 396) nr_threads = 128;
  else if (sh_sz > 396 && sh_sz <= 528) nr_threads = 192;
  else if (sh_sz > 528 && sh_sz <= 840) nr_threads = 256;
  else if (sh_sz > 840 && sh_sz <= 1187) nr_threads = 384;
  else if (sh_sz > 1187 && sh_sz <= 1776) nr_threads = 512;
  else nr_threads = 1024;
  // Virtual warps (according to tuning tests)
  if (sh_sz <= 1) warp_size = 1;
  else if (sh_sz > 1 && sh_sz <= 3) warp_size = 2;
  else if (sh_sz > 3 && sh_sz <= 6) warp_size = 4;
  else if (sh_sz > 6 && sh_sz <= 12) warp_size = 8;
  else if (sh_sz > 12 && sh_sz <= 24) warp_size = 16;
  else warp_size = 32;
#endif
#else
#if SP
  // Block size (according to tuning tests)
  if (sh_sz <= 256) nr_threads = 64;
  else if (sh_sz > 256 && sh_sz <= 384) nr_threads = 128;
  else if (sh_sz > 384 && sh_sz <= 576) nr_threads = 192;
  else nr_threads = 256;
  // Virtual warps (according to tuning tests)
  if (sh_sz <= 1) warp_size = 1;
  else if (sh_sz > 1 && sh_sz <= 2) warp_size = 2;
  else if (sh_sz > 2 && sh_sz <= 12) warp_size = 4;
  else if (sh_sz > 12 && sh_sz <= 24) warp_size = 8;
  else if (sh_sz > 24 && sh_sz <= 48) warp_size = 16;
  else if (sh_sz > 48 && sh_sz <= 96) warp_size = 32;
  else warp_size = 64;
#else
  // Block size (according to tuning tests)
  if (sh_sz <= 128) nr_threads = 64;
  else if (sh_sz > 128 && sh_sz <= 192) nr_threads = 128;
  else if (sh_sz > 192 && sh_sz <= 288) nr_threads = 192;
  else nr_threads = 256;
  // Virtual warps (according to tuning tests)
  if (sh_sz <= 1) warp_size = 1;
  else if (sh_sz > 1 && sh_sz <= 2) warp_size = 2;
  else if (sh_sz > 2 && sh_sz <= 6) warp_size = 4;
  else if (sh_sz > 6 && sh_sz <= 12) warp_size = 8;
  else if (sh_sz > 12 && sh_sz <= 24) warp_size = 16;
  else if (sh_sz > 24 && sh_sz <= 48) warp_size = 32;
  else warp_size = 64;
#endif
#endif

  int warps = nr_threads / warp_size;
#if SP
  err = kernel.setArg(3, sh_sz<256?(warps*B*a)*sizeof(cl_float):(B*a*sizeof(cl_float)), NULL);
#else
  err = kernel.setArg(3, sh_sz<128?(warps*B*a)*sizeof(cl_double):(B*a*sizeof(cl_double)), NULL);
#endif
  err |= kernel.setArg(4, warp_size);
  err |= kernel.setArg(5, A);
  if (err != CL_SUCCESS)
    return true;

  // NDRange and kernel call
#if SP
  if (sh_sz < 256){
#else
  if (sh_sz < 128){
#endif
#if PRINT
    std::cerr << "nr_threads_vwarp = " << warp_size << "\t"; // Print warp_size
#endif
    cl::NDRange global(std::min((A/warps+1)*nr_threads, (8192/warps+1)*nr_threads)), local(nr_threads);
    err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local, NULL,
      prof.GetEvent());
    if (err != CL_SUCCESS)
      return true;
  }
  else{
#if PRINT
    std::cerr << "nr_threads =  " << nr_threads << "\t"; // Print nr_threads
#endif
    cl::NDRange global(std::min(A*nr_threads, 8192*nr_threads)), local(nr_threads);
    err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local, NULL,
      prof.GetEvent());
    if (err != CL_SUCCESS)
      return true;
  }

#ifdef LIBMARSHAL_OCL_PROFILE
  if (elapsed_time) {
    *elapsed_time += prof.Report();
  } else {
    prof.Report(A*a*B*sizeof(T)*2);
  }
#endif
  return false;
}

// Transformation 010, or AaB to ABa
extern "C" bool cl_transpose_010_pttwac(cl_command_queue cl_queue,
    cl_mem src, int A, int a, int B, cl_ulong *elapsed_time, int R, int P) {
  // Standard preparation of invoking a kernel
  cl::CommandQueue queue = cl::CommandQueue(cl_queue);
  Profiling prof(queue, "AOS-ASTA PTTWAC");
  cl::Buffer buffer = cl::Buffer(src);
  clRetainMemObject(src);
  cl::Context context;
  if(buffer.getInfo(CL_MEM_CONTEXT, &context) != CL_SUCCESS)
    return true;
  MarshalProg *marshalprog = MarshalProgSingleton::Instance();
  marshalprog->Init(context());

  cl::Kernel kernel(marshalprog->program, "transpose_010_PTTWAC");
  if (CL_SUCCESS != kernel.setArg(0, buffer))
    return true;
  cl_int err = kernel.setArg(1, A);
  if (err != CL_SUCCESS)
    return true;
  err |= kernel.setArg(2, a);
  err = kernel.setArg(3, B);
  if (err != CL_SUCCESS)
    return true;

  int sh_sz = R * ((a*B+31)/32);
  sh_sz += (sh_sz >> 5) * P;
  err = kernel.setArg(4, sh_sz*sizeof(cl_uint), NULL);
  if (err != CL_SUCCESS)
    return true;
  err = kernel.setArg(5, R);
  if (err != CL_SUCCESS)
    return true;
  err = kernel.setArg(6, (int)(5 - log2(R)));
  if (err != CL_SUCCESS)
    return true;
  err = kernel.setArg(7, P);
  if (err != CL_SUCCESS)
    return true;

  // NDRange and kernel call
#if PRINT
  std::cerr << "NR_THREADS = " << NR_THREADS << "\t"; // Print nr_threads
#endif
  cl::NDRange global(A*NR_THREADS), local(NR_THREADS);
  err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local, NULL,
    prof.GetEvent());
  if (err != CL_SUCCESS)
    return true;
#ifdef LIBMARSHAL_OCL_PROFILE
  if (elapsed_time) {
    *elapsed_time += prof.Report();
  } else {
    prof.Report(A*a*B*sizeof(T)*2);
  }
#endif
  return false;
}

// Generic transformation 0100, or AaBb to ABab. Used by both
// transformation 100 and 0100.
bool _cl_transpose_0100(cl_command_queue cl_queue,
    cl_mem src, int A, int a, int B, int b, cl_ulong *elapsed_time) {
    // Standard preparation of invoking a kernel
  cl::CommandQueue queue = cl::CommandQueue(cl_queue);
  Profiling prof(queue, "Transpose 0100/100 PTTWAC");
  cl::Buffer buffer = cl::Buffer(src);
  clRetainMemObject(src);
  cl::Context context;
  if(buffer.getInfo(CL_MEM_CONTEXT, &context) != CL_SUCCESS)
    return true;
  MarshalProg *marshalprog = MarshalProgSingleton::Instance();
  marshalprog->Init(context());

  cl_int *finished = (cl_int *)calloc(sizeof(cl_int), A*a*B);
  cl_int err;
  cl::Buffer d_finished = cl::Buffer(context, CL_MEM_READ_WRITE,
      sizeof(cl_int)*A*a*B, NULL, &err);
  if (err != CL_SUCCESS)
    return true;
  err = queue.enqueueWriteBuffer(d_finished, CL_TRUE, 0,
      sizeof(cl_int)*A*a*B, finished);
  free(finished);
  if (err != CL_SUCCESS)
    return true;

  cl::Kernel kernel(marshalprog->program, 
#if SP
    b<192?(A==1?"transpose_100":"transpose_0100"):(A==1?"transpose_100_b":"transpose_0100_b"));
#else
    b<96?(A==1?"transpose_100":"transpose_0100"):(A==1?"transpose_100_b":"transpose_0100_b"));
#endif

  err = kernel.setArg(0, buffer);
  if (err != CL_SUCCESS)
    return true;
  err = kernel.setArg(1, a);
  if (err != CL_SUCCESS)
    return true;
  err = kernel.setArg(2, B);
  if (err != CL_SUCCESS)
    return true;
  err = kernel.setArg(3, b);
  if (err != CL_SUCCESS)
    return true;
  err = kernel.setArg(4, d_finished);
  if (err != CL_SUCCESS)
    return true;

  // Select work-group size and virtual-SIMD-unit size
  // This version uses shared memory tiling
#if NVIDIA
#define WARPS 4
#define WARP_SIZE 32
#if SP
  // Virtual warps (according to tuning tests)
  int v_warp_size;
  if (b <= 1) v_warp_size = 4;
  else if (b > 1 && b <= 2) v_warp_size = 4;
  else if (b > 2 && b <= 4) v_warp_size = 4;
  else if (b > 4 && b <= 24) v_warp_size = 8;
  else if (b > 24 && b <= 48) v_warp_size = 16;
  else v_warp_size = 32;
  // Block size (according to tuning tests)
  int block_size;
  if (b <= 511) block_size = 128;                  
  else if (b > 511 && b <= 608) block_size = 192;  
  else if (b > 608 && b <= 1023) block_size = 256; 
  else if (b > 1023 && b <= 1215) block_size = 384; 
  else if (b > 1215 && b <= 2047) block_size = 512; 
  else if (b > 2047 && b <= 2304) block_size = 768; 
  else block_size = 1024;
#else
  // Virtual warps (according to tuning tests)
  int v_warp_size;
  if (b <= 1) v_warp_size = 4;
  else if (b > 1 && b <= 2) v_warp_size = 4;
  else if (b > 2 && b <= 12) v_warp_size = 8; 
  else if (b > 12 && b <= 24) v_warp_size = 16; 
  else v_warp_size = 32;
  // Block size (according to tuning tests)
  int block_size;
  if (b <= 255) block_size = 128;                   
  else if (b > 255 && b <= 304) block_size = 192;  
  else if (b > 304 && b <= 511) block_size = 256;  
  else if (b > 511 && b <= 607) block_size = 384;  
  else if (b > 607 && b <= 1023) block_size = 512; 
  else if (b > 1023 && b <= 1152) block_size = 768; 
  else block_size = 1024;
#endif
#else
#define WARPS 1
#define WARP_SIZE 64
#if SP
  // Virtual warps (according to tuning tests) -> Typically 3 x number_of_warps
  int v_warp_size;
  if (b <= 1) v_warp_size = 1;
  else if (b > 1 && b <= 2) v_warp_size = 2;
  else if (b > 2 && b <= 4) v_warp_size = 4;
  else if (b > 4 && b <= 8) v_warp_size = 8;
  else if (b > 8 && b <= 48) v_warp_size = 16;
  else if (b > 48 && b <= 96) v_warp_size = 32;
  else v_warp_size = 64;
  // Block size (according to tuning tests)
  int block_size;
  if (b <= 192) block_size = 64;
  else if (b > 192 && b <= 384) block_size = 128;
  else if (b > 384 && b <= 576) block_size = 192;
  else block_size = 256;
#else
  // Virtual warps (according to tuning tests)
  int v_warp_size;
  if (b <= 1) v_warp_size = 1;
  else if (b > 1 && b <= 2) v_warp_size = 2;
  else if (b > 2 && b <= 4) v_warp_size = 4;
  else if (b > 4 && b <= 5) v_warp_size = 8;
  else if (b > 5 && b <= 24) v_warp_size = 16;
  else if (b > 24 && b <= 48) v_warp_size = 32;
  else v_warp_size = 64;
  // Block size (according to tuning tests)
  int block_size;
  if (b <= 96) block_size = 64;
  else if (b > 96 && b <= 192) block_size = 128;
  else if (b > 192 && b <= 288) block_size = 192;
  else block_size = 256;
#endif
#endif

#if NVIDIA
#define LOCALMEM_TILING 0
#else
#define LOCALMEM_TILING 1
#endif

#if SP
#if LOCALMEM_TILING
  err = kernel.setArg(5, b<192?(b*(WARPS*WARP_SIZE/v_warp_size)*sizeof(cl_float)):(b*sizeof(cl_float)), NULL);
  if (err != CL_SUCCESS)
    return true;
  err = kernel.setArg(6, b<192?(b*(WARPS*WARP_SIZE/v_warp_size)*sizeof(cl_float)):(b*sizeof(cl_float)), NULL);
  if (err != CL_SUCCESS)
    return true;
  err = kernel.setArg(7, v_warp_size);
  if (err != CL_SUCCESS)
    return true;
  err = kernel.setArg(8, b<192?((WARPS*WARP_SIZE/v_warp_size)*sizeof(cl_int)):(sizeof(cl_int)), NULL);
  if (err != CL_SUCCESS)
    return true;
#else
  err = kernel.setArg(5, v_warp_size);
  if (err != CL_SUCCESS)
    return true;
  err = kernel.setArg(6, b<192?((WARPS*WARP_SIZE/v_warp_size)*sizeof(cl_int)):(sizeof(cl_int)), NULL);
  if (err != CL_SUCCESS)
    return true;
#endif
#else
#if LOCALMEM_TILING
  err = kernel.setArg(5, b<96?(b*(WARPS*WARP_SIZE/v_warp_size)*sizeof(cl_double)):(b*sizeof(cl_double)), NULL);
  if (err != CL_SUCCESS)
    return true;
  err = kernel.setArg(6, b<96?(b*(WARPS*WARP_SIZE/v_warp_size)*sizeof(cl_double)):(b*sizeof(cl_double)), NULL);
  if (err != CL_SUCCESS)
    return true;
  err = kernel.setArg(7, v_warp_size);
  if (err != CL_SUCCESS)
    return true;
  err = kernel.setArg(8, b<96?((WARPS*WARP_SIZE/v_warp_size)*sizeof(cl_int)):(sizeof(cl_int)), NULL);
  if (err != CL_SUCCESS)
    return true;
#else
  err = kernel.setArg(5, v_warp_size);
  if (err != CL_SUCCESS)
    return true;
  err = kernel.setArg(6, b<96?((WARPS*WARP_SIZE/v_warp_size)*sizeof(cl_int)):(sizeof(cl_int)), NULL);
  if (err != CL_SUCCESS)
    return true;
#endif
#endif

  // NDRange and kernel call
#if SP
  if (b < 192){
#else
  if (b < 96){
#endif
#if PRINT
    std::cerr << "vwarp = " << v_warp_size << "\t"; // Print v_warp_size
#endif
    // NDRange - PPoPP'2014 + use of virtual warps
    long int aux = A==1?(B<1024?(long int)1024*WARP_SIZE:(long int)B*WARP_SIZE):(A*B<1024?(long int)1024*WARP_SIZE:(long int)B*WARP_SIZE);
    cl::NDRange global(std::min((long int)a*B*WARP_SIZE, aux), WARPS, A),
      local(WARP_SIZE, WARPS, 1);
    err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local,
      NULL, prof.GetEvent());
    if (err != CL_SUCCESS)
      return true;
  }
  else{
#if PRINT
    std::cerr << "blocksize =  " << block_size << "\t"; // Print block size
#endif
    // NDRange - Block-centric using shared memory tiling
    cl::NDRange global(std::min(a*B*block_size, 1024*block_size), 1, A),
      local(block_size, 1, 1);
    err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local,
      NULL, prof.GetEvent());
    if (err != CL_SUCCESS)
      return true;
  }

#ifdef LIBMARSHAL_OCL_PROFILE
  if (elapsed_time) {
    *elapsed_time += prof.Report();
  } else {
    prof.Report(A*a*B*b*sizeof(T)*2);
  }
#endif
  return false;
}


extern "C" bool cl_transpose(cl_command_queue queue, cl_mem src, int A, int a,
  int B, int b, int R, int stages, cl_ulong *elapsed_time) {
  cl::Buffer buffer = cl::Buffer(src);
  clRetainMemObject(src);
  cl::Context context;
  if(buffer.getInfo(CL_MEM_CONTEXT, &context) != CL_SUCCESS)
    return true;
  cl_ulong et = 0;
  if (stages == 2){
    // Method 1: Aa >> Bb
#if SP
    if (((a*B*b+31)/32) + ((((a*B*b+31)/32)>>5)*1) <= MAX_MEM && a < (MAX_MEM - 64)/2){
#else
    if (((a*B*b+31)/32) + ((((a*B*b+31)/32)>>5)*1) <= MAX_MEM && a < (MAX_MEM - 32)/2){
#endif
      bool r1;
      if (a > 1){
        if (B*b*a <= MAX_MEM){
          std::cerr << "010_BS-";
          r1 = cl_transpose_010_bs(queue, src, A, a, B*b, &et);
        }
        else{
          std::cerr << "010_PTTWAC-";
          r1 = cl_transpose_010_pttwac(queue, src, A, a, B*b, &et, R, 1);
        }
        if (r1) {
          std::cerr << "cl_transpose: step 1 failed\n";
          return r1;
        }
      }
      bool r2 = cl_transpose_100(queue, src, A, B*b, a, &et);
      if (r2) {
        std::cerr << "cl_transpose: step 2 failed\n";
      }
#ifdef LIBMARSHAL_OCL_PROFILE
      std::cerr << "2.1-";
#endif
      *elapsed_time += et;
      return r2;
    }
  }
  if (stages == 22){
    // Method 1: Aa >> Bb
#if SP
    if (((b*A*a+31)/32) + ((((b*A*a+31)/32)>>5)*1) <= MAX_MEM && b < (MAX_MEM - 64)/2){
#else
    if (((b*A*a+31)/32) + ((((b*A*a+31)/32)>>5)*1) <= MAX_MEM && b < (MAX_MEM - 32)/2){
#endif
      bool r1 = cl_transpose_100(queue, src, A*a, B, b, &et);
      if (r1) {
        std::cerr << "cl_transpose: step 1 failed\n";
      }
      bool r2;
      if (b > 1){
        if (A*a*b <= MAX_MEM){
          std::cerr << "010_BS-";
          r2 = cl_transpose_010_bs(queue, src, B, A*a, b, &et);
        }
        else{
          std::cerr << "010_PTTWAC-";
          r2 = cl_transpose_010_pttwac(queue, src, B, A*a, b, &et, R, 1);
        }
        if (r2) {
          std::cerr << "cl_transpose: step 2 failed\n";
          return r2;
        }
      }
#ifdef LIBMARSHAL_OCL_PROFILE
      std::cerr << "2.2-";
#endif
      *elapsed_time += et;
      return r2;
    }
  }
  // 3-step approach
  if (stages == 3) {
    // Method 2: a, b < TILE_SIZE 
    // AaBb to BAab (step 1)
    // to BAba (step 2)
    // to BbAa (step 3)
#if SP
    if (((a*b+31)/32) + ((((a*b+31)/32)>>5)*1) <= MAX_MEM && b < (MAX_MEM - 64)/2 && a < (MAX_MEM - 64)/2 && a > 1 && b > 1){
#else
    if (((a*b+31)/32) + ((((a*b+31)/32)>>5)*1) <= MAX_MEM && b < (MAX_MEM - 32)/2 && a < (MAX_MEM - 32)/2 && a > 1 && b > 1){
#endif
      bool r1 = cl_transpose_100(queue, src, A*a, B, b, &et);
      if (r1) {
        std::cerr << "cl_transpose: step 2.1 failed\n";
        return r1;
      }
      bool r2;
      if (a > 1){ // If A is prime, it is not necessary to run 010
        if (b*a <= MAX_MEM){
          std::cerr << "010_BS-";
          r2 = cl_transpose_010_bs(queue, src, B*A, a, b, &et);
        }
        else{
          std::cerr << "010_PTTWAC-";
          r2 = cl_transpose_010_pttwac(queue, src, B*A, a, b, &et, R, 1);
        }
      }
      if (r2) {
        std::cerr << "cl_transpose: step 2.2 failed\n";
        return r2;
      }
      bool r3 = cl_transpose_0100(queue, src, B, A, b, a, &et);
      if (r3) {
        std::cerr << "cl_transpose: step 2.3 failed\n";
        return r3;
      }
#ifdef LIBMARSHAL_OCL_PROFILE
      std::cerr << "3.1-";
#endif
      *elapsed_time += et;
      return r1 || r2 || r3;
    }
  }
  // 3-step approach
  if (stages == 32) {
    // Method 2: a, b < TILE_SIZE 
    // AaBb to ABab (step 1)
    // to ABba (step 2)
    // to BbAa (step 3)
#if SP
    if (((a*b+31)/32) + ((((a*b+31)/32)>>5)*1) <= MAX_MEM && b < (MAX_MEM - 64)/2 && a < (MAX_MEM - 64)/2 && a > 1 && b > 1){
#else
    if (((a*b+31)/32) + ((((a*b+31)/32)>>5)*1) <= MAX_MEM && b < (MAX_MEM - 32)/2 && a < (MAX_MEM - 32)/2 && a > 1 && b > 1){
#endif
      bool r1 = cl_transpose_0100(queue, src, A, a, B, b, &et);
      if (r1) {
        std::cerr << "cl_transpose: step 3.2.1 failed\n";
        return r1;
      }
      bool r2;
      if (a > 1){ // If A is prime, it is not necessary to run 010
        if (b*a <= MAX_MEM){
          std::cerr << "010_BS-";
          r2 = cl_transpose_010_bs(queue, src, A*B, a, b, &et);
        }
        else{
          std::cerr << "010_PTTWAC-";
          r2 = cl_transpose_010_pttwac(queue, src, A*B, a, b, &et, R, 1);
        }
      }
      if (r2) {
        std::cerr << "cl_transpose: step 3.2.2 failed\n";
        return r2;
      }
      bool r3 = cl_transpose_100(queue, src, A, B*b, a, &et);
      if (r3) {
        std::cerr << "cl_transpose: step 3.2.3 failed\n";
        return r3;
      }
#ifdef LIBMARSHAL_OCL_PROFILE
      std::cerr << "3.2-";
#endif
      *elapsed_time += et;
      return r1 || r2 || r3;
    }
  }
  // 4-step approach
  if (stages == 4) {
    // Karlsson's method: a, b < TILE_SIZE 
    // AaBb to ABab (0100)
    // ABab ABba    (0010)
    // ABba BAba    (1000)
    // BAba to BbAa (0100)
    T0100_PTTWAC step1(A, a, B, b, context());
    T010_BS step2(a, b, context());
    T010_PTTWAC step2p(a, b, context());
    T0100_PTTWAC step3(1, A, B, b*a, context());
    T0100_PTTWAC step4(B, A, b, a, context());
    if (step1.IsFeasible() && (step2.IsFeasible()||step2p.IsFeasible()) 
        && step3.IsFeasible() && step4.IsFeasible()) {
      bool r1 = cl_transpose_0100(queue, src, A, a, B, b, &et);
      if (r1) {
        std::cerr << "cl_transpose: step 4.1 failed\n";
        return r1;
      }
      bool r2;
      if (step2.IsFeasible())
        r2 = cl_transpose_010_bs(queue, src, B*A, a, b, &et);
      else
        r2 = cl_transpose_010_pttwac(queue, src, B*A, a, b, &et, R, 1);
      if (r2) {
        std::cerr << "cl_transpose: step 4.2 failed\n";
        return r2;
      }
      bool r3 = cl_transpose_100(queue, src, A, B, b*a, &et);
      if (r3) {
        std::cerr << "cl_transpose: step 4.3 failed\n";
        return r3;
      }
      bool r4 = cl_transpose_0100(queue, src, B, A, b, a, &et);
      if (r4) {
        std::cerr << "cl_transpose: step 4.4 failed\n";
        return r4;
      }
#ifdef LIBMARSHAL_OCL_PROFILE
      //std::cerr << "[cl_transpose] Karlsson's method; "<< 
      std::cerr<<
        float(A*a*B*b*2*sizeof(T))/et << "\n";
#endif
      *elapsed_time += et;
      return r1 || r2 || r3 || r4;
    }
  }

  // fallback
  bool r = cl_transpose_0100(queue, src, 1, A*a, B*b, 1, &et);
#ifdef LIBMARSHAL_OCL_PROFILE
  std::cerr << "Fallback-"; 
#endif
  *elapsed_time += et;
  return r;
}

// Transformation 100, or ABb to BAb
extern "C" bool cl_transpose_100(cl_command_queue cl_queue,
    cl_mem src, int A, int B, int b, cl_ulong *elapsed_time) {
  return _cl_transpose_0100(cl_queue, src, 1, A, B, b, elapsed_time);
}

// Transformation 0100, or AaBb to ABab
extern "C" bool cl_transpose_0100(cl_command_queue cl_queue, cl_mem src,
  int A, int a, int B, int b, cl_ulong *elapsed_time) {
  return _cl_transpose_0100(cl_queue, src, A, a, B, b, elapsed_time);
}


// Wrappers for old API compatibility
// Transformation 010, or AaB to ABa
extern "C" bool cl_aos_asta_bs(cl_command_queue cl_queue,
    cl_mem src, int height, int width,
    int tile_size) {
  return cl_transpose_010_bs(cl_queue, src,
    height/tile_size /*A*/,
    tile_size /*a*/,
    width /*B*/, NULL);
}

// Transformation 010, or AaB to ABa
extern "C" bool cl_aos_asta_pttwac(cl_command_queue cl_queue,
    cl_mem src, int height, int width, int tile_size, int R, int P) {
  assert ((height/tile_size)*tile_size == height);
  return cl_transpose_010_pttwac(cl_queue, src, height/tile_size/*A*/, 
    tile_size /*a*/,
    width /*B*/,
    NULL,
    R,
    P);
}

extern "C" bool cl_aos_asta(cl_command_queue queue, cl_mem src, int height,
  int width, int tile_size, int R) {
  return cl_aos_asta_bs(queue, src, height, width, tile_size) &&
    cl_aos_asta_pttwac(queue, src, height, width, tile_size, R, 1);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////
extern "C" bool cl_padding(cl_command_queue cl_queue,
    cl_mem src, int x_size, int y_size, int pad_size, cl_ulong *elapsed_time) {
  // Standard preparation of invoking a kernel
  cl::CommandQueue queue = cl::CommandQueue(cl_queue);
  Profiling prof(queue, "Padding");
  cl::Buffer buffer = cl::Buffer(src);
  clRetainMemObject(src);
  cl::Context context;
  if(buffer.getInfo(CL_MEM_CONTEXT, &context) != CL_SUCCESS)
    return true;
  MarshalProg *marshalprog = MarshalProgSingleton::Instance();
  marshalprog->Init(context());

#if NVIDIA
#define REGS 8 //16
#else
#define REGS 64
#endif
  int ldim = NR_THREADS;
  // Atomic flags
  const int num_flags = (y_size * pad_size) / (ldim * REGS);
  cl_int *flags = (cl_int *)calloc(sizeof(cl_int), num_flags + 2);
  flags[0] = 1;
  flags[num_flags + 1] = 0;
  cl_int err;
  cl::Buffer d_flags = cl::Buffer(context, CL_MEM_READ_WRITE,
      sizeof(cl_int)*(num_flags + 2), NULL, &err);
  if (err != CL_SUCCESS)
    return true;
  err = queue.enqueueWriteBuffer(d_flags, CL_TRUE, 0,
      sizeof(cl_int)*(num_flags + 2), flags);
  free(flags);
  if (err != CL_SUCCESS)
    return true;

  int num_wg = num_flags + 1;
  int rows = 0, shm_size = 0;
  cl::Kernel kernel(marshalprog->program, "padding");

  if (CL_SUCCESS != kernel.setArg(0, buffer))
    return true;
  err = kernel.setArg(1, x_size);
  if (err != CL_SUCCESS)
    return true;
  err = kernel.setArg(2, pad_size);
  err |= kernel.setArg(3, y_size);
  err |= kernel.setArg(4, rows);
  if (err != CL_SUCCESS)
    return true;
  err = kernel.setArg(5, d_flags);
  if (err != CL_SUCCESS)
    return true;

  cl::NDRange global(num_wg*ldim), local(ldim);
  err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local, NULL,
    prof.GetEvent());
  if (err != CL_SUCCESS)
    return true;
#ifdef LIBMARSHAL_OCL_PROFILE
  if (elapsed_time) {
    *elapsed_time += prof.Report();
  } else {
    prof.Report(x_size*y_size*sizeof(T)*2);
  }
#endif
  return false;
}

extern "C" bool cl_unpadding(cl_command_queue cl_queue,
    cl_mem src, int x_size, int y_size, int pad_size, cl_ulong *elapsed_time) {
  // Standard preparation of invoking a kernel
  cl::CommandQueue queue = cl::CommandQueue(cl_queue);
  Profiling prof(queue, "Unpadding");
  cl::Buffer buffer = cl::Buffer(src);
  clRetainMemObject(src);
  cl::Context context;
  if(buffer.getInfo(CL_MEM_CONTEXT, &context) != CL_SUCCESS)
    return true;
  MarshalProg *marshalprog = MarshalProgSingleton::Instance();
  marshalprog->Init(context());

#undef REGS
#if NVIDIA
#define REGS 8 //16
#else
#define REGS 64
#endif
  int ldim = NR_THREADS;
  // Atomic flags
  const int num_flags = (y_size * pad_size) / (ldim * REGS);
  cl_int *flags = (cl_int *)calloc(sizeof(cl_int), num_flags + 2);
  flags[0] = 1;
  flags[num_flags + 1] = 0;
  cl_int err;
  cl::Buffer d_flags = cl::Buffer(context, CL_MEM_READ_WRITE,
      sizeof(cl_int)*(num_flags + 2), NULL, &err);
  if (err != CL_SUCCESS)
    return true;
  err = queue.enqueueWriteBuffer(d_flags, CL_TRUE, 0,
      sizeof(cl_int)*(num_flags + 2), flags);
  free(flags);
  if (err != CL_SUCCESS)
    return true;

  int num_wg = num_flags + 1;
  int rows, shm_size = 0;
  cl::Kernel kernel(marshalprog->program, "unpadding");

  if (CL_SUCCESS != kernel.setArg(0, buffer))
    return true;
  err = kernel.setArg(1, x_size);
  if (err != CL_SUCCESS)
    return true;
  err = kernel.setArg(2, pad_size);
  err |= kernel.setArg(3, y_size);
  err |= kernel.setArg(4, rows);
  if (err != CL_SUCCESS)
    return true;
  err = kernel.setArg(5, d_flags);
  if (err != CL_SUCCESS)
    return true;

  cl::NDRange global(num_wg*ldim), local(ldim);
  err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local, NULL,
    prof.GetEvent());
  if (err != CL_SUCCESS)
    return true;
#ifdef LIBMARSHAL_OCL_PROFILE
  if (elapsed_time) {
    *elapsed_time += prof.Report();
  } else {
    prof.Report(x_size*y_size*sizeof(T)*2);
  }
#endif
  return false;
}

