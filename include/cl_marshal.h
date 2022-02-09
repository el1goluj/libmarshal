//===--- marshal.h - GPU in-place marshaling library          ----------===//
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
//  This file declares the interface of the libmarshal 
//
//===---------------------------------------------------------------------===//

#ifndef _LIBMARSHAL_INCLUDE_MARSHAL_H_
#define _LIBMARSHAL_INCLUDE_MARSHAL_H_

#include <cl.h>
extern "C" {
bool cl_aos_asta_bs(cl_command_queue queue,
    cl_mem src, int height, int width, int tile_size);
bool cl_aos_asta_pttwac(cl_command_queue queue, cl_mem src, int height,
    int width, int tile_size, int R, int P);
bool cl_aos_asta(cl_command_queue queue,
    cl_mem src, int height, int width, int tile_size, int R);
bool cl_transpose(cl_command_queue queue,
    cl_mem src, int A, int a, int B, int b, int R, int stages, cl_ulong *);
// Transformation 010, or AaB to ABa
bool cl_transpose_010_bs(cl_command_queue cl_queue, cl_mem src,
  int A, int a, int B, cl_ulong *);
bool cl_transpose_010_pttwac(cl_command_queue cl_queue, cl_mem src,
  int A, int a, int B, cl_ulong *, int R, int P);
// Transformation 100, or ABb to BAb
bool cl_transpose_100(cl_command_queue cl_queue, cl_mem src,
  int A, int B, int b, cl_ulong *);
// Transformation 0100, or AaBb to ABab
bool cl_transpose_0100(cl_command_queue queue, cl_mem src,
  int A, int a, int B, int b, cl_ulong *);
// Padding and unpadding
bool cl_padding(cl_command_queue cl_queue, cl_mem src,
  int x_size, int y_size, int pad_size, cl_ulong *);
bool cl_unpadding(cl_command_queue cl_queue, cl_mem src,
  int x_size, int y_size, int pad_size, cl_ulong *);
}

#endif // _LIBMARSHAL_INCLUDE_MARSHAL_H_
