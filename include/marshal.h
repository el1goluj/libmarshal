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

extern "C" {

bool gpu_aos_asta_bs_float(float *src, int height, int width, int tile_size, clock_t *timer);
bool gpu_aos_asta_bs_double(double *src, int height, int width, int tile_size, clock_t *timer);

bool gpu_aos_asta_pttwac(float *src, int height, int width, int tile_size, clock_t *timer);

bool gpu_soa_asta_pttwac(float *src, int height, int width, int tile_size, clock_t *timer);

};

#endif // _LIBMARSHAL_INCLUDE_MARSHAL_H_
