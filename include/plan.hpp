#ifndef PLAN_HPP_
#define PLAN_HPP_
#include <algorithm>

class GPUInfo {
 public:
  GPUInfo(cl_context context=NULL):ctx_(context) {
    bool successful = false;
    if (context) {
      size_t ctsize;
      cl_int r;
      r = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, 0, &ctsize);
      if (r == CL_SUCCESS && ctsize/sizeof(cl_device_id*) == 1) {
        cl_device_id *devices = (cl_device_id*)malloc(ctsize);
        r = clGetContextInfo(context, CL_CONTEXT_DEVICES, ctsize, devices, 0);
        if (r == CL_SUCCESS) {
          cl_int r2;
          r = clGetDeviceInfo(devices[0], CL_DEVICE_LOCAL_MEM_SIZE, 
            sizeof(cl_ulong), &max_local_mem_, 0);
          r2 = clGetDeviceInfo(devices[0], CL_DEVICE_MAX_WORK_GROUP_SIZE, 
            sizeof(size_t), &max_work_items_, 0);
          successful = ((r == CL_SUCCESS) && (r2 == CL_SUCCESS));
        }
        free(devices);
      }
    } 
    if (!successful) {
      max_work_items_ = 256;
      max_local_mem_ = 16*1024; // mininum allowed in OpenCL
    }
#if 0
    std::cerr << "GPUInfo: max work items = " << max_work_items_<<"\n";
    std::cerr << "GPUInfo: max local mem = " << max_local_mem_<<"\n";
#endif
  }
  size_t GetMaxWorkItems(void) const {
    return max_work_items_;
  }
  cl_ulong GetMaxLocalMemSize(void) const {
    return max_local_mem_;
  }
 private:
  size_t max_work_items_;
  cl_ulong max_local_mem_;
  cl_context ctx_;
};

class Transposition {
 public:
  const int m, n; // M by N in row major. i.e. A[i,j]=i*N+j and i<M
  Transposition(int mm, int nn, cl_context ctx=NULL):
    m(mm), n(nn), mn_(mm*nn-1),
    nr_cycles_(-1), cache_line_(8), gpu_info_(ctx) {}
  unsigned Next(unsigned i) const {
    return (i*m)-mn_* (i/n);
  }
  unsigned GetNumCycles() {
    if (nr_cycles_ >= 0)
      return nr_cycles_;
    unsigned nontrivial = 0;
    for (int i = 1; i <= mn_; i ++) {
      int j = Next(i);
      if (j == i)
        continue;
      for (; j > i; j = Next(j))
        ;
      if (j != i)
        continue;
      nontrivial ++;
    }
    nr_cycles_=nontrivial;
    return nontrivial;
  }
  virtual bool IsFeasible(void) const {
    return true;
  }
  virtual unsigned GetEstimatedGlobalMemOps(void) {
    return 0;
  }
  virtual unsigned GetEstimatedGlobalAtomicOps(void) {
    return 0;
  }
  virtual unsigned GetEstimatedLocalAtomicOps(void) {
    return 0;
  }
 protected:
  const int cache_line_;
  int nr_cycles_;
  const int mn_;
  GPUInfo gpu_info_;
  Transposition();
};

/// Barrier-synchnorization
class T010_BS: public Transposition {
 public:
  T010_BS(int m, int n, cl_context ctx=NULL):
    Transposition(m, n, ctx) {}
  unsigned GetEstimatedGlobalMemOps(void) {
    return float(2*m*n)/cache_line_;
  }
  unsigned GetEstimatedGlobalAtomicOps(void) {
    return 0;
  }
  unsigned GetEstimatedLocalAtomicOps(void) {
    return 0;
  }
  bool IsFeasible(void) const {
    /* a small fraction of local memory is not really usable */
    return m*n < gpu_info_.GetMaxLocalMemSize()/4-512;
  }
};

/// local memory based PTTWAC for transpostion 010
class T010_PTTWAC: public Transposition {
 public:
  T010_PTTWAC(int m, int n, cl_context ctx=NULL):
    Transposition(m, n, ctx) {}
  unsigned GetEstimatedGlobalMemOps(void) {
    unsigned(2*m*n);
    return 0;
  }
  unsigned GetEstimatedGlobalAtomicOps(void) {
    return 0;
  }
  unsigned GetEstimatedLocalAtomicOps(void) {
    return m*n;
  }
  bool IsFeasible(void) const {
    return (m*n/16) < (gpu_info_.GetMaxLocalMemSize()/4);
  }
};

/// global memory based PTTWAC for transposition 0100
// when A == 1 this is equivalent to transposition 100
class T0100_PTTWAC: public Transposition {
 public:
  T0100_PTTWAC(int A, int a, int B, int b, cl_context ctx):
    Transposition(a, B, ctx), A_(A), a_(a), B_(B), b_(b) {}
  unsigned GetEstimatedGlobalMemOps(void) {
    return unsigned(2*A_*a_*B_*std::max(float(cache_line_)/b_, 1.0f));
  }
  unsigned GetEstimatedGlobalAtomicOps(void) {
    return m*n;
  }
  unsigned GetEstimatedLocalAtomicOps(void) {
    return 0;
  }
  bool IsFeasible(void) const {
    // the algorithm takes two shared memory buffers of b*WARPS*sizeof(float)
    unsigned shm_sz = b_*6*sizeof(float)*2;
    shm_sz += 6*sizeof(int); // done[WARP_SIZE];
    return shm_sz < gpu_info_.GetMaxLocalMemSize()-512; // hardcoded for now
  }
 private:
  int A_, a_, B_, b_;
};

#include <map>
#include <iostream>
#include <sstream>
//http://www.daniweb.com/software-development/c/code/237001/prime-factorization#
class Factorize {
 public:
  typedef std::map<int, int> Factors;
  Factorize(int n):n_(n) {
    int d = 2;  
    if(n < 2) return;
    /* while the factor being tested
     * is lower than the number to factorize */
    while(d < n) {
      /* valid prime factor */
      if(n % d == 0) {
	factors[d] += 1;
	n /= d;
      }
      /* invalid prime factor */
      else {
	if(d == 2) d = 3;
	else d += 2;
      }
    }
    /* last prime factor */
    factors[d] += 1;
  }
  const std::map<int, int> &get_factors() const {
    return factors;
  }
  void tiling_options(void) {
    tiling_options(std::string(""), factors.begin(), 1);
  }
  std::vector<int> &get_tile_sizes(void) { return tile_sizes_; }
 private:
  Factorize():n_(0) {}
  void tiling_options(std::string s, Factors::const_iterator current, int t) {
    if (current == factors.end()) {
      if (t > 1) {
//	std::cout << s << ";" << t << "*"<< n_/t << "\n";
        tile_sizes_.push_back(t);
      }
      return;
    } else {
      s += "*";
    }
    for (int i = 0; i<=current->second; i++) {
      std::stringstream ss;
      ss << current->first << "^" << i;
      Factors::const_iterator inc = current;
      ++inc;
      if (i)
	t*=current->first;
      tiling_options(s+ss.str(), inc, t); 
    }
  }
  const int n_;
  Factors factors;
  std::vector<int> tile_sizes_;

};


#endif // PLAN_HPP_
