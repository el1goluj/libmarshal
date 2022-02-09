#include "local_cl.hpp"
#include <gtest/gtest.h>
#include <cstdlib>
#include <string>
#include <vector>
#include <math.h>
#include <limits.h>
#include "cl_marshal.h"
#include "plan.hpp"
#include "/usr/include/gsl/gsl_sort.h"

#define NVIDIA 1 // Choose compilation for NVIDIA or other (e.g., AMD)
#define SP 1 // SP = 1 -> Single Precision; SP = 0 -> Double Precision 
// Make sure these two flags have the same value in ~/src/cl_marshal.cc and ~/src/cl/cl_aos_asta.cl

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
#define CHECK_RESULTS 1 

namespace {
class libmarshal_cl_test : public ::testing::Test {
 public:
  cl_uint GetCtxRef(void) const {
    cl_uint rc;
    rc = context_->getInfo<CL_CONTEXT_REFERENCE_COUNT>();
    return rc;
  }
  cl_uint GetQRef(void) const {
    cl_uint rc = queue_->getInfo<CL_QUEUE_REFERENCE_COUNT>();
    return rc;
  }
 protected:
  virtual void SetUp(void);
  virtual void TearDown(void);
  libmarshal_cl_test() {}
  std::string device_name_;
  cl::Context *context_;
  cl::CommandQueue *queue_;
};

void libmarshal_cl_test::SetUp(void) {
  cl_int err;

  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  if (platforms.size() == 0) {
    std::cerr << "Platform size 0\n";
    return;
  }
  // Choose Platform (e.g., NVIDIA or AMD)
  int i = 0;
  //int i = 1;
  for (; i < platforms.size(); i++) {
    std::vector<cl::Device> devices;
    platforms[i].getDevices(CL_DEVICE_TYPE_GPU, &devices);
    if (devices.size() == 0)
      continue;
    else
      break;
  }
  if (i == platforms.size()) {
    std::cerr << "None of the platforms have GPU\n";
    return;
  }
  cl_context_properties properties[] = 
  { CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[i])(), 0};

  context_ = new cl::Context(CL_DEVICE_TYPE_GPU, properties, NULL, NULL, &err);
  queue_ = NULL;
  ASSERT_EQ(err, CL_SUCCESS);
  std::vector<cl::Device> devices = context_->getInfo<CL_CONTEXT_DEVICES>();
  // Get name of the devices
  devices[0].getInfo(CL_DEVICE_NAME, &device_name_);
  std::cerr << "Testing on device " << device_name_ << std::endl;

  // Create a command queue on the first GPU device
  queue_ = new cl::CommandQueue(*context_, devices[0], CL_QUEUE_PROFILING_ENABLE);
}

extern "C" void cl_marshal_finalize(void);
void libmarshal_cl_test::TearDown(void) {
  cl_marshal_finalize();
  delete queue_;
  delete context_;
}

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
int generate_vector(T *x_vector, int dim) {       
  srand(5432);
  for(int i=0;i<dim;i++) {
    x_vector[i] = ((T) (rand() % 100) / 100);
  }
  return 0;
}

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
//[w][h/t][t] to [h/t][w][t] 
void cpu_soa_asta(T *src, T *dst, int height, int width,
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

}

// Bug537 - Test Transpose 100 (PTTWAC)
TEST_F(libmarshal_cl_test, bug537) {
  int ws[2] = {197, 215};
  int hs[2] = {35588, 44609};
  for (int i = 0; i < 2; i++)
  for (int t = 1; t <= 1024; t*=2) {
    int w = ws[i];
    int h = (hs[i]+t-1)/t*t;

    std::cerr << "w = "<<w<< "; h/t = " <<h/t<< "; t = " <<t<< ", ";

    T *src = (T*)malloc(sizeof(T)*h*w);
    T *dst = (T*)malloc(sizeof(T)*h*w);
    T *dst_gpu = (T*)malloc(sizeof(T)*h*w);
    generate_vector(src, h*w);
    cl_int err;
    cl::Buffer d_dst = cl::Buffer(*context_, CL_MEM_READ_WRITE,
        sizeof(T)*h*w, NULL, &err);
    ASSERT_EQ(err, CL_SUCCESS);
    cl_uint oldqref = GetQRef();
    ASSERT_EQ(queue_->enqueueWriteBuffer(
          d_dst, CL_TRUE, 0, sizeof(T)*h*w, src), CL_SUCCESS);
    cl_uint oldref = GetCtxRef();
    // Change N to something > 1 to compute average performance (and use some WARM_UP runs).
    const int N = 1;
    const int WARM_UP = 0;
    cl_ulong et = 0;
    for (int n = 0; n < N+WARM_UP; n++) {
      if (n == WARM_UP)
        et = 0;
      bool r = cl_transpose_100((*queue_)(), d_dst(), w, h/t, t, &et);
      EXPECT_EQ(oldref, GetCtxRef());
      EXPECT_EQ(oldqref, GetQRef());
      ASSERT_EQ(false, r);
      ASSERT_EQ(queue_->enqueueReadBuffer(d_dst, CL_TRUE, 0,
            sizeof(T)*h*w, dst_gpu), CL_SUCCESS);
      if ((n % 2) == 0) {
        cpu_soa_asta(src, dst, h, w, t);
        EXPECT_EQ(0, compare_output(dst_gpu, dst, h*w));
      } else {
        cpu_soa_asta(dst, src, h, w, t);
        EXPECT_EQ(0, compare_output(dst_gpu, src, h*w));
      }
    }
    Transposition tx(w,h/t);
    std::cerr << "Throughput = " << float(2*h*w*sizeof(T)*N)/et;
    std::cerr << " GB/s\t";
    std::cerr << "Num cycles:"<<tx.GetNumCycles()<< "; percentage = " <<
      (float)tx.GetNumCycles()/(float)(h*w/t)*100 << "\n";
    free(src);
    free(dst);
    free(dst_gpu);
  }
}

// Bug536 - Test Transpose 010 (PTTWAC)
TEST_F(libmarshal_cl_test, bug536) {
  int ws[2] = {197, 215};
  int hs[2] = {35588, 44609};
  for (int i = 0; i < 2; i++)
  for (int t = 16; t <= 64; t*=2) {
    int w = ws[i];
    int h = (hs[i]+t-1)/t*t;

#define P 1 // Padding size
    bool r;
    for(int S_f = 1; S_f < 32; S_f *=2){ // (Use S_f <= 1 for testing IPT) - S_f = Spreading factor

      int sh_sz2 = S_f * ((t*w+31)/32);
      sh_sz2 += (sh_sz2 >> 5) * P;

      std::cerr << "w = " << w << ", t = " << t << ", w*t = " << w*t << ", S_f = " << S_f;
      std::cerr << ", P = " << P << ", " << (int)(5-log2(S_f)) << ", " << sh_sz2 << ", ";

      if(sh_sz2 > MAX_MEM) std::cerr << "\n";
      else{
        T *src = (T*)malloc(sizeof(T)*h*w);
        T *dst = (T*)malloc(sizeof(T)*h*w);
        T *dst_gpu = (T*)malloc(sizeof(T)*h*w);
        generate_vector(src, h*w);
        cl_int err;
        cl::Buffer d_dst = cl::Buffer(*context_, CL_MEM_READ_WRITE,
            sizeof(T)*h*w, NULL, &err);
        ASSERT_EQ(err, CL_SUCCESS);
        ASSERT_EQ(queue_->enqueueWriteBuffer(
            d_dst, CL_TRUE, 0, sizeof(T)*h*w, src), CL_SUCCESS);
        cl_uint oldref = GetCtxRef();
        cl_uint oldqref = GetQRef();
        cl_ulong et = 0;
        // Change N to something > 1 to compute average performance (and use some WARM_UP runs).
        const int N = 1;
        const int WARM_UP = 0;
        for (int n = 0; n < N+WARM_UP; n++) {
          if (n == WARM_UP)
            et = 0;
          r = cl_transpose_010_pttwac((*queue_)(), d_dst(), h/t, t, w, &et, S_f, P);
          EXPECT_EQ(oldref, GetCtxRef());
          EXPECT_EQ(oldqref, GetQRef());
          ASSERT_EQ(false, r);
          ASSERT_EQ(queue_->enqueueReadBuffer(d_dst, CL_TRUE, 0,
              sizeof(T)*h*w, dst_gpu), CL_SUCCESS);
          if ((n%2) == 0) {
            cpu_aos_asta(src, dst, h, w, t);
            EXPECT_EQ(0, compare_output(dst_gpu, dst, h*w));
          } else {
            cpu_aos_asta(dst, src, h, w, t);
            EXPECT_EQ(0, compare_output(dst_gpu, src, h*w));
          }
        }
        std::cerr << "Throughput = " << float(h*w*2*sizeof(T)*N) / et;
        std::cerr << " GB/s\t";

        Transposition tx(t,w);
        std::cerr << "Num cycles:"<<tx.GetNumCycles()<< "; percentage = " <<
          (float)tx.GetNumCycles()/(float)(w*t)*100 << "\n";

        free(src);
        free(dst);
        free(dst_gpu);
      }
    }
  }
}

// Bug533 - Test Transpose 010 (BS)
TEST_F(libmarshal_cl_test, bug533) {
  for (int w = 3; w <= 384; w*=2)
  for (int t=1; t<=8; t+=1) {
    int h = (500/w+1)*(100*130+t-1)/t*t;
    std::cerr << "A = " << h/t << ", a = " << t << ", B = " << w << ", w*t = " << w*t << "\t";
    T *src = (T*)malloc(sizeof(T)*h*w);
    T *dst = (T*)malloc(sizeof(T)*h*w);
    T *dst_gpu = (T*)malloc(sizeof(T)*h*w);
    generate_vector(src, h*w);
    cpu_aos_asta(src, dst, h, w, t);
    cl_int err;
    cl::Buffer d_dst = cl::Buffer(*context_, CL_MEM_READ_WRITE,
	sizeof(T)*h*w, NULL, &err);
    ASSERT_EQ(err, CL_SUCCESS);
    ASSERT_EQ(queue_->enqueueWriteBuffer(
	  d_dst, CL_TRUE, 0, sizeof(T)*h*w, src), CL_SUCCESS);
    cl_uint oldref = GetCtxRef();
    cl_uint oldqref = GetQRef();
    bool r = cl_aos_asta_bs((*queue_)(), d_dst(), h, w, t);
    EXPECT_EQ(oldref, GetCtxRef());
    EXPECT_EQ(oldqref, GetQRef());
    ASSERT_EQ(false, r);
    ASSERT_EQ(queue_->enqueueReadBuffer(d_dst, CL_TRUE, 0, sizeof(T)*h*w,
    	  dst_gpu), CL_SUCCESS);
    EXPECT_EQ(0, compare_output(dst_gpu, dst, h*w));

    free(src);
    free(dst);
    free(dst_gpu);
  }
}

void tile(int x) {
  Factorize f(x);
  std::cout << "factors = ";
  for (std::map<int, int>::const_iterator it = f.get_factors().begin(), e = f.get_factors().end();
    it != e; it++) {
    std::cout << it->first << "^" << it->second<<"*";
  }
  std::cout << "\n";
  f.tiling_options();
  std::cout << "\n";
}


// Bug_full - Test full transposition of general matrices or AoS-SoA conversions
#define RANDOM 1
#define SoA 0 // SoA to AoS conversion
#define AoS 0 // AoS to SoA conversion
TEST_F(libmarshal_cl_test, full) {
#if RANDOM
  // Matrix sizes
  // For general matrices
#if SP
  // Single precision
#if SoA
  // For skinny matrices (SoA-AoS)
  const int w_max = 1e7; const int w_min = 10000;
  const int h_max = 32; const int h_min = 2;
#elif AoS
  // For skinny matrices (AoS-SoA)
  const int h_max = 1e7; const int h_min = 10000;
  const int w_max = 32; const int w_min = 2;
#else
  // General matrices
  const int h_max = 20000; const int h_min = 1000;
  const int w_max = 20000; const int w_min = 1000;
#endif
#else
  // Double precision
#if SoA
  // For skinny matrices (SoA-AoS)
  const int w_max = 1e7; const int w_min = 10000;
  const int h_max = 32; const int h_min = 2;
#elif AoS
  // For skinny matrices (AoS-SoA)
  const int h_max = 1e7; const int h_min = 10000;
  const int w_max = 32; const int w_min = 2;
#else
  // General matrices
  const int h_max = 15000; const int h_min = 1000;
  const int w_max = 15000; const int w_min = 1000;
#endif
#endif

  for (int n = 0; n < 15; n++){
  // Generate random dimensions
  srand(n+1);
  int h = rand() % (h_max-h_min) + h_min;
  int w = rand() % (w_max-w_min) + w_min;
#else 
  int ws[] = {1800, 2500, 3200, 3900, 5100, 7200}; // Matrix sizes in PPoPP2014 paper
  int hs[] = {7200, 5100, 4000, 3300, 2500, 1800};
  for (int n = 0; n < 6; n++) {
  int w = ws[n];
  int h = hs[n];
#endif

  // Print matrix dimensions
  std::cerr << "" << n << "\t" << "" << h << "," << w << "\t";

  // Choose super-element sizes (a and b) - Algorithm 5 (TPDS paper)
  std::vector<int> hoptions;
  std::vector<int> woptions;
  int pad_h = 0; int pad_w = 0;
  bool done_h = false; bool done_w = false;
  // Minimum and maximum for super-element sizes (they might be different for float and double)
#if SP
  int min_limit = 24; 
#if SoA || AoS
  int max_limit = MAX_MEM / 32; //32 is w_max or h_max in these experiments 
#else
  int max_limit = (int)sqrt(MAX_MEM); 
#endif
#else
  int min_limit = 24;
#if SoA || AoS
  int max_limit = MAX_MEM / 32; //32 is w_max or h_max in these experiments 
#else
  int max_limit = (int)sqrt(MAX_MEM); 
#endif
#endif
  int aa = MAX_MEM;
  int bb = MAX_MEM;
  do{
    // Factorize dimensions
    Factorize hf(h), wf(w);
    hf.tiling_options();
    wf.tiling_options();
    hoptions = hf.get_tile_sizes();
    woptions = wf.get_tile_sizes();
    // Sort factors
    size_t hf_sorted2[hoptions.size()];
    size_t wf_sorted2[woptions.size()];
    gsl_sort_int_index((size_t *)hf_sorted2, &hoptions[0], 1, hoptions.size());
    gsl_sort_int_index((size_t *)wf_sorted2, &woptions[0], 1, woptions.size());

    // Desired minimum and maximum for a and b
    if (!done_h)
#if SoA
      {done_h = true;
      aa = h;}
#else
      for (int j = hoptions.size() - 1; j >= 0; j--)
        if (hoptions[hf_sorted2[j]] >= min_limit && hoptions[hf_sorted2[j]] <= max_limit){
          aa = hoptions[hf_sorted2[j]];
          done_h = true;
          break;
        }
#endif

    if (!done_w)
#if AoS
      {done_w = true;
      bb = w;}
#else
      for (int j = woptions.size() - 1; j >= 0; j--)
        if (woptions[wf_sorted2[j]] >= min_limit && woptions[wf_sorted2[j]] <= max_limit){
          bb = woptions[wf_sorted2[j]];
          done_w = true;
          break;
        }
#endif

    if (!done_h){
      pad_h++;
      h++;
    }
    if (!done_w){
      pad_w++;
      w++;
    }
  } while(!done_h || !done_w);

  // Print padding sizes
  std::cerr << "" << hoptions.size() << "," << woptions.size() << "\t";
  std::cerr << "percent_pad " << ((float)((w-(w-pad_w))*100)/(float)(w-pad_w)) << "%\t";
  std::cerr << "percent_unpad " << ((float)((h-(h-pad_h))*100)/(float)(h-pad_h)) << "%\t";
  std::cerr << "pad_h = " << pad_h << ", pad_w = " << pad_w << ",";
  std::cerr << " h + pad_h = " << h << ", w + pad_w = " << w << ",";

  // Host memory allocation
  T *src = (T*)malloc(sizeof(T)*h*w);
  T *dst = (T*)malloc(sizeof(T)*h*w);
  T *dst_gpu = (T*)malloc(sizeof(T)*h*w);
  generate_vector(src, (h - pad_h) * (w - pad_w));

#define BRUTE 0 // Brute-force search (all possible tile sizes)
#if BRUTE
  // Brute-force search
  cl_ulong max_et = ULONG_MAX;
  for (int i = 0 ; i < hoptions.size(); i++) {
    int A = h/hoptions[hf_sorted[i]], a = hoptions[hf_sorted[i]];
    for (int j = 0; j < woptions.size(); j++) {
      int B = w/woptions[wf_sorted[j]], b = woptions[wf_sorted[j]];
#else
  // Heuristic for determining tile dimensions (using Algs. 3 and 5 in TPDS paper)
  int A, a, B, b;
  A=h/aa; a=aa;
  B=w/bb; b=bb;
#endif

  cl_int err;
  cl::Buffer d_dst = cl::Buffer(*context_, CL_MEM_READ_WRITE,
      sizeof(T)*h*w, NULL, &err);
  ASSERT_EQ(err, CL_SUCCESS);
  err = queue_->enqueueWriteBuffer(
        d_dst, CL_TRUE, 0, sizeof(T)*h*w, src);
  EXPECT_EQ(err, CL_SUCCESS);
  if (err != CL_SUCCESS)
    continue;

  bool r = false;
  cl_ulong et = 0;
  cl_ulong et2 = 0;
  cl_ulong et3 = 0;
  // Change N to something > 1 to compute average performance (and use some WARM_UP runs).
  const int N = 1; 
  const int WARM_UP = 0;

  // Once selected super-element sizes (and tile size) the appropriate sequence of elementary transpositions is choosen (Algorithm 3 - TPDS paper)
    if(a >= 6 && a*B*b <= MAX_MEM){
      std::cerr << "" << A << "," << a << ",";
      std::cerr << "" << B*b << ",";

      // 2-stage approach
      for (int n = 0; n < N+WARM_UP; n++) {
        if (n == WARM_UP){
          et = 0; et2 = 0; et3 = 0;
        }
        // Padding
        if (pad_w > 0){
          r = cl_padding((*queue_)(), d_dst(), w - pad_w, h - pad_h, w, &et2);
          EXPECT_EQ(false, r);
          if (r != false)
            continue;
        }
        // Transpose
        r = cl_transpose((*queue_)(), d_dst(), A, a, B, b, 1, 2, &et); 
        EXPECT_EQ(false, r);
        if (r != false)
          continue;
        // Unpadding
        if (pad_h > 0){
          r = cl_unpadding((*queue_)(), d_dst(), h - pad_h, w - pad_w, h, &et3);
          EXPECT_EQ(false, r);
          if (r != false)
            continue;
        }
#if CHECK_RESULTS
        ASSERT_EQ(queue_->enqueueReadBuffer(d_dst, CL_TRUE, 0, sizeof(T)*(h-pad_h)*(w-pad_w),
              dst_gpu), CL_SUCCESS);
        // compute golden
        // [h/t][t][w] to [h/t][w][t]
        cpu_aos_asta(src, dst, h-pad_h, w-pad_w, 1);
        // [h/t][w][t] to [h/t][t][w]
        cpu_soa_asta(dst, src, (w-pad_w)*1, h-pad_h, 1);
        EXPECT_EQ(0, compare_output(dst_gpu, src, (h-pad_h)*(w-pad_w)));
#endif
      }
    }

    else if(b >= 6 && b*A*a <= MAX_MEM){
      std::cerr << "" << A*a << ",";
      std::cerr << "" << B << "," << b << ",";

      // 2-stage approach
      for (int n = 0; n < N+WARM_UP; n++) {
        if (n == WARM_UP){
          et = 0; et2 = 0; et3 = 0;
        }
        // Padding
        if (pad_w > 0){
          r = cl_padding((*queue_)(), d_dst(), w - pad_w, h - pad_h, w, &et2);
          EXPECT_EQ(false, r);
          if (r != false)
            continue;
        }
        // Transpose
        r = cl_transpose((*queue_)(), d_dst(), A, a, B, b, 1, 22, &et);
        EXPECT_EQ(false, r);
        if (r != false)
          continue;
        // Unpadding
        if (pad_h > 0){
          r = cl_unpadding((*queue_)(), d_dst(), h - pad_h, w - pad_w, h, &et3);
          EXPECT_EQ(false, r);
          if (r != false)
            continue;
        }
#if CHECK_RESULTS
        ASSERT_EQ(queue_->enqueueReadBuffer(d_dst, CL_TRUE, 0, sizeof(T)*(h-pad_h)*(w-pad_w),
              dst_gpu), CL_SUCCESS);
        // compute golden
        // [h/t][t][w] to [h/t][w][t]
        cpu_aos_asta(src, dst, h-pad_h, w-pad_w, 1);
        // [h/t][w][t] to [h/t][t][w]
        cpu_soa_asta(dst, src, (w-pad_w)*1, h-pad_h, 1);
        EXPECT_EQ(0, compare_output(dst_gpu, src, (h-pad_h)*(w-pad_w)));
#endif
      }
    }

    else{
      std::cerr << "" << A << "," << a << ",";
      std::cerr << "" << B << "," << b <<",";

      // 3-stage approach
      for (int n = 0; n < N+WARM_UP; n++) {
        if (n == WARM_UP){
          et = 0; et2 = 0; et3 = 0;
        }
        // Padding
        if (pad_w > 0){
          r = cl_padding((*queue_)(), d_dst(), w - pad_w, h - pad_h, w, &et2);
          EXPECT_EQ(false, r);
          if (r != false)
            continue;
        }
        // Transpose
        if (a <= b)
          r = cl_transpose((*queue_)(), d_dst(), A, a, B, b, 1, 3, &et);
        else
          r = cl_transpose((*queue_)(), d_dst(), A, a, B, b, 1, 32, &et);
        EXPECT_EQ(false, r);
        if (r != false)
          continue;
        // Unpadding
        if (pad_h > 0){
          r = cl_unpadding((*queue_)(), d_dst(), h - pad_h, w - pad_w, h, &et3);
          EXPECT_EQ(false, r);
          if (r != false)
            continue;
        }
#if CHECK_RESULTS
        ASSERT_EQ(queue_->enqueueReadBuffer(d_dst, CL_TRUE, 0, sizeof(T)*(h-pad_h)*(w-pad_w),
              dst_gpu), CL_SUCCESS);
        // compute golden
        // [h/t][t][w] to [h/t][w][t]
        cpu_aos_asta(src, dst, h-pad_h, w-pad_w, 1);
        // [h/t][w][t] to [h/t][t][w]
        cpu_soa_asta(dst, src, (w-pad_w)*1, h-pad_h, 1);
        EXPECT_EQ(0, compare_output(dst_gpu, src, (h-pad_h)*(w-pad_w)));
#endif
      }
    }

#if BRUTE
    if(et < max_et) max_et = et;
    }
  }
  std::cerr << "" << h << "," << w << "\t";
  std::cerr << "Max_Throughput = " << float(h*w*2*sizeof(T)) / max_et;
  std::cerr << " GB/s\n";
#else
  std::cerr << "Padding_Throughput = " << float(2*h*w*sizeof(T)*N)/et2;
  std::cerr << " GB/s\t";
  std::cerr << "Transpose_Throughput = " << float(2*h*w*sizeof(T)*N)/et;
  std::cerr << " GB/s\t";
  std::cerr << "Unpadding_Throughput = " << float(2*h*w*sizeof(T)*N)/et3;
  std::cerr << " GB/s\t";
  std::cerr << "Throughput = " << float(2*h*w*sizeof(T)*N)/(et + et2 + et3);
  std::cerr << " GB/s\n";
#endif
  free(src);
  free(dst);
  free(dst_gpu);
  }
}

// testing 0100 transformation AaBb->ABab
TEST_F(libmarshal_cl_test, test_0100) {
  int bs[] = {256}; //{32};
  int Bs[] = {57};
  int as[] = {62};
  int As[] = {128};
  int b = bs[0];
  int B = Bs[0];
  int a = as[0];
  int A = As[0];
  size_t size = A*a*B*b;

  T *src = (T*)malloc(sizeof(T)*size);
  T *dst = (T*)malloc(sizeof(T)*size);
  T *dst_gpu = (T*)malloc(sizeof(T)*size);
  generate_vector(src, size);

  cl_int err;
  cl::Buffer d_dst = cl::Buffer(*context_, CL_MEM_READ_WRITE,
      sizeof(T)*size, NULL, &err);
  ASSERT_EQ(err, CL_SUCCESS);
  ASSERT_EQ(queue_->enqueueWriteBuffer(
        d_dst, CL_TRUE, 0, sizeof(T)*size, src), CL_SUCCESS);
  bool r = false;
  r = cl_transpose_0100((*queue_)(), d_dst(), A, a, B, b, NULL);
  // This may fail
  EXPECT_EQ(false, r);
  // compute golden: A instances of aBb->Bab transformation
  for (int i = 0; i < A; i++) {
    cpu_soa_asta(src+i*(a*B*b),
      dst+i*(a*B*b), B*b /*h*/, a /*w*/, b /*t*/);
  }
  ASSERT_EQ(queue_->enqueueReadBuffer(d_dst, CL_TRUE, 0,
    sizeof(T)*size, dst_gpu), CL_SUCCESS);
  EXPECT_EQ(0, compare_output(dst_gpu, dst, size));
  free(src);
  free(dst);
  free(dst_gpu);
}

TEST(libmarshal_plan_test, cycle) {
  Transposition tx(2,3), tx2(3,5);
  EXPECT_EQ(1, tx.GetNumCycles());
  EXPECT_EQ(2, tx2.GetNumCycles());
}
