#ifndef LIBMARSHAL_CL_PROFILE_H_
#define LIBMARSHAL_CL_PROFILE_H_

#include "local_cl.hpp"
namespace {
class Profiling {
 public:
  Profiling(cl::CommandQueue q, std::string id): id_(id),
    profiling_status_(CL_SUCCESS), queue_(q) {
    clRetainCommandQueue(q());
    cl_command_queue_properties cp;
    profiling_status_ = clGetCommandQueueInfo(queue_(),
      CL_QUEUE_PROPERTIES, sizeof(cl_command_queue_properties),
        &cp, NULL);
    if (!(cp & CL_QUEUE_PROFILING_ENABLE)) {
      std::cerr << "[Profile] Command queue is not set for profiling\n";
    }
  }
  cl::Event *GetEvent(void) { return &event_; }
  
  ~Profiling() {}

  cl_ulong Report(void) {
    profiling_status_ = queue_.finish();
    if (profiling_status_ != CL_SUCCESS) {
      std::cerr << "[Profile] "<<id_ << ": clFinish() failed on the queue: ";
      std::cerr << profiling_status_ << "\n";
      return 0;
    }
    cl_ulong start_time, end_time;
    profiling_status_ |= clGetEventProfilingInfo(event_(),
        CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time, NULL);
    profiling_status_ |= clGetEventProfilingInfo(event_(),
        CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, NULL);
    if (profiling_status_ == CL_SUCCESS) {
      return end_time - start_time;
    } else {
      std::cerr << "[Profile] " << id_ << " profiling failed: " <<
        profiling_status_ << std::endl;
      return 0;
    }
  }
  void Report(size_t amount_s) {
    cl_ulong elapsed_time = Report();
    if (elapsed_time) {
      std::cerr << "[Profile] " << id_ << " "<< elapsed_time
          << "ns"; 
      if (amount_s) {
        double amount = amount_s; 
        double throughput = (amount)/(elapsed_time);
        std::cerr << "; " << throughput << " GB/s" << std::endl;
      } else {
        std::cerr << std::endl;
      }
    }
  }
 private:
  std::string id_;
  cl_int profiling_status_;
  cl::CommandQueue queue_;
  cl::Event event_;
  cl_command_queue_properties old_priority_;
};
}

#endif // LIBMARSHAL_CL_PROFILE_H_
