# Basic Transposition routines
import pyopencl as cl
import numpy as np
import inspect, os

#initialization -- obtain the CL file path
print inspect.getfile(inspect.currentframe()) # script filename (usually with path)
scriptdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
program_path = os.path.join(scriptdir,'..','..','src','cl','cl_aos_asta.cl')

def LoadAndCompileCL(ctx, path):
  f = open(path, 'r')
  program = f.read();
  return cl.Program(ctx, program).build()

def Transpose0100(queue, buf, A, a, B, b):
  ctx = queue.get_info(cl.command_queue_info.CONTEXT)
  program = LoadAndCompileCL(ctx, program_path)
  ev=program.transpose_0100(queue, (a*B*b, 256/b), (b, 256/b), buf,
    np.uint32(A), np.uint32(a), np.uint32(B), np.uint32(b))
  return ev

# PTTWAC
def Transpose100(queue, buf, A, B, b):
  ctx = queue.get_info(cl.command_queue_info.CONTEXT)
  program = LoadAndCompileCL(ctx, program_path)
  finished = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, A*B*4)
  evf=program.mymemset(queue, (A*B,), None, finished);
  queue.finish()
  ev=program.transpose_100(queue,
    (min(A*B*b, b*1024),), (b,), buf,
    np.uint32(A), np.uint32(B), np.uint32(b), finished)
  return ev

# IPT
def Transpose100IPT(queue, buf, A, B, b):
  return Transpose0100(queue, buf, 1, A, B, b)

def Transpose010BS(queue, buf, A, a, B):
  ctx = queue.get_info(cl.command_queue_info.CONTEXT)
  program = LoadAndCompileCL(ctx, program_path)
  ev=program.BS_marshal(queue, (A*256,), (256,), 
    buf, np.uint32(a), np.uint32(B), cl.LocalMemory(B*(a+1)*4))
  return ev


