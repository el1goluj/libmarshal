import pyopencl as cl
import pyopencl.array as cla
import numpy as np
import basic
import transpose 
import unittest

def CreateFirstGPUContext():
  platforms = cl.get_platforms()
  devices=[]
  for t in platforms:
    try:
      devices = t.get_devices(cl.device_type.GPU)
      break
    except:
      pass
  return cl.Context([devices[0],])


# Profiling version
def Prof(func, queue, *args):
  ev = func(queue, *args)
  ev.wait();
  prof1 = ev.get_profiling_info(cl.profiling_info.START)
  prof2 = ev.get_profiling_info(cl.profiling_info.END)
  return prof2 - prof1

class TransposeTestCase(unittest.TestCase):
  def setUp(self):
    self.ctx = CreateFirstGPUContext()
    self.queue = cl.CommandQueue(self.ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
  def NumpyTranspose0100(A, a, B, b):
    for AA in range(A):
      for aa in range(a):
        for BB in range(B):
          for bb in range(b):
            dst[AA][BB][aa] = src[AA][aa][BB].copy()

class TestFullTranspose(TransposeTestCase):
  def runTest(self):
    A = 40
    a_array = [4, 8, 16, 32, 64]
    B = 37
    b = 30;

    for a in a_array:
      src = np.random.rand(A*a, B*b)
      dst = np.empty((B*b, A*a))
      dst = src.transpose()
      hsrc = src.astype(np.float32)
      hdst = dst.astype(np.float32)
      src_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE|cl.mem_flags.COPY_HOST_PTR,
        hostbuf = hsrc)
      t = transpose.FullTranspose(self.queue, src_buf, A, a, B, b) 
      cl.enqueue_read_buffer(self.queue, src_buf, hsrc).wait()
      hsrct = hsrc.reshape(B*b, A*a)
      self.assertEqual((hdst.astype(np.float32) == hsrct).all(), True, 'Comparison failed')
    
class Test010(TransposeTestCase):
  def runTest(self):
    A = 40 #64*100
    a_array = [4, 8, 16, 32, 64]
    B = 37

    perf=[]
    for a in a_array:
      src = np.random.rand(A, a, B)
      dst = np.empty((A, B, a))
      for ii in range(A):
        dst[ii] = src[ii].transpose()
      hsrc = src.astype(np.float32)
      hdst = dst.astype(np.float32)
      src_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE|cl.mem_flags.COPY_HOST_PTR,
        hostbuf = hsrc)
      t = Prof(basic.Transpose010BS, self.queue, src_buf, A, a, B) 
      cl.enqueue_read_buffer(self.queue, src_buf, hsrc).wait()
      hsrct = hsrc.reshape(A, B, a)
      self.assertEqual((hdst.astype(np.float32) == hsrct).all(), True, 'Comparison failed')
      gbs = float(A*a*B*4*2)/t
      perf.append(gbs)

class TestArray(TransposeTestCase):
  def runTest(self):
    A = 40 #64*100
    a = 16
    B = 37
    b = 10
    src = np.random.rand(A*a, B*b).astype(np.float32)
    expected = src.transpose()
    dsrc = cla.to_device(self.queue, src)
    hdst = transpose.TransposeArrayInPlace(dsrc, a, b) 
    self.assertEqual(hdst.shape[0], B*b, 'Dimension mismatch')
    self.assertEqual(hdst.shape[1], A*a, 'Dimension mismatch')
    self.assertEqual((hdst.get() == expected).all(), True, 'Comparison failed')

if __name__ == '__main__':
  unittest.main()
