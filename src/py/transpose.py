import pyopencl as cl
import basic

class Transposition:
  def __init__(self, mm, nn):
    self.m = mm
    self.n = nn
    self.nr_cycles = -1 # # of nontrivial cycles

  def Next(self, i):
    return i*self.m - (self.m*self.n-1)*(i/self.n);

  def GetNumCycles(self):
    if (self.nr_cycles >= 0):
      return self.nr_cycles
    else:
      self.nr_cycles = 0;
    for i in range(1, self.m*self.n):
      j = self.Next(i)
      if j == i:
        continue
      while j > i:
        j = self.Next(j)
      if j <> i:
        continue
      self.nr_cycles = self.nr_cycles+1
    return self.nr_cycles
  def IsFeasible():
    return False

class T010_BS(Transposition):
  def __init__(self, AA, aa, BB):
    Transposition.__init__(self, aa, BB)
    self.A = AA
    self.a = aa
    self.B = BB
  def IsFeasible(self, buf):
    ctx = buf.get_info(cl.mem_info.CONTEXT)
    local_sz = ctx.get_info(cl.context_info.DEVICES)[0].get_info(cl.device_info.LOCAL_MEM_SIZE)
    return self.m*self.n < local_sz/4-128;
  def Transpose(self, queue, buf):
    self.event = basic.Transpose010BS(queue, buf, self.A, self.a, self.B)
    return self
  def Throughput(self):
    self.event.wait()
    st = self.event.get_profiling_info(cl.profiling_info.START)
    e = self.event.get_profiling_info(cl.profiling_info.END)
    return float(self.A*self.a*self.B*4*2)/(e-st)

class T100(Transposition):
  def __init__(self, AA, BB, bb):
    Transposition.__init__(self, AA, BB)
    self.A=AA
    self.B=BB
    self.b=bb
  def IsFeasible(self, buf):
    ctx = buf.get_info(cl.mem_info.CONTEXT)
    max_local = ctx.get_info(cl.context_info.DEVICES)[0].get_info(cl.device_info.MAX_WORK_GROUP_SIZE)
    return self.b < max_local
  def Transpose(self, queue, buf):
    self.event = basic.Transpose100(queue, buf, self.A, self.B, self.b)
    return self
  def Throughput(self):
    self.event.wait()
    st = self.event.get_profiling_info(cl.profiling_info.START)
    e = self.event.get_profiling_info(cl.profiling_info.END)
    return float(self.A*self.B*self.b*4*2)/(e-st)

class T0100(Transposition):
  def __init__(self, AA, aa, BB, bb):
    Transposition.__init__(self, aa, BB)
    self.A=AA
    self.B=BB
    self.a=aa
    self.b=bb
  def IsFeasible(self, buf):
    ctx = buf.get_info(cl.mem_info.CONTEXT)
    max_local = ctx.get_info(cl.context_info.DEVICES)[0].get_info(cl.device_info.MAX_WORK_GROUP_SIZE)
    return self.b < max_local
  def Transpose(self, queue, buf):
    self.event = basic.Transpose0100(queue, buf, self.A, self.a, self.B, self.b)
    return self
  def Throughput(self):
    self.event.wait()
    st = self.event.get_profiling_info(cl.profiling_info.START)
    e = self.event.get_profiling_info(cl.profiling_info.END)
    return float(self.A*self.a*self.B*self.b*4*2)/(e-st)

def FullTranspose(queue, buf, A, a, B, b):
  # Method 1: Aa >> Bb  (TBD)
  # Method 2: a < max work group size
  method2a = T100(A*a, B, b) # AaBb to BAab
  method2b = T010_BS(B*A, a, b) #BAab to BAba
  method2c = T0100(B, A, b, a) # BAba to BbAa
  if (method2a.IsFeasible(buf) and method2b.IsFeasible(buf) and method2c.IsFeasible(buf)):
    t1 = method2a.Transpose(queue, buf).Throughput();
    t2 = method2b.Transpose(queue, buf).Throughput();
    t3 = method2c.Transpose(queue, buf).Throughput();
    #print "Method2:"+str([t1, t2, t3])
    #print 1/(1/t1+1/t2+1/t3)
  else:
    # Fallback
    fallback = T0100(1, A*a, B*b, 1)
    fallback.Transpose(queue, buf)
    #print fallback.Throughput()

# Operations on pyopencl.array.Array
def TransposeArrayInPlace(arr, a=1, b=1):
  if (len(arr.shape) <> 2):
    raise TypeError("only allows transposing 2D arrays")
  if (arr.shape[0]%a <> 0 or arr.shape[1]%b <> 0):
    raise TypeError("tile size must evenly divide array dimensions")
  FullTranspose(arr.queue, arr.data, arr.shape[0]/a, a, arr.shape[1]/b, b)
  new_strides = (arr.size/(arr.strides[0]/arr.strides[1])*arr.strides[1], arr.strides[1])
  return arr._new_with_changes(data=arr.data,
    shape=(arr.shape[1], arr.shape[0]), strides=new_strides)
