import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import numpy as np

a = np.random.randn(16)
a = a.astype(np.float32)

a_gpu = cuda.mem_alloc(a.nbytes)

cuda.memcpy_htod(a_gpu, a)

module = SourceModule("""
                __global__ void double_array(float *a){
                      int idx = blockIdx.x * blockDim.x + threadIdx.x;
                      a[idx] *= 2;
                }
                """)

funct = module.get_function("double_array")
funct(a_gpu, block=(16, 1, 1), grid=(1, 1, 1))

a_doubled = np.empty_like(a)

cuda.memcpy_dtoh(a_doubled, a_gpu)

print(a)
print(a_doubled)

# Inicializar PyCUDA
cuda.init()

# Seleccionar el dispositivo (GPU)
device = cuda.Device(0)

# Obtener las propiedades del dispositivo
props = device.get_attributes()

# Imprimir la cantidad máxima de hilos por bloque y la cantidad máxima de bloques en una dimensión
print("Máximo de hilos por bloque:", props.get(cuda.device_attribute.MAX_THREADS_PER_BLOCK))
print("Máximo de bloques en una dimensión:", props.get(cuda.device_attribute.MAX_GRID_DIM_X))
