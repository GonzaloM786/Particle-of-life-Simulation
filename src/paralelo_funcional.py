import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

# Código del kernel para actualizar velocidades
kernel_code_velocity = """
__device__ float fuerza(float r, float beta, float a) {
    if (r < beta) {
        return r / beta - 1.0f;
    } else if (beta < r && r < 1.0f) {
        return a * (1.0f - abs(2.0f * r - 1.0f - beta) / (1.0f - beta));
    } else {
        return 0.0f;
    }
}

__global__ void actualizar_velocidad_kernel(float* posX, float* posY, float* velX, float* velY, float* colores, float* matriz_atraccion, int num_particulas, float r_max, float dt, float mu, float force_factor, float beta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_particulas) {
        float totalForceX = 0.0f;
        float totalForceY = 0.0f;

        for (int j = 0; j < num_particulas; j++) {
            if (idx == j) continue;
            float rX = posX[j] - posX[idx];
            float rY = posY[j] - posY[idx];
            float r = sqrtf(rX * rX + rY * rY);

            if (r > 0.05f && r < r_max) {
                int color_idx = (int)(colores[idx] * num_particulas + colores[j]);
                float atraccion = matriz_atraccion[color_idx];
                float f = fuerza(r / r_max, beta, atraccion);
                totalForceX += rX / r * f;
                totalForceY += rY / r * f;
            }
        }

        totalForceX *= r_max * force_factor;
        totalForceY *= r_max * force_factor;

        velX[idx] *= mu;
        velY[idx] *= mu;

        velX[idx] += totalForceX * dt;
        velY[idx] += totalForceY * dt;
    }
}
"""

# Compilar los kernels
mod_velocity = SourceModule(kernel_code_velocity)

# Obtener la función del kernel compilado
func_velocity = mod_velocity.get_function("actualizar_velocidad_kernel")

# Definir los parámetros del kernel
num_particulas = 1000  # Número de partículas
dt = 0.01  # Delta de tiempo
r_max = 10.0  # Radio máximo de interacción
mu = 0.9  # Coeficiente de fricción
force_factor = 1.0  # Factor de fuerza
beta = 0.1  # Parámetro beta

# Crear los arreglos de numpy para las posiciones, velocidades y colores
posicionesX = np.random.rand(num_particulas).astype(np.float32)
posicionesY = np.random.rand(num_particulas).astype(np.float32)
velocidadesX = np.random.rand(num_particulas).astype(np.float32)
velocidadesY = np.random.rand(num_particulas).astype(np.float32)
colores = np.random.randint(0, 3, size=num_particulas).astype(np.float32)
matriz_atraccion = np.random.rand(3, 3).astype(np.float32).flatten()

# Definir los argumentos del kernel
args_velocity = [cuda.InOut(posicionesX), cuda.InOut(posicionesY), cuda.InOut(velocidadesX), cuda.InOut(velocidadesY), cuda.In(colores), cuda.In(matriz_atraccion), np.int32(num_particulas), np.float32(r_max), np.float32(dt), np.float32(mu), np.float32(force_factor), np.float32(beta)]

# Configurar la cantidad de bloques y hilos por bloque
block_size = 1024  # Máximo de hilos por bloque
grid_size = (num_particulas + block_size - 1) // block_size  # Calcula la cantidad de bloques necesarios

# Ejecutar el kernel de actualización de velocidades
func_velocity(*args_velocity, block=(block_size, 1, 1), grid=(grid_size, 1))

# Imprimir las nuevas posiciones y velocidades para verificar que todo funcione correctamente
print("Nuevas posiciones X:", posicionesX)
print("Nuevas posiciones Y:", posicionesY)
print("Nuevas velocidades X:", velocidadesX)
print("Nuevas velocidades Y:", velocidadesY)
