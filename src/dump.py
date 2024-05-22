import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
from matriz_atraccion import Matriz_atraccion
from particula import Particula



# Código del kernel para actualizar velocidades
kernel_code_velocity = """
        __global__ void actualizar_velocidad_kernel(
    float *posX, float *posY, float *velX, float *velY, int *colores, float *matriz_atraccion, 
    int n, float r_max, float beta, float mu, float dt, float force_factor) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float totalForceX = 0;
        float totalForceY = 0;
        float px = posX[idx];
        float py = posY[idx];
        int color_idx = colores[idx];

        for (int j = 0; j < n; ++j) {
            if (idx == j) continue;

            float dx = posX[j] - px;
            float dy = posY[j] - py;
            float r = sqrt(dx * dx + dy * dy);

            if (r > 0.05 && r < r_max) {
                int other_color_idx = colores[j];
                float atraccion = matriz_atraccion[color_idx * 4 + other_color_idx];
                
                float f;
                float r_normalized = r / r_max;
                if (r_normalized < beta) {
                    f = r_normalized / beta - 1;
                } else if (beta < r_normalized && r_normalized < 1) {
                    f = atraccion * (1 - abs(2 * r_normalized - 1 - beta) / (1 - beta));
                } else {
                    f = 0;
                }

                totalForceX += dx / r * f;
                totalForceY += dy / r * f;
            }
        }

        totalForceX *= r_max * force_factor;
        totalForceY *= r_max * force_factor;

        velX[idx] = velX[idx] * mu + totalForceX * dt;
        velY[idx] = velY[idx] * mu + totalForceY * dt;

        //posX[idx] += velX[idx] * dt;
        //posY[idx] += velY[idx] * dt;

        //if (posX[idx] < -mitad_tam_pantalla) posX[idx] += tam_pantalla;
        //else if (posX[idx] > mitad_tam_pantalla) posX[idx] -= tam_pantalla;

        //if (posY[idx] < -mitad_tam_pantalla) posY[idx] += tam_pantalla;
        //else if (posY[idx] > mitad_tam_pantalla) posY[idx] -= tam_pantalla;
    }
}

        """

# Kernel de Actualización de Posición:
kernel_code_position = """
__global__ void actualizar_posicion_kernel(float* posX, float* posY, float* velX, float* velY, int num_particulas, float dt, float tam_pantalla, float mitad_tam_pantalla) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_particulas) {
        posX[idx] += velX[idx] * dt;
        posY[idx] += velY[idx] * dt;

        if (posX[idx] < -mitad_tam_pantalla)
            posX[idx] += tam_pantalla;
        else if (posX[idx] > mitad_tam_pantalla)
            posX[idx] -= tam_pantalla;

        if (posY[idx] < -mitad_tam_pantalla)
            posY[idx] += tam_pantalla;
        else if (posY[idx] > mitad_tam_pantalla)
            posY[idx] -= tam_pantalla;
    }
}
"""

# Compilar los kernels
mod_velocity = SourceModule(kernel_code_velocity)
mod_position = SourceModule(kernel_code_position)

# Obtener las funciones del kernel compilado
func_velocity = mod_velocity.get_function("actualizar_velocidad_kernel")
func_position = mod_position.get_function("actualizar_posicion_kernel")
'''
# Definir los parámetros del kernel
num_particulas = 1000  # Número de partículas
dt = 0.01  # Delta de tiempo
tam_pantalla = 100.0  # Tamaño de la pantalla
mitad_tam_pantalla = tam_pantalla / 2  # Mitad del tamaño de la pantalla
r_max = 10.0  # Radio máximo de interacción
mu = 0.9  # Coeficiente de fricción
force_factor = 1.0  # Factor de fuerza


# Crear los arreglos de numpy para las posiciones, velocidades y colores
posicionesX = np.random.rand(num_particulas).astype(np.float32)
posicionesY = np.random.rand(num_particulas).astype(np.float32)
velocidadesX = np.random.rand(num_particulas).astype(np.float32)
velocidadesY = np.random.rand(num_particulas).astype(np.float32)
colores = np.random.randint(0, 3, size=num_particulas).astype(np.float32)
matriz_atraccion = np.random.rand(3, 3).astype(np.float32).flatten()
'''

matriz_atraccion = Matriz_atraccion().matriz_atraccion
matriz_atraccion = matriz_atraccion.values.flatten().astype(np.float32)
particulas = [Particula() for _ in range(5000)]
tam_pantalla = 10
mitad_tam_pantalla = tam_pantalla/2
tam_punto = 3
r_max = 20
beta = 0.01 # Corte con el eje x de la funcion de fuerza
mu = 0.9 # Rozamiento
dt = 0.03 # Intervalo de tiempo
force_factor = 1 # Factor de fuerza

def color_to_index(color):
    color_dict = {'r': 0, 'lime': 1, 'c': 2, 'y': 3}
    return color_dict.get(color)

posicionesX = np.array([p.posicionX for p in particulas])
posicionesY = np.array([p.posicionY for p in particulas])
velocidadesX = np.array([p.velocidadX for p in particulas])
velocidadesY = np.array([p.velocidadY for p in particulas])
colores = np.array([color_to_index(p.color) for p in particulas], dtype=np.int32)


'''
# Definir los argumentos del kernel
#args_velocity = [cuda.InOut(posicionesX), cuda.InOut(posicionesY), cuda.InOut(velocidadesX), cuda.InOut(velocidadesY), cuda.In(colores), cuda.In(matriz_atraccion), np.int32(len(particulas)), np.float32(r_max), np.float32(dt), np.float32(mu), np.float32(force_factor)]
args_velocity = [cuda.InOut(posicionesX), cuda.InOut(posicionesY),
                cuda.InOut(velocidadesX), cuda.InOut(velocidadesY),
                cuda.In(colores), cuda.In(matriz_atraccion),
                np.int32(len(particulas)), np.float32(r_max), 
                np.float32(beta), np.float32(mu), np.float32(dt),
                np.float32(force_factor), np.float32(tam_pantalla), 
                np.float32(mitad_tam_pantalla)]
args_position = [cuda.InOut(posicionesX), cuda.InOut(posicionesY),
                  cuda.In(velocidadesX), cuda.In(velocidadesY),
                    np.int32(len(particulas)), np.float32(dt),
                      np.float32(tam_pantalla), np.float32(mitad_tam_pantalla)]
'''
# Configurar la cantidad de bloques y hilos por bloque
block_size = 1024  # Máximo de hilos por bloque
grid_size = (len(particulas) + block_size - 1) // block_size  # Calcula la cantidad de bloques necesarios


print("Antiguas posiciones X:", posicionesX)
print("Antiguas posiciones Y:", posicionesY)
print("Antiguas velocidades X:", velocidadesX)
print("Antiguas velocidades Y:", velocidadesY)


# Ejecutar el kernel de actualización de velocidades
func_velocity(cuda.In(posicionesX),
                          cuda.In(posicionesY),
                         cuda.InOut(velocidadesX),
                           cuda.InOut(velocidadesY),
                         cuda.In(colores),
                           cuda.In(matriz_atraccion),
            np.int32(len(particulas)), np.float32(r_max), 
            np.float32(beta), np.float32(mu), np.float32(dt),
            np.float32(force_factor),
            block=(block_size, 1, 1), grid=(grid_size, 1))

# Ejecutar el kernel de actualización de posiciones
func_position(cuda.InOut(posicionesX),
                        cuda.InOut(posicionesY),
                            cuda.In(velocidadesX),
                          cuda.In(velocidadesY),
                            np.int32(len(particulas)), np.float32(dt),
                            np.float32(tam_pantalla), np.float32(mitad_tam_pantalla),
                            block=(block_size, 1, 1), grid=(grid_size, 1))




# Imprimir las nuevas posiciones y velocidades
print("Nuevas posiciones X:", posicionesX)
print("Nuevas posiciones Y:", posicionesY)
print("Nuevas velocidades X:", velocidadesX)
print("Nuevas velocidades Y:", velocidadesY)
