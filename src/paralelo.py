import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Paralelo:
    def __init__(self, matriz_atraccion, particulas, tam_pantalla = 10, tam_punto = 3, r_max = 4, 
                 beta = 0.01, mu = 0.9, dt = 0.03, force_factor = 1):
         # Parametros
        self.matriz_atraccion = matriz_atraccion
        self.particulas = particulas
        self.tam_pantalla = tam_pantalla 
        self.mitad_tam_pantalla = tam_pantalla/2
        self.tam_punto = tam_punto
        self.r_max = r_max
        self.beta = beta # Corte con el eje x de la funcion de fuerza
        self.mu = mu # Rozamiento
        self.dt = dt # Intervalo de tiempo
        self.force_factor = force_factor # Factor de fuerza
        self.kernel_velocidad = self.kernel_velocidad()
        self.kernel_posicion = self.kernel_posicion()
        self.block_size = 1024  # Máximo de hilos por bloque
        self.grid_size = (len(particulas) + self.block_size - 1) // self.block_size  # Calcula la cantidad de bloques necesarios

        # Pantalla
        plt.style.use('dark_background') # Fondo negro
        plt.rcParams['toolbar'] = 'None' # Ocultar barra de tareas
        self.fig, self.ax = fig, ax = plt.subplots()

    def visualizar(self, save = False):
        self.fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None) 

        self.ax.axis('off') # Eliminar ejes
        self.ax.set_xlim(-self.mitad_tam_pantalla, self.mitad_tam_pantalla)
        self.ax.set_ylim(-self.mitad_tam_pantalla, self.mitad_tam_pantalla)

        # Crear la animación
        ani = FuncAnimation(self.fig, self.update, frames=np.arange(1000), init_func=self.init, blit=True, interval = 1/self.dt)
        if save:
            ani.save('animation.gif', writer='imagemagick')

        # Mostrar la animación
        plt.show()
    
    def init(self):
        for particula in self.particulas:
            self.ax.plot(particula.posicionX, particula.posicionY, marker='o', color=particula.color, markersize=self.tam_punto)
        return []
    
    def kernel_velocidad(self):
        # Código del kernel para actualizar velocidades
        mod_velocidad = SourceModule("""
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
        """)
        return mod_velocidad.get_function("actualizar_velocidad_kernel")
    
    def kernel_posicion(self):
        mod_posicion = SourceModule("""
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
        """)
        return mod_posicion.get_function("actualizar_posicion_kernel")

    def actualizar_posicion(self):
        posicionesX = np.array([p.posicionX for p in self.particulas])
        posicionesY = np.array([p.posicionY for p in self.particulas])
        velocidadesX = np.array([p.velocidadX for p in self.particulas])
        velocidadesY = np.array([p.velocidadY for p in self.particulas])

        args_position = [cuda.InOut(posicionesX),
                        cuda.InOut(posicionesY),
                            cuda.In(velocidadesX),
                          cuda.In(velocidadesY),
                            np.int32(len(self.particulas)),
                          np.float32(self.dt),
                            np.float32(self.tam_pantalla),
                         np.float32(self.mitad_tam_pantalla)]
        
        self.kernel_posicion(*args_position, block=(self.block_size, 1, 1), grid=(self.grid_size, 1))

        for i, p in enumerate(self.particulas):
            p.posicionX = posicionesX[i]
            p.posicionY = posicionesY[i]

    def actualizar_velocidad(self):
        posicionesX = np.array([p.posicionX for p in self.particulas])
        posicionesY = np.array([p.posicionY for p in self.particulas])
        velocidadesX = np.array([p.velocidadX for p in self.particulas])
        velocidadesY = np.array([p.velocidadY for p in self.particulas])

        args_velocity = [cuda.InOut(posicionesX),
                          cuda.InOut(posicionesY),
                         cuda.InOut(velocidadesX),
                           cuda.InOut(velocidadesY),
                         cuda.In(self.colores),
                           cuda.In(self.matriz_atraccion),
                         np.int32(len(self.particulas)),
                           np.float32(self.r_max),
                             np.float32(self.dt),
                         np.float32(self.mu),
                           np.float32(self.force_factor),
                             np.float32(self.beta)]

        self.kernel_velocidad(*args_velocity, block=(self.block_size, 1, 1), grid=(self.grid_size, 1))

        for i, v in enumerate(self.particulas):
            v.velocidadX = velocidadesX[i]
            v.velocidadY = velocidadesY[i]