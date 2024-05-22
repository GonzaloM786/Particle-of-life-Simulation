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
        self.matriz_atraccion = matriz_atraccion.values.flatten().astype(np.float32)
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
        self.colores = np.array([self.color_to_index(p.color) for p in self.particulas], dtype=np.int32)

        # Pantalla
        plt.style.use('dark_background') # Fondo negro
        plt.rcParams['toolbar'] = 'None' # Ocultar barra de tareas
        self.fig, self.ax = plt.subplots()

    def color_to_index(self, color):
        color_dict = {'r': 0, 'lime': 1, 'c': 2, 'y': 3}
        return color_dict.get(color)

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

                }
            }

            """)
        return mod_velocidad.get_function("actualizar_velocidad_kernel")
    
    def kernel_posicion(self):
        mod_posicion = SourceModule("""
            __global__ void actualizar_posicion_kernel(float* posX, float* posY, float* velX, float* velY,
                                     int num_particulas, float dt, float tam_pantalla, float mitad_tam_pantalla) {
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

        args_position = [cuda.InOut(posicionesX), cuda.InOut(posicionesY),
                            cuda.In(velocidadesX), cuda.In(velocidadesY),
                            np.int32(len(self.particulas)), np.float32(self.dt),
                            np.float32(self.tam_pantalla), np.float32(self.mitad_tam_pantalla)]
        
        self.kernel_posicion(*args_position, block=(self.block_size, 1, 1), grid=(self.grid_size, 1))

        for i, p in enumerate(self.particulas):
            p.posicionX = posicionesX[i]
            p.posicionY = posicionesY[i]

    def actualizar_velocidad(self):
        posicionesX = np.array([p.posicionX for p in self.particulas])
        posicionesY = np.array([p.posicionY for p in self.particulas])
        velocidadesX = np.array([p.velocidadX for p in self.particulas])
        velocidadesY = np.array([p.velocidadY for p in self.particulas])

        args_velocity = [cuda.In(posicionesX), cuda.In(posicionesY),
                         cuda.InOut(velocidadesX), cuda.InOut(velocidadesY),
                         cuda.In(self.colores), cuda.In(self.matriz_atraccion),
                         np.int32(len(self.particulas)), np.float32(self.r_max),
                         np.float32(self.beta), np.float32(self.mu),
                             np.float32(self.dt), np.float32(self.force_factor)]

        self.kernel_velocidad(*args_velocity, block=(self.block_size, 1, 1), grid=(self.grid_size, 1))

        for i, v in enumerate(self.particulas):
            v.velocidadX = velocidadesX[i]
            v.velocidadY = velocidadesY[i]


    def update(self, frame):
        self.ax.clear()
        self.ax.axis('off')
        self.ax.set_xlim(-self.mitad_tam_pantalla, self.mitad_tam_pantalla)
        self.ax.set_ylim(-self.mitad_tam_pantalla, self.mitad_tam_pantalla)
        self.actualizar_velocidad()
        self.actualizar_posicion()
        for n_particula in range(len(self.particulas)):
            self.ax.plot(self.particulas[n_particula].posicionX, self.particulas[n_particula].posicionY,
                          marker='o', color=self.particulas[n_particula].color, markersize=self.tam_punto)
        return []