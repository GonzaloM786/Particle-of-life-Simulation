import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from particula import Particula

class Secuencial:

    def __init__(self, matriz_atraccion, particulas, tam_pantalla = 10, tam_punto = 3, r_max = 4, 
                 beta = 0.01, mu = 0.9, dt = 0.03, force_factor = 1, generar_figura = True):
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

        if generar_figura:
            plt.style.use('dark_background') # Fondo negro
            plt.rcParams['toolbar'] = 'None' # Ocultar barra de tareas
            self.fig, self.ax = plt.subplots()

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

    # Función de inicialización (opcional)
    def init(self):
        for particula in self.particulas:
            self.ax.plot(particula.posicionX, particula.posicionY, marker='o', color=particula.color, markersize=self.tam_punto)
        return []

    def fuerza(self, r, a):
        if r < self.beta:
            return (r/self.beta) - 1
        else:
            return a*(1-(abs(2*r -1 - self.beta))/(1-self.beta))


    def actualizar_velocidad(self, n_particula):
        # Inicializar fuerzas
        totalForceX = 0
        totalForceY = 0
        # Obtener partícula
        particula = self.particulas[n_particula]

        # Computar la fuerza con el resto de partículas
        for j in range(len(self.particulas)):
            if n_particula == j: continue
            rX, rY, r = particula.distancia(self.particulas[j])
            if r < self.r_max:
                atraccion = self.matriz_atraccion.at[particula.color, self.particulas[j].color]
                f = self.fuerza(r/self.r_max, atraccion)
                totalForceX += rX / r * f
                totalForceY += rY / r * f
        
        # Ponderar fuerza
        totalForceX *= self.r_max*self.force_factor
        totalForceY *= self.r_max*self.force_factor

        # Rozamiento
        particula.velocidadX *= self.mu
        particula.velocidadY *= self.mu

        # Actualizar velociada
        particula.velocidadX += totalForceX*self.dt
        particula.velocidadY += totalForceY*self.dt

    def actualizar_posicion(self, n_particula):
        # Obtener partícula
        particula = self.particulas[n_particula]
        # Actualizar posición
        particula.posicionX += particula.velocidadX * self.dt
        particula.posicionY += particula.velocidadY * self.dt
        
        # Hacer que la partícula no se salga de la pantalla
        if particula.posicionX < -self.mitad_tam_pantalla:
            particula.posicionX += self.tam_pantalla 
        elif particula.posicionX > self.mitad_tam_pantalla:
            particula.posicionX -= self.tam_pantalla

        if particula.posicionY < -self.mitad_tam_pantalla:
            particula.posicionY += self.tam_pantalla 
        elif particula.posicionY > self.mitad_tam_pantalla:
            particula.posicionY -= self.tam_pantalla
        

    # Función de actualización para animar el punto
    def update(self, frame):
        self.ax.clear()
        self.ax.axis('off')
        self.ax.set_xlim(-self.mitad_tam_pantalla, self.mitad_tam_pantalla)
        self.ax.set_ylim(-self.mitad_tam_pantalla, self.mitad_tam_pantalla)
        for n_particula in range(len(self.particulas)):
            self.actualizar_velocidad(n_particula)
            self.actualizar_posicion(n_particula)
            self.ax.plot(self.particulas[n_particula].posicionX, self.particulas[n_particula].posicionY, marker='o', color=self.particulas[n_particula].color, markersize=self.tam_punto)
        return []





