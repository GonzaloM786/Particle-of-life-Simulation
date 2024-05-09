import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from particula import Particula

# Parametros
tam_pantalla = 10
tam_punto = 4
n = 20 # numero de particulas
r_max = 5
beta = 0.3 # Corte con el eje x de la funcion de fuerza
mu = 0.8 # Rozamiento
dt = 0.1 

# Matriz de atraccion
matriz_atraccion = np.array([[1, 0.3, -0.6, -0.8],
                             [0.3, -0.5, 0.8, -0.5],
                             [-0.6, 0.8, 1, 0.1],
                             [-0.8, -0.5, 0.1, 0.2]])

filas_columnas = ['r', 'lime', 'c', 'y']

matriz_atraccion = pd.DataFrame(matriz_atraccion, index=filas_columnas, columns=filas_columnas)




# Pantalla
plt.style.use('dark_background') # Fondo negro
plt.rcParams['toolbar'] = 'None' # Ocultar barra de tareas

fig, ax = plt.subplots()
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None) 

ax.axis('off') # Eliminar ejes
ax.set_xlim(0, tam_pantalla)
ax.set_ylim(0, tam_pantalla)


# Puntos
particulas = [Particula(tam_pantalla) for _ in range(n)]



# Función de inicialización (opcional)
def init():
    for particula in particulas:
        ax.plot(particula.posicionX, particula.posicionY, marker='o', color=particula.color, markersize=tam_punto)
    return []

def mover(particula):
    posX = particula.posicionX
    posY = particula.posicionY
    for p in particulas:
        distX, distY, distancia = particula.distancia(p)
        distX *= 1/r_max
        distY *= 1/r_max
        distancia *= 1/r_max
        if distancia != 0 and distancia < 1:
            if distancia < beta:
                particula.velocidadX = mu*particula.velocidadX + ((distX/beta)-1)*dt*r_max
                particula.velocidadY = mu*particula.velocidadY + ((distY/beta)-1)*dt*r_max
                posX = posX + ((distX/beta)-1)*dt*dt*r_max
                posY = posY + ((distY/beta)-1)*dt*dt*r_max
            else:
                atraccion = matriz_atraccion.at[particula.color, p.color]
                particula.velocidadX = mu*particula.velocidadX + (atraccion*(1-(abs(2*distX-1-beta)/(1-beta))))*dt*r_max
                particula.velocidadY = mu*particula.velocidadY + (atraccion*(1-(abs(2*distY-1-beta)/(1-beta))))*dt*r_max
                posX = posX + (atraccion*(1-(abs(2*distX-1-beta)/(1-beta))))*dt*dt*r_max
                posY = posY + (atraccion*(1-(abs(2*distY-1-beta)/(1-beta))))*dt*dt*r_max
    particula.posicionX = posX
    particula.posicionY = posY


# Función de actualización para animar el punto
def update(frame):
    ax.clear()
    ax.axis('off')
    ax.set_xlim(0, tam_pantalla)
    ax.set_ylim(0, tam_pantalla)
    for particula in particulas:
        mover(particula)
        ax.plot(particula.posicionX, particula.posicionY, marker='o', color=particula.color, markersize=tam_punto)

    return []


# Crear el punto inicialmente vacío
point, = ax.plot([], [], marker='o', color='r')

# Crear la animación
ani = FuncAnimation(fig, update, frames=np.arange(100), init_func=init, blit=True, interval = 1/dt)
#ani.save('animation.gif', writer='imagemagick')


# Mostrar la animación
plt.show()


