import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from particula import Particula

# Parametros
tam_pantalla = 10
tam_punto = 3
n = 100 # numero de particulas
r_max = 4
beta = 0.01 # Corte con el eje x de la funcion de fuerza
mu = 0.9 # Rozamiento
dt = 0.03
force_factor = 1

# Matriz de atraccion
def generar_matriz_aleatoria():
    matriz = np.random.uniform(-1, 1, size=(4, 4))
    matriz_simetrica = (matriz + matriz.T) / 2
    filas_columnas = ['r', 'lime', 'c', 'y']
    matriz_atraccion = pd.DataFrame(matriz_simetrica, index=filas_columnas, columns=filas_columnas)
    return matriz_atraccion

matriz_atraccion = generar_matriz_aleatoria()


# Pantalla
plt.style.use('dark_background') # Fondo negro
plt.rcParams['toolbar'] = 'None' # Ocultar barra de tareas

fig, ax = plt.subplots()
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None) 

ax.axis('off') # Eliminar ejes
mitad_tam_pantalla = tam_pantalla/2
ax.set_xlim(-mitad_tam_pantalla, mitad_tam_pantalla)
ax.set_ylim(-mitad_tam_pantalla, mitad_tam_pantalla)

# Puntos
particulas = [Particula(mitad_tam_pantalla) for _ in range(n)]

# Función de inicialización (opcional)
def init():
    for particula in particulas:
        ax.plot(particula.posicionX, particula.posicionY, marker='o', color=particula.color, markersize=tam_punto)
    return []

def fuerza(r, a):
    if r < beta:
        return r/beta - 1
    elif beta < r and r < 1:
        return a*(1-(abs(2*r -1 -beta))/(1-beta))
    else:
        return 0

def actualizar_velocidad(n_particula):
    totalForceX = 0
    totalForceY = 0
    particula = particulas[n_particula]

    for j in range(len(particulas)):
        if n_particula == j: continue
        rX, rY, r = particula.distancia(particulas[j])
        if r > 0.05 and r < r_max:
            atraccion = matriz_atraccion.at[particula.color, particulas[j].color]
            #fuerza = atraccion/r
            f = fuerza(r/r_max, atraccion)
            totalForceX += rX / r * f
            totalForceY += rY / r * f
    
    totalForceX *= r_max*force_factor
    totalForceY *= r_max*force_factor

    particula.velocidadX *= mu
    particula.velocidadY *= mu

    particula.velocidadX += totalForceX*dt
    particula.velocidadY += totalForceY*dt

def actualizar_posicion(n_particula):
    particula = particulas[n_particula]
    particula.posicionX += particula.velocidadX * dt
    particula.posicionY += particula.velocidadY * dt
    
    if particula.posicionX < -mitad_tam_pantalla:
        particula.posicionX += tam_pantalla 
    elif particula.posicionX > mitad_tam_pantalla:
        particula.posicionX -= tam_pantalla

    if particula.posicionY < -mitad_tam_pantalla:
        particula.posicionY += tam_pantalla 
    elif particula.posicionY > mitad_tam_pantalla:
        particula.posicionY -= tam_pantalla
    

# Función de actualización para animar el punto
def update(frame):
    ax.clear()
    ax.axis('off')
    ax.set_xlim(-mitad_tam_pantalla, mitad_tam_pantalla)
    ax.set_ylim(-mitad_tam_pantalla, mitad_tam_pantalla)
    for n_particula in range(len(particulas)):
        actualizar_velocidad(n_particula)
        actualizar_posicion(n_particula)
        ax.plot(particulas[n_particula].posicionX, particulas[n_particula].posicionY, marker='o', color=particulas[n_particula].color, markersize=tam_punto)

    return []


# Crear el punto inicialmente vacío
#point, = ax.plot([], [], marker='o', color='r')

# Crear la animación
ani = FuncAnimation(fig, update, frames=np.arange(100), init_func=init, blit=True, interval = 1/dt)
#ani.save('animation.gif', writer='imagemagick')

# Mostrar la animación
plt.show()


