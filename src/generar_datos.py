from particula import Particula
from secuencial import Secuencial
from paralelo import Paralelo
from matriz_atraccion import Matriz_atraccion
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_line(x, y):
    _, ax = plt.subplots()

    ax.plot(x, y)
    ax.set_title('Tiempo vs Número de Partículas')
    ax.set_ylabel('Tiempo (s)')
    ax.set_xlabel('Número de Partículas')

    plt.show() 

def mostrar_graficas():
    df1 = pd.read_csv("datos_paralelo.csv")
    x1 = df1['Numero de Particulas'].tolist()
    y1 = df1['Tiempo'].tolist()

    df2 = pd.read_csv("datos_secuencial.csv")
    x2 = df2['Numero de Particulas'].tolist()
    y2 = df2['Tiempo'].tolist()

    datos = [(x1, y1), (x2, y2)]

    _, ax = plt.subplots()

    for (x, y) in datos:
        ax.plot(x, y)

    ax.set_title('Tiempo vs Número de Partículas')
    ax.set_ylabel('Tiempo (s)')
    ax.set_xlabel('Número de Partículas')

    plt.show()

def mostrar_grafica(name):
    df = pd.read_csv(name)
    x = df['Numero de Particulas'].tolist()
    y = df['Tiempo'].tolist()

    plot_line(x, y)

def generar_datos_secuencial(matriz, rango):
    x_secuencial = list(rango)
    y_secuencial = []
    for n_part in rango:
        particulas = [Particula() for _ in range(n_part)]
        anim_secuencial = Secuencial(matriz_atraccion=matriz, particulas=particulas,
                                     r_max = 7, beta = 0.01, mu = 0.9, force_factor = 1, generar_figura=False)
        start = time.time()
        for _ in range(50):
            for j in range(len(particulas)):
                anim_secuencial.actualizar_velocidad(j)
                anim_secuencial.actualizar_posicion(j)
        y_secuencial.append(time.time() - start)

    df = pd.DataFrame({'Numero de Particulas': x_secuencial, 'Tiempo': y_secuencial})
    df.to_csv('datos_secuencial.csv', index=False)

def generar_datos_paralelo(matriz, rango):
    x_paralelo = list(rango)
    y_paralelo = []
    for n_part in rango:
        particulas = [Particula() for _ in range(n_part)]
        anim_paralela = Paralelo(matriz_atraccion=matriz, particulas=particulas,
                                     r_max = 7, beta = 0.01, mu = 0.9, force_factor = 1, generar_figura=False)
        start = time.time()
        for _ in range(50):
            anim_paralela.actualizar_velocidad()
            anim_paralela.actualizar_posicion()
        y_paralelo.append(time.time() - start)

    df = pd.DataFrame({'Numero de Particulas': x_paralelo, 'Tiempo': y_paralelo})
    df.to_csv('datos_paralelo.csv', index=False)



# =========================== PARAMETROS ===========================

matriz_atraccion = Matriz_atraccion().matriz_atraccion
intervalo = range(100, 10001, 100)

# =========================== GENERAR DATOS ===========================

#generar_datos_secuencial(matriz_atraccion, intervalo)
#generar_datos_paralelo(matriz_atraccion, intervalo)

# =========================== VISUALIZAR DATOS ===========================

#mostrar_grafica("datos_paralelo.csv")
#mostrar_grafica("datos_secuencial.csv")
mostrar_graficas()