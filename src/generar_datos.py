from particula import Particula
from secuencial import Secuencial
from matriz_atraccion import Matriz_atraccion
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_line(x, y):
    _, ax = plt.subplots()

    ax.plot(x, y)
    ax.set_title('Número de Partículas vs Tiempo')
    ax.set_xlabel('Tiempo')
    ax.set_ylabel('Número de Partículas')

    plt.show() 

def generar_datos_secuencial():
    matriz = Matriz_atraccion().matriz_atraccion
    x_secuencial = list(range(100, 1001, 100))
    y_secuencial = []
    for n_part in range(100, 1001, 100):
        particulas = [Particula() for _ in range(n_part)]
        anim_secuencial = Secuencial(matriz_atraccion=matriz, particulas=particulas,
                                     r_max = 7, beta = 0.01, mu = 0.9, force_factor = 1)
        start = time.time()
        for i in range(50):
            for j in range(len(particulas)):
                anim_secuencial.actualizar_velocidad(j)
                anim_secuencial.actualizar_posicion(j)
        y_secuencial.append(time.time() - start)

    df = pd.DataFrame({'Numero de Particulas': x_secuencial, 'Tiempo': y_secuencial})
    df.to_csv('simulacion.csv', index=False)


#generar_datos_secuencial()


df = pd.read_csv('simulacion.csv')
x_secuencial = df['Numero de Particulas'].tolist()
y_secuencial = df['Tiempo'].tolist()

plot_line(x_secuencial, y_secuencial)
