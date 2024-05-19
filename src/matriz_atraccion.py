import numpy as np
import pandas as pd

class Matriz_atraccion:
    def __init__(self):
        self.matriz_atraccion = self.generar_matriz_aleatoria()
    
    def generar_matriz_aleatoria(self):
        matriz = np.random.uniform(-1, 1, size=(4, 4))
        matriz_simetrica = (matriz + matriz.T) / 2
        filas_columnas = ['r', 'lime', 'c', 'y']
        matriz_atraccion = pd.DataFrame(matriz_simetrica, index=filas_columnas, columns=filas_columnas)
        return matriz_atraccion