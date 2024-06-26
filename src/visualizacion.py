from particula import Particula
from secuencial import Secuencial
from matriz_atraccion import Matriz_atraccion
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from paralelo import Paralelo

matriz = Matriz_atraccion().matriz_atraccion
particulas = [Particula() for _ in range(200)]


anim_secuencial = Secuencial(matriz_atraccion=matriz, particulas=particulas,
                                     r_max = 7, beta = 0.01, mu = 0.9, force_factor = 0.04)

anim_secuencial.visualizar(save=True)

'''
anim_paralela = Paralelo(matriz_atraccion=matriz, particulas=particulas,
                                     r_max = 15, beta = 0.01, mu = 0.9, force_factor = 0.5, tam_punto=1)
anim_paralela.visualizar()
'''