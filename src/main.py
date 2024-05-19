from particula import Particula
from secuencial import Secuencial
from matriz_atraccion import Matriz_atraccion

matriz = Matriz_atraccion().matriz_atraccion


particulas = [Particula() for _ in range(50)]

anim_secuencial = Secuencial(matriz_atraccion=matriz, particulas=particulas)

anim_secuencial.visualizar()