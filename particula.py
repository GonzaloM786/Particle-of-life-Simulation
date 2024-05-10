import random

class Particula:

    def __init__(self, tam_pantalla):
        self.color = random.choice(['r', 'lime', 'c', 'y'])
        self.posicionX = random.uniform(-tam_pantalla, tam_pantalla)
        self.posicionY = random.uniform(-tam_pantalla, tam_pantalla)
        self.velocidadX = 0
        self.velocidadY = 0

    
    def distancia(self, punto):
        distX = punto.posicionX - self.posicionX 
        distY = punto.posicionY - self.posicionY
        dist_euclidea = ((distX**2) + (distY**2))**0.5
        return distX, distY, dist_euclidea
    
