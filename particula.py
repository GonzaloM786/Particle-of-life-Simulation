import random

class Particula:

    def __init__(self, tam_pantalla):
        self.color = random.choice(['r', 'g', 'b', 'y'])
        self.posicionX = random.randint(0, tam_pantalla)
        self.posicionY = random.randint(0, tam_pantalla)
        self.velocidadX = 0
        self.velocidadY = 0

    
    def distancia(self, punto):
        distX = punto.posicionX - self.posicionX 
        distY = punto.posicionY - self.posicionY
        dist_euclidea = ((distX**2) + (distY**2))**0.5
        return distX, distY, dist_euclidea
    
