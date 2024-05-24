import numpy as np
import matplotlib.pyplot as plt

def fuerza(r, beta, a):
    if r < beta:
        return (r/beta) - 1
    elif(r < 1):
        return a*(1-(abs(2*r -1 - beta))/(1-beta))
    else:
        return 0


def plot_fuerza(beta, a, filename):
    r_values = np.linspace(0, 1, 400)
    fuerza_values = np.array([fuerza(r, beta, a) for r in r_values])

    fig, ax = plt.subplots()
    ax.plot(r_values, fuerza_values, label=f'β = {beta}, a = {a}')
    ax.axhline(0, color='black', linestyle='--', linewidth=0.7)  # Línea horizontal en y=0
    ax.set_title('Fuerza en función de la distancia')
    ax.set_xlabel('Distancia (r)')
    ax.set_ylabel('Fuerza')
    ax.legend()
    ax.grid(True)

    ax.grid(False)

    fig.savefig(filename)
    plt.show()

plot_fuerza(beta=0.5, a=1, filename="fuerza-distancia.png")