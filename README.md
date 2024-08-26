# Particle-of-life-Simulation

## Introduction

This project implements a simulation "**Particle of Life**" style, which is intended to make an approximation to **Miller-Urey experiment**. This experiment simulated the atmosferic conditions of prebiotic earth, before life emerged. From a set of just inorganic molecules and physical conditions, it is seen that simple organic compounds are synthesized, suggesting a **possible origin of life**.

Based on this astonishing discovery, the question that led to this project was: **Is it possible to simulate the creation of organic molecules and structures from scratch?**. Moreover, in the traditional Particle of Life simulation, structures and patterns are formed depending on the parameters set. This analogy makes one think that with the valid conditions and adjustments, the PoL simulation may be able to simulate the Miller-Urey experiment.

The objective of this project is ambitious and just an idea. The main goal is to learn the different techniques used for the implementation while having a real and very interesting scalable project. 

![GIF](https://github.com/GonzaloM786/Particle-of-life-Simulation/blob/main/simulaciones/animation4.gif?raw=true)

## Characteristics

### Properties and force

The particles in the bidimentional space have three properties: **position, speed and color**. The force of attraction between them is calculated according to the following formula:

![image](https://github.com/user-attachments/assets/6b78d5a8-2833-4c87-b840-9105eacb0616)
 
Where **β** is a constant, **r** is the distance between the particles and **a** is a factor that depends of the color of the two particles, it varies between [-1, 1]. A graphical example of the attraction force as a function of the distance is represented below:

![image](https://github.com/user-attachments/assets/82fb3e72-09d4-41e3-b2b5-b08884a8d4d6)

Note that the force is always repulsive when the particles are very close, avoiding overlapping particles. Moreover, it can be attractive or repulsive, depending on **a**. This formula was chosen since it offers a simple, fast-computable calculus and pseudorealistic first approach. This will need to be changend in future versions to better approximate the elemtal forces between the particles.

### Maximum distance generalization

As in electromagnetic force, the formula used in this experiment decreases the force with increasing distance. Therefore, a maximum distance was parameterized in order to severely reduce the number of calculations yet offering a minimal loss of information.

In addition, the formula has a default maximum distance of 1 unit, so the ecuation was scaled to the maximum distance set.

### Position and speed update

Setting a new parameter **dt** that states the time interval between istances of the simulation, the new position and speed of the particles can be easily updated:

- **New_position = Speed * dt**
- **New_speed = Force * dt**

Since (force = acceleration * mass) and (mass = 1; for all particles)
 
### Friction

Friction was simply implemented this way:

- **Speed_after_friction = Speed_before_friction * μ**

Mu is a parameter in the interval [0, 1]


