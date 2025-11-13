import numpy as np

class Particle:
    def __init__(self,swarmsize):
        self.position = np.random.uniform(-swarmsize,swarmsize)
        self.velocity = np.random.uniform(-swarmsize,swarmsize)
        self.best_position = self.position
        self.best_fitness = -np.inf
        
class PSO:
    def __init__(self,swarmsize,a,b,g,l,s,e,assess_fitness):
        
        self.particles = {Particle(swarmsize) for _ in range(swarmsize)}
        self.best = None
        self.assess_fitness = assess_fitness
        self.best_fittest = None
    
    def optimize(self):
        while True:
            for particle in self.particles:
                fitness = self.assess_fitness(particle)
                 
                if self.best is None or fitness > self.best_fittest:
                    self.best = particle
                    particle.
            
            for particle in self.particles
                
        



     
            
            
            
    
    