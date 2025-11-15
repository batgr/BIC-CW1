# import numpy as np

# class Particle:
#     def __init__(self,dim,low,high):
#         self.dim = dim
#         self.pos =  np.random.uniform(low, high, dim)
#         self.velocity = np.array([np.random.uniform(-1,1) for _ in range(dim)])
        
#         self.best_pos = self.pos.copy()
#         self.best_fit = None
        
#         self.informants = []
    
#     def get_best_informants_pos(self):
#         valid = [b for b in self.informants if b.best_fit is not None]
#         if not valid:
#             return self.best_pos
#         idx = np.argmax([b.best_fit for b in valid])
#         return valid[idx].best_pos

    
#     def update_best(self, fitness):
#         if self.best_fit is None or fitness > self.best_fit:
#             self.best_fit = fitness
#             self.best_pos = self.pos.copy()

        
#     def update_velocity(self,global_best_pos,a,b,g,s):
#         best_particle_pos = self.best_pos
#         global_best_pos = global_best_pos
#         best_informant_pos = self.get_best_informants_pos()
        
#         for i in range(self.dim):
#                     p1= np.random.uniform(0.0,b)
#                     p2= np.random.uniform(0.0,g)
#                     p3= np.random.uniform(0.0,s)
                    
#                     v = self.velocity[i]
#                     pos = self.pos[i]
#                     self.velocity[i] = (a*v +p1*(best_particle_pos[i] - pos) +
#                                         p3*(global_best_pos[i]-pos) + 
#                                         p2*(best_informant_pos[i]-pos)
#                                         )
            

              
        
# class PSO:
#     def __init__(self,swarmsize,low,high,alpha,beta,gamma,sigma,epsilon,assess_fitness,dim,iter,n_informants):
#         self.swarmsize = swarmsize
#         self.assess_fitness = assess_fitness
#         self.dim = dim
#         self.iter = iter
#         self.n_informants = n_informants if n_informants < swarmsize else swarmsize
#         self.a, self.b, self.g, self.s, self.e = alpha, beta, gamma, sigma, epsilon
#         self.low,self.high = low,high

#         self.init_swarm()
        
#         self.global_best_pos = None
#         self.global_best_fit = None
        
#         self.update_global_best()
       
#     def init_swarm(self):
        
#         self.swarm = [Particle(self.dim, self.low, self.high) for _ in range(self.swarmsize)]    
#         self.init_informants()
        
#     def init_informants(self):
#         for particle in self.swarm:
#             other_particles = [p for p in self.swarm if p is not particle]
#             selected = np.random.choice(other_particles, min(self.n_informants, len(other_particles)),replace=False)
#             particle.informants.extend(selected)


#     def optimize(self):
#         for _ in range(self.iter):
            
#             for particle in self.swarm:    
#                 particle.update_velocity(self.global_best_pos,self.a,self.b,self.g,self.s)
                
#             for particle in self.swarm:
#                 particle.pos = particle.pos+ self.e*particle.velocity
                
                
#             self.update_global_best()
            
#         return self.global_best_pos
    
#     def update_global_best(self):
#         for particle in self.swarm:
#                 fit = self.assess_fitness(particle.pos)
#                 particle.update_best(fit)
                 
#                 if self.global_best_pos is None or fit > self.global_best_fit:
#                     self.global_best_pos = particle.pos
#                     self.global_best_fit = fit
                    
    
        
import numpy as np
def pso(swarmsize,dim,n_iter,alpha,beta,gamma,sigma,epsilon,assess_fitness):
    
    
    particles = np.random.uniform(-1, 1, (swarmsize, dim))

    velocities = np.random.uniform(-1, 1, (swarmsize, dim))

    best_particles = np.copy(particles)
    best_fitness = np.full(swarmsize, -np.inf)
    
    informants = [[i,(i-1)%swarmsize,(i+1)%swarmsize] for i in range(swarmsize)]

    

    best_global_particle = None
    best_global_fitness = None
    
    for _ in range(n_iter):
        
        for pos in range(swarmsize):
            particle = particles[pos]
            fitness = assess_fitness(particle)
            
            if fitness > best_fitness[pos]:
                best_particles[pos] = particle
                best_fitness[pos] = fitness
                
            if best_global_particle is None or fitness > best_global_fitness:
                best_global_particle = particle
                best_global_fitness = fitness
                 
        for pos in range(swarmsize):
            particle = particles[pos]
            prev_fittest_pos = best_particles[pos]
            prev_global_fittest = best_global_particle
            inf = informants[pos]
            best_idx = inf[np.argmax(best_fitness[i] for i in inf)]
            prev_best_fittest_informants = best_particles[best_idx]
            
            for i in range(dim):
                b = np.random.uniform(0.0,beta)
                c = np.random.uniform(0.0,gamma)
                d = np.random.uniform(0.0,sigma)
                
                velocity = velocities[pos][i]
                velocities[pos][i] = (
                    alpha * velocity + 
                    b*(prev_fittest_pos[i] - particle[i]) + 
                    c*(prev_best_fittest_informants[i] - particle[i]) +
                    d*(prev_global_fittest[i]- particle[i])
                    )
                
        for pos in range(swarmsize):
            particles[pos] = particles[pos] + epsilon*velocities[pos]
            
    return best_global_particle
                





