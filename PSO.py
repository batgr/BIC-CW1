import numpy as np

class Particle:
    def __init__(self,swarmsize,dim):
        self.dim = dim
        self.pos = np.array([np.random.uniform(-swarmsize,swarmsize) for _ in range(dim)])
        self.velocity = np.array([np.random.uniform(-swarmsize,swarmsize) for _ in range(dim)])
        
        self.best_pos = self.pos.copy()
        self.best_fit = None
        
        self.informants = []
    
    def get_best_informants_pos(self):
        idx = np.argmax([b.best_fit for b in self.informants])
        return self.informants[idx].best_pos
    
    def update_best(self,fitness):
        if self.best_fit is None:
            self.best_fit = fitness
            return
        
        if fitness > self.best_fit:
            self.best_fit = fitness
            self.best_pos = self.pos.copy()
        
    def update_velocity(self,global_best_pos,a,b,g,s):
        best_particle_pos = self.best_pos
        global_best_pos = global_best_pos
        best_informant_pos = self.get_best_informants_pos()
        
        for i in range(self.dim):
                    p1= np.random.uniform(0.0,b)
                    p2= np.random.uniform(0.0,g)
                    p3= np.random.uniform(0.0,s)
                    
                    v = self.velocity[i]
                    pos = self.pos[i]
                    self.velocity[i] = (a*v +p1*(best_particle_pos[i] - pos) +
                                        p3*(global_best_pos[i]-pos) + 
                                        p2*(best_informant_pos[i]-pos)
                                        )
        
        
       
        
class PSO:
    def __init__(self,swarmsize,alpha,beta,gamma,sigma,epsilon,assess_fitness,dim,iter,n_informants):
        self.swarmsize = swarmsize
        self.assess_fitness = assess_fitness
        self.dim = dim
        self.iter = iter
        self.n_informants = n_informants if n_informants < swarmsize else swarmsize
        self.a, self.b, self.g, self.s, self.e = alpha, beta, gamma, sigma, epsilon

        self.init_swarm()
        
        self.global_best_pos = None
        self.global_best_fit = None
       
    def init_swarm(self):
        self.swarm = [Particle(self.swarmsize,self.dim) for _ in range(self.swarmsize)]    
        self.init_informants()
        
    def init_informants(self):
        for particle in self.swarm:
            particle.informants.extend(np.random.choice(self.swarm,self.n_informants,False))


    def optimize(self):
        for _ in range(self.iter):
            self.update_global_best()
                    
            for particle in self.swarm:
               
                particle.update_velocity(self.global_best_pos,self.a,self.b,self.g,self.s)
                
            for particle in self.swarm:
                particle.pos = particle.pos+ self.e*particle.velocity
            
        return self.global_best_pos,self.global_best_fit
    
    def update_global_best(self):
        for particle in self.swarm:
                fit = self.assess_fitness(particle.pos)
                particle.update_best(fit)
                 
                if self.global_best_pos is None or fit > self.global_best_fit:
                    self.global_best_pos = particle.pos
                    self.global_best_fit = fit
                    
    
        
    

