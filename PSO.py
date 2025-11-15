import numpy as np
def pso(swarmsize,dim,n_iter,alpha,beta,gamma,sigma,epsilon,min_bound, max_bound,assess_fitness):
    
    
    particles = np.random.uniform(-1, 1, (swarmsize, dim))

    velocities = np.random.uniform(-1, 1, (swarmsize, dim))

    best_particles = np.copy(particles)
    best_fitness = np.full(swarmsize, -np.inf)
    
    informants = [[i,(i-1)%swarmsize,(i+1)%swarmsize ] for i in range(swarmsize)]

    

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
            particles[pos] = np.clip(particles[pos], min_bound, max_bound)
            
    return best_global_particle
                





