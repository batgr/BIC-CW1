import numpy as np
from PSO import pso


class Neuron:
    
    def __init__(self,n_inputs,activation):
        self.n_inputs = n_inputs
        self.bias = None
        self.weights = None
        activations = {"logistic":self.logistic,"relu":self.relu,"tanh":self.hyperbolic_tangent,"linear":self.linear}
        self.activation = activations[activation]

    def output(self,inputs):
        z = np.dot(inputs,self.weights) + self.bias
        out = self.activation(z)
        return out
    
    def logistic(self,z):
        return 1 / (1 + np.exp(-z))
    def relu(self,z):
        return np.maximum(0, z)
    def hyperbolic_tangent(self,z):
        return np.tanh(z)
    def linear(self,z):
        return z

class Layer:
    def __init__(self,n_inputs,n_neurons,activation):
        
        self.neurons = [Neuron(n_inputs,activation) for _ in range(n_neurons)]
    
    def output(self,inputs):
        outs =  np.column_stack([n.output(inputs) for n in self.neurons])
        return outs

class ANN_PSO:
    def __init__(self,n_nodes_layer,hidden_activation, output_activation,swarmsize,n_iter,alpha,beta,gamma,sigma,epsilon,min_bound,max_bound):
        self.n_nodes_layer = n_nodes_layer
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.swarmsize = swarmsize
        self.n_iter = n_iter
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.sigma = sigma
        self.epsilon = epsilon
        self.min_bound = min_bound
        self.max_bound = max_bound
        self.layers = None
        self.y = None
        self.X = None
    
    def output(self,X):
        out = X
        for layer in self.layers:
            out = layer.output(out)
        return out
    
    def init_layers(self,dim_layers):
        
        n_layers = len(dim_layers) - 1
        
        self.layers = []
        
        for i in range(n_layers):
            n_i = dim_layers[i]
            n_o = dim_layers[i + 1]
            
            if i == n_layers - 1:
                activation = self.output_activation
            else:
                activation = self.hidden_activation
            
            self.layers.append(Layer(n_i, n_o, activation))
    
    def init_params(self,params):
        i = 0
        for layer in self.layers:
            for neuron in layer.neurons:
                n_inputs = neuron.n_inputs
                weights = params[i:n_inputs+i]
                bias = params[n_inputs+i]
                neuron.weights = np.array(weights)
                neuron.bias = float(bias)
                i+=n_inputs+1
                
    
    def fit(self,X,y):
        self.X = X
        self.y = y
        
        n_inputs = X.shape[1]
        
        dim_layers =[n_inputs] + self.n_nodes_layer
        self.init_layers(dim_layers)
        
        dim = sum((i+1)*j for i,j in zip(dim_layers[:-1],dim_layers[1:]))
        params_opt = pso(self.swarmsize,dim,self.n_iter,self.alpha,self.beta,self.gamma,self.sigma,self.epsilon,self.min_bound,self.max_bound,self.assess_fitness)
        self.init_params(params_opt)
        
    
    def predict(self,X):
        return self.output(X) 
    
    def assess_fitness(self,params):
        self.init_params(params)
        preds = self.output(self.X)
        return -self.mse(preds)
    
    def mse(self,preds):
         return np.mean(np.square(preds - self.y))
        
        



