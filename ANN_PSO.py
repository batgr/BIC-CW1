import numpy as np
from PSO import PSO

class Neuron:
    
    def __init__(self,n_inputs,activation):
        self.n_inputs = n_inputs
        self.bias = None
        self.weights = None
        activations = {
            "logistic":self.logistic,
            "relu":self.relu,
            "hyperbolic_tangent":self.hyperbolic_tangent,
            "linear":self.linear}
        self.activation = activations[activation]

    def output(self,inputs):
        z = np.dot(inputs,self.weights) + self.bias
        out = self.activation(z)
        return out
    
    def set_params(self,weights,bias):
        self.weights = np.array(weights)
        self.bias = bias
    
    def logistic(self,z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    def relu(self,z):
        return np.maximum(0, z)
    def hyperbolic_tangent(self,z):
        return np.tanh(z)
    def linear(self,z):
        return z
class Layer:
    def __init__(self,n_inputs,n_neurons,activation):
        
        self.neurons = [Neuron(n_inputs,activation) for _ in range(n_neurons)]
        self.n_inputs = n_inputs
    
    def output(self,inputs):
        outs = np.column_stack([n.output(inputs) for n in self.neurons])
        return outs
class MLP:
    def __init__(self,n_nodes_layer,n_inputs,hidden_activation, output_activation):
        
        dim = [n_inputs] + n_nodes_layer
        self.layers = []
        for n_i, n_o in zip(dim[:-2], dim[1:-1]):
            self.layers.append(Layer(n_i, n_o, hidden_activation))

        self.layers.append(Layer(dim[-2], dim[-1], output_activation))
        self.n_params = sum((n_i + 1) * n_o for n_i, n_o in zip(dim[:-1], dim[1:]))

    def output(self,inputs):
        out = inputs
        for layer in self.layers:
            out = layer.output(out)
        return out
    
    def set_params(self,params):
         idx = 0
         for layer in self.layers:
            for neuron in layer.neurons:
                weights = params[idx : idx + layer.n_inputs]
                idx += layer.n_inputs
                bias = params[idx]
                idx += 1
                neuron.set_params(weights, bias)
    


class ANN_PSO:
    def __init__(self,n_nodes_layer,hidden_activation, output_activation,swarmsize,alpha,beta,gamma,sigma,epsilon,iter,n_informants,metric,low=-2.0, high=2.0):
        self.n_nodes_layer = n_nodes_layer
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.swarmsize = swarmsize
        self.alpha=alpha
        self.beta = beta
        self.gamma = gamma
        self.sigma = sigma
        self.epsilon = epsilon
        self.iter = iter
        self.n_informants = n_informants
        self.mlp = None
        self.pso = None
        self.y = None
        self.X = None
        metrics = {"mse":self.mse,"rmse":self.rmse}
        self.metric = metrics[metric]
        
    
    def assess_fitness(self,params):
        self.mlp.set_params(params)
        out = self.mlp.output(self.X)
        return -self.metric(out)
    
    def fit(self,X,y):
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        self.y = y
        self.X=X
        n_inputs = X.shape[1]
        self.mlp = MLP(self.n_nodes_layer,n_inputs,self.hidden_activation,self.output_activation)
        dim = self.mlp.n_params
        
        pso = PSO(
            self.swarmsize,-5,5,self.alpha,self.beta,
            self.gamma,self.sigma,self.epsilon,
            self.assess_fitness,dim,self.iter,
            self.n_informants)
        params = pso.optimize()
        self.mlp.set_params(params)
       
    def mse(self,preds):
        return np.mean(np.square(preds - self.y))
    
    def rmse(self,preds):
        return np.sqrt(self.mse(preds))
        
    
    def predict(self,X):
       
        return self.mlp.output(X)






