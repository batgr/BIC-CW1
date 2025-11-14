import numpy as np
from PSO import PSO
class Neuron:
    
    def __init__(self,n_inputs,activation):
        self.n_inputs = n_inputs
        self.bias = None
        self.weights = None
        self.activation = activation

    def output(self,inputs):
        z = np.dot(inputs,self.weights) + self.bias
        out = self.activation(z)
        return out
    
    def set_params(self,weights,bias):
        self.weights = weights
        self.bias = bias

class Layer:
    def __init__(self,n_inputs,n_neurons,activation):
        
        self.neurons = [Neuron(n_inputs,activation) for _ in range(n_neurons)]
    
    def output(self,inputs):
        outs = [n.output(inputs) for n in self.neurons]
        return outs

class MLP:
    def __init__(self,n_layer_nodes,inputs,activation):
        self.inputs = inputs
        dim = [len(inputs)] + n_layer_nodes
        self.layers = [Layer(n_i,n_o,activation) for n_i,n_o in zip(dim[:-1],dim[1:])]
        self.n_params = sum((n_i*n_o )+1 for n_i,n_o in zip(dim[:-1],dim[1:]))
    
    def output(self):
        out = self.inputs
        for layer in self.layers:
            out = layer.output(out)
        return out
    
    def set_params(self,params):
         i = 0
         for layer in self.layers:
            for neuron in layer.neurons:
                n = neuron.n_inputs
                weights = params[i : i+n]
                bias = params[i+n]
                neuron.set_params(weights, bias)
                i += n + 1
    
  
    
    





