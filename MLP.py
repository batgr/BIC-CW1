import numpy as np
class Neuron:
    
    def __init__(self,n_inputs,activation):
        self.n_inputs = n_inputs
        self.bias = np.random.uniform(-1,1)
        self.weights = [np.random.uniform(-1,1) for _ in range(n_inputs)]
        self.activation = activation

    def output(self,inputs):
        z = np.dot(inputs,self.weights) + self.bias
        out = self.activation(z)
        return out

class Layer:
    def __init__(self,n_inputs,n_neurons,activation):
        
        self.neurons = [Neuron(n_inputs,activation) for _ in range(n_neurons)]
    
    def output(self,inputs):
        outs = [n.output(inputs) for n in self.neurons]
        return outs

class MLP:
    def __init__(self,n_outputs_layer,inputs,activation):
        self.inputs = inputs
        dim = [len(inputs)] + n_outputs_layer
        self.layers = [Layer(n_i,n_o,activation) for n_i,n_o in zip(dim[:-1],dim[1:])]
    
    def output(self):
        out = self.inputs
        for layer in self.layers:
            out = layer.output(out)
        return out
            

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


x = [1.0,2.0]
dim_layer = [2,2]

mlp = MLP(dim_layer,x,sigmoid)
print(mlp.output())
