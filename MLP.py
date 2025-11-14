import numpy as np
class Neuron:
    
    def __init__(self,n_inputs,activation):
        self.n_inputs = n_inputs
        self.bias = np.random.uniform(-1,1)
        self.weights = [np.random.uniform(-1,1) for _ in range(n_inputs)]
        activations = {"logistic":self.logistic,"relu":self.relu,"hyperbolic_tangent":self.hyperbolic_tangent}
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

class Layer:
    def __init__(self,n_inputs,n_neurons,activation):
        
        self.neurons = [Neuron(n_inputs,activation) for _ in range(n_neurons)]
    
    def output(self,inputs):
        outs =  np.column_stack([n.output(inputs) for n in self.neurons])
        return outs

class MLP:
    def __init__(self,n_nodes_layer,inputs,activation):
        self.inputs = inputs
        dim = [inputs.shape[1]] + n_nodes_layer
        self.layers = [Layer(n_i,n_o,activation) for n_i,n_o in zip(dim[:-1],dim[1:])]
    
    def output(self):
        out = self.inputs
        for layer in self.layers:
            out = layer.output(out)
        return out
    
    
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])
mlp = MLP([2,1],X,"logistic")
print(mlp.output())
