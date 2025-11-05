import numpy as np
class Neuron:
    
    def __init__(self,inputs,activation):
        self.inputs = inputs
        self.n_inputs = len(inputs)
        self.bias = np.random.uniform(-1,1)
        self.weights = [np.random.uniform(-1,1) for _ in range(self.n_inputs)]
        self.activation = activation

    def output(self):
        z = sum((x*w for x,w in zip(self.inputs,self.weights)),self.bias)
        out = self.activation(z)
        return

