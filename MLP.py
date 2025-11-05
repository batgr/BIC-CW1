import numpy as np
class Neuron:
    
    def __init__(self,inputs):
        self.inputs = inputs
        self.n_inputs = len(inputs)
        self.bias = np.random.uniform(-1,1)
        self.weight = [np.random.uniform(-1,1) for _ in range(self.n_inputs)]



neuron = Neuron([1.0,2.0,3.0])
print(neuron.inputs)
print(neuron.bias)
print(neuron.weight)
    