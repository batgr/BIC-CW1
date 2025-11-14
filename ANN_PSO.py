import numpy as np
from PSO import PSO

class Neuron:
    
    def __init__(self,n_inputs,activation):
        self.n_inputs = n_inputs
        self.bias = None
        self.weights = None
        activations = {"logistic":self.logistic,"relu":self.relu,"hyperbolic_tangent":self.hyperbolic_tangent}
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
class Layer:
    def __init__(self,n_inputs,n_neurons,activation):
        
        self.neurons = [Neuron(n_inputs,activation) for _ in range(n_neurons)]
        self.n_inputs = n_inputs
    
    def output(self,inputs):
        outs = np.column_stack([n.output(inputs) for n in self.neurons])
        return outs
class MLP:
    def __init__(self,n_nodes_layer,inputs,activation):
        self.inputs = inputs
        dim = [inputs.shape[1]] + n_nodes_layer
        self.layers = [Layer(n_i,n_o,activation) for n_i,n_o in zip(dim[:-1],dim[1:])]
        self.n_params = sum((n_i + 1) * n_o for n_i, n_o in zip(dim[:-1], dim[1:]))

    def output(self):
        out = self.inputs
        for layer in self.layers:
            out = layer.output(out)
        return out
    
    def set_params(self,params):
         i = 0
         for layer in self.layers:
            for neuron in layer.neurons:
                w = params[i : i + layer.n_inputs]
                b = params[i + layer.n_inputs]
                neuron.set_params(w, b)
                i += layer.n_inputs + 1
    


class ANN_PSO:
    def __init__(self,n_nodes_layer,activation,swarmsize,alpha,beta,gamma,sigma,epsilon,iter,n_informants,metric):
        self.n_nodes_layer = n_nodes_layer
        self.activation = activation
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
        metrics = {"mse":self.mse,"rmse":self.rmse}
        self.metric = metrics[metric]
        
    
    def assess_fitness(self,params):
        self.mlp.set_params(params)
        out = self.mlp.output()
        return -self.metric(out)
    
    def fit(self,X,y):
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        self.y = y
        self.mlp = MLP(self.n_nodes_layer,X,self.activation)
        dim = self.mlp.n_params
        
        pso = PSO(
            self.swarmsize,self.alpha,self.beta,
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
        self.mlp.inputs = X
        return self.mlp.output()


X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

y = np.array([[0], [1], [1], [0]])

ann_pso = ANN_PSO([4,1],"logistic",50,0.7,1.5,1.5,1.5,1.0,1000,3,"mse")
ann_pso.fit(X,y)
print(ann_pso.predict(X))





