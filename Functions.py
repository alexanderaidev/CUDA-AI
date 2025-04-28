import cupy as cp

class activation_functions():
    def __init__(self, value):

        pass

    
    @staticmethod
    def ReLu(x):
        return cp.maximum(0, x)
    
    @staticmethod
    def ReLu_Derivative(x):
        return cp.where(x < 0, 0.0, 1.0)
    

    @staticmethod
    def LeakyReLU(x, alpha):
        x = cp.maximum(x, x * alpha, dtype=cp.float32)
        return x

    @staticmethod
    def LeakyReLU_derivative(x, alpha):
        x = cp.where(x < 0, alpha, 1.0)
        return x
    

    @staticmethod
    def Tanh(x):
        return cp.tanh(x)
    
    @staticmethod
    def Tanh_derivative(x):
        return 1 - activation_functions.Tanh(x) **2


    @staticmethod
    def Sigmoid(x):
        x = 1 / (1 + cp.exp(-x))
        return x
    
    @staticmethod
    def Sigmoid_Derivative(x):
        s = activation_functions.Sigmoid(x)
        return s * (1 - s)
    

    @staticmethod
    def Softmax(x):
        exp_x = cp.exp(x - cp.max(x))
        return exp_x / cp.sum(exp_x, axis= -1, keepdims=True)
    
    @staticmethod
    def Softmax_Derivative(x):
        s = activation_functions.Softmax(x)
        return s * (1 - s)
    
