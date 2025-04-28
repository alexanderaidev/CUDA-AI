import cupy as cp
from Functions import activation_functions as F
import Optimizers as optim


class Agent_Model():
    def __init__ (self, state_dim, lr, momentum):  
        self.state_dim = state_dim
        self.ffn = []
        self.ffn_for_optimizers = []
        self.results = []
        self.lr = lr
        self.momentum = momentum
        
        
        self.layer1 = cp.random.normal(0, cp.sqrt(2 / state_dim), (state_dim, 128))
        self.layer2 = cp.random.normal(0, cp.sqrt(2 / 128), (128, 256))  
        self.layer3 = cp.random.normal(0, cp.sqrt(2 / 256), (256, 512))
        self.layer4 = cp.random.normal(0, cp.sqrt(2 / 512), (512, 512))
        self.layer5 = cp.random.normal(0, cp.sqrt(2 / 512), (512, 512))
        self.layer6 = cp.random.normal(0, cp.sqrt(2 / 512), (512, 512))
        self.layer7 = cp.random.normal(0, cp.sqrt(2 / 512), (512, 512))
        self.layer8 = cp.random.normal(0, cp.sqrt(2 / 512), (512, 1024))
        self.layer9 = cp.random.normal(0, cp.sqrt(2 / 1024), (1024, 1024))
        self.layer10 = cp.random.normal(0, cp.sqrt(2 / 1024), (1024, 1024))
        self.layer11 = cp.random.normal(0, cp.sqrt(2 / 1024), (1024, 512))
        self.layer12 = cp.random.normal(0, cp.sqrt(2 / 512), (512, 512))
        self.layer13 = cp.random.normal(0, cp.sqrt(2 / 512), (512, 512))
        self.layer14 = cp.random.normal(0, cp.sqrt(2 / 256), (512, 256))
        self.layer15 = cp.random.normal(0, cp.sqrt(2 / 256), (256, 128))
        self.layer16 = cp.random.normal(0, cp.sqrt(2 / 128), (128, 128))   
        self.layer17 = cp.random.normal(0, cp.sqrt(2 / 128), (128, 64))
        self.layer18 = cp.random.normal(0, cp.sqrt(2 / 64), (64, 32))
        self.layer19 = cp.random.normal(0, cp.sqrt(3 / 32), (32, 2))
        
        
        
        self.ffn.extend([self.layer1, self.layer2, self.layer3, self.layer4, self.layer5, self.layer6, self.layer7, self.layer8, self.layer9, self.layer10, self.layer11,
                        self.layer12, self.layer13, self.layer14, self.layer15, self.layer16, self.layer17, self.layer18])
       
       
        self.ffn_for_optimizers.extend([self.layer1, self.layer2, self.layer3, self.layer4, self.layer5, self.layer6, self.layer7, self.layer8, self.layer9, self.layer10, self.layer11,
                        self.layer12, self.layer13, self.layer14, self.layer15, self.layer16, self.layer17, self.layer18, self.layer19])
        

        self.optimizer = optim.SGDMomentum(lr=lr, momentum=momentum, parameters=self.ffn_for_optimizers)
    
    
    def layer_norm(self, x, eps=1e-5):
        # x: Eingabe mit Form (features,)
        # gamma, beta: trainierbare Parameter mit gleicher Form wie x
        mu = cp.mean(x)
        var = cp.var(x)
        x_hat = (x - mu) / cp.sqrt(var + eps)
        return x_hat      
                    
                    
                    
    def forward(self, x):
        
        self.results.clear()
        i = 0
        for layer in self.ffn:
            self.results.append(x)
            
            x = cp.matmul(x, layer)
            x = self.layer_norm(x)
            x = F.LeakyReLU(x, alpha=0.01)
            #print(f"Layer{i+1}: ", x.min(), x.max())
            i += 1
        
        self.results.append(x)
            

        # Layer 19 muss separat gehandhabt werden, da es Sigmoid braucht
        
        x = cp.matmul(x, self.layer19)
        
        #x = self.layer_norm(x)         # Layernorm wurde rausgenommen, da sonst die sigmoid in einer Sättigung wäre.
        
        self.results.append(x)
        print("Input vor Sigmoid: ", x.min(), x.max())
        x = F.Sigmoid(x)
        #print(f"Layer19: ", self.layer19.min(), self.layer19.max())
        
        
        return x
    
    
    def backward(self, delta):
        
        #gradients = []

        derivated = F.Sigmoid_Derivative(self.results[len(self.results) - 1])
        x = delta * derivated
        
        gradients_layer19 = cp.matmul(self.results[18].transpose(), x)                  # len(results)=20, daher -2 für input aus layer 18
        gradients_learning = F.Tanh(gradients_layer19)
        #print("Gradients_Layer19: ", gradients_learning.min(), gradients_learning.max())
        self.layer19 -= self.lr * (gradients_learning + 0.01 * self.layer19)
        
        #gradients.insert(0, gradients_layer19)
        
        # TODO Optimizer erstellen und integrieren
        mistake_next_layer = cp.matmul(x, self.layer19.transpose())      # Verlust für schicht 18
        #mistake_next_layer = self.layer_norm(mistake_next_layer)
        
        
        for i in range(len(self.results) - 2, 0, -1):           # Rückwärts über die liste iterieren
            
            derivated = F.LeakyReLU_derivative(self.results[i], alpha=0.01)
            x = mistake_next_layer * derivated
            #print(mistake_next_layer.min(), mistake_next_layer.max())
            #print("Results: ", self.results[i-1].min(), self.results[i-1].max())
            gradients_layer = cp.matmul(self.results[i-1].transpose(), x)
            gradients_layer_learning = F.Tanh(gradients_layer)
            self.ffn[i-1] -= self.lr * (gradients_layer_learning + 0.01 * self.ffn[i-1])
            #self.ffn[i-1] = cp.clip(self.ffn[i-1], -0.5, 0.5)
            
            #print(self.ffn[i-1].min(), self.ffn[i-1].max())
            #print(f"Layer{i}: ", gradients_layer_learning.min(), gradients_layer_learning.max())  
            #gradients.insert(0, gradients_layer)
            
            # TODO Optimizer erstellen und integrieren
                       
            mistake_next_layer = cp.matmul(x, self.ffn[i-1].transpose())
            
            
        #self.optimizer.step(gradients)
        
