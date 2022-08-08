import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, layers, num_lines_predict = 1, max_char=102):
        super().__init__()
        
        self.num_lines_predict = num_lines_predict 
        self.fc = [nn.Linear(max_char*self.num_lines_predict, layers[0])]
        for i in range(len(layers)-1):
            self.fc.append(nn.Linear(layers[i],layers[i+1]))
        self.fc.append(nn.Linear(layers[-1],max_char))
        
        self.fc1 = self.fc[0]
        self.fc2 = self.fc[1]
        self.fc3 = self.fc[2]
        self.fc4 = self.fc[3]
        self.fc5 = self.fc[4]
    
    def forward(self, x):
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        
        return x