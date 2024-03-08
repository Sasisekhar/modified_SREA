import torch.nn as nn

class NonLinClassifier(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.hidden1 = nn.Linear(d_in, 128)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(128, 64)
        self.act2 = nn.ReLU()
        self.output = nn.Linear(64, 1)
        self.act_output = nn.Sigmoid()
 
    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.act_output(self.output(x))
        return x