import torch
import torch.nn as nn

class VIMESemi(nn.Module):
    def __init__(self, encoder, predictor, K):
        super(VIMESemi, self).__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.K = K
        self.encoder.requires_grad = False
    
    def forward(self, x, xu):
        x = self.encoder(x)
        y_logit = self.predictor(x)
        xu = self.encoder(xu)
        yu_logit = self.predictor(xu)
        
        y = torch.softmax(y_logit, dim=1)
        yu = torch.softmax(yu_logit, dim=1)
        return y_logit, yu_logit, y, yu
    
    def predict(self, x):
        x = self.encoder(x)
        y_logit = self.predictor(x)
        y = torch.softmax(y_logit, dim=1)
        return y_logit, y