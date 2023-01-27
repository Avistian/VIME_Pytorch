import torch.nn as nn

class VIMESelf(nn.Module):
    def __init__(self, encoder, mask_estimator, feature_estimator):
        super(VIMESelf, self).__init__()
        self.encoder = encoder
        self.mask_estimator = mask_estimator
        self.feature_estimator = feature_estimator
    
    def forward(self, x):
        x = self.encoder(x)
        x_mask = self.mask_estimator(x)
        x_feature = self.feature_estimator(x)
        
        return x_mask, x_feature