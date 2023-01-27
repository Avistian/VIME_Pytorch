import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from vime_semi import VIMESemi
from vime_self import VIMESelf

class VIME(nn.Module):
    def __init__(self, alpha, K, device, encoder_backbone=None, feature_estimator_backbone=None, mask_estimator_backbone=None, predictor_backbone=None):
        super(VIME, self).__init__()
        self.encoder_backbone = encoder_backbone
        self.feature_estimator_backbone = feature_estimator_backbone
        self.mask_estimator_backbone = mask_estimator_backbone
        self.predictor_backbone = predictor_backbone
        self.K = K
        self.alpha = alpha
        self.device = device
        self.VIME_self = VIMESelf(self.encoder_backbone, self.mask_estimator_backbone, self.feature_estimator_backbone)
        self.VIME_semi = VIMESemi(self.encoder_backbone, self.predictor_backbone, self.K)
        
    def fit(self, train_loader, unsup_dataset, test_loader, epochs, lr, batch_size, K, beta):
        pass
    
    def predict(self, x):
        pass
        
    def _generate_mask(self, shape, p_corr):
        mask = np.random.binomial(1, p_corr, shape)
        return mask

    def _generate_pretext_task(self, x, mask):
        x = x.reshape(1, x.shape[0]) if x.ndim != 2 else x
        n, dim = x.shape
        x_bar = np.zeros((n, dim))
        for d in range(dim):
            x_bar[:, d] = np.random.permutation(x[:, d])
        x_tilde = (1 - mask) * x + mask * x_bar
        m = 1 * (x != x_tilde)
        return m, x_tilde

    def _loss_fn(self, x_mask, x_feature, mask, x_unlab, alpha):
        mse = torch.mean((x_unlab - x_feature) ** 2)
        bce = torch.mean(F.binary_cross_entropy(x_mask, mask))
        return alpha * mse + bce
    
    def _loss_fn_semi(self, y_pred, yu_pred, y_true, beta):
        loss = F.cross_entropy(y_pred, y_true)
        loss += beta * torch.var(yu_pred, dim=1).mean()
        return loss
    
    def _generate_pretext_k_examples(self, row, k, p_corr):
        result = np.zeros((1, k, row.shape[0]))
        for k in range(k):
            mask_semi = self._generate_mask(row.shape, p_corr)
            m_semi_label, X_semi_unsup_tilde = self._generate_pretext_task(row, mask_semi)
            result[0, k, :] = X_semi_unsup_tilde
        return result
    
    def _train_loop_self(self, dataloader, model, loss_fn, optimizer, report_iters=10):
        size = len(dataloader)
        for batch, x in enumerate(dataloader):
            # Transfer to device
            X = x[0].to(self.device)
            unlab = x[1].to(self.device)
            m_label = x[3].to(self.device)
            mask = x[2].to(self.device)
            batch_size = X.shape[0]
            
            # Compute prediction and loss
            x_mask, x_feature = model(X)
            loss = loss_fn(x_mask, x_feature, m_label, unlab, self.alpha)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % report_iters == 0:
                loss, current = loss.item(), batch
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                
    def _test_loop_self(self, dataloader, model, loss_fn):
        size = len(dataloader)
        num_batches = len(dataloader)
        test_loss = 0
        batch_size = size
        with torch.no_grad():
            for x in dataloader:
                X = x[0].to(self.device)
                unlab = x[1].to(self.device)
                m_label = x[3].to(self.device)
                mask = x[2].to(self.device)
                batch_size = X.shape[0]
                x_mask, x_feature = model(X)
                test_loss += loss_fn(x_mask, x_feature, m_label, unlab, self.alpha)

        test_loss /= num_batches
        print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")
        
    def _train_loop_semi(self, train_loader, unsup_dataset, model, loss_fn, beta, optimizer, batch_size, K, report_iters=10):
        model.train()
        size = len(train_loader.dataset)
        for batch, (x, y) in enumerate(train_loader):
            x, y = x.to(self.device), y.to(self.device)
            xu = unsup_dataset[batch * batch_size * K : (batch + 1) * batch_size * K, :].to(self.device)
            y_logit_pred, yu_logit_pred, y_pred, yu_pred = model(x, xu)
            loss = self._loss_fn_semi(y_pred, yu_logit_pred, y, beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch % report_iters == 0:
                loss, current = loss.item(), batch
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                
    def _test_loop_semi(self, test_loader, unsup_dataset, model, loss_fn, beta, batch_size, K):
        size_sup = len(test_loader.dataset)
        size_unsup = unsup_dataset.shape[0]
        size = min(size_sup, size_unsup)
        num_batches = len(test_loader)
        test_loss = 0
        with torch.no_grad():
            for i, (x, y) in enumerate(test_loader):
                x, y = x.to(self.device), y.to(self.device)
                xu = unsup_dataset[i * batch_size * K : (i + 1) * batch_size * K, :].to(self.device)
                y_logit_pred, yu_logit_pred, y_pred, yu_pred = model(x, xu)
                test_loss += self._loss_fn_semi(y_pred, yu_logit_pred, y, beta).item()

        test_loss /= num_batches
        print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")
        return test_loss
        
    def _get_default_encoder_backbone(input_width, hidden_width, output_width):
        def get_network():
            if len(hidden_width) == 0:
                return [nn.Linear(input_width, output_width), nn.ReLU()]
            layers = [nn.Linear(input_width, hidden_width[0]), nn.ReLU()]
            for layer_width, next_layer_width in zip(hidden_width[0:len(hidden_width)-1], hidden_width[1:]):
                layers.append(nn.Linear(layer_width, next_layer_width))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_width[-1], output_width))
            layers.append(nn.ReLU())
            return layers
        return nn.Sequential(*get_network())

    def _get_default_mask_estimator_backbone(input_width, hidden_width, output_width):
        def get_network():
            if len(hidden_width) == 0:
                return [nn.Linear(input_width, output_width), nn.Sigmoid()]
            layers = [nn.Linear(input_width, hidden_width[0]), nn.ReLU()]
            for layer_width, next_layer_width in zip(hidden_width[0:len(hidden_width)-1], hidden_width[1:]):
                layers.append(nn.Linear(layer_width, next_layer_width))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_width[-1], output_width))
            layers.append(nn.Sigmoid())
            return layers
        return nn.Sequential(*get_network())

    def _get_default_feature_estimator_backbone(input_width, hidden_width, output_width):
        def get_network():
            if len(hidden_width) == 0:
                return [nn.Linear(input_width, output_width)]
            layers = [nn.Linear(input_width, hidden_width[0]), nn.ReLU()]
            for layer_width, next_layer_width in zip(hidden_width[0:len(hidden_width)-1], hidden_width[1:]):
                layers.append(nn.Linear(layer_width, next_layer_width))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_width[-1], output_width))
            return layers
        return nn.Sequential(*get_network())
    
    def _get_default_predictor_backbone(self, input_width, hidden_width, output_width):
        def get_network():
            layers = [nn.Linear(input_width, hidden_width[0]), nn.ReLU()]
            for layer_width, next_layer_width in zip(hidden_width[0:len(hidden_width)-1], hidden_width[1:]):
                layers.append(nn.Linear(layer_width, next_layer_width))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_width[-1], output_width))
            return layers
        return nn.Sequential(*get_network())