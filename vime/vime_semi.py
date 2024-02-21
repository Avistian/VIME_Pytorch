import torch
import torch.nn as nn


class VIMESemi(nn.Module):
    """
    VIMESemi class represents a VIME model with semi-supervised learning.

    Args:
        encoder (nn.Module): The encoder module.
        predictor (nn.Module): The predictor module.
        K (int): The number of classes.

    Attributes:
        encoder (nn.Module): The encoder module.
        predictor (nn.Module): The predictor module.
        K (int): The number of classes.
    """

    def __init__(self, encoder: nn.Module, predictor: nn.Module, K: int):
        super(VIMESemi, self).__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.K = K
        self.encoder.requires_grad = False

    def forward(
        self, x: torch.Tensor, xu: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the VIMESemi model.

        Args:
            x (torch.Tensor): The input tensor.
            xu (torch.Tensor): The unlabeled input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: The logits and softmax outputs for labeled and unlabeled inputs.
        """
        x = self.encoder(x)
        y_logit = self.predictor(x)
        xu = self.encoder(xu)
        yu_logit = self.predictor(xu)

        y = torch.softmax(y_logit, dim=1)
        yu = torch.softmax(yu_logit, dim=1)
        return y_logit, yu_logit, y, yu

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict the output for the given input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The logits and softmax output.
        """
        x = self.encoder(x)
        y_logit = self.predictor(x)
        y = torch.softmax(y_logit, dim=1)
        return y_logit, y
