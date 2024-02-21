import torch
import torch.nn as nn


class VIMESelf(nn.Module):
    """
    VIMESelf module for VIME.

    Args:
        encoder (nn.Module): The encoder module.
        mask_estimator (nn.Module): The mask estimator module.
        feature_estimator (nn.Module): The feature estimator module.
    """

    def __init__(
        self,
        encoder: nn.Module,
        mask_estimator: nn.Module,
        feature_estimator: nn.Module,
    ):
        super(VIMESelf, self).__init__()
        self.encoder = encoder
        self.mask_estimator = mask_estimator
        self.feature_estimator = feature_estimator

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the VIMESelf module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The output tensors (x_mask, x_feature).
        """
        x = self.encoder(x)
        x_mask = self.mask_estimator(x)
        x_feature = self.feature_estimator(x)

        return x_mask, x_feature
