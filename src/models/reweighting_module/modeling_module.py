from torch import nn

from src.models.reweighting_module.configuration_module import ReweightAttentionConfig


class ReweightAttentionModule(nn.Module):

    def __init__(self, config: ReweightAttentionConfig):
        self.config = config
        super().__init__()

    def forward(self, ):
        raise NotImplementedError
