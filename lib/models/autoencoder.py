import torch
from torch import nn
from lib.models.generic_UNet import Generic_UNet
import ipdb


class AutoEnc(nn.Module):
    def __init__(self, **net_params):
        super(AutoEnc, self).__init__()
        self.pred_module = Generic_UNet(**net_params)
        net_params['gnn_type'] = None
        self.ae_module = Generic_UNet(**net_params)
    
    def forward(self, x):
        out = self.pred_module(x)
        preds = nn.Softmax(dim=1)(out[0])[:, 1, ...]  # preds (B, H, W)
        preds = preds.unsqueeze(dim=-4)
        out = self.ae_module(preds)
        return out
