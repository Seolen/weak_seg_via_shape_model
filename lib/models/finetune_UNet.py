import torch
from torch import nn
from lib.models.generic_UNet import Generic_UNet
import ipdb


class FinetuneUNet(nn.Module):
    def __init__(self, **net_params):
        super(FinetuneUNet, self).__init__()

        # model1: pretrained


        # model2: to train [input C=32]


        self.pred_module = Generic_UNet(**net_params)
        net_params['gnn_type'] = None
        self.ae_module = Generic_UNet(**net_params)
    
    def forward(self, x):
        # generate feature: no grad

        # UNet prediction




        out = self.pred_module(x)
        preds = nn.Softmax(dim=1)(out[0])[:, 1, ...]  # preds (B, H, W)
        preds = preds.unsqueeze(dim=-4)
        out = self.ae_module(preds)
        return out
