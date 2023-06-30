import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from .layers.SelfAttention_Family import ReformerLayer
from .layers.Embed import DataEmbedding
import numpy as np


class Model(nn.Module):
    """
    Reformer with O(LlogL) complexity
    Reformer is not proposed for time series forecasting, in that it cannot accomplish the cross attention.
    - Here is only one adaption in BERT-style, other possible implementations can also be acceptable.
    - The hyper-parameters, such as bucket_size and n_hashes, need to be further tuned.
    The official repo of Reformer (https://github.com/lucidrains/reformer-pytorch)
    """

    def __init__(self, configs):
        super(Model, self).__init__()

        self.config_arch = configs['arch_G']['args']
        self.n_hashes = 4           # Unique parameter for Reformer
        self.bucket_size = 4           # Unique parameter for Reformer
        
        self.pred_len = configs['data_loader']['args']['pred_len']
        
        self.output_attention = self.config_arch['output_attention']

        # Embedding
        self.enc_embedding = DataEmbedding(self.config_arch['enc_in'], self.config_arch['d_model'], self.config_arch['embed'], configs['data_loader']['args']['freq'],
                                           self.config_arch['dropout'])
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    ReformerLayer(None, self.config_arch['d_model'], self.config_arch['n_heads'], bucket_size=self.bucket_size,
                                  n_hashes=self.n_hashes),
                    self.config_arch['d_model'],
                    self.config_arch['d_ff'],
                    dropout=self.config_arch['dropout'],
                    activation=self.config_arch['activation']
                ) for l in range(self.config_arch['e_layers'])
            ],
            norm_layer=torch.nn.LayerNorm(self.config_arch['d_model'])
        )
        self.projection = nn.Linear(self.config_arch['d_model'], 1, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # add placeholder
        x_enc = torch.cat([x_enc, x_dec[:, -self.pred_len:, :]], dim=1)
        x_mark_enc = torch.cat([x_mark_enc, x_mark_dec[:, -self.pred_len:, :]], dim=1)
        # Reformer: encoder only
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        enc_out = self.projection(enc_out)

        if self.output_attention:
            return torch.unsqueeze(dec_out[:, -self.pred_len:],-1), attns
        else:
            return torch.unsqueeze(dec_out[:, -self.pred_len:],-1)  # [B, L, D]
