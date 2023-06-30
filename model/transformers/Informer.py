import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.masking import TriangularCausalMask, ProbMask
from .layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from .layers.SelfAttention_Family import FullAttention, ProbAttention, AttentionLayer
from .layers.Embed import DataEmbedding
import numpy as np



class Model(nn.Module):
    """
    Propspare attention in O(LlogL) complexity
    """
    def __init__(self, configs):
        super(Model, self).__init__()

        self.config_arch = configs['arch_G']['args']
        self.pred_len = configs['data_loader']['args']['pred_len']
        self.output_attention = self.config_arch['output_attention']

        # Embedding
        self.enc_embedding = DataEmbedding(self.config_arch['enc_in'], self.config_arch['d_model'], self.config_arch['embed'], configs['data_loader']['args']['freq'],
                                           self.config_arch['dropout'])
        self.dec_embedding = DataEmbedding(self.config_arch['dec_in'], self.config_arch['d_model'], self.config_arch['embed'], configs['data_loader']['args']['freq'],
                                           self.config_arch['dropout'])

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, self.config_arch['factor'], attention_dropout=self.config_arch['dropout'],
                                      output_attention=self.config_arch['output_attention']),
                        self.config_arch['d_model'], self.config_arch['n_heads']),
                    self.config_arch['d_model'],
                    self.config_arch['d_ff'],
                    dropout=self.config_arch['dropout'],
                    activation=self.config_arch['activation']
                ) for l in range(self.config_arch['e_layers'])
            ],
            [
                ConvLayer(
                    self.config_arch['d_model']
                ) for l in range(self.config_arch['e_layers'] - 1)
            ] if self.config_arch['distil'] else None,
            norm_layer=torch.nn.LayerNorm(self.config_arch['d_model'])
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(True, self.config_arch['factor'], attention_dropout=self.config_arch['dropout'], output_attention=False),
                        self.config_arch['d_model'], self.config_arch['n_heads']),
                    AttentionLayer(
                        ProbAttention(False, self.config_arch['factor'], attention_dropout=self.config_arch['dropout'], output_attention=False),
                        self.config_arch['d_model'], self.config_arch['n_heads']),
                    self.config_arch['d_model'],
                    self.config_arch['d_ff'],
                    dropout=self.config_arch['dropout'],
                    activation=self.config_arch['activation'],
                )
                for l in range(self.config_arch['d_layers'])
            ],
            norm_layer=torch.nn.LayerNorm(self.config_arch['d_model']),
            projection=nn.Linear(self.config_arch['d_model'], 1, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        if self.output_attention:
            return torch.unsqueeze(dec_out[:, -self.pred_len:],-1), attns
        else:
            return torch.unsqueeze(dec_out[:, -self.pred_len:],-1)  # [B, L, D]
