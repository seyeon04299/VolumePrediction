import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from .layers.SelfAttention_Family import FullAttention, AttentionLayer
from .layers.Embed import DataEmbedding
import numpy as np


class Model(nn.Module):
    """
    Vanilla Transformer with O(L^2) complexity
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        print("Transformer Called")

        self.config_arch = configs['arch_G']['args']
        self.pred_len = configs['data_loader']['args']['pred_len']
        self.output_attention = self.config_arch['output_attention']

        # Embedding
        self.enc_embedding = DataEmbedding(self.config_arch['enc_in'], self.config_arch['d_model'], self.config_arch['embed'], configs['data_loader']['args']['freq'],
                                           self.config_arch['dropout'])
        self.dec_embedding = DataEmbedding(self.config_arch['dec_in'], self.config_arch['d_model'], self.config_arch['embed'], configs['data_loader']['args']['freq'],
                                           self.config_arch['dropout'])
        
        self.criterion = configs['trainer']['criterion']
        if self.criterion == "GEV":
            self.out = 3
        
        
        
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, self.config_arch['factor'], attention_dropout=self.config_arch['dropout'],
                                      output_attention=self.config_arch['output_attention']), self.config_arch['d_model'], self.config_arch['n_heads']),
                    self.config_arch['d_model'],
                    self.config_arch['d_ff'],
                    dropout=self.config_arch['dropout'],
                    activation=self.config_arch['activation']
                ) for l in range(self.config_arch['e_layers'])
            ],
            norm_layer=torch.nn.LayerNorm(self.config_arch['d_model'])
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, self.config_arch['factor'], attention_dropout=self.config_arch['dropout'], output_attention=False),
                        self.config_arch['d_model'], self.config_arch['n_heads']),
                    AttentionLayer(
                        FullAttention(False, self.config_arch['factor'], attention_dropout=self.config_arch['dropout'], output_attention=False),
                        self.config_arch['d_model'], self.config_arch['n_heads']),
                    self.config_arch['d_model'],
                    self.config_arch['d_ff'],
                    dropout=self.config_arch['dropout'],
                    activation=self.config_arch['activation'],
                )
                for l in range(self.config_arch['d_layers'])
            ],
            norm_layer=torch.nn.LayerNorm(self.config_arch['d_model']),
            projection=nn.Linear(self.config_arch['d_model'], self.config_arch['c_out'], bias=True)
        )
        self.out_gamma = nn.Sequential(
			nn.Linear(self.config_arch['c_out'],48),
			nn.Linear(48,1),
			nn.ReLU()
		)
        self.out_sigma = nn.Sequential(
			nn.Linear(self.config_arch['c_out'],48),
			nn.Linear(48,1),
			nn.ReLU()
		)
        self.final = nn.Linear(self.config_arch['c_out'],1)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        if self.out == 3:
            dec_out_gamma = self.out_gamma(dec_out)
            dec_out_sigma = self.out_sigma(dec_out)
            dec_out = self.final(dec_out)

        if self.output_attention:
            if self.out==3:
                # print('dec_out[:, -self.pred_len:,:] : ', dec_out[:, -self.pred_len:].shape)
                return dec_out[:, -self.pred_len:,:], dec_out_gamma[:, -self.pred_len:,:], dec_out_sigma[:, -self.pred_len:,:], attns
            else:
                
                return torch.unsqueeze(dec_out[:, -self.pred_len:],-1), attns
        else:
            if self.out==3:
                
                return dec_out[:, -self.pred_len:,:], dec_out_gamma[:, -self.pred_len:,:], dec_out_sigma[:, -self.pred_len:,:]
            else:
                
                return torch.unsqueeze(dec_out[:, -self.pred_len:],-1)  # [B, L, D]




class Proformer(nn.Module):
    """
    Vanilla Transformer with O(L^2) complexity
    """
    def __init__(self, configs, step, final=False):
        super(Proformer, self).__init__()
        print("Transformer Called")

        self.config_arch = configs['arch_G'+str(step)]['args']
        # self.pred_len = configs['data_loader']['args']['pred_len']
        self.output_attention = self.config_arch['output_attention']
        self.criterion = configs['trainer']['criterion']
        if self.criterion == "GEV":
            self.out = 3

        # Embedding
        self.enc_embedding = DataEmbedding(self.config_arch['enc_in'], self.config_arch['d_model'], self.config_arch['embed'], configs['data_loader']['args']['freq'],
                                           self.config_arch['dropout'])
        self.dec_embedding = DataEmbedding(self.config_arch['dec_in'], self.config_arch['d_model'], self.config_arch['embed'], configs['data_loader']['args']['freq'],
                                           self.config_arch['dropout'])

        self.final = final
        if self.final:
            self.final_out=self.config_arch['c_out']            ## could change to 1 for later investigations
        else:
            self.final_out = self.config_arch['c_out']

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, self.config_arch['factor'], attention_dropout=self.config_arch['dropout'],
                                      output_attention=self.config_arch['output_attention']), self.config_arch['d_model'], self.config_arch['n_heads']),
                    self.config_arch['d_model'],
                    self.config_arch['d_ff'],
                    dropout=self.config_arch['dropout'],
                    activation=self.config_arch['activation']
                ) for l in range(self.config_arch['e_layers'])
            ],
            norm_layer=torch.nn.LayerNorm(self.config_arch['d_model'])
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, self.config_arch['factor'], attention_dropout=self.config_arch['dropout'], output_attention=False),
                        self.config_arch['d_model'], self.config_arch['n_heads']),
                    AttentionLayer(
                        FullAttention(False, self.config_arch['factor'], attention_dropout=self.config_arch['dropout'], output_attention=False),
                        self.config_arch['d_model'], self.config_arch['n_heads']),
                    self.config_arch['d_model'],
                    self.config_arch['d_ff'],
                    dropout=self.config_arch['dropout'],
                    activation=self.config_arch['activation'],
                )
                for l in range(self.config_arch['d_layers'])
            ],
            norm_layer=torch.nn.LayerNorm(self.config_arch['d_model']),
            projection=nn.Linear(self.config_arch['d_model'], self.config_arch['c_out'], bias=True)
        )

        self.out_gamma = nn.Sequential(
			nn.Linear(self.config_arch['c_out'],48),
			nn.Linear(48,1),
			nn.ReLU()
		)
        self.out_sigma = nn.Sequential(
			nn.Linear(self.config_arch['c_out'],48),
			nn.Linear(48,1),
			nn.ReLU()
		)
        self.final = nn.Linear(self.config_arch['c_out'],1)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,label_len, pred_len, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        if self.criterion=='GEV':
            dec_out_gamma = self.out_gamma(dec_out)
            dec_out_sigma = self.out_sigma(dec_out)
            dec_out = self.final(dec_out)
        # print('dec_out.shape : ', dec_out.shape)

        if self.output_attention:
            if self.criterion=='GEV':
                # print('dec_out[:, -self.pred_len:,:] : ', dec_out[:, -self.pred_len:].shape)
                return dec_out[:, -self.pred_len:,:], dec_out_gamma[:, -self.pred_len:,:], dec_out_sigma[:, -self.pred_len:,:], attns
            else:
                return dec_out[:, -pred_len:,:], attns
        else:
            if self.criterion=='GEV':
                return dec_out[:, -self.pred_len:,:], dec_out_gamma[:, -self.pred_len:,:], dec_out_sigma[:, -self.pred_len:,:]
            else:
                return dec_out[:, -pred_len:,:]  # [B, L, D]