import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from .layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from .layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
import math
import numpy as np


class Model(nn.Module):
    """
    Series-wise connection, ( inherent O(LlogL) complexity)
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.config_arch = configs['arch_G']['args']

        self.seq_len = configs['data_loader']['args']['seq_len']
        self.label_len = configs['data_loader']['args']['label_len']
        self.pred_len = configs['data_loader']['args']['pred_len']
        self.output_attention = self.config_arch['output_attention']

        # Decomp
        kernel_size = self.config_arch['moving_avg']
        self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_wo_pos(self.config_arch['enc_in'], self.config_arch['d_model'], self.config_arch['embed'], configs['data_loader']['args']['freq'],
                                                  self.config_arch['dropout'])
        self.dec_embedding = DataEmbedding_wo_pos(self.config_arch['dec_in'], self.config_arch['d_model'], self.config_arch['embed'], configs['data_loader']['args']['freq'],
                                                  self.config_arch['dropout'])

        
        self.criterion = configs['trainer']['criterion']
        if self.criterion == "GEV":
            self.out = 3
        
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, self.config_arch['factor'], attention_dropout=self.config_arch['dropout'],
                                        output_attention=self.config_arch['output_attention']),
                        self.config_arch['d_model'], self.config_arch['n_heads']),
                    self.config_arch['d_model'],
                    self.config_arch['d_ff'],
                    moving_avg=self.config_arch['moving_avg'],
                    dropout=self.config_arch['dropout'],
                    activation=self.config_arch['activation']
                ) for l in range(self.config_arch['e_layers'])
            ],
            norm_layer=my_Layernorm(self.config_arch['d_model'])
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, self.config_arch['factor'], attention_dropout=self.config_arch['dropout'],
                                        output_attention=False),
                        self.config_arch['d_model'], self.config_arch['n_heads']),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, self.config_arch['factor'], attention_dropout=self.config_arch['dropout'],
                                        output_attention=False),
                        self.config_arch['d_model'], self.config_arch['n_heads']),
                    self.config_arch['d_model'],
                    self.config_arch['c_out'],
                    self.config_arch['d_ff'],
                    moving_avg=self.config_arch['moving_avg'],
                    dropout=self.config_arch['dropout'],
                    activation=self.config_arch['activation'],
                )
                for l in range(self.config_arch['d_layers'])
            ],
            norm_layer=my_Layernorm(self.config_arch['d_model']),
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
                # batch_x, batch_x_mark, dec_inp, batch_y_mark

        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # dec
        # print('seasonal_init.shape',seasonal_init.shape)
        # print('x_mark_dec.shape',x_mark_dec.shape)
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part

        if self.criterion=='GEV':
            dec_out_gamma = self.out_gamma(dec_out)
            dec_out_sigma = self.out_sigma(dec_out)
            dec_out = self.final(dec_out)

        if self.output_attention:
            if self.criterion=='GEV':
                # print('dec_out[:, -self.pred_len:,:] : ', dec_out[:, -self.pred_len:].shape)
                return dec_out[:, -self.pred_len:,:], dec_out_gamma[:, -self.pred_len:,:], dec_out_sigma[:, -self.pred_len:,:], attns
            else:
                return torch.unsqueeze(dec_out[:, -self.pred_len:],-1), attns
        else:
            if self.criterion=='GEV':
                return dec_out[:, -self.pred_len:,:], dec_out_gamma[:, -self.pred_len:,:], dec_out_sigma[:, -self.pred_len:,:]
            else:
                return torch.unsqueeze(dec_out[:, -self.pred_len:],-1)  # [B, L, D]


class Proformer(nn.Module):
    """
    Series-wise connection, ( inherent O(LlogL) complexity)
    """
    def __init__(self, configs, step, final=False):
        super(Proformer, self).__init__()
        self.config_arch = configs['arch_G'+str(step)]['args']

        # self.seq_len = configs['data_loader']['args']['seq_len']
        # self.label_len = configs['data_loader']['args']['label_len']
        # self.pred_len = configs['data_loader']['args']['pred_len']
        self.output_attention = self.config_arch['output_attention']

        # Decomp
        kernel_size = self.config_arch['moving_avg']
        self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_wo_pos(self.config_arch['enc_in'], self.config_arch['d_model'], self.config_arch['embed'], configs['data_loader']['args']['freq'],
                                                  self.config_arch['dropout'])
        self.dec_embedding = DataEmbedding_wo_pos(self.config_arch['dec_in'], self.config_arch['d_model'], self.config_arch['embed'], configs['data_loader']['args']['freq'],
                                                  self.config_arch['dropout'])

        self.final = final
        if self.final:
            self.final_out = 1
        else:
            self.final_out = self.config_arch['c_out']
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, self.config_arch['factor'], attention_dropout=self.config_arch['dropout'],
                                        output_attention=self.config_arch['output_attention']),
                        self.config_arch['d_model'], self.config_arch['n_heads']),
                    self.config_arch['d_model'],
                    self.config_arch['d_ff'],
                    moving_avg=self.config_arch['moving_avg'],
                    dropout=self.config_arch['dropout'],
                    activation=self.config_arch['activation']
                ) for l in range(self.config_arch['e_layers'])
            ],
            norm_layer=my_Layernorm(self.config_arch['d_model'])
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, self.config_arch['factor'], attention_dropout=self.config_arch['dropout'],
                                        output_attention=False),
                        self.config_arch['d_model'], self.config_arch['n_heads']),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, self.config_arch['factor'], attention_dropout=self.config_arch['dropout'],
                                        output_attention=False),
                        self.config_arch['d_model'], self.config_arch['n_heads']),
                    self.config_arch['d_model'],
                    self.config_arch['c_out'],
                    self.config_arch['d_ff'],
                    moving_avg=self.config_arch['moving_avg'],
                    dropout=self.config_arch['dropout'],
                    activation=self.config_arch['activation'],
                )
                for l in range(self.config_arch['d_layers'])
            ],
            norm_layer=my_Layernorm(self.config_arch['d_model']),
            projection=nn.Linear(self.config_arch['d_model'], self.final_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, label_len, pred_len, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
                # batch_x, batch_x_mark, dec_inp, batch_y_mark

        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], pred_len, x_dec.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, -label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -label_len:, :], zeros], dim=1)
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # dec
        # print('seasonal_init.shape',seasonal_init.shape)
        # print('x_mark_dec.shape',x_mark_dec.shape)
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part

        if self.output_attention:
            if self.final:
                # print('final')
                # print('dec_out[:, -pred_len:]', dec_out[:, -pred_len:].shape)
                # return torch.unsqueeze(dec_out[:, -pred_len:],-1), attns
                return dec_out[:, -pred_len:], attns
            return dec_out[:, -pred_len:,:], attns
        else:
            if self.final:
                # print('final')
                # print('dec_out[:, -pred_len:]', dec_out[:, -pred_len:].shape)
                # return torch.unsqueeze(dec_out[:, -pred_len:],-1)
                return dec_out[:, -pred_len:], attns
            return dec_out[:, -pred_len:,:]  # [B, L, D]

