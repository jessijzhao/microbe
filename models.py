#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from abc import abstractmethod

import torch.nn as nn
from opacus.layers import DPGRU, DPRNN, DPLSTM, DPMultiheadAttention
import torch
import torch.nn.functional as F


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ModelType:
    RNN: str = "rnn"
    DPRNN: str = "dprnn"
    GRU: str = "gru"
    DPGRU: str = "dpgru"
    LSTM: str = "lstm"
    DPLSTM: str = "dplstm"
    MHA: str = "mha"
    DPMHA: str = "dpmha"

class ModelFactory:

    class Model:

        def prepare_inference(self):
            self.model.eval()
            self.input_tensor.requires_grad = False

        def prepare_training(self):
            self.model.train()
            self.input_tensor.requires_grad = True
        
        @abstractmethod
        def inference(self):
            pass

        def training(self) -> torch.Tensor:
            preds, _ = self.inference()
            loss = F.cross_entropy(preds, self.labels)
            loss.backward()

        
    class MHABase(Model):
        def __init__(
            self,
            *,
            model,
            batch_size,
            source_seq_len,
            targ_seq_len,
            embed_dim,
            num_heads,
            dropout=0.0,
            bias=True,
            add_bias_kv=False,
            add_zero_attn=False,
            kdim=None,
            vdim=None,
            batch_first=False,
        ):
            kdim = kdim if kdim else embed_dim
            vdim = vdim if vdim else embed_dim
            
            self.input_tensor = (torch.randn(
                targ_seq_len, batch_size, embed_dim, device=device
            ) if not batch_first
            else torch.randn(batch_size, targ_seq_len, embed_dim, device=device))
            
            self.key = (torch.randn(
                source_seq_len, batch_size, kdim, device=device
            ) if not batch_first
            else torch.randn(batch_size, source_seq_len, kdim, device=device))

            self.value = (torch.randn(
                source_seq_len, batch_size, vdim, device=device
            ) if not batch_first
            else torch.randn(batch_size, source_seq_len, vdim, device=device))
            
            self.labels = (torch.randn(
                targ_seq_len, batch_size, embed_dim, device=device
            ) if not batch_first
            else torch.randn(batch_size, targ_seq_len, embed_dim, device=device))

            self.model = model(
                embed_dim,
                num_heads,
                dropout=dropout,
                bias=bias,
                add_bias_kv=add_bias_kv,
                add_zero_attn=add_zero_attn,
                kdim=kdim,
                vdim=vdim
            )


        def inference(self):
            return self.model(self.input_tensor, self.key, self.value)

    class MHA(MHABase):
        def __init__(self, **kwargs):
            super().__init__(model=nn.MultiheadAttention, **kwargs)

    class DPMHA(MHABase):
        def __init__(self, **kwargs):
            super().__init__(model=DPMultiheadAttention, **kwargs)


    class RNNBase(Model):
        def __init__(
            self,
            *,
            model,
            batch_size,
            seq_len,
            input_size,
            hidden_size,
            num_layers=1,
            bias=False,
            batch_first=False,
            dropout=0,
            bidirectional=False,
            **kwargs
        ):
            self.input_tensor = (torch.randn(
                seq_len, batch_size, input_size,
                device=device, 
            ) if not batch_first 
            else torch.randn(batch_size, seq_len, input_size, device=device))
            
            self.D = 2 if bidirectional else 1
            self.h_0 = torch.randn(self.D * num_layers, batch_size, hidden_size, device=device)
            
            self.labels = (torch.randn(
                seq_len, batch_size, self.D * hidden_size,
                device=device
            ) if not batch_first
            else torch.randn(batch_size, seq_len, self.D * hidden_size, device=device))

            self.model = model(
                input_size,
                hidden_size,
                num_layers=num_layers,
                bias=bias,
                batch_first=batch_first,
                dropout=dropout,
                bidirectional=bidirectional,
                **kwargs
            )

        def inference(self) -> torch.Tensor:
            return self.model(self.input_tensor, self.h_0)


    class RNN(RNNBase):
        def __init__(self, **kwargs):
            super().__init__(model=nn.RNN, **kwargs)

    class DPRNN(RNNBase):
        def __init__(self, **kwargs):
            super().__init__(model=DPRNN, **kwargs)

    class GRU(RNNBase):
        def __init__(self, **kwargs):
            super().__init__(model=nn.GRU, **kwargs)

    class DPGRU(RNNBase):
        def __init__(self, **kwargs):
            super().__init__(model=DPGRU, **kwargs)

    class LSTMBase(RNNBase):
        def __init__(
            self,
            *,
            model,
            batch_size,
            seq_len,
            input_size,
            hidden_size,
            num_layers=1,
            bias=False,
            batch_first=False,
            dropout=0,
            bidirectional=False,
            proj_size=0
        ):
            super().__init__(
                model=model,
                batch_size=batch_size,
                seq_len=seq_len,
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bias=bias,
                batch_first=batch_first,
                dropout=dropout,
                bidirectional=bidirectional,
                proj_size=proj_size
            )
            h_out = proj_size if proj_size > 0 else hidden_size
            self.h_0 = torch.randn(self.D * num_layers, batch_size, h_out, device=device)
            self.c_0 = torch.randn(self.D * num_layers, batch_size, hidden_size, device=device)
            self.labels = (torch.randn(
                seq_len, batch_size, self.D * h_out,
                device=device
            ) if not batch_first
            else torch.randn(batch_size, seq_len, self.D * h_out, device=device))

        def inference(self)-> torch.Tensor:
            return self.model(self.input_tensor, (self.h_0, self.c_0))


    class LSTM(LSTMBase):
        def __init__(self, **kwargs):
            super().__init__(model=nn.LSTM, **kwargs)


    class DPLSTM(LSTMBase):
        def __init__(self, **kwargs):
            super().__init__(model=DPLSTM, **kwargs)


    @staticmethod
    def create(model_type: str, **kwargs):
        if model_type == ModelType.RNN:
            return ModelFactory.RNN(**kwargs)
        elif model_type == ModelType.DPRNN:
            return ModelFactory.DPRNN(**kwargs)
        elif model_type == ModelType.GRU:
            return ModelFactory.GRU(**kwargs)
        elif model_type == ModelType.DPGRU:
            return ModelFactory.DPGRU(**kwargs)
        elif model_type == ModelType.LSTM:
            return ModelFactory.LSTM(**kwargs)
        elif model_type == ModelType.DPLSTM:
            return ModelFactory.DPLSTM(**kwargs)
        elif model_type == ModelType.MHA:
            return ModelFactory.MHA(**kwargs)
        elif model_type == ModelType.DPMHA:
            return ModelFactory.DPMHA(**kwargs)
        else:
            print(f"Invalid model type: {model_type}.")
