#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from opacus.grad_sample import GradSampleModule
from opacus.layers import DPGRU, DPRNN, DPLSTM, DPMultiheadAttention

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class LayerType:
    LINEAR: str = "linear"
    GSM_LINEAR: str = "gsm_linear"
    CONV: str = "conv"
    GSM_CONV: str = "gsm_conv"
    LAYERNORM: str = "layernorm"
    GSM_LAYERNORM: str = "gsm_layernorm"
    INSTANCENORM: str = "instancenorm"
    GSM_INSTANCENORM: str = "gsm_instancenorm"
    GROUPNORM: str = "groupnorm"
    GSM_GROUPNORM: str = "gsm_groupnorm"
    EMBEDDING: str = "embedding"
    GSM_EMBEDDING: str = "gsm_embedding"
    MHA: str = "mha"
    DPMHA: str = "dpmha"
    GSM_DPMHA: str = "gsm_dpmha"
    RNN: str = "rnn"
    DPRNN: str = "dprnn"
    GSM_DPRNN: str = "gsm_dprnn"
    GRU: str = "gru"
    DPGRU: str = "dpgru"
    GSM_DPGRU: str = "gsm_dpgru"
    LSTM: str = "lstm"
    DPLSTM: str = "dplstm"
    GSM_DPLSTM: str = "gsm_dplstm"


class LayerFactory:

    class Layer:
        def __init__(
            self,
            *,
            criterion=F.cross_entropy
        ):
            self.criterion = criterion
        
        def prepare_forward_only(self):
            self.layer.eval()

        def prepare_forward_backward(self):
            self.layer.train()
        
        def forward_only(self):
            return self.layer(*self.layer_inputs)

        def forward_only_no_hooks(self):
            return self.layer.forward(*self.layer_inputs)

        def forward_backward(self) -> torch.Tensor:
            preds = self.forward_only()
            loss = self.criterion(preds, self.labels)
            loss.backward()


    class LinearBase(Layer):
        def __init__(
            self,
            *,
            batch_size,
            input_shape,
            in_features,
            out_features,
            bias=True,
        ):
            super().__init__()
            self.input_tensor = torch.randn(batch_size, *input_shape, in_features, device=device)
            self.layer_inputs = [self.input_tensor]
            self.layer = nn.Linear(
                in_features=in_features,
                out_features=out_features,
                bias=bias,
            )
            self.layer.to(device)
            self.labels = torch.randn(batch_size, *input_shape, out_features, device=device)
    
    class ConvBase(Layer):
        def __init__(
            self,
            *,
            batch_size,
            in_channels,
            input_shape,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            padding_mode="zeros"
        ):
            super().__init__()
            D = len(input_shape)
            if D == 1:
                self.layer_name = nn.Conv1d
            elif D ==2:
                self.layer_name = nn.Conv2d
            elif D == 3:
                self.layer_name = nn.Conv3d
            else:
                raise Exception("Input shape must be between 1 and 3 long")

            self.input_tensor = torch.randn(
                batch_size, in_channels, *input_shape, device=device
            )
            self.layer_inputs = [self.input_tensor]
            self.layer = self.layer_name(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
                padding_mode=padding_mode
            )
            self.layer.to(device)
            outputs = self.layer(self.input_tensor)
            self.labels = torch.randn(outputs.shape, device=device)

    class LayerNormBase(Layer):
        def __init__(
            self,
            *,
            batch_size,
            input_shape,
            D,
            eps=1e-05,
            elementwise_affine=True
        ):
            super().__init__()
            self.input_tensor = torch.randn(batch_size, *input_shape, device=device)
            self.layer_inputs = [self.input_tensor]
            self.layer = nn.LayerNorm(
                normalized_shape=self.input_tensor.shape[-D:],
                eps=eps,
                elementwise_affine=elementwise_affine
            )
            self.layer.to(device)
            self.labels = torch.randn(self.input_tensor.shape, device=device)
    
    class InstanceNormBase(Layer):
        def __init__(
            self,
            *,
            batch_size,
            num_features,
            input_shape,
            eps=1e-05,
            affine=False,
            track_running_stats=False
        ):
            super().__init__()
            D = len(input_shape)
            if D == 1:
                self.layer_name = nn.InstanceNorm1d
            elif D ==2:
                self.layer_name = nn.InstanceNorm2d
            elif D == 3:
                self.layer_name = nn.InstanceNorm3d
            else:
                raise Exception("Input shape must be between 1 and 3 long")

            self.input_tensor = torch.randn(batch_size, num_features, *input_shape, device=device)
            self.layer_inputs = [self.input_tensor]
            self.layer = self.layer_name(
                num_features=num_features,
                eps=eps,
                affine=affine,
                track_running_stats=track_running_stats
            )
            self.layer.to(device)
            self.labels = torch.randn(self.input_tensor.shape, device=device)

    class GroupNormBase(Layer):
        def __init__(
            self,
            *,
            batch_size,
            input_shape,
            num_groups,
            num_channels,
            eps=1e-05,
            affine=True
        ):
            super().__init__()
            self.input_tensor = torch.randn(batch_size, num_channels, *input_shape, device=device)
            self.layer_inputs = [self.input_tensor]
            self.layer = nn.GroupNorm(
                num_groups=num_groups,
                num_channels=num_channels,
                eps=eps,
                affine=affine
            )
            self.layer.to(device)
            self.labels = torch.randn(self.input_tensor.shape, device=device)

    class EmbeddingBase(Layer):
        def __init__(
            self,
            *,
            batch_size,
            input_shape,
            num_embeddings,
            embedding_dim,
            padding_idx=None,
            max_norm=None,
            norm_type=2.0,
            scale_grad_by_freq=False,
            sparse=False,
        ):
            super().__init__()
            self.input_tensor = torch.randint(
                high=num_embeddings, 
                size=(batch_size, *input_shape), 
                dtype=torch.long,
                device=device
            )
            self.layer_inputs = [self.input_tensor]
            self.layer = nn.Embedding(
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                padding_idx=padding_idx,
                max_norm=max_norm,
                norm_type=norm_type,
                scale_grad_by_freq=scale_grad_by_freq,
                sparse=sparse,
            )
            self.layer.to(device)
            self.labels = torch.randn(batch_size, *input_shape, embedding_dim, device=device)
    
    class CLayer(Layer):
        def forward_backward(self) -> torch.Tensor:
            preds, _ = self.forward_only()
            loss = F.cross_entropy(preds, self.labels)
            loss.backward()

    class MHABase(CLayer):
        def __init__(
            self,
            *,
            layer,
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
            super().__init__()
            kdim = kdim if kdim else embed_dim
            vdim = vdim if vdim else embed_dim
            
            self.input_tensor = (
                torch.randn(
                    targ_seq_len, batch_size, embed_dim, device=device
                ) if not batch_first
                else torch.randn(batch_size, targ_seq_len, embed_dim, device=device)
            )
            self.key = (
                torch.randn(
                    source_seq_len, batch_size, kdim, device=device
                ) if not batch_first
                else torch.randn(batch_size, source_seq_len, kdim, device=device)
            )
            self.value = (
                torch.randn(
                    source_seq_len, batch_size, vdim, device=device
                ) if not batch_first
                else torch.randn(batch_size, source_seq_len, vdim, device=device)
            )
            self.layer_inputs = [self.input_tensor, self.key, self.value]

            self.layer = layer(
                embed_dim,
                num_heads,
                dropout=dropout,
                bias=bias,
                add_bias_kv=add_bias_kv,
                add_zero_attn=add_zero_attn,
                kdim=kdim,
                vdim=vdim
            )
            self.layer.to(device)
            self.labels = (
                torch.randn(
                    targ_seq_len, batch_size, embed_dim, device=device
                ) if not batch_first
                else torch.randn(batch_size, targ_seq_len, embed_dim, device=device)
            )            
    

    class RNNBase(CLayer):
        def __init__(
            self,
            *,
            layer,
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
            super().__init__()
            self.input_tensor = (
                torch.randn(
                    seq_len, batch_size, input_size, device=device, 
                ) if not batch_first 
                else torch.randn(
                    batch_size, seq_len, input_size, device=device
                )
            )
            
            self.D = 2 if bidirectional else 1
            self.h_0 = torch.randn(
                self.D * num_layers, batch_size, hidden_size, device=device
            )
            self.layer_inputs = [self.input_tensor, self.h_0]

            self.layer = layer(
                input_size,
                hidden_size,
                num_layers=num_layers,
                bias=bias,
                batch_first=batch_first,
                dropout=dropout,
                bidirectional=bidirectional,
                **kwargs
            )
            self.layer.to(device)
            
            self.labels = (
                torch.randn(
                    seq_len, batch_size, self.D * hidden_size,device=device
                ) if not batch_first
                else torch.randn(
                    batch_size, seq_len, self.D * hidden_size, device=device
                )
            )


    class LSTMBase(RNNBase):
        def __init__(
            self,
            *,
            layer,
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
                layer=layer,
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
            self.h_0 = torch.randn(
                self.D * num_layers, batch_size, h_out, device=device
            )
            self.c_0 = torch.randn(
                self.D * num_layers, batch_size, hidden_size, device=device
            )
            self.layer_inputs = [self.input_tensor, (self.h_0, self.c_0)]

            self.labels = (
                torch.randn(
                    seq_len, batch_size, self.D * h_out, device=device
                ) if not batch_first
                else torch.randn(batch_size, seq_len, self.D * h_out, device=device)
            )
            

    def make_private(layer: Layer) -> Layer:
        layer.layer = GradSampleModule(layer.layer)
        return layer


    @staticmethod
    def create(layer_name: str, **kwargs):
        if layer_name == LayerType.LINEAR:
            return LayerFactory.LinearBase(**kwargs)
        elif layer_name == LayerType.GSM_LINEAR:
            return LayerFactory.make_private(
                LayerFactory.LinearBase(**kwargs)
            )
        if layer_name == LayerType.CONV:
            return LayerFactory.ConvBase(**kwargs)
        elif layer_name == LayerType.GSM_CONV:
            return LayerFactory.make_private(
                LayerFactory.ConvBase(**kwargs)
            )
        if layer_name == LayerType.LAYERNORM:
            return LayerFactory.LayerNormBase(**kwargs)
        elif layer_name == LayerType.GSM_LAYERNORM:
            return LayerFactory.make_private(
                LayerFactory.LayerNormBase(**kwargs)
            )
        if layer_name == LayerType.INSTANCENORM:
            return LayerFactory.InstanceNormBase(**kwargs)
        elif layer_name == LayerType.GSM_INSTANCENORM:
            return LayerFactory.make_private(
                LayerFactory.InstanceNormBase(**kwargs)
            )
        if layer_name == LayerType.GROUPNORM:
            return LayerFactory.GroupNormBase(**kwargs)
        elif layer_name == LayerType.GSM_GROUPNORM:
            return LayerFactory.make_private(
                LayerFactory.GroupNormBase(**kwargs)
            )
        elif layer_name == LayerType.EMBEDDING:
            return LayerFactory.EmbeddingBase(**kwargs)
        elif layer_name == LayerType.GSM_EMBEDDING:
            return LayerFactory.make_private(
                LayerFactory.EmbeddingBase(**kwargs)
            )
        elif layer_name == LayerType.RNN:
            return LayerFactory.RNNBase(layer=nn.RNN, **kwargs)
        elif layer_name == LayerType.DPRNN:
            return LayerFactory.RNNBase(layer=DPRNN, **kwargs)
        elif layer_name == LayerType.GSM_DPRNN:
            return LayerFactory.make_private(
                LayerFactory.RNNBase(layer=DPRNN, **kwargs)
            )
        elif layer_name == LayerType.GRU:
            return LayerFactory.RNNBase(layer=nn.GRU, **kwargs)
        elif layer_name == LayerType.DPGRU:
            return LayerFactory.RNNBase(layer=DPGRU, **kwargs)
        elif layer_name == LayerType.GSM_DPGRU:
            return LayerFactory.make_private(
                LayerFactory.RNNBase(layer=DPGRU, **kwargs)
        )
        elif layer_name == LayerType.LSTM:
            return LayerFactory.LSTMBase(layer=nn.LSTM, **kwargs)
        elif layer_name == LayerType.DPLSTM:
            return LayerFactory.LSTMBase(layer=DPLSTM, **kwargs)
        elif layer_name == LayerType.GSM_DPLSTM:
            return LayerFactory.make_private(
                LayerFactory.LSTMBase(layer=DPLSTM, **kwargs)
            )
        elif layer_name == LayerType.MHA:
            return LayerFactory.MHABase(layer=nn.MultiheadAttention, **kwargs)
        elif layer_name == LayerType.DPMHA:
            return LayerFactory.MHABase(layer=DPMultiheadAttention, **kwargs)
        elif layer_name == LayerType.GSM_DPMHA:
            return LayerFactory.make_private(
                LayerFactory.MHABase(layer=DPMultiheadAttention, **kwargs)
            )
        else:
            print(f"Invalid layer type: {layer_name}.")
