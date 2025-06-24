import polars as pl
import numpy as np

import tomlkit

import torch
from torch import nn
import torch.nn.functional as F

import math
from dataclasses import dataclass
#from typing import Self
import os

CFG_FILE_NAME   = 'config.toml' # this is the one that is reported in the save dir
MODEL_FILE_NAME = 'model.torch'

__all__ = [
    'llama2_Filler', 'llama2_Predictor', 'load_llama2_for_inference', 'prepare_batch_for_inference',
    'llama2_Config', 'load_llama2_for_generation', 'prepare_batch_for_generation'
]

@dataclass(kw_only=True)
class llama2_Config:
    """
    Configuration class for the Llama2 model.

    Attributes:
        vocab_size (int): Size of the vocabulary.
        output_size (int): Size of the model output.
        hidden_size (int): Dimensionality of the hidden layers.
        num_layers (int): Number of transformer layers in the model.
        num_heads (int): Number of attention heads in each layer.
        head_dim (int): Dimensionality of each attention head.
        pos_base (float): Base value for positional encoding.
        dropout (float): Dropout probability for regularization.
        device (str | torch.device): Device to run the model on (e.g., 'cpu', 'cuda', or torch.device).
        mlp_intermediate_size (int): Size of the intermediate layer in the MLP block.
        parametrized_head (bool): Whether to use a parametrized output head.
    """
    vocab_size:  int
    output_size: int
    hidden_size: int
    num_layers:  int
    num_heads:   int
    head_dim:    int
    pos_base:    float
    dropout:     float
    device:      str | torch.device
    mlp_intermediate_size: int
    parametrized_head: bool


class llama2_Model(nn.Module):
    """
    Llama2_Model is a PyTorch neural network module implementing a transformer-based architecture
    inspired by Llama2. It supports configurable embedding, rotary positional encoding, multiple
    decoder layers, pooling, and a final linear head for output projection.

    Args:
        config (llama2_Config): Configuration object containing model hyperparameters such as
            vocab_size, hidden_size, num_layers, head_dim, pos_base, output_size, dropout, and device.
        decoder_mask (bool): If True, applies a decoder (causal) mask for autoregressive modeling.

    Attributes:
        config (llama2_Config): Model configuration.
        decoder_mask (bool): Whether to use decoder mask.
        embedding (nn.Embedding): Token embedding layer.
        decoder_layers (nn.ModuleList): List of transformer decoder layers.
        pooling (llama2_Pooling): Pooling layer.
        head (nn.Linear): Output projection layer.

    Methods:
        forward(batch, positions, lengths, return_attention=False):
            Performs a forward pass through the model.

            Args:
                batch (torch.Tensor): Input token IDs of shape (batch_size, seq_len).
                positions (torch.LongTensor): Positional indices of shape (batch_size, seq_len).
                lengths (torch.LongTensor): Lengths of each sequence in the batch, shape (batch_size,).
                return_attention (bool, optional): If True, returns attention weights from all layers.

            Returns:
                torch.Tensor: Output tensor after passing through the model.
                If return_attention is True, also returns attention weights as a tensor of shape
                (batch_size, num_layers, num_heads, seq_len, seq_len).

            Raises:
                ValueError: If input shapes are inconsistent.
    """
    def __init__(self, config: llama2_Config, decoder_mask: bool):
        super().__init__()
        self.config = config
        self.decoder_mask = decoder_mask
        with torch.device(config.device):
            self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
            rotary_embedding = Rotary_Embedding (
                config.head_dim,
                config.pos_base,
                device = config.device,
            )
            self.decoder_layers = nn.ModuleList (
                [llama2_Decoder_Layer(config, rotary_embedding) for _ in range(config.num_layers)]
            ) 
            self.pooling = llama2_Pooling(config)
            self.head = nn.Linear(config.hidden_size, config.output_size, bias=True)

    def forward (
        self,
        batch:     torch.Tensor,
        positions: torch.LongTensor,
        lengths:   torch.LongTensor,
        return_attention: bool = False,
    ) -> list[torch.Tensor]:
        # batch, position size is (bsz, b_n)  
        if batch.shape != positions.shape:
            raise ValueError(
                f'position shape should be equal to batch shape. Expected: {batch.shape}, '
                f'received: {positions.shape}'
            )
        # lenghts size is (bsz,)
        if batch.shape[0] != lengths.shape[0] or len(lengths.shape) >  1:
            raise ValueError(
                f'lengths shape should be ({batch.shape[0]},) but it is {lengths.shape}' 
            )

        bsz, b_n = positions.shape

        # PHASE 1: embed the code_ids

        # embedding has dim (bsz, n, hidden_size)
        embeddings = self.embedding(batch)

        # PHASE 2: create mask

        # attention mask has dim (bsz, 1, b_n, b_n)
        # decoder mask has dim (bsz, b_n, b_n)
        # pad mask has dim (bsz, b_n)

        minus_inf = torch.finfo(torch.float).min

        filter   = torch.arange(b_n, device=self.config.device).view((1, -1))
        pad_mask = torch.full((bsz, b_n), minus_inf, device=self.config.device)
        pad_mask = pad_mask.masked_fill_(filter < lengths.view((-1, 1)), 0)

        pad_mask = pad_mask[:, None, None, :]

        if self.decoder_mask:
            decoder_mask = torch.full((bsz, b_n, b_n), 0.0, device=self.config.device)
            decoder_mask = decoder_mask.masked_fill_(positions.unsqueeze(2) < positions.unsqueeze(1), minus_inf)
            decoder_mask = decoder_mask[:, None, :, :]
            mask = pad_mask + decoder_mask
        else:
            mask = pad_mask

        # PHASE 3: transformer

        all_attentions = []

        batch = embeddings
        for layer in self.decoder_layers:
            batch, attention = layer(batch, positions, mask) 
            all_attentions.append(attention)

        batch = F.dropout(batch, p=self.config.dropout, training=self.training)

        if return_attention:
            # this is of shape (n_layers, bsz, num_heads, b_n, b_n)
            all_attentions = torch.stack(all_attentions)
            # this is of shape (bsz, n_layers, num_heads, b_n, b_n)
            return batch, all_attentions.transpose(0, 1)
        else:
            return batch


class llama2_Filler(nn.Module):
    """
    Generator module that wraps a llama2_Model and applies a linear head for output transformation.

    Args:
        config (llama2_Config): Configuration object containing model parameters such as device, hidden size, and output size.

    Attributes:
        model (llama2_Model): The underlying llama2 model used for feature extraction.
        config (llama2_Config): The configuration object.
        head (nn.Linear): A linear layer mapping from hidden_size to output_size.

    Methods:
        forward(batch: torch.Tensor, positions: torch.LongTensor, lengths: torch.LongTensor) -> list[torch.Tensor]:
            Processes the input batch through the llama2 model and applies the linear head to produce the final output.

            Args:
                batch (torch.Tensor): Input tensor containing the batch data.
                positions (torch.LongTensor): Tensor containing position indices for the input data.
                lengths (torch.LongTensor): Tensor containing the lengths of each sequence in the batch.

            Returns:
                list[torch.Tensor]: The output of the linear head applied to the model's output.
    """
    def __init__(self, config: llama2_Config):
        super().__init__()
        self.model  = llama2_Model(config, decoder_mask=False)
        self.config = config
        with torch.device(config.device):
            self.head = nn.Linear(config.hidden_size, config.output_size, bias=True)

    def forward (
        self,
        batch:     torch.Tensor,
        positions: torch.LongTensor,
        lengths:   torch.LongTensor,
    ) -> list[torch.Tensor]:
        batch = self.model(batch, positions, lengths)
        result = self.head(batch)
        return result

    

class llama2_Predictor(nn.Module):
    """
    Neural network module for sequence prediction tasks using a Llama2-based model architecture.

    Args:
        config (llama2_Config): Configuration object containing model hyperparameters and device information.

    Attributes:
        config (llama2_Config): Stores the configuration for the model.
        model (llama2_Model): The main Llama2-based model with decoder mask enabled.
        pooling (llama2_Pooling): Pooling layer for aggregating sequence representations.
        head (nn.Linear): Linear layer mapping hidden representations to output size.

    Methods:
        forward(batch, positions, lengths, return_attention=False):
            Performs a forward pass through the model.
            
            Args:
                batch (torch.Tensor): Input batch tensor of token indices.
                positions (torch.LongTensor): Tensor indicating positions for each sequence in the batch.
                lengths (torch.LongTensor): Tensor indicating lengths of each sequence in the batch.
                return_attention (bool, optional): If True, also returns attention weights. Defaults to False.
            
            Returns:
                list[torch.Tensor]: List of output tensors for each sequence in the batch.
                If return_attention is True, also returns attention weights.
    """
    def __init__(self, config: llama2_Config):
        super().__init__()
        self.config = config
        self.model  = llama2_Model(config, decoder_mask=True)
        with torch.device(config.device):
            self.pooling = llama2_Pooling(config)
            self.head    = nn.Linear(config.hidden_size, config.output_size, bias=True)

    def forward (
        self,
        batch:     torch.Tensor,
        positions: torch.LongTensor,
        lengths:   torch.LongTensor,
        return_attention: bool = False,
    ) -> list[torch.Tensor]:
        bsz = batch.shape[0]
        batch = self.model(batch, positions, lengths, return_attention)

        if return_attention:
            batch, attention = batch

        # this is of shape (bsz, b_m, hidden)
        batch_pooled = self.pooling(batch, positions, lengths)
        output = self.head(batch_pooled)

        result = []
        for it in range(bsz):
            t = output[it][:positions[it].max() + 1]
            result.append(t)
        if return_attention:
            return result, attention
        else:
            return result

# See this [article](http://arxiv.org/abs/2104.09864)
class Rotary_Embedding:
    """
    Rotary_Embedding implements rotary positional embeddings for transformer models.

    Attributes:
        dim (int): The dimensionality of the embedding. Must be even.
        base (float): The base used for computing inverse frequencies.
        max_seq_len (int): The maximum sequence length cached for embeddings.
        inv_freq (torch.Tensor): Inverse frequency tensor for rotary embeddings.
        cos_buffer (torch.Tensor): Cached cosine values for rotary embeddings.
        sin_buffer (torch.Tensor): Cached sine values for rotary embeddings.

    Methods:
        __init__(dim: int, base: float, *, device):
            Initializes the Rotary_Embedding instance with the given dimension, base, and device.
            Raises ValueError if dim is not even.

        increase_cache(seq_len: int):
            Increases the cached cosine and sine buffers to support a new maximum sequence length.

        __call__(batch: torch.Tensor, positions: torch.LongTensor) -> torch.Tensor:
            Applies rotary positional embeddings to the input batch at the specified positions.

    Args:
        batch (torch.Tensor): Input tensor of shape (batch_size, num_heads, seq_len, head_dim).
        positions (torch.LongTensor): Tensor of positions for which to apply the embeddings.

    Returns:
        torch.Tensor: The input batch with rotary positional embeddings applied.
    """
    dim:    int
    base:   float

    def __init__(self, dim: int, base: float, *, device):
        if dim % 2 != 0:
            raise ValueError (f'Tried to instanciate a rotary embedding with a odd dimension ({dim})')
        self.dim  = dim
        self.base = base

        self.max_seq_len = 0
        # @rubustness when we call the .to() method on the parent module, this should be moved too
        self.inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=device).float() / self.dim))

    def increase_cache(self, seq_len: int):
        self.max_seq_len = seq_len
        t     = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb   = torch.cat((freqs, freqs), dim=-1)
        self.cos_buffer = emb.cos()
        self.sin_buffer = emb.sin()

    def __call__(self, batch: torch.Tensor, positions: torch.LongTensor) -> torch.Tensor:
        bsz, num_heads, b_n, head_dim = batch.shape
        if self.max_seq_len < b_n:
            self.increase_cache(max(b_n, 2*self.max_seq_len))
        batch_x = batch[..., : head_dim // 2  ] 
        batch_y = batch[...,   head_dim // 2 :] 
        rotated = torch.cat((-batch_y, batch_x), dim=-1)
        cos_f = self.cos_buffer[positions].unsqueeze(1) # we will broadcast over dim 1
        sin_f = self.sin_buffer[positions].unsqueeze(1) # we will broadcast over dim 1
        return batch*cos_f + rotated*sin_f


class llama2_Decoder_Layer(nn.Module):
    """
    A single decoder layer for a Llama2-like Transformer model.
    This layer consists of a multi-head self-attention mechanism with rotary embeddings,
    followed by a feed-forward MLP block. Layer normalization and dropout are applied
    before and after each sub-layer, with residual connections.
    Args:
        config (llama2_Config): Configuration object containing model hyperparameters.
        rotary_embedding (Rotary_Embedding): Rotary positional embedding module.
    Attributes:
        attention (llama2_Attention): Multi-head self-attention module with rotary embeddings.
        mlp (llama2_MLP): Feed-forward neural network module.
        normalization_pre (nn.LayerNorm): Layer normalization before the MLP block.
        normalization_post (nn.LayerNorm): Layer normalization after the MLP block.
        dropout (float): Dropout probability.
    Forward Args:
        batch (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_size).
        positions (torch.LongTensor): Position indices for rotary embeddings.
        mask (torch.Tensor): Attention mask tensor.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 
            - Output tensor after applying attention, MLP, normalization, and residual connections.
            - Attention weights from the attention module.
    """
    def __init__(self, config: llama2_Config, rotary_embedding: Rotary_Embedding):
        super().__init__()
        self.attention = llama2_Attention(config, rotary_embedding)
        self.mlp = llama2_MLP(config.hidden_size, config.mlp_intermediate_size)
        self.normalization_pre  = nn.LayerNorm(config.hidden_size)
        self.normalization_post = nn.LayerNorm(config.hidden_size)
        self.dropout = config.dropout
    
    def forward(self, batch: torch.Tensor, positions: torch.LongTensor, mask: torch.Tensor):
        residual = batch
        batch, attn = self.attention(batch, mask, positions)
        batch = F.dropout(batch, p=self.dropout, training=self.training)
        batch = batch + residual
        residual = batch
        batch = self.normalization_pre(batch)
        batch = self.mlp(batch)
        batch = F.dropout(batch, p=self.dropout, training=self.training)
        batch = batch + residual
        batch = self.normalization_post(batch)
        return batch, attn


class llama2_Attention(nn.Module):
    """
    Implements the multi-head self-attention mechanism with optional rotary positional embeddings for the Llama2 model.

    Args:
        config (llama2_Config): Configuration object containing model hyperparameters such as head_dim, num_heads, hidden_size, and dropout.
        rotary_embedding (Rotary_Embedding, optional): Optional rotary embedding module for applying positional encodings to queries and keys.

    Attributes:
        head_dim (int): Dimension of each attention head.
        num_heads (int): Number of attention heads.
        hidden_size (int): Size of the input and output hidden states.
        rotary_embedding (Rotary_Embedding or None): Rotary embedding module if provided.
        q_proj (nn.Linear): Linear projection for queries.
        k_proj (nn.Linear): Linear projection for keys.
        v_proj (nn.Linear): Linear projection for values.
        o_proj (nn.Linear): Linear projection for output.
        dropout (float): Dropout probability.

    Methods:
        forward(hidden_states, mask, positions=None):
            Computes the attention output and attention weights.

            Args:
                hidden_states (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_size).
                mask (torch.Tensor): Attention mask tensor broadcastable to (batch_size, num_heads, seq_len, seq_len).
                positions (torch.LongTensor, optional): Positional indices for rotary embeddings.

            Returns:
                attn_output (torch.Tensor): Output tensor after attention and projection, shape (batch_size, seq_len, hidden_size).
                attn_weights (torch.Tensor): Attention weights, shape (batch_size, num_heads, seq_len, seq_len).

            Raises:
                ValueError: If positions are provided but no rotary embedding is set, or if the attention output shape is incorrect.
    """
    def __init__(self, config: llama2_Config, rotary_embedding: Rotary_Embedding|None = None):
        super().__init__()
        self.head_dim         = config.head_dim
        self.num_heads        = config.num_heads
        self.hidden_size      = config.hidden_size
        self.rotary_embedding = rotary_embedding

        self.q_proj = nn.Linear(self.hidden_size,               self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size,               self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size,               self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.dropout = config.dropout

    def forward (
        self,
        hidden_states: torch.Tensor,
        mask:          torch.Tensor,
        positions:     torch.LongTensor|None = None,
    ):
        bsz, b_n, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states   = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        query_states = query_states.view(bsz, b_n, self.num_heads, self.head_dim).transpose(1, 2)
        key_states   = key_states  .view(bsz, b_n, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, b_n, self.num_heads, self.head_dim).transpose(1, 2)

        if self.rotary_embedding is not None:
            query_states = self.rotary_embedding(query_states, positions)
            key_states   = self.rotary_embedding(key_states,   positions)
        else:
            if positions is not None:
                raise ValueError('positions provided but no rotary embedding is set')

        attn_logits = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        attn_logits = attn_logits + mask

        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_output  = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, b_n, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, b_n, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, b_n, self.num_heads * self.head_dim)
        attn_output = F.dropout(attn_output, p=self.dropout, training=self.training)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights

class llama2_MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) module used in Llama2 architecture.

    Args:
        hidden_size (int): The size of the input and output feature dimension.
        intermediate_size (int): The size of the intermediate feature dimension.

    Attributes:
        gate_proj (nn.Linear): Linear layer projecting from hidden_size to intermediate_size (no bias).
        up_proj (nn.Linear): Linear layer projecting from hidden_size to intermediate_size (no bias).
        down_proj (nn.Linear): Linear layer projecting from intermediate_size back to hidden_size (no bias).
        act_fn (Callable): Activation function (SiLU).

    Forward Args:
        x (torch.Tensor): Input tensor of shape (..., hidden_size).

    Returns:
        torch.Tensor: Output tensor of shape (..., hidden_size) after applying the gated MLP transformation.
    """
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj   = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn    = F.silu

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

class llama2_Pooling(nn.Module):
    """
    Pooling module for the Llama2 model, supporting both parametrized and non-parametrized pooling heads.
    Args:
        config (llama2_Config): Configuration object containing model hyperparameters.
            - parametrized_head (bool): If True, uses attention and MLP-based pooling; otherwise, uses uniform pooling.
            - hidden_size (int): Dimensionality of the hidden representations.
            - mlp_intermediate_size (int): Size of the intermediate layer in the MLP.
    Attributes:
        parametrized_head (bool): Indicates whether to use parametrized pooling.
        scorer_attn (llama2_Attention, optional): Attention module for computing pooling scores (if parametrized_head is True).
        scorer_gmlp (llama2_MLP, optional): MLP module for further processing pooling scores (if parametrized_head is True).
    Forward Args:
        batch (torch.Tensor): Input tensor of shape (batch_size, num_tokens, hidden_size) containing token representations.
        pos (torch.LongTensor): Tensor of shape (batch_size, num_tokens) indicating position/grouping of each token (e.g., visit indices).
        lengths (torch.LongTensor): Tensor of shape (batch_size,) indicating the valid length for each sequence in the batch.
    Returns:
        torch.Tensor: Pooled output tensor of shape (batch_size, max_num_groups, hidden_size), where max_num_groups is determined by the maximum value in `pos` plus one.
    Notes:
        - If `parametrized_head` is True, pooling scores are computed using attention and an MLP; otherwise, uniform pooling is applied.
        - Handles padding and masking to ensure only valid tokens contribute to the pooled output.
        - Supports visit-wise (or group-wise) pooling based on the `pos` tensor.
    """
    def __init__(self, config: llama2_Config):
        super().__init__()
        self.parametrized_head = config.parametrized_head
        if config.parametrized_head:
            self.scorer_attn = llama2_Attention(config)
            self.scorer_gmlp = llama2_MLP(config.hidden_size, config.mlp_intermediate_size)

    def forward(self, batch: torch.Tensor, pos: torch.LongTensor, lengths: torch.LongTensor) -> torch.Tensor:
        bsz, b_n, h = batch.shape
        b_m = pos.max() + 1
        minus_inf = torch.finfo(torch.float).min
        
        # compute pooling scores
        with torch.device(batch.device):
            decoder_mask = torch.full((bsz, b_n, b_n), 0.0)
            decoder_mask = decoder_mask.masked_fill_(pos.unsqueeze(2) != pos.unsqueeze(1), minus_inf)

            filter   = torch.arange(b_n).view((1, -1))
            pad_mask = torch.full((bsz, b_n), minus_inf)
            pad_mask = pad_mask.masked_fill_(filter < lengths.view((-1, 1)), 0)

        decoder_mask = decoder_mask[:, None, :, :]
        pad_mask = pad_mask[:, None, None, :]
        mask = pad_mask + decoder_mask

        if self.parametrized_head:
            pooling_score = self.scorer_attn(batch, mask)
            pooling_score = self.scorer_gmlp(batch)
        else:
            with torch.device(batch.device):
                pooling_score = torch.ones((bsz, b_n, h))

        # visit-wise sums helpers
        ids = pos.clamp(0, None) + torch.arange(bsz, device=pos.device).unsqueeze(1) * b_m
        ids = ids.flatten()

        # Compute visit-wise softmax
        pooling_mask = torch.zeros((bsz, b_n, 1), device=batch.device)
        pooling_mask.masked_fill_((pos == -1).unsqueeze(2), minus_inf)

        pooling_score += pooling_mask
        pooling_score = torch.exp(pooling_score)
        pooling_score = pooling_score.reshape(-1, h)

        sums = torch.zeros((bsz * b_m, h), device=batch.device)
        sums = sums.index_add_(0, ids, pooling_score)
        pooling_probs = pooling_score / sums[ids]

        # now pool
        batch = batch.reshape((-1, h)) * pooling_probs
        result = torch.zeros((bsz * b_m, h), device=batch.device)
        result = result.index_add_(0, ids, batch)
        result = result.reshape((bsz, b_m, h))

        return result


@dataclass
class Inference_Batch:
    codes:     torch.Tensor
    positions: torch.Tensor
    lenghts:   torch.Tensor

    def unpack(self) -> dict:
        return {
            'batch':     self.codes,
            'positions': self.positions,
            'lengths':   self.lenghts,
        }

def prepare_batch_for_inference (
    
    codes:     list[np.ndarray],
    counts:    list[np.ndarray],
    positions: list[np.ndarray],
    device:    torch.device,
) -> Inference_Batch:
    """
    Prepares a batch of input data for inference by padding and converting numpy arrays to PyTorch tensors.

    Args:
        codes (list[np.ndarray]): List of numpy arrays containing code sequences for each sample in the batch.
        counts (list[np.ndarray]): List of numpy arrays containing count information for each sample (not used in output).
        positions (list[np.ndarray]): List of numpy arrays containing position information for each sample in the batch.
        device (torch.device): The device (CPU or CUDA) to which the tensors will be moved.

    Returns:
        Inference_Batch: An object containing the following fields:
            - codes (torch.Tensor): Padded tensor of code sequences, shape (batch_size, max_seq_len).
            - positions (torch.Tensor): Padded tensor of position sequences, shape (batch_size, max_seq_len).
            - lenghts (torch.Tensor): Tensor of original sequence lengths, shape (batch_size,).

    Notes:
        - Sequences in `codes` are padded with 0.
        - Sequences in `positions` are padded with -1.
        - All output tensors are moved to the specified device.
        - The `counts` argument is converted but not used in the returned batch.
    """
    codes     = [x.astype(np.int64) for x in codes]
    counts    = [x.astype(np.int64) for x in counts]
    positions = [x.astype(np.int64) for x in positions]

    lengths = [len(x) for x in codes]
    b_n = max(lengths)

    b_codes     = np.array([np.pad(x, (0, b_n - len(x)), constant_values=0 ) for x in codes])
    b_positions = np.array([np.pad(x, (0, b_n - len(x)), constant_values=-1) for x in positions])

    with torch.device(device):
        b_codes     = torch.from_numpy(b_codes)    .to(device)
        b_positions = torch.from_numpy(b_positions).to(device)
        b_lengths   = torch.LongTensor(lengths)    .to(device)

    return Inference_Batch (
        codes     = b_codes,
        positions = b_positions,
        lenghts   = b_lengths,
    )

@dataclass
class Generation_Batch:
    codes:     torch.Tensor
    positions: torch.Tensor
    lenghts:   torch.Tensor

    def unpack(self) -> dict:
        return {
            'batch':     self.codes,
            'positions': self.positions,
            'lengths':   self.lenghts,
        }

def prepare_batch_for_generation (
    
    codes:     list[np.ndarray],
    counts:    list[np.ndarray],
    positions: list[np.ndarray],
    hole_prob: float,
    hole_token_id: int,
    device:    torch.device,
) -> Generation_Batch:
    """
    Prepares a batch of input data for sequence generation by padding, masking, and converting arrays to tensors.

    Args:
        codes (list[np.ndarray]): List of integer arrays representing code sequences for each sample in the batch.
        counts (list[np.ndarray]): List of integer arrays representing counts for each sample (currently unused in function).
        positions (list[np.ndarray]): List of integer arrays representing positional information for each sample.
        hole_prob (float): Probability of masking (replacing) each token in the input with the hole token.
        hole_token_id (int): The integer ID to use for masked (hole) tokens.
        device (torch.device): The device (CPU or GPU) to which the resulting tensors will be moved.

    Returns:
        Inference_Batch: An object containing the padded, masked, and tensorized codes, positions, and sequence lengths for the batch.

    Notes:
        - All input arrays are padded to the length of the longest sequence in the batch.
        - Positions are padded with -1, codes are padded with 0.
        - Random masking is applied to codes with probability `hole_prob`.
        - The `counts` argument is currently not used in the returned batch.
    """
    codes     = [x.astype(np.int64) for x in codes]
    counts    = [x.astype(np.int64) for x in counts]
    positions = [x.astype(np.int64) for x in positions]

    lengths = [len(x) for x in codes]
    b_n = max(lengths)

    b_codes     = np.array([np.pad(x, (0, b_n - len(x)), constant_values=0 ) for x in codes])
    b_positions = np.array([np.pad(x, (0, b_n - len(x)), constant_values=-1) for x in positions])

    mask = np.random.rand(*b_codes.shape) < hole_prob
    b_codes[mask] = hole_token_id

    with torch.device(device):
        b_codes     = torch.from_numpy(b_codes)    .to(device)
        b_positions = torch.from_numpy(b_positions).to(device)
        b_lengths   = torch.LongTensor(lengths)    .to(device)

    return Inference_Batch (
        codes     = b_codes,
        positions = b_positions,
        lenghts   = b_lengths,
    )




def load_llama2_for_inference(path: str) -> llama2_Predictor:
    """
    Loads a Llama2 model for inference from the specified directory.

    Args:
        path (str): The directory path containing the model configuration and weights files.

    Returns:
        llama2_Predictor: An instance of the Llama2 predictor model loaded with the specified configuration and weights.

    Raises:
        FileNotFoundError: If the configuration or model file does not exist at the specified path.
        tomlkit.exceptions.ParseError: If the configuration file cannot be parsed.
        RuntimeError: If the model state dictionary cannot be loaded.
    """
    config_path = os.path.join(path, CFG_FILE_NAME)
    model_path  = os.path.join(path, MODEL_FILE_NAME)

    with open(config_path, 'r') as f:
        txt = f.read()
    config = tomlkit.parse(txt)['model']
    config = llama2_Config(**config)

    model = llama2_Predictor(config) 
    state_dict = torch.load(model_path, map_location=config.device, weights_only=False)
    model.load_state_dict(state_dict)
    return model

def load_llama2_for_generation(path: str, device: str) -> llama2_Filler:
    """
    Loads a Llama2 model for text generation from the specified path and prepares it for use on the given device.

    Args:
        path (str): The directory path containing the model and configuration files.
        device (str): The device identifier (e.g., 'cpu', 'cuda:0') to load the model onto.

    Returns:
        tuple:
            - llama2_Filler: The loaded Llama2 model ready for generation.
            - float: The probability of a "hole" (mask) used during training.
            - int: The token ID used to represent a "hole" in the input.

    Raises:
        FileNotFoundError: If the configuration or model file does not exist at the specified path.
        tomlkit.exceptions.ParseError: If the configuration file cannot be parsed.
        RuntimeError: If the model state dictionary cannot be loaded.
    """
    config_path = os.path.join(path, CFG_FILE_NAME)
    model_path  = os.path.join(path, MODEL_FILE_NAME)

    with open(config_path, 'r') as f:
        txt = f.read()
    config = tomlkit.parse(txt)
    hole_prob = config['trainer']['hole_prob']
    hole_token_id = config['trainer']['hole_token_id']
    config['model']['device'] = str(device)  # Aggiorna il device nel config)

    config = llama2_Config(**config['model'])

    model = llama2_Filler(config) 
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)

    return model, hole_prob, hole_token_id

