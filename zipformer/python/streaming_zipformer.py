"""
zipformer_streaming_model.py

Streaming DLA 适配版 Zipformer 模型定义。

修改内容 (相比原始 icefall 代码):
1. BasicNorm.forward:  ** -0.5  → torch.rsqrt()   (RSQRT 在 NPU 5.3 支持)
2. PoolingModule.streaming_forward:
   - x.cumsum(dim=0) → matmul(tril_ones, x)  (CUMSUM 不支持)
   - cached_len 全程使用 float32              (int64 不支持)
3. RelPositionMultiheadAttention.streaming_multi_head_attention_forward:
   - torch.gather(pos_weights, ...) → slice + stack (GATHER 不支持)
4. EncoderStreaming: 封装 Zipformer.streaming_forward，cached_len float32 输入/输出
5. DecoderNPU: 去掉 Embedding，接受 embedded float32 输入
6. JoinerStreaming:  直接包装 Joiner (无修改)

使用方式:
    from streaming_zipformer import EncoderStreaming, DecoderNPU, JoinerStreaming
"""

import sys
import types
import contextlib
import warnings
import math
import copy
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# icefall.utils 最小 stub（只需 make_pad_mask, subsequent_chunk_mask）
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def _torch_autocast(device_type="cuda", **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with torch.cuda.amp.autocast(enabled=False):
            yield

def _make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    assert lengths.ndim == 1
    max_len = max_len if max_len > 0 else int(lengths.max())
    n = lengths.size(0)
    seq_range = torch.arange(0, max_len, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(n, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask

def _subsequent_chunk_mask(size, chunk_size, num_left_chunks=-1, device=torch.device("cpu")):
    ret = torch.zeros(size, size, device=device, dtype=torch.bool)
    for i in range(size):
        if num_left_chunks < 0:
            start = 0
        else:
            start = max(0, (i // chunk_size - num_left_chunks) * chunk_size)
        ending = min(size, (i // chunk_size + 1) * chunk_size)
        ret[i, start:ending] = True
    return ret

if "icefall" not in sys.modules:
    icefall_utils_stub = types.ModuleType("icefall.utils")
    icefall_utils_stub.torch_autocast        = _torch_autocast
    icefall_utils_stub.make_pad_mask         = _make_pad_mask
    icefall_utils_stub.subsequent_chunk_mask = _subsequent_chunk_mask
    icefall_stub = types.ModuleType("icefall")
    icefall_stub.utils = icefall_utils_stub
    sys.modules["icefall"]       = icefall_stub
    sys.modules["icefall.utils"] = icefall_utils_stub
    for _mod in ["icefall.checkpoint", "icefall.decode", "icefall.dist",
                 "icefall.env", "icefall.lm_wrapper", "icefall.rnn_lm",
                 "icefall.rnn_lm.model"]:
        sys.modules[_mod] = types.ModuleType(_mod)

# ---------------------------------------------------------------------------
# 导入 icefall 模型
# （icefall egs 路径已由调用方 export_onnx.py 通过 sys.path 注入，这里不再自行解析）
# ---------------------------------------------------------------------------
from zipformer import Zipformer  # noqa
from decoder   import Decoder    # noqa
from joiner    import Joiner     # noqa
from scaling   import BasicNorm  # noqa


# ===========================================================================
# 修改 1: BasicNorm — Pow(-0.5) → rsqrt
# ===========================================================================
class BasicNormStreaming(nn.Module):
    """
    BasicNorm 的 Streaming 适配版。
    原始代码:  (mean(x^2) + eps.exp()) ** -0.5
    改为:      torch.rsqrt(mean(x^2) + eps.exp())
    RSQRT 在 NPU 5.3 支持。
    """

    def __init__(self, orig: BasicNorm):
        super().__init__()
        self.num_channels = orig.num_channels
        self.channel_dim  = orig.channel_dim
        # 复制 eps 参数
        if isinstance(orig.eps, nn.Parameter):
            self.eps = nn.Parameter(orig.eps.detach().clone())
        else:
            self.register_buffer("eps", orig.eps.detach().clone())
        self.eps_min = orig.eps_min
        self.eps_max = orig.eps_max

    def forward(self, x: Tensor) -> Tensor:
        eps = self.eps
        scales = torch.rsqrt(
            torch.mean(x ** 2, dim=self.channel_dim, keepdim=True) + eps.exp()
        )
        return x * scales


# ===========================================================================
# 修改 2: PoolingModuleStreaming — cumsum → matmul, cached_len float32
# ===========================================================================
class PoolingModuleStreaming(nn.Module):
    """
    PoolingModule 的 Streaming 适配版 (仅实现 streaming_forward)。
    修改:
    1. x.cumsum(dim=0) → matmul(tril_ones, x)  避免 CUMSUM 算子
    2. cached_len 使用 float32 (而非 int64)
    """

    def __init__(self, orig_pooling, T: int):
        """
        Args:
            orig_pooling: 原始 PoolingModule 实例
            T: 该 stack 的 streaming_forward 输出帧数（固定值）
        """
        super().__init__()
        self.proj = orig_pooling.proj

        # 预计算下三角全1矩阵 [T, T]
        # tril_ones[i, j] = 1 if j <= i else 0
        tril_ones = torch.tril(torch.ones(T, T, dtype=torch.float32))  # [T, T]
        self.register_buffer("tril_ones", tril_ones)
        self.T = T

    def streaming_forward(
        self,
        x: Tensor,        # [T, N, C]
        cached_len: Tensor,  # [N] or scalar, float32
        cached_avg: Tensor,  # [N, C], float32
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Returns: (output [T,N,C], updated_cached_len, updated_cached_avg)
        All float32, no int64.
        """
        T_actual, N, C = x.shape

        # Cumsum via matmul: x [T,N,C] → [N,C,T] → matmul → [N,C,T] → [T,N,C]
        # tril_ones: [T,T]
        # x_perm: [N, C, T]
        x_perm = x.permute(1, 2, 0)  # [N, C, T]
        # matmul: [N, C, T] @ [T, T] → [N, C, T]
        x_cumsum = torch.matmul(x_perm, self.tril_ones.t())  # [N, C, T]
        # Note: (tril_ones @ x_perm^T)^T = x_perm @ tril_ones^T
        # But tril_ones.t() = triu, so matmul(x_perm, tril_ones.t()) gives cumsum along T
        # Let's verify: output[n,c,t] = sum_j(x_perm[n,c,j] * tril_ones.t()[j,t])
        #             = sum_j(x_perm[n,c,j] * tril_ones[t,j])
        #             = sum_{j<=t} x_perm[n,c,j]  ← correct cumsum
        x = x_cumsum.permute(2, 0, 1)  # [T, N, C]

        # Add cached contribution: cached_avg * cached_len
        # cached_len: [N] (scalar per batch), cached_avg: [N, C]
        x = x + (cached_avg * cached_len.unsqueeze(1)).unsqueeze(0)  # [T, N, C]

        # Compute pooling mask: 1/(t+1+cached_len)
        # cum_mask: [T, N]
        t_arange = torch.arange(1, T_actual + 1, dtype=x.dtype, device=x.device)  # [T]
        cum_mask = t_arange.unsqueeze(1) + cached_len.unsqueeze(0)  # [T, N]
        pooling_mask = (1.0 / cum_mask).unsqueeze(2)  # [T, N, 1]
        x = x * pooling_mask  # [T, N, C]

        # Update cache
        # cached_len += T (float32 arithmetic)
        cached_len = cached_len + float(T_actual)
        cached_avg = x[-1]  # [N, C]

        x = self.proj(x)
        return x, cached_len, cached_avg


# ===========================================================================
# 修改 3: RelPositionMultiheadAttentionStreaming
# streaming_multi_head_attention_forward 中替换 torch.gather
# ===========================================================================
class RelPositionMultiheadAttentionStreaming(nn.Module):
    """
    RelPositionMultiheadAttention 的 Streaming 适配版。
    仅修改 streaming_multi_head_attention_forward 中的 torch.gather：
    用预计算的固定 gather 替换（slice + stack 方式）。
    """

    def __init__(self, orig_attn, time1: int, kv_len: int, batch_size: int = 1):
        """
        Args:
            orig_attn: 原始 RelPositionMultiheadAttention 实例
            time1: 查询序列长度（streaming forward 的固定 seq_len）
            kv_len: key/value 长度 = time1 + left_context_len//ds
            batch_size: 推理时 batch_size（固定为 1）
        """
        super().__init__()
        # 复制所有权重
        self.embed_dim    = orig_attn.embed_dim
        self.attention_dim = orig_attn.attention_dim
        self.num_heads    = orig_attn.num_heads
        self.dropout      = orig_attn.dropout
        self.head_dim     = orig_attn.head_dim
        self.pos_dim      = orig_attn.pos_dim

        self.in_proj      = orig_attn.in_proj
        self.whiten_values  = orig_attn.whiten_values
        self.whiten_keys    = orig_attn.whiten_keys
        self.linear_pos     = orig_attn.linear_pos
        self.copy_pos_query = orig_attn.copy_pos_query
        self.copy_query     = orig_attn.copy_query
        self.out_proj       = orig_attn.out_proj
        self.in_proj2       = orig_attn.in_proj2
        self.out_proj2      = orig_attn.out_proj2
        self.whiten_values2 = orig_attn.whiten_values2

        self.time1    = time1
        self.kv_len   = kv_len

        # 预计算 gather 的 slice 起始索引（用于 streaming forward）
        # pos_weights: [B, H, T, N] where N = 2*T-1 + left_context_len
        # 对行 i (0..T-1): 起始位置 = T-1-i，长度 = kv_len
        # 预存每行的 start offset
        slice_starts = [time1 - 1 - i for i in range(time1)]
        self.register_buffer(
            "slice_starts",
            torch.tensor(slice_starts, dtype=torch.long)
        )

    def forward(
        self,
        x: Tensor,
        pos_emb: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        # 使用原始的 multi_head_attention_forward（非 streaming）
        # 这个路径在我们的 Streaming trace 中不会被调用
        raise NotImplementedError("Use streaming_forward instead")

    def streaming_forward(
        self,
        x: Tensor,
        pos_emb: Tensor,
        cached_key: Tensor,
        cached_val: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        (
            x,
            weights,
            cached_key,
            cached_val,
        ) = self.streaming_multi_head_attention_forward_streaming(
            self.in_proj(x),
            self.linear_pos(pos_emb),
            self.attention_dim,
            self.num_heads,
            self.out_proj.weight,
            self.out_proj.bias,
            cached_key=cached_key,
            cached_val=cached_val,
        )
        return x, weights, cached_key, cached_val

    def streaming_multi_head_attention_forward_streaming(
        self,
        x_proj: Tensor,
        pos: Tensor,
        attention_dim: int,
        num_heads: int,
        out_proj_weight: Tensor,
        out_proj_bias: Tensor,
        cached_key: Tensor,
        cached_val: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        streaming_multi_head_attention_forward 的 Streaming 版本。
        用 slice+stack 替换 torch.gather。
        """
        seq_len, bsz, _ = x_proj.size()

        head_dim = attention_dim // num_heads
        pos_dim  = self.pos_dim

        # Split projections
        q = x_proj[..., 0:attention_dim]
        k = x_proj[..., attention_dim : 2 * attention_dim]
        value_dim = attention_dim // 2
        v = x_proj[..., 2 * attention_dim : 2 * attention_dim + value_dim]
        p = x_proj[..., 2 * attention_dim + value_dim :]

        left_context_len = cached_key.shape[0]

        # Concatenate with cached context
        k = torch.cat([cached_key, k], dim=0)
        v = torch.cat([cached_val, v], dim=0)
        # Update cache
        cached_key = k[-left_context_len:, ...]
        cached_val = v[-left_context_len:, ...]

        kv_len = k.shape[0]

        q = q.reshape(seq_len, bsz, num_heads, head_dim)
        p = p.reshape(seq_len, bsz, num_heads, pos_dim)
        k = k.reshape(kv_len, bsz, num_heads, head_dim)
        v = v.reshape(kv_len, bsz * num_heads, head_dim // 2).transpose(0, 1)

        q = q.permute(1, 2, 0, 3)   # [B, H, T, head_dim]
        p = p.permute(1, 2, 0, 3)   # [B, H, T, pos_dim]
        k = k.permute(1, 2, 3, 0)   # [B, H, head_dim, kv_len]

        seq_len2 = 2 * seq_len - 1 + left_context_len
        pos = pos.reshape(1, seq_len2, num_heads, pos_dim).permute(0, 2, 3, 1)
        # pos: [1, H, pos_dim, seq_len2]

        # pos_weights: [B, H, T, seq_len2]
        pos_weights = torch.matmul(p, pos)

        # ---- Replace torch.gather with slice+stack ----
        # pos_weights[b, h, i, :] has relative positions
        # We need: output[b,h,i,j] = pos_weights[b,h,i, T-1-i+j] for j=0..kv_len-1
        # slice_starts[i] = T-1-i
        slices = []
        for i in range(seq_len):
            start = self.time1 - 1 - i
            slices.append(pos_weights[:, :, i:i+1, start:start+kv_len])
        pos_weights = torch.cat(slices, dim=2)  # [B, H, T, kv_len]

        # Compute attention scores
        attn_output_weights = torch.matmul(q, k) + pos_weights
        # [B, H, T, kv_len] → [B*H, T, kv_len]
        attn_output_weights = attn_output_weights.view(bsz * num_heads, seq_len, kv_len)

        attn_output_weights = F.softmax(attn_output_weights, dim=-1)

        attn_output = torch.bmm(attn_output_weights, v)
        assert list(attn_output.size()) == [bsz * num_heads, seq_len, head_dim // 2]
        attn_output = (
            attn_output.transpose(0, 1)
            .contiguous()
            .view(seq_len, bsz, attention_dim // 2)
        )
        attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)

        return attn_output, attn_output_weights, cached_key, cached_val

    def streaming_forward2(
        self,
        x: Tensor,
        attn_weights: Tensor,
        cached_val: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Second forward (reuse attn_weights with different input)."""
        num_heads = self.num_heads
        (seq_len, bsz, embed_dim) = x.shape
        head_dim = self.attention_dim // num_heads

        v = self.in_proj2(x)

        left_context_len = cached_val.shape[0]
        v = torch.cat([cached_val, v], dim=0)
        cached_val = v[-left_context_len:]

        seq_len2 = left_context_len + seq_len
        v = v.reshape(seq_len2, bsz * num_heads, head_dim // 2).transpose(0, 1)

        attn_output = torch.bmm(attn_weights, v)
        attn_output = (
            attn_output.transpose(0, 1)
            .contiguous()
            .view(seq_len, bsz, self.attention_dim // 2)
        )
        return self.out_proj2(attn_output), cached_val


# ===========================================================================
# 修改 4: ZipformerEncoderLayerStreaming — 组合修改后的子模块
# ===========================================================================
class ZipformerEncoderLayerStreaming(nn.Module):
    """
    ZipformerEncoderLayer 的 Streaming 适配版。
    只修改 streaming_forward 路径中使用的子模块。
    """

    def __init__(self, orig_layer, time1: int, kv_len: int):
        super().__init__()
        # 不修改的模块直接复用原始引用
        self.feed_forward1  = orig_layer.feed_forward1
        self.feed_forward2  = orig_layer.feed_forward2
        self.feed_forward3  = orig_layer.feed_forward3
        self.conv_module1   = orig_layer.conv_module1
        self.conv_module2   = orig_layer.conv_module2
        self.balancer       = orig_layer.balancer
        self.bypass_scale   = orig_layer.bypass_scale

        # BasicNorm Streaming: Pow(-0.5) → rsqrt
        self.norm_final = BasicNormStreaming(orig_layer.norm_final)

        # PoolingModule Streaming: cumsum → matmul, cached_len float32
        self.pooling = PoolingModuleStreaming(orig_layer.pooling, T=time1)

        # RelPositionMultiheadAttention Streaming: gather → slice+stack
        self.self_attn = RelPositionMultiheadAttentionStreaming(
            orig_layer.self_attn, time1=time1, kv_len=kv_len
        )

    def streaming_forward(
        self,
        src: Tensor,
        pos_emb: Tensor,
        cached_len: Tensor,     # float32
        cached_avg: Tensor,
        cached_key: Tensor,
        cached_val: Tensor,
        cached_val2: Tensor,
        cached_conv1: Tensor,
        cached_conv2: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        src_orig = src

        src = src + self.feed_forward1(src)

        src_pool, cached_len, cached_avg = self.pooling.streaming_forward(
            src, cached_len=cached_len, cached_avg=cached_avg,
        )
        src = src + src_pool

        (
            src_attn,
            attn_weights,
            cached_key,
            cached_val,
        ) = self.self_attn.streaming_forward(
            src, pos_emb=pos_emb, cached_key=cached_key, cached_val=cached_val,
        )
        src = src + src_attn

        src_conv, cached_conv1 = self.conv_module1.streaming_forward(
            src, cache=cached_conv1,
        )
        src = src + src_conv

        src = src + self.feed_forward2(src)

        src_attn, cached_val2 = self.self_attn.streaming_forward2(
            src, attn_weights, cached_val=cached_val2,
        )
        src = src + src_attn

        src_conv, cached_conv2 = self.conv_module2.streaming_forward(
            src, cache=cached_conv2,
        )
        src = src + src_conv

        src = src + self.feed_forward3(src)

        src = self.norm_final(self.balancer(src))

        delta = src - src_orig
        src   = src_orig + delta * self.bypass_scale

        return (
            src,
            cached_len,
            cached_avg,
            cached_key,
            cached_val,
            cached_val2,
            cached_conv1,
            cached_conv2,
        )


# ===========================================================================
# 修改 5: ZipformerEncoderStreaming — 替换 layers 并转发 streaming_forward
# ===========================================================================
class ZipformerEncoderStreaming(nn.Module):
    """ZipformerEncoder 的 Streaming 适配版（streaming_forward only）"""

    def __init__(self, orig_encoder, time1_per_stack: int, kv_len: int):
        """
        Args:
            orig_encoder: 原始 ZipformerEncoder 实例（num_layers层）
            time1_per_stack: 本 encoder stack 在 streaming 时的 seq_len
            kv_len: key/value length = time1 + left_context_len//ds
        """
        super().__init__()
        self.encoder_pos = orig_encoder.encoder_pos
        self.num_layers  = orig_encoder.num_layers
        self.d_model     = orig_encoder.d_model
        self.attention_dim = orig_encoder.attention_dim
        self.cnn_module_kernel = orig_encoder.cnn_module_kernel

        self.layers = nn.ModuleList([
            ZipformerEncoderLayerStreaming(layer, time1=time1_per_stack, kv_len=kv_len)
            for layer in orig_encoder.layers
        ])

    def streaming_forward(
        self,
        src: Tensor,
        cached_len: Tensor,
        cached_avg: Tensor,
        cached_key: Tensor,
        cached_val: Tensor,
        cached_val2: Tensor,
        cached_conv1: Tensor,
        cached_conv2: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        left_context_len = cached_key.shape[1]
        pos_emb = self.encoder_pos(src, left_context_len)

        output = src

        new_cached_len   = []
        new_cached_avg   = []
        new_cached_key   = []
        new_cached_val   = []
        new_cached_val2  = []
        new_cached_conv1 = []
        new_cached_conv2 = []

        for i, mod in enumerate(self.layers):
            output, len_avg, avg, key, val, val2, conv1, conv2 = mod.streaming_forward(
                output,
                pos_emb,
                cached_len=cached_len[i],
                cached_avg=cached_avg[i],
                cached_key=cached_key[i],
                cached_val=cached_val[i],
                cached_val2=cached_val2[i],
                cached_conv1=cached_conv1[i],
                cached_conv2=cached_conv2[i],
            )
            new_cached_len.append(len_avg)
            new_cached_avg.append(avg)
            new_cached_key.append(key)
            new_cached_val.append(val)
            new_cached_val2.append(val2)
            new_cached_conv1.append(conv1)
            new_cached_conv2.append(conv2)

        return (
            output,
            torch.stack(new_cached_len,   dim=0),
            torch.stack(new_cached_avg,   dim=0),
            torch.stack(new_cached_key,   dim=0),
            torch.stack(new_cached_val,   dim=0),
            torch.stack(new_cached_val2,  dim=0),
            torch.stack(new_cached_conv1, dim=0),
            torch.stack(new_cached_conv2, dim=0),
        )


# ===========================================================================
# 修改 6: DownsampledZipformerEncoderStreaming
# ===========================================================================
class DownsampledZipformerEncoderStreaming(nn.Module):
    """DownsampledZipformerEncoder 的 Streaming 适配版"""

    def __init__(self, orig_ds_encoder, time1_per_stack: int, kv_len: int):
        super().__init__()
        self.downsample_factor = orig_ds_encoder.downsample_factor
        self.downsample  = orig_ds_encoder.downsample
        self.upsample    = orig_ds_encoder.upsample
        self.out_combiner = orig_ds_encoder.out_combiner
        self.num_layers  = orig_ds_encoder.num_layers
        self.d_model     = orig_ds_encoder.d_model
        self.attention_dim = orig_ds_encoder.attention_dim
        self.cnn_module_kernel = orig_ds_encoder.cnn_module_kernel

        # 内部 encoder 也替换为 Streaming 版本
        self.encoder = ZipformerEncoderStreaming(
            orig_ds_encoder.encoder, time1_per_stack, kv_len
        )

    def streaming_forward(
        self,
        src: Tensor,
        cached_len: Tensor,
        cached_avg: Tensor,
        cached_key: Tensor,
        cached_val: Tensor,
        cached_val2: Tensor,
        cached_conv1: Tensor,
        cached_conv2: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        src_orig = src
        src = self.downsample(src)

        (
            src,
            cached_len,
            cached_avg,
            cached_key,
            cached_val,
            cached_val2,
            cached_conv1,
            cached_conv2,
        ) = self.encoder.streaming_forward(
            src,
            cached_len=cached_len,
            cached_avg=cached_avg,
            cached_key=cached_key,
            cached_val=cached_val,
            cached_val2=cached_val2,
            cached_conv1=cached_conv1,
            cached_conv2=cached_conv2,
        )
        src = self.upsample(src)
        src = src[: src_orig.shape[0]]

        return (
            self.out_combiner(src_orig, src),
            cached_len,
            cached_avg,
            cached_key,
            cached_val,
            cached_val2,
            cached_conv1,
            cached_conv2,
        )


# ===========================================================================
# EncoderStreaming: 顶层封装，接受 flat 列表 states（含 float32 cached_len）
# ===========================================================================
class EncoderStreaming(nn.Module):
    """
    Zipformer Encoder Streaming 封装。

    接口（flat tensors，方便 TorchScript trace）:
    输入 (1 + 35 = 36 个):
        x:           [1, 103, 80]  float32
        (x_lens 已移除 — SEGMENT=103 固定，无需传入)
        cached_len_0..4:   [2, 1]                 float32
        cached_avg_0..4:   [2, 1, 256]            float32
        cached_key_0..4:   [2, left_ctx//ds, 1, 192] float32
        cached_val_0..4:   [2, left_ctx//ds, 1, 96]  float32
        cached_val2_0..4:  [2, left_ctx//ds, 1, 96]  float32
        cached_conv1_0..4: [2, 1, 256, 30]        float32
        cached_conv2_0..4: [2, 1, 256, 30]        float32

    输出 (1 + 35 = 36 个):
        encoder_out: [1, 24, 256]  float32
        (out_lens 已移除 — 始终为 24)
        + 35 个更新后的 states（same shapes）
    """

    # 模型固定参数
    SEGMENT              = 103
    DECODE_CHUNK_SIZE    = 32
    NUM_LEFT_CHUNKS      = 4
    LEFT_CONTEXT_LEN     = DECODE_CHUNK_SIZE * NUM_LEFT_CHUNKS  # 128
    DS_FACTORS           = (1, 2, 4, 8, 2)
    NUM_ENCODERS         = 5

    def __init__(self, orig_zipformer: Zipformer):
        super().__init__()

        self.encoder_embed    = orig_zipformer.encoder_embed
        self.downsample_output = orig_zipformer.downsample_output
        self.skip_layers      = orig_zipformer.skip_layers
        self.skip_modules     = orig_zipformer.skip_modules

        # 计算每个 stack 的 time1 和 kv_len
        # After Conv2dSubsampling: T_base = (SEGMENT - 7) // 2 = 48
        T_base = (self.SEGMENT - 7) // 2  # 48

        self.encoders = nn.ModuleList()
        for i, orig_enc in enumerate(orig_zipformer.encoders):
            ds = self.DS_FACTORS[i]
            # T after downsampling within this stack
            time1 = (T_base + ds - 1) // ds
            # kv_len = time1 + left_context_len // ds
            kv_len = time1 + self.LEFT_CONTEXT_LEN // ds

            if ds == 1:
                # First stack: no downsampling wrapper
                enc_streaming = ZipformerEncoderStreaming(orig_enc, time1, kv_len)
            else:
                enc_streaming = DownsampledZipformerEncoderStreaming(orig_enc, time1, kv_len)
            self.encoders.append(enc_streaming)

    def get_init_state(self) -> List[Tensor]:
        """返回初始 states（全 float32）"""
        cached_len   = []
        cached_avg   = []
        cached_key   = []
        cached_val   = []
        cached_val2  = []
        cached_conv1 = []
        cached_conv2 = []

        for i in range(self.NUM_ENCODERS):
            ds        = self.DS_FACTORS[i]
            num_layers = 2  # hardcoded for this model
            lc        = self.LEFT_CONTEXT_LEN // ds

            # cached_len: float32 (was int64)
            cached_len.append(
                torch.zeros(num_layers, 1, dtype=torch.float32)
            )
            cached_avg.append(torch.zeros(num_layers, 1, 256))
            cached_key.append(torch.zeros(num_layers, lc, 1, 192))
            cached_val.append(torch.zeros(num_layers, lc, 1, 96))
            cached_val2.append(torch.zeros(num_layers, lc, 1, 96))
            cached_conv1.append(torch.zeros(num_layers, 1, 256, 30))
            cached_conv2.append(torch.zeros(num_layers, 1, 256, 30))

        return (
            cached_len + cached_avg + cached_key + cached_val
            + cached_val2 + cached_conv1 + cached_conv2
        )

    def forward(
        self,
        x: Tensor,       # [1, 103, 80]
        # NOTE: x_lens is removed — SEGMENT=103 is fixed, out_lens is always 24.
        # 35 states: cached_len×5, cached_avg×5, cached_key×5,
        #            cached_val×5, cached_val2×5, cached_conv1×5, cached_conv2×5
        cached_len_0: Tensor, cached_len_1: Tensor, cached_len_2: Tensor,
        cached_len_3: Tensor, cached_len_4: Tensor,
        cached_avg_0: Tensor, cached_avg_1: Tensor, cached_avg_2: Tensor,
        cached_avg_3: Tensor, cached_avg_4: Tensor,
        cached_key_0: Tensor, cached_key_1: Tensor, cached_key_2: Tensor,
        cached_key_3: Tensor, cached_key_4: Tensor,
        cached_val_0: Tensor, cached_val_1: Tensor, cached_val_2: Tensor,
        cached_val_3: Tensor, cached_val_4: Tensor,
        cached_val2_0: Tensor, cached_val2_1: Tensor, cached_val2_2: Tensor,
        cached_val2_3: Tensor, cached_val2_4: Tensor,
        cached_conv1_0: Tensor, cached_conv1_1: Tensor, cached_conv1_2: Tensor,
        cached_conv1_3: Tensor, cached_conv1_4: Tensor,
        cached_conv2_0: Tensor, cached_conv2_1: Tensor, cached_conv2_2: Tensor,
        cached_conv2_3: Tensor, cached_conv2_4: Tensor,
    ) -> Tuple[
        Tensor,
        Tensor, Tensor, Tensor, Tensor, Tensor,
        Tensor, Tensor, Tensor, Tensor, Tensor,
        Tensor, Tensor, Tensor, Tensor, Tensor,
        Tensor, Tensor, Tensor, Tensor, Tensor,
        Tensor, Tensor, Tensor, Tensor, Tensor,
        Tensor, Tensor, Tensor, Tensor, Tensor,
        Tensor, Tensor, Tensor, Tensor, Tensor,
    ]:
        cached_len  = [cached_len_0,  cached_len_1,  cached_len_2,  cached_len_3,  cached_len_4]
        cached_avg  = [cached_avg_0,  cached_avg_1,  cached_avg_2,  cached_avg_3,  cached_avg_4]
        cached_key  = [cached_key_0,  cached_key_1,  cached_key_2,  cached_key_3,  cached_key_4]
        cached_val  = [cached_val_0,  cached_val_1,  cached_val_2,  cached_val_3,  cached_val_4]
        cached_val2 = [cached_val2_0, cached_val2_1, cached_val2_2, cached_val2_3, cached_val2_4]
        cached_conv1 = [cached_conv1_0, cached_conv1_1, cached_conv1_2, cached_conv1_3, cached_conv1_4]
        cached_conv2 = [cached_conv2_0, cached_conv2_1, cached_conv2_2, cached_conv2_3, cached_conv2_4]

        # --- Conv2d subsampling ---
        x = self.encoder_embed(x)
        x = x.permute(1, 0, 2)  # [N, T, C] → [T, N, C]
        # lengths is constant: (SEGMENT-7)//2 = 48, then (48+1)//2 = 24
        # Do NOT compute from x_lens to avoid int64 rshift operator

        # --- 5 encoder stacks ---
        outputs = []
        new_cached_len   = []
        new_cached_avg   = []
        new_cached_key   = []
        new_cached_val   = []
        new_cached_val2  = []
        new_cached_conv1 = []
        new_cached_conv2 = []

        for i, (module, skip_module) in enumerate(
            zip(self.encoders, self.skip_modules)
        ):
            k = self.skip_layers[i]
            if isinstance(k, int):
                x = skip_module(outputs[k], x)
            x, llen, lavg, lkey, lval, lval2, lconv1, lconv2 = module.streaming_forward(
                x,
                cached_len=cached_len[i],
                cached_avg=cached_avg[i],
                cached_key=cached_key[i],
                cached_val=cached_val[i],
                cached_val2=cached_val2[i],
                cached_conv1=cached_conv1[i],
                cached_conv2=cached_conv2[i],
            )
            outputs.append(x)
            new_cached_len.append(llen)
            new_cached_avg.append(lavg)
            new_cached_key.append(lkey)
            new_cached_val.append(lval)
            new_cached_val2.append(lval2)
            new_cached_conv1.append(lconv1)
            new_cached_conv2.append(lconv2)

        x = self.downsample_output(x)
        x = x.permute(1, 0, 2)  # [T, N, C] → [N, T, C]

        return (
            x,
            new_cached_len[0],   new_cached_len[1],   new_cached_len[2],   new_cached_len[3],   new_cached_len[4],
            new_cached_avg[0],   new_cached_avg[1],   new_cached_avg[2],   new_cached_avg[3],   new_cached_avg[4],
            new_cached_key[0],   new_cached_key[1],   new_cached_key[2],   new_cached_key[3],   new_cached_key[4],
            new_cached_val[0],   new_cached_val[1],   new_cached_val[2],   new_cached_val[3],   new_cached_val[4],
            new_cached_val2[0],  new_cached_val2[1],  new_cached_val2[2],  new_cached_val2[3],  new_cached_val2[4],
            new_cached_conv1[0], new_cached_conv1[1], new_cached_conv1[2], new_cached_conv1[3], new_cached_conv1[4],
            new_cached_conv2[0], new_cached_conv2[1], new_cached_conv2[2], new_cached_conv2[3], new_cached_conv2[4],
        )


# ===========================================================================
# DecoderNPU: 去掉 Embedding，接受 embedded float32 输入
# ===========================================================================
class DecoderNPU(nn.Module):
    """
    Decoder 的 NPU 部分（去掉 Embedding 层）。

    输入:  embedded [1, context_size, decoder_dim] = [1, 2, 512] float32
    输出:  decoder_out [1, 512] float32
    """

    def __init__(self, orig_decoder: Decoder):
        super().__init__()
        assert orig_decoder.context_size > 1, "context_size must be > 1"
        self.conv    = orig_decoder.conv
        self.context_size = orig_decoder.context_size

    def forward(self, embedded: Tensor) -> Tensor:
        """
        Args:
            embedded: [N, context_size, decoder_dim] float32
        Returns:
            [N, decoder_dim] float32
        """
        # embedded: [N, U, C] → permute to [N, C, U] for Conv1d
        x = embedded.permute(0, 2, 1)  # [N, C, context_size]
        # No padding needed (inference mode: input already has context_size frames)
        x = self.conv(x)               # [N, C, 1]
        x = x.permute(0, 2, 1)        # [N, 1, C]
        x = F.relu(x)
        x = x.squeeze(1)              # [N, C]
        return x


# ===========================================================================
# JoinerStreaming: 直接包装 Joiner
# ===========================================================================
class JoinerStreaming(nn.Module):
    """
    Joiner 的 Streaming 包装（无修改，所有算子均支持）。

    输入:  encoder_out [1, 256], decoder_out [1, 512]
    输出:  logits [1, 6254]
    """

    def __init__(self, orig_joiner: Joiner):
        super().__init__()
        self.encoder_proj  = orig_joiner.encoder_proj
        self.decoder_proj  = orig_joiner.decoder_proj
        self.output_linear = orig_joiner.output_linear

    def forward(self, encoder_out: Tensor, decoder_out: Tensor) -> Tensor:
        logit = self.encoder_proj(encoder_out) + self.decoder_proj(decoder_out)
        return self.output_linear(torch.tanh(logit))


# ===========================================================================
# 工具函数：加载 checkpoint 并构建 Streaming 模型
# ===========================================================================
def build_streaming_models(checkpoint_path: str):
    """
    从 checkpoint 加载权重，构建 EncoderStreaming, DecoderNPU, JoinerStreaming。

    Returns:
        encoder_streaming, decoder_npu, joiner_streaming
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt  = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt["model"]

    # 构建原始模型
    encoder = Zipformer(
        num_features=80,
        output_downsampling_factor=2,
        encoder_dims=(256, 256, 256, 256, 256),
        attention_dim=(192, 192, 192, 192, 192),
        encoder_unmasked_dims=(192, 192, 192, 192, 192),
        zipformer_downsampling_factors=(1, 2, 4, 8, 2),
        nhead=(4, 4, 4, 4, 4),
        feedforward_dim=(768, 768, 768, 768, 768),
        num_encoder_layers=(2, 2, 2, 2, 2),
        cnn_module_kernels=(31, 31, 31, 31, 31),
        pos_dim=4,
        num_left_chunks=4,
        short_chunk_size=50,
        decode_chunk_size=32,
    )
    decoder = Decoder(
        vocab_size=6254,
        decoder_dim=512,
        blank_id=0,
        context_size=2,
    )
    joiner = Joiner(
        encoder_dim=256,
        decoder_dim=512,
        joiner_dim=512,
        vocab_size=6254,
    )

    # 加载权重
    enc_state = {k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")}
    encoder.load_state_dict(enc_state, strict=False)

    dec_state = {k[len("decoder."):]: v for k, v in state.items() if k.startswith("decoder.")}
    decoder.load_state_dict(dec_state, strict=False)

    j_state = {k[len("joiner."):]: v for k, v in state.items() if k.startswith("joiner.")}
    joiner.load_state_dict(j_state, strict=False)

    encoder.eval()
    decoder.eval()
    joiner.eval()

    # 构建 Streaming 版本
    encoder_streaming  = EncoderStreaming(encoder).eval()
    decoder_npu  = DecoderNPU(decoder).eval()
    joiner_streaming   = JoinerStreaming(joiner).eval()

    print("  Streaming models built OK.")
    return encoder_streaming, decoder_npu, joiner_streaming
