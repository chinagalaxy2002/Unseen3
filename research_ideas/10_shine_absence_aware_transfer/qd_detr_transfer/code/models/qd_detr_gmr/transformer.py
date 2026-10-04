"""
Transformer modules for QD-DETR.
"""
from __future__ import annotations

import copy
import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from models.qd_detr_gmr.attention import MultiheadAttention


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def inverse_sigmoid(x: Tensor, eps: float = 1e-3) -> Tensor:
    x = x.clamp(min=0.0, max=1.0)
    x1 = x.clamp(min=eps)
    x2 = (1.0 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def gen_sineembed_for_position(pos_tensor: Tensor, d_model: int = 256) -> Tensor:
    scale = 2 * math.pi
    dim_feats = d_model // 2
    dim_t = torch.arange(dim_feats, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000.0 ** (2 * (dim_t // 2) / dim_feats)

    center_embed = pos_tensor[:, :, 0] * scale
    pos_x = center_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)

    span_embed = pos_tensor[:, :, 1] * scale
    pos_w = span_embed[:, :, None] / dim_t
    pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

    pos = torch.cat((pos_x, pos_w), dim=2)
    return pos


class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_queries: int = 10,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: str = "prelu",
        normalize_before: bool = False,
        return_intermediate_dec: bool = True,
        query_dim: int = 2,
        keep_query_pos: bool = False,
        query_scale_type: str = "cond_elewise",
        num_patterns: int = 0,
        modulate_t_attn: bool = True,
        bbox_embed_diff_each_layer: bool = False,
    ):
        super().__init__()

        t2v_encoder_layer = T2V_TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.t2v_encoder = TransformerEncoder(t2v_encoder_layer, num_encoder_layers, encoder_norm)

        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before,
            keep_query_pos=keep_query_pos
        )
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
            d_model=d_model,
            query_dim=query_dim,
            keep_query_pos=keep_query_pos,
            query_scale_type=query_scale_type,
            modulate_t_attn=modulate_t_attn,
            bbox_embed_diff_each_layer=bbox_embed_diff_each_layer,
        )

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        self.dec_layers = num_decoder_layers
        self.num_queries = num_queries
        self.num_patterns = num_patterns

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: Tensor, mask: Tensor, query_embed: Tensor, pos_embed: Tensor, video_length: int):
        """
        Args:
            src: (batch_size, L, d)
            mask: (batch_size, L)
            query_embed: (#queries, 2)
            pos_embed: (batch_size, L, d)
            video_length: int
        """
        bs, l, d = src.shape
        src = src.permute(1, 0, 2)  # (L, bs, d)
        pos_embed = pos_embed.permute(1, 0, 2)  # (L, bs, d)
        refpoint_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)  # (#queries, bs, 2)

        src = self.t2v_encoder(src, src_key_padding_mask=mask, pos=pos_embed, video_length=video_length)
        src = src[:video_length + 1]
        mask = mask[:, :video_length + 1]
        pos_embed = pos_embed[:video_length + 1]

        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)  # (L, bs, d)
        memory_global, memory_local = memory[0], memory[1:]
        mask_local = mask[:, 1:]
        pos_embed_local = pos_embed[1:]

        tgt = torch.zeros(refpoint_embed.shape[0], bs, d, device=src.device)
        hs, references = self.decoder(
            tgt,
            memory_local,
            memory_key_padding_mask=mask_local,
            pos=pos_embed_local,
            refpoints_unsigmoid=refpoint_embed,
        )
        memory_local = memory_local.transpose(0, 1)  # (bs, L_vid, d)
        return hs, references, memory_local, memory_global


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, src: Tensor, mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None, **kwargs):
        output = src
        intermediate = []

        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos, **kwargs)
            if self.return_intermediate:
                intermediate.append(output)

        if self.norm is not None:
            output = self.norm(output)

        if self.return_intermediate:
            return torch.stack(intermediate)
        return output


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        decoder_layer,
        num_layers,
        norm=None,
        return_intermediate=False,
        d_model=256,
        query_dim=2,
        keep_query_pos=False,
        query_scale_type="cond_elewise",
        modulate_t_attn=False,
        bbox_embed_diff_each_layer=False,
    ):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        assert return_intermediate
        self.query_dim = query_dim

        assert query_scale_type in ["cond_elewise", "cond_scalar", "fix_elewise"]
        self.query_scale_type = query_scale_type
        if query_scale_type == "cond_elewise":
            self.query_scale = MLP(d_model, d_model, d_model, 2)
        elif query_scale_type == "cond_scalar":
            self.query_scale = MLP(d_model, d_model, 1, 2)
        elif query_scale_type == "fix_elewise":
            self.query_scale = nn.Embedding(num_layers, d_model)

        self.ref_point_head = MLP(d_model, d_model, d_model, 2)

        if bbox_embed_diff_each_layer:
            self.bbox_embed = nn.ModuleList([MLP(d_model, d_model, 2, 3) for _ in range(num_layers)])
            for bbox_embed in self.bbox_embed:
                nn.init.constant_(bbox_embed.layers[-1].weight.data, 0)
                nn.init.constant_(bbox_embed.layers[-1].bias.data, 0)
        else:
            self.bbox_embed = MLP(d_model, d_model, 2, 3)
            nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        self.d_model = d_model
        self.modulate_t_attn = modulate_t_attn
        self.bbox_embed_diff_each_layer = bbox_embed_diff_each_layer

        if modulate_t_attn:
            self.ref_anchor_head = MLP(d_model, d_model, 1, 2)

        if not keep_query_pos:
            for layer_id in range(num_layers - 1):
                self.layers[layer_id + 1].ca_qpos_proj = None

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        refpoints_unsigmoid: Optional[Tensor] = None,
    ):
        output = tgt
        intermediate = []
        reference_points = refpoints_unsigmoid.sigmoid()
        ref_points = [reference_points]

        for layer_id, layer in enumerate(self.layers):
            obj_center = reference_points[..., :self.query_dim]
            query_sine_embed = gen_sineembed_for_position(obj_center, d_model=self.d_model)
            query_pos = self.ref_point_head(query_sine_embed)

            if self.query_scale_type != "fix_elewise":
                if layer_id == 0:
                    pos_transformation = 1.0
                else:
                    pos_transformation = self.query_scale(output)
            else:
                pos_transformation = self.query_scale.weight[layer_id]

            query_sine_embed = query_sine_embed * pos_transformation

            if self.modulate_t_attn:
                reft_cond = self.ref_anchor_head(output).sigmoid()
                query_sine_embed = query_sine_embed * (reft_cond[..., 0] / obj_center[..., 1].clamp(min=1e-5)).unsqueeze(-1)

            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
                query_sine_embed=query_sine_embed,
                is_first=(layer_id == 0),
            )

            if self.bbox_embed is not None:
                if self.bbox_embed_diff_each_layer:
                    tmp = self.bbox_embed[layer_id](output)
                else:
                    tmp = self.bbox_embed(output)
                tmp[..., :self.query_dim] += inverse_sigmoid(reference_points)
                new_reference_points = tmp[..., :self.query_dim].sigmoid()
                if layer_id != self.num_layers - 1:
                    ref_points.append(new_reference_points)
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            if self.bbox_embed is not None:
                return [
                    torch.stack(intermediate).transpose(1, 2),
                    torch.stack(ref_points).transpose(1, 2),
                ]
            return [
                torch.stack(intermediate).transpose(1, 2),
                reference_points.unsqueeze(0).transpose(1, 2),
            ]
        return output.unsqueeze(0)


class T2V_TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "relu", normalize_before: bool = False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.nhead = nhead

    def with_pos_embed(self, tensor: Tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src: Tensor, src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None, video_length: Optional[int] = None):
        assert video_length is not None
        pos_src = self.with_pos_embed(src, pos)
        global_token = src[0].unsqueeze(0)
        q = pos_src[1:video_length + 1]
        k = pos_src[video_length + 1:]
        v = src[video_length + 1:]

        qmask = src_key_padding_mask[:, 1:video_length + 1].unsqueeze(2)
        kmask = src_key_padding_mask[:, video_length + 1:].unsqueeze(1)
        attn_mask = torch.matmul(qmask.float(), kmask.float()).bool().repeat(self.nhead, 1, 1)

        src2 = self.self_attn(
            q, k, value=v, attn_mask=attn_mask,
            key_padding_mask=src_key_padding_mask[:, video_length + 1:]
        )[0]
        src2 = src[1:video_length + 1] + self.dropout1(src2)
        src3 = self.norm1(src2)
        src3 = self.linear2(self.dropout(self.activation(self.linear1(src3))))
        src2 = src2 + self.dropout2(src3)
        src2 = self.norm2(src2)
        src2 = torch.cat([global_token, src2], dim=0)
        src = torch.cat([src2, src[video_length + 1:]], dim=0)
        return src

    def forward_pre(self, src: Tensor, src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None, video_length: Optional[int] = None):
        assert video_length is not None
        src2 = self.norm1(src)
        pos_src = self.with_pos_embed(src2, pos)
        global_token = src[0].unsqueeze(0)
        q = pos_src[1:video_length + 1]
        k = pos_src[video_length + 1:]
        v = src2[video_length + 1:]

        qmask = src_key_padding_mask[:, 1:video_length + 1].unsqueeze(2)
        kmask = src_key_padding_mask[:, video_length + 1:].unsqueeze(1)
        attn_mask = torch.matmul(qmask.float(), kmask.float()).bool().repeat(self.nhead, 1, 1)

        src2 = self.self_attn(
            q, k, value=v, attn_mask=attn_mask,
            key_padding_mask=src_key_padding_mask[:, video_length + 1:]
        )[0]
        src2 = src[1:video_length + 1] + self.dropout1(src2)
        src3 = self.norm1(src2)
        src3 = self.linear2(self.dropout(self.activation(self.linear1(src3))))
        src2 = src2 + self.dropout2(src3)
        src2 = self.norm2(src2)
        src2 = torch.cat([global_token, src2], dim=0)
        src = torch.cat([src2, src[video_length + 1:]], dim=0)
        return src

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None, **kwargs):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos, **kwargs)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos, **kwargs)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "relu", normalize_before: bool = False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor: Tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src: Tensor, src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None, **kwargs):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src: Tensor, src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None, **kwargs):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None, **kwargs):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos, **kwargs)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos, **kwargs)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "relu", normalize_before: bool = False, keep_query_pos: bool = False,
                 rm_self_attn_decoder: bool = False):
        super().__init__()
        if not rm_self_attn_decoder:
            self.sa_qcontent_proj = nn.Linear(d_model, d_model)
            self.sa_qpos_proj = nn.Linear(d_model, d_model)
            self.sa_kcontent_proj = nn.Linear(d_model, d_model)
            self.sa_kpos_proj = nn.Linear(d_model, d_model)
            self.sa_v_proj = nn.Linear(d_model, d_model)
            self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, vdim=d_model)

            self.norm1 = nn.LayerNorm(d_model)
            self.dropout1 = nn.Dropout(dropout)

        self.ca_qcontent_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_proj = nn.Linear(d_model, d_model)
        self.ca_kcontent_proj = nn.Linear(d_model, d_model)
        self.ca_kpos_proj = nn.Linear(d_model, d_model)
        self.ca_v_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_sine_proj = nn.Linear(d_model, d_model)
        self.cross_attn = MultiheadAttention(d_model * 2, nhead, dropout=dropout, vdim=d_model)

        self.nhead = nhead
        self.rm_self_attn_decoder = rm_self_attn_decoder

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.keep_query_pos = keep_query_pos

    def with_pos_embed(self, tensor: Tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        query_sine_embed: Optional[Tensor] = None,
        is_first: bool = False,
    ):
        if not self.rm_self_attn_decoder:
            q_content = self.sa_qcontent_proj(tgt)
            q_pos = self.sa_qpos_proj(query_pos)
            k_content = self.sa_kcontent_proj(tgt)
            k_pos = self.sa_kpos_proj(query_pos)
            v = self.sa_v_proj(tgt)

            q = q_content + q_pos
            k = k_content + k_pos

            tgt2 = self.self_attn(q, k, value=v, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask)[0]
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)

        q_content = self.ca_qcontent_proj(tgt)
        k_content = self.ca_kcontent_proj(memory)
        v = self.ca_v_proj(memory)

        num_queries, bs, n_model = q_content.shape
        hw, _, _ = k_content.shape

        k_pos = self.ca_kpos_proj(pos)

        if is_first or self.keep_query_pos:
            q_pos = self.ca_qpos_proj(query_pos)
            q = q_content + q_pos
            k = k_content + k_pos
        else:
            q = q_content
            k = k_content

        q = q.view(num_queries, bs, self.nhead, n_model // self.nhead)
        query_sine_embed = self.ca_qpos_sine_proj(query_sine_embed)
        query_sine_embed = query_sine_embed.view(num_queries, bs, self.nhead, n_model // self.nhead)
        q = torch.cat([q, query_sine_embed], dim=3).view(num_queries, bs, n_model * 2)

        k = k.view(hw, bs, self.nhead, n_model // self.nhead)
        k_pos = k_pos.view(hw, bs, self.nhead, n_model // self.nhead)
        k = torch.cat([k, k_pos], dim=3).view(hw, bs, n_model * 2)

        tgt2 = self.cross_attn(query=q, key=k, value=v, attn_mask=memory_mask,
                               key_padding_mask=memory_key_padding_mask)[0]

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def _get_activation_fn(activation: str):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "prelu":
        return nn.PReLU()
    if activation == "selu":
        return F.selu
    raise RuntimeError(f"activation should be relu/gelu/prelu, not {activation}.")


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=False,
        return_intermediate_dec=True,
        activation=getattr(args, "activation", "prelu"),
    )
