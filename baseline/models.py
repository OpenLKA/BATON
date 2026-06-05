"""
models.py — GRU and TCN architectures for PassingCtrl baseline.

Key design choices:
- Gated multimodal fusion: each modality gets a learned gate (sigmoid)
  so the model can suppress noisy branches (e.g. video) when struct is strong.
- Video MLP projection: 1280 → hidden with ReLU, not raw linear compression.
- All branches project to the same hidden_dim for symmetric gating.
"""
import torch
import torch.nn as nn

from config import (
    GRU_HIDDEN, GRU_LAYERS_STRUCT, GRU_LAYERS_VIDEO, GRU_LAYERS_GPS,
    FUSION_DIM, DROPOUT, VIDEO_FEATURE_DIM, NUM_CLASSES_TASK1, GPS_COLS,
    STRUCT_SEQ_LEN, VIDEO_SEQ_LEN,
)


class GatedFusion(nn.Module):
    """Gated fusion: each branch gets a sigmoid gate, outputs are summed."""

    def __init__(self, hidden_dim, n_branches):
        super().__init__()
        self.gates = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(n_branches)
        ])
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, FUSION_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
        )

    def forward(self, branch_outputs):
        # branch_outputs: list of [B, H] tensors
        gated = [torch.sigmoid(g(h)) * h for g, h in zip(self.gates, branch_outputs)]
        fused = sum(gated)
        return self.proj(fused)


class ResidualGatedFusion(nn.Module):
    """Residual fusion: primary branch is backbone, auxiliary branches add gated residuals.

    Gates initialized near zero so model starts ≈ primary-only performance.
    Adding modalities can only help (or be neutral), architecturally.
    """

    def __init__(self, hidden_dim, n_aux_branches):
        super().__init__()
        self.n_aux = n_aux_branches
        self.gates = nn.ModuleList([
            nn.Linear(hidden_dim * 2, hidden_dim)
            for _ in range(n_aux_branches)
        ])
        # Init gates near zero: sigmoid(-3) ≈ 0.05
        for g in self.gates:
            nn.init.zeros_(g.weight)
            nn.init.constant_(g.bias, -3.0)

        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, FUSION_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
        )

    def forward(self, primary, aux_list):
        """primary: [B, H], aux_list: list of [B, H]"""
        fused = primary
        for gate, aux in zip(self.gates, aux_list):
            g = torch.sigmoid(gate(torch.cat([primary, aux], dim=1)))
            fused = fused + g * aux
        return self.proj(fused)


class GRUBackbone(nn.Module):
    """GRU-based baseline with gated multimodal fusion.

    When struct is present AND auxiliary modalities exist, uses ResidualGatedFusion:
    struct is the backbone, other modalities add gated residuals (init ≈ 0).
    Otherwise falls back to standard GatedFusion.
    """

    def __init__(self, struct_dim, use_gps=False, use_front_video=False,
                 use_cabin_video=False, task="task1", video_feature_dim=None,
                 video_dropout=None):
        super().__init__()
        self.struct_dim = struct_dim
        self.use_gps = use_gps
        self.use_front_video = use_front_video
        self.use_cabin_video = use_cabin_video
        self.task = task
        vf_dim = video_feature_dim or VIDEO_FEATURE_DIM
        vdrop = video_dropout if video_dropout is not None else DROPOUT

        # Structured branch (Veh + Int + Drv + IMU, no GPS)
        if struct_dim > 0:
            self.struct_norm = nn.LayerNorm(struct_dim)
            self.struct_gru = nn.GRU(
                input_size=struct_dim, hidden_size=GRU_HIDDEN,
                num_layers=GRU_LAYERS_STRUCT, batch_first=True,
                dropout=DROPOUT if GRU_LAYERS_STRUCT > 1 else 0,
            )

        # GPS branch — separate from struct, own gate in fusion
        if use_gps:
            gps_dim = len(GPS_COLS)
            self.gps_norm = nn.LayerNorm(gps_dim)
            self.gps_gru = nn.GRU(
                input_size=gps_dim, hidden_size=GRU_HIDDEN,
                num_layers=GRU_LAYERS_GPS, batch_first=True,
            )

        # Video branches — LayerNorm + MLP projection
        if use_front_video:
            self.fv_feat_norm = nn.LayerNorm(vf_dim)
            self.fv_proj = nn.Sequential(
                nn.Linear(vf_dim, GRU_HIDDEN),
                nn.ReLU(),
                nn.Dropout(vdrop),
            )
            self.fv_gru = nn.GRU(
                input_size=GRU_HIDDEN, hidden_size=GRU_HIDDEN,
                num_layers=GRU_LAYERS_VIDEO, batch_first=True,
            )

        if use_cabin_video:
            self.cv_feat_norm = nn.LayerNorm(vf_dim)
            self.cv_proj = nn.Sequential(
                nn.Linear(vf_dim, GRU_HIDDEN),
                nn.ReLU(),
                nn.Dropout(vdrop),
            )
            self.cv_gru = nn.GRU(
                input_size=GRU_HIDDEN, hidden_size=GRU_HIDDEN,
                num_layers=GRU_LAYERS_VIDEO, batch_first=True,
            )

        # Fusion strategy
        n_aux = use_gps + use_front_video + use_cabin_video
        self._use_residual_fusion = (struct_dim > 0 and n_aux > 0)

        if self._use_residual_fusion:
            self.fusion = ResidualGatedFusion(GRU_HIDDEN, n_aux)
        else:
            n_branches = ((1 if struct_dim > 0 else 0) + n_aux)
            self.fusion = GatedFusion(GRU_HIDDEN, n_branches)

        # Task head
        if task == "task1":
            self.head = nn.Linear(FUSION_DIM, NUM_CLASSES_TASK1)
        else:
            self.head = nn.Linear(FUSION_DIM, 1)

    def forward(self, struct=None, gps=None, front_video=None, cabin_video=None):
        struct_out = None
        aux_parts = []

        if self.struct_dim > 0 and struct is not None:
            x = self.struct_norm(struct)
            out, _ = self.struct_gru(x)
            struct_out = out[:, -1, :]

        if self.use_gps and gps is not None:
            g = self.gps_norm(gps)
            out, _ = self.gps_gru(g)
            aux_parts.append(out[:, -1, :])

        if self.use_front_video and front_video is not None:
            fv = self.fv_feat_norm(front_video)
            fv = self.fv_proj(fv)
            out, _ = self.fv_gru(fv)
            aux_parts.append(out[:, -1, :])

        if self.use_cabin_video and cabin_video is not None:
            cv = self.cv_feat_norm(cabin_video)
            cv = self.cv_proj(cv)
            out, _ = self.cv_gru(cv)
            aux_parts.append(out[:, -1, :])

        if self._use_residual_fusion:
            fused = self.fusion(struct_out, aux_parts)
        else:
            # Video-only or struct-only: use standard gated fusion
            parts = ([struct_out] if struct_out is not None else []) + aux_parts
            fused = self.fusion(parts)

        return self.head(fused)


# ═══════════════════════════════════════════════════════════
# TCN
# ═══════════════════════════════════════════════════════════

class _TCNBlock(nn.Module):
    """Single residual TCN block with dilated causal convolution."""

    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation  # causal padding
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size,
                               padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size,
                               padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        # x: [B, C, T]
        out = self.conv1(x)
        out = out[:, :, :x.size(2)]  # causal trim
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = out[:, :, :x.size(2)]  # causal trim
        out = self.relu(out)
        out = self.dropout(out)
        return self.relu(out + self.residual(x))


class TCNBackbone(nn.Module):
    """TCN-based baseline with gated fusion."""

    def __init__(self, struct_dim, use_gps=False, use_front_video=False,
                 use_cabin_video=False, task="task1", video_feature_dim=None,
                 video_dropout=None):
        super().__init__()
        self.struct_dim = struct_dim
        self.use_gps = use_gps
        self.use_front_video = use_front_video
        self.use_cabin_video = use_cabin_video
        self.task = task
        vf_dim = video_feature_dim or VIDEO_FEATURE_DIM
        vdrop = video_dropout if video_dropout is not None else DROPOUT

        if struct_dim > 0:
            self.struct_norm = nn.LayerNorm(struct_dim)
            channels = [GRU_HIDDEN // 2, GRU_HIDDEN, GRU_HIDDEN]
            dilations = [1, 2, 4]
            layers = []
            in_ch = struct_dim
            for ch, d in zip(channels, dilations):
                layers.append(_TCNBlock(in_ch, ch, kernel_size=3,
                                        dilation=d, dropout=DROPOUT))
                in_ch = ch
            self.tcn = nn.Sequential(*layers)

        # GPS branch
        if use_gps:
            gps_dim = len(GPS_COLS)
            self.gps_norm = nn.LayerNorm(gps_dim)
            self.gps_gru = nn.GRU(
                input_size=gps_dim, hidden_size=GRU_HIDDEN,
                num_layers=GRU_LAYERS_GPS, batch_first=True,
            )

        # Video branches
        if use_front_video:
            self.fv_feat_norm = nn.LayerNorm(vf_dim)
            self.fv_proj = nn.Sequential(
                nn.Linear(vf_dim, GRU_HIDDEN),
                nn.ReLU(),
                nn.Dropout(vdrop),
            )
            self.fv_gru = nn.GRU(GRU_HIDDEN, GRU_HIDDEN, 1, batch_first=True)

        if use_cabin_video:
            self.cv_feat_norm = nn.LayerNorm(vf_dim)
            self.cv_proj = nn.Sequential(
                nn.Linear(vf_dim, GRU_HIDDEN),
                nn.ReLU(),
                nn.Dropout(vdrop),
            )
            self.cv_gru = nn.GRU(GRU_HIDDEN, GRU_HIDDEN, 1, batch_first=True)

        n_aux = use_gps + use_front_video + use_cabin_video
        self._use_residual_fusion = (struct_dim > 0 and n_aux > 0)

        if self._use_residual_fusion:
            self.fusion = ResidualGatedFusion(GRU_HIDDEN, n_aux)
        else:
            n_branches = ((1 if struct_dim > 0 else 0) + n_aux)
            self.fusion = GatedFusion(GRU_HIDDEN, n_branches)

        if task == "task1":
            self.head = nn.Linear(FUSION_DIM, NUM_CLASSES_TASK1)
        else:
            self.head = nn.Linear(FUSION_DIM, 1)

    def forward(self, struct=None, gps=None, front_video=None, cabin_video=None):
        struct_out = None
        aux_parts = []

        if self.struct_dim > 0 and struct is not None:
            x = self.struct_norm(struct)
            x = x.permute(0, 2, 1)  # [B, C, T]
            x = self.tcn(x)
            struct_out = x.mean(dim=2)  # global average pooling

        if self.use_gps and gps is not None:
            g = self.gps_norm(gps)
            out, _ = self.gps_gru(g)
            aux_parts.append(out[:, -1, :])

        if self.use_front_video and front_video is not None:
            fv = self.fv_feat_norm(front_video)
            fv = self.fv_proj(fv)
            out, _ = self.fv_gru(fv)
            aux_parts.append(out[:, -1, :])

        if self.use_cabin_video and cabin_video is not None:
            cv = self.cv_feat_norm(cabin_video)
            cv = self.cv_proj(cv)
            out, _ = self.cv_gru(cv)
            aux_parts.append(out[:, -1, :])

        if self._use_residual_fusion:
            fused = self.fusion(struct_out, aux_parts)
        else:
            parts = ([struct_out] if struct_out is not None else []) + aux_parts
            fused = self.fusion(parts)

        return self.head(fused)


# ═══════════════════════════════════════════════════════════
# CROSS-MODAL TRANSFORMER (stronger multimodal fusion baseline)
# ═══════════════════════════════════════════════════════════

# Transformer fusion hyperparameters
TF_DMODEL = GRU_HIDDEN          # 256
TF_LAYERS = 4
TF_HEADS = 8
TF_STRUCT_PATCH = 10            # strided-conv tokenization: 250 -> 25 tokens
# modality-type ids for the type embedding
_MOD_IDS = {"cls": 0, "struct": 1, "gps": 2, "front": 3, "cabin": 4}


class CrossModalTransformer(nn.Module):
    """Token-based cross-modal Transformer fusion.

    Each modality is tokenized into a short sequence, tagged with a learned
    modality-type embedding + a learned positional embedding, concatenated with a
    [CLS] token, and processed by N pre-norm Transformer encoder layers whose
    self-attention is cross-modal over the joint token set. The [CLS] state feeds
    the task head. Same __init__/forward contract as GRUBackbone/TCNBackbone.
    """

    def __init__(self, struct_dim, use_gps=False, use_front_video=False,
                 use_cabin_video=False, task="task1", video_feature_dim=None,
                 video_dropout=None):
        super().__init__()
        self.struct_dim = struct_dim
        self.use_gps = use_gps
        self.use_front_video = use_front_video
        self.use_cabin_video = use_cabin_video
        self.task = task
        d = TF_DMODEL
        vf_dim = video_feature_dim or VIDEO_FEATURE_DIM
        vdrop = video_dropout if video_dropout is not None else DROPOUT

        # Per-modality tokenizers → [B, n_tokens, d]
        if struct_dim > 0:
            self.struct_norm = nn.LayerNorm(struct_dim)
            # strided conv temporally downsamples 250 -> ~25 tokens
            self.struct_tok = nn.Conv1d(struct_dim, d, kernel_size=TF_STRUCT_PATCH,
                                        stride=TF_STRUCT_PATCH)
            self.n_struct_tok = STRUCT_SEQ_LEN // TF_STRUCT_PATCH
        if use_gps:
            self.gps_norm = nn.LayerNorm(len(GPS_COLS))
            self.gps_tok = nn.Conv1d(len(GPS_COLS), d, kernel_size=TF_STRUCT_PATCH,
                                     stride=TF_STRUCT_PATCH)
            self.n_gps_tok = STRUCT_SEQ_LEN // TF_STRUCT_PATCH
        if use_front_video:
            self.fv_norm = nn.LayerNorm(vf_dim)
            self.fv_tok = nn.Linear(vf_dim, d)
        if use_cabin_video:
            self.cv_norm = nn.LayerNorm(vf_dim)
            self.cv_tok = nn.Linear(vf_dim, d)

        # [CLS], modality-type embedding, learned positional embedding
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.type_emb = nn.Embedding(len(_MOD_IDS), d)
        max_pos = max(STRUCT_SEQ_LEN // TF_STRUCT_PATCH, VIDEO_SEQ_LEN)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_pos, d))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        self.in_drop = nn.Dropout(vdrop)

        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=TF_HEADS, dim_feedforward=4 * d,
            dropout=DROPOUT, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=TF_LAYERS)
        self.out_norm = nn.LayerNorm(d)

        if task == "task1":
            self.head = nn.Linear(d, NUM_CLASSES_TASK1)
        else:
            self.head = nn.Linear(d, 1)

    def _tag(self, tokens, mod_name):
        """Add modality-type + positional embedding to a [B, n, d] token block."""
        n = tokens.size(1)
        t = self.type_emb(torch.tensor(_MOD_IDS[mod_name], device=tokens.device))
        return tokens + t.view(1, 1, -1) + self.pos_emb[:, :n, :]

    def forward(self, struct=None, gps=None, front_video=None, cabin_video=None):
        B = None
        toks = []

        if self.struct_dim > 0 and struct is not None:
            B = struct.size(0)
            x = self.struct_norm(struct).permute(0, 2, 1)      # [B, C, T]
            x = self.struct_tok(x).permute(0, 2, 1)            # [B, n, d]
            toks.append(self._tag(x, "struct"))
        if self.use_gps and gps is not None:
            B = gps.size(0)
            g = self.gps_norm(gps).permute(0, 2, 1)
            g = self.gps_tok(g).permute(0, 2, 1)
            toks.append(self._tag(g, "gps"))
        if self.use_front_video and front_video is not None:
            B = front_video.size(0)
            toks.append(self._tag(self.fv_tok(self.fv_norm(front_video)), "front"))
        if self.use_cabin_video and cabin_video is not None:
            B = cabin_video.size(0)
            toks.append(self._tag(self.cv_tok(self.cv_norm(cabin_video)), "cabin"))

        cls = self.cls.expand(B, -1, -1) + self.type_emb(
            torch.tensor(_MOD_IDS["cls"], device=self.cls.device)).view(1, 1, -1)
        seq = torch.cat([cls] + toks, dim=1)                   # [B, 1+sum, d]
        seq = self.in_drop(seq)
        seq = self.encoder(seq)
        return self.head(self.out_norm(seq[:, 0]))             # [CLS] -> logits


# ═══════════════════════════════════════════════════════════
# RG-HBT-Q: Reliability-Gated Hierarchical Bottleneck Transformer
#           with Transition Queries
# ═══════════════════════════════════════════════════════════
#
# References: MBT — Attention Bottlenecks for Multimodal Fusion (Nagrani et al.,
# NeurIPS 2021); enhanced with a per-branch reliability gate (pooled / masked
# bottleneck aggregation) and a small set of task-specific transition queries.
#
# Pipeline:
#   A) per-modality temporal encoders
#        - each leak-safe CAN category: Conv1d patch (250->25) + shared encoder,
#          attention-pooled to 2 summary tokens (keeps short pre-transition cues)
#        - front / cabin video: Linear proj + [CLS] + temporal encoder
#   B) intra-CAN fusion: 6x2 category tokens + [CLS_CAN] -> cross-category encoder
#   C) reliability-gated bottleneck fusion across {can, front, cabin}: each branch
#      attends to itself + shared bottleneck; bottleneck = sum_m r_m B_m / sum_m r_m
#      with r_m = sigmoid(MLP(pool(tokens_m))) * valid_m  (valid_m=0 if branch zero)
#   D) transition queries cross-attend the fused tokens -> pooled -> head.
#
# Missing-modality robustness: per-sample modality dropout during training
# (front/cabin p=0.2, CAN p=0.1, >=1 branch kept); the reliability gate + a
# decoder memory padding-mask make a missing branch contribute nothing.

HBT_DMODEL = GRU_HIDDEN          # 256
HBT_HEADS = 8
HBT_FFN = 512                    # explicit (NOT the 2048 default) to bound params
HBT_DROPOUT = 0.1
HBT_PATCH = TF_STRUCT_PATCH      # 10  -> 250/10 = 25 tokens per CAN category
HBT_CAT_LAYERS = 1
HBT_CAT_SUMMARY_TOKENS = 2
HBT_STAT_TOKENS = 2              # tabular (per-channel mean/std/min/max/last) tokens
HBT_CAN_FUSION_LAYERS = 1
HBT_VIDEO_LAYERS = 1
HBT_FUSION_LAYERS = 2
HBT_BOTTLENECK = 4
HBT_QUERIES = 4
HBT_DEC_LAYERS = 1
HBT_MODROP = {"front": 0.2, "cabin": 0.2, "can": 0.1}
_BRANCH_IDS = {"can": 0, "front": 1, "cabin": 2}


def _hbt_layer(d=HBT_DMODEL, heads=HBT_HEADS, ffn=HBT_FFN, drop=HBT_DROPOUT):
    return nn.TransformerEncoderLayer(
        d_model=d, nhead=heads, dim_feedforward=ffn, dropout=drop,
        activation="gelu", batch_first=True, norm_first=True,
    )


class _AttnPool(nn.Module):
    """Pool a token sequence into n_queries summary tokens via cross-attention."""

    def __init__(self, d, n_queries, heads=HBT_HEADS, drop=HBT_DROPOUT):
        super().__init__()
        self.q = nn.Parameter(torch.zeros(1, n_queries, d))
        nn.init.trunc_normal_(self.q, std=0.02)
        self.attn = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):  # x: [B, n, d] -> [B, n_queries, d]
        q = self.q.expand(x.size(0), -1, -1)
        out, _ = self.attn(q, x, x, need_weights=False)
        return self.norm(out)


class RGHBTQ(nn.Module):
    """Reliability-Gated Hierarchical Bottleneck Transformer with Transition Queries.

    Same __init__/forward contract as the other backbones, with two extra optional
    kwargs supplied by the training script:
      struct_group_dims  : list[int] channel count of each CAN category (sums to struct_dim)
      struct_group_names : list[str] category names (for logging only)
    forward() additionally accepts apply_modality_dropout / return_repr (training controls).
    """

    def __init__(self, struct_dim, use_gps=False, use_front_video=False,
                 use_cabin_video=False, task="task1", video_feature_dim=None,
                 video_dropout=None, struct_group_dims=None, struct_group_names=None):
        super().__init__()
        d = HBT_DMODEL
        self.task = task
        self.struct_dim = struct_dim
        self.use_front_video = use_front_video
        self.use_cabin_video = use_cabin_video
        vf_dim = video_feature_dim or VIDEO_FEATURE_DIM

        if struct_group_dims is None:               # fallback: whole struct = 1 category
            struct_group_dims = [struct_dim]
            struct_group_names = ["all"]
        assert sum(struct_group_dims) == struct_dim, \
            f"group dims {struct_group_dims} != struct_dim {struct_dim}"
        self.group_dims = list(struct_group_dims)
        self.group_names = list(struct_group_names) if struct_group_names else \
            [f"can{i}" for i in range(len(self.group_dims))]
        self.n_cat = len(self.group_dims)
        self.n_patch = STRUCT_SEQ_LEN // HBT_PATCH  # 25

        # ----- Stage A: per-CAN-category encoders (tokenizer per category; encoder
        #       + pooler SHARED across categories, category identity via type emb) -----
        self.cat_norm = nn.ModuleList([nn.LayerNorm(dc) for dc in self.group_dims])
        self.cat_tok = nn.ModuleList([
            nn.Conv1d(dc, d, kernel_size=HBT_PATCH, stride=HBT_PATCH)
            for dc in self.group_dims
        ])
        self.cat_pos = nn.Parameter(torch.zeros(1, self.n_patch, d))
        nn.init.trunc_normal_(self.cat_pos, std=0.02)
        self.cat_type = nn.Embedding(self.n_cat, d)
        self.cat_enc = nn.TransformerEncoder(_hbt_layer(), num_layers=HBT_CAT_LAYERS)
        self.cat_pool = _AttnPool(d, HBT_CAT_SUMMARY_TOKENS)

        # ----- tabular (statistical) branch: per-channel mean/std/min/max/last of
        #       the leak-safe controller signals — the same summary features that
        #       gradient-boosted trees exploit, so the model is not weaker than
        #       XGBoost on the structured cues (e.g. max brake, last steering torque).
        self.stat_dim = struct_dim * 5
        self.stat_norm = nn.LayerNorm(self.stat_dim)
        self.stat_mlp = nn.Sequential(
            nn.Linear(self.stat_dim, d), nn.GELU(), nn.Dropout(HBT_DROPOUT),
            nn.Linear(d, HBT_STAT_TOKENS * d),
        )
        self.stat_type = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.stat_type, std=0.02)

        # ----- Stage B: intra-CAN fusion (category tokens + tabular tokens + CLS) -----
        n_can_tok = self.n_cat * HBT_CAT_SUMMARY_TOKENS + HBT_STAT_TOKENS + 1
        self.can_cls = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.can_cls, std=0.02)
        self.can_pos = nn.Parameter(torch.zeros(1, n_can_tok, d))
        nn.init.trunc_normal_(self.can_pos, std=0.02)
        self.can_fusion = nn.TransformerEncoder(_hbt_layer(),
                                                num_layers=HBT_CAN_FUSION_LAYERS)

        # ----- video temporal encoders -----
        if use_front_video:
            self.fv_norm = nn.LayerNorm(vf_dim)
            self.fv_proj = nn.Linear(vf_dim, d)
            self.fv_cls = nn.Parameter(torch.zeros(1, 1, d))
            nn.init.trunc_normal_(self.fv_cls, std=0.02)
            self.fv_pos = nn.Parameter(torch.zeros(1, VIDEO_SEQ_LEN + 1, d))
            nn.init.trunc_normal_(self.fv_pos, std=0.02)
            self.fv_enc = nn.TransformerEncoder(_hbt_layer(), num_layers=HBT_VIDEO_LAYERS)
        if use_cabin_video:
            self.cv_norm = nn.LayerNorm(vf_dim)
            self.cv_proj = nn.Linear(vf_dim, d)
            self.cv_cls = nn.Parameter(torch.zeros(1, 1, d))
            nn.init.trunc_normal_(self.cv_cls, std=0.02)
            self.cv_pos = nn.Parameter(torch.zeros(1, VIDEO_SEQ_LEN + 1, d))
            nn.init.trunc_normal_(self.cv_pos, std=0.02)
            self.cv_enc = nn.TransformerEncoder(_hbt_layer(), num_layers=HBT_VIDEO_LAYERS)

        self.branch_names = ["can"] + (["front"] if use_front_video else []) \
            + (["cabin"] if use_cabin_video else [])
        self.branch_type = nn.Embedding(len(_BRANCH_IDS), d)

        # ----- Stage C: reliability-gated bottleneck fusion -----
        self.bottleneck = nn.Parameter(torch.zeros(1, HBT_BOTTLENECK, d))
        nn.init.trunc_normal_(self.bottleneck, std=0.02)
        self.fusion_layers = nn.ModuleList([
            nn.ModuleDict({m: _hbt_layer() for m in self.branch_names})
            for _ in range(HBT_FUSION_LAYERS)
        ])
        self.gate_mlp = nn.ModuleDict({
            m: nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
            for m in self.branch_names
        })

        # ----- Stage D: transition-query decoder + head -----
        self.queries = nn.Parameter(torch.zeros(1, HBT_QUERIES, d))
        nn.init.trunc_normal_(self.queries, std=0.02)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d, nhead=HBT_HEADS, dim_feedforward=HBT_FFN, dropout=HBT_DROPOUT,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=HBT_DEC_LAYERS)
        self.in_drop = nn.Dropout(HBT_DROPOUT)
        self.out_norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, NUM_CLASSES_TASK1 if task == "task1" else 1)

        # gate statistics from the most recent forward (filled in eval)
        self.last_gate_mean = {}
        self.last_gate_std = {}

    # ----- branch encoders -----
    def _stat_tokens(self, struct):
        """Per-channel summary statistics -> HBT_STAT_TOKENS tokens (tabular cues)."""
        B = struct.size(0)
        feats = torch.cat([struct.mean(1), struct.std(1), struct.amin(1),
                           struct.amax(1), struct[:, -1, :]], dim=-1)   # [B, 5C]
        tok = self.stat_mlp(self.stat_norm(feats)).view(B, HBT_STAT_TOKENS, -1)
        return tok + self.stat_type

    def _encode_can(self, struct):
        B, device = struct.size(0), struct.device
        offset = 0
        cat_tokens = []
        for i in range(self.n_cat):
            dc = self.group_dims[i]
            xc = struct[:, :, offset:offset + dc]               # [B, 250, dc]
            offset += dc
            xc = self.cat_norm[i](xc).permute(0, 2, 1)          # [B, dc, 250]
            xc = self.cat_tok[i](xc).permute(0, 2, 1)           # [B, 25, d]
            xc = xc + self.cat_pos + \
                self.cat_type(torch.tensor(i, device=device)).view(1, 1, -1)
            xc = self.cat_enc(xc)                               # [B, 25, d]
            cat_tokens.append(self.cat_pool(xc))                # [B, 2, d]
        can = torch.cat(cat_tokens, dim=1)                      # [B, 12, d]
        stat = self._stat_tokens(struct)                       # [B, 2, d]
        can = torch.cat([self.can_cls.expand(B, -1, -1), can, stat], dim=1)  # [B, 15, d]
        can = can + self.can_pos
        can = self.can_fusion(can)
        return can + self.branch_type(
            torch.tensor(_BRANCH_IDS["can"], device=device)).view(1, 1, -1)

    def _encode_video(self, x, which):
        B, device = x.size(0), x.device
        if which == "front":
            h = self.fv_proj(self.fv_norm(x)); cls, pos, enc = self.fv_cls, self.fv_pos, self.fv_enc
        else:
            h = self.cv_proj(self.cv_norm(x)); cls, pos, enc = self.cv_cls, self.cv_pos, self.cv_enc
        h = torch.cat([cls.expand(B, -1, -1), h], dim=1)        # [B, 11, d]
        h = enc(h + pos[:, :h.size(1), :])
        return h + self.branch_type(
            torch.tensor(_BRANCH_IDS[which], device=device)).view(1, 1, -1)

    def forward(self, struct=None, gps=None, front_video=None, cabin_video=None,
                apply_modality_dropout=True, return_repr=False):
        B, device = struct.size(0), struct.device
        inputs = {"can": struct}
        if self.use_front_video:
            inputs["front"] = front_video if front_video is not None \
                else torch.zeros(B, VIDEO_SEQ_LEN, self.fv_norm.normalized_shape[0], device=device)
        if self.use_cabin_video:
            inputs["cabin"] = cabin_video if cabin_video is not None \
                else torch.zeros(B, VIDEO_SEQ_LEN, self.cv_norm.normalized_shape[0], device=device)

        # ----- per-sample modality dropout (training only) -----
        if self.training and apply_modality_dropout:
            keep = {m: (torch.rand(B, device=device) > HBT_MODROP[m]).float()
                    for m in self.branch_names}
            none_kept = torch.stack([keep[m] for m in self.branch_names], 1).sum(1) == 0
            keep["can"] = torch.where(none_kept, torch.ones_like(keep["can"]), keep["can"])
            inputs = {m: x * keep[m].view(B, 1, 1) for m, x in inputs.items()}

        # validity from the (possibly zeroed) inputs: 0 -> branch absent
        valid = {m: (inputs[m].abs().sum(dim=(1, 2)) > 0).float() for m in self.branch_names}

        # ----- encode branches -----
        tokens = {"can": self._encode_can(inputs["can"])}
        if self.use_front_video:
            tokens["front"] = self._encode_video(inputs["front"], "front")
        if self.use_cabin_video:
            tokens["cabin"] = self._encode_video(inputs["cabin"], "cabin")
        tokens = {m: self.in_drop(t) for m, t in tokens.items()}

        # ----- Stage C: reliability-gated bottleneck fusion -----
        bott = self.bottleneck.expand(B, -1, -1)
        gates = {}
        for layer in self.fusion_layers:
            updates = {}
            for m in self.branch_names:
                n_m = tokens[m].size(1)
                fused = layer[m](torch.cat([tokens[m], bott], dim=1))
                tokens[m] = fused[:, :n_m]
                updates[m] = fused[:, n_m:]
                r = torch.sigmoid(self.gate_mlp[m](tokens[m].mean(dim=1))).squeeze(-1)
                gates[m] = r * valid[m]                          # [B]
            rsum = sum(gates[m] for m in self.branch_names).view(B, 1, 1) + 1e-6
            bott = sum(gates[m].view(B, 1, 1) * updates[m]
                       for m in self.branch_names) / rsum

        if not self.training:                                    # record gate stats
            self.last_gate_mean = {m: float(gates[m].mean()) for m in self.branch_names}
            self.last_gate_std = {m: float(gates[m].std()) for m in self.branch_names}

        # ----- Stage D: transition-query decoder -----
        memory = torch.cat([tokens[m] for m in self.branch_names] + [bott], dim=1)
        pad = [(valid[m] < 0.5).view(B, 1).expand(B, tokens[m].size(1))
               for m in self.branch_names]
        pad.append(torch.zeros(B, bott.size(1), dtype=torch.bool, device=device))
        key_padding_mask = torch.cat(pad, dim=1)                 # [B, T] True = ignore
        q = self.queries.expand(B, -1, -1)
        dec = self.decoder(q, memory, memory_key_padding_mask=key_padding_mask)
        z = self.out_norm(dec.mean(dim=1))
        logits = self.head(z)
        return (logits, z) if return_repr else logits


# ═══════════════════════════════════════════════════════════
# DI-RG-HBT-Q: Driver-Input-Guided multimodal fusion
# ═══════════════════════════════════════════════════════════
#
# Motivation: for takeover, the driver-override (driver-input) is the dominant,
# physically-causal signal (gradient-boosted trees exploit it via max/last over a
# sparse binary spike). Instead of burying driver-input as one of several CAN
# categories, we promote it to the fusion HUB: driver-input forms per-sample event
# queries that conditionally read the front video, cabin video, and the remaining
# CAN context. Front/cabin video thus EXPLAIN the driver-input cue rather than being
# fused on an equal footing. (DMS / driver-monitoring is dropped from the inputs.)
#
#   driver_input --(spike-preserving encoder)--> event queries Q
#        Q --cross-attn--> front video tokens   -> Q_front
#        Q --cross-attn--> cabin video tokens   -> Q_cabin
#        Q --cross-attn--> aux CAN (ego/lead/road/imu) -> Q_aux
#   fusion Transformer over [Q, Q_front, Q_cabin, Q_aux]
#   interaction head: [z_di, z_*, z_di*z_*, |z_di - z_*|] -> logit
#   + auxiliary front/cabin heads (training signal so video is actually used)

DI_DMODEL = GRU_HIDDEN          # 256
DI_HEADS = 8
DI_FFN = 512
DI_DROPOUT = 0.2                # higher reg to delay the fast overfitting (best epoch ~3)
DI_PATCH = 10                   # 250 -> 25 patches
DI_EVENT_TOKENS = 4
DI_AUX_TOKENS = 2
DI_VIDEO_LAYERS = 1
DI_FUSION_LAYERS = 2
DI_NAME = "Veh_drvinput"        # which struct group is the driver-input hub
DI_DCN_DIM = 128                # DCNv2 tabular-cross head width
DI_DCN_CROSS = 3                # number of cross layers


class _CrossNetV2(nn.Module):
    """DCNv2 cross network: x_{l+1} = x0 * (W_l x_l + b_l) + x_l.
    Explicitly models bounded-degree feature crosses (axis-aligned interactions),
    the inductive bias gradient-boosted trees exploit."""

    def __init__(self, dim, n_layers):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_layers)])

    def forward(self, x0):
        x = x0
        for lin in self.layers:
            x = x0 * lin(x) + x
        return x


class _DCNTabHead(nn.Module):
    """DCNv2 (parallel cross + deep) tabular head -> logit, bypassing attention."""

    def __init__(self, in_dim, dim=DI_DCN_DIM, n_cross=DI_DCN_CROSS, out_dim=1,
                 drop=DI_DROPOUT):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, dim)
        self.cross = _CrossNetV2(dim, n_cross)
        self.deep = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Dropout(drop),
                                  nn.Linear(dim, dim), nn.GELU())
        self.out = nn.Linear(dim * 2, out_dim)

    def forward(self, feats):
        x = self.proj(self.norm(feats))
        return self.out(torch.cat([self.cross(x), self.deep(x)], dim=-1))


class _CrossAttn(nn.Module):
    """Pre-norm cross-attention block: queries q attend to key/value kv, + FFN."""

    def __init__(self, d=DI_DMODEL, heads=DI_HEADS, ffn=DI_FFN, drop=DI_DROPOUT):
        super().__init__()
        self.nq = nn.LayerNorm(d)
        self.nk = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.nf = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Dropout(drop),
                                nn.Linear(ffn, d))

    def forward(self, q, kv):
        a, _ = self.attn(self.nq(q), self.nk(kv), self.nk(kv), need_weights=False)
        q = q + a
        return q + self.ff(self.nf(q))


class DIRGHBTQ(nn.Module):
    """Driver-Input-Guided RG-HBT-Q (v1). Same construction contract as RGHBTQ
    (struct_group_dims / struct_group_names supplied by the trainer). forward()
    accepts return_aux to also return the front/cabin auxiliary logits for the
    video auxiliary loss."""

    def __init__(self, struct_dim, use_gps=False, use_front_video=False,
                 use_cabin_video=False, task="task1", video_feature_dim=None,
                 video_dropout=None, struct_group_dims=None, struct_group_names=None):
        super().__init__()
        d = DI_DMODEL
        self.task = task
        self.struct_dim = struct_dim
        self.use_front_video = use_front_video
        self.use_cabin_video = use_cabin_video
        vf_dim = video_feature_dim or VIDEO_FEATURE_DIM
        out_dim = NUM_CLASSES_TASK1 if task == "task1" else 1

        assert struct_group_dims is not None and struct_group_names is not None
        self.group_dims = list(struct_group_dims)
        self.group_names = list(struct_group_names)
        assert DI_NAME in self.group_names, f"{DI_NAME} must be a struct group"
        self.di_idx = self.group_names.index(DI_NAME)
        self.aux_idx = [i for i in range(len(self.group_dims)) if i != self.di_idx]
        # cumulative channel offsets to slice the concatenated struct tensor
        self.offsets = [0]
        for dc in self.group_dims:
            self.offsets.append(self.offsets[-1] + dc)
        self.n_patch = STRUCT_SEQ_LEN // DI_PATCH

        # ----- driver-input event encoder (spike-preserving) -----
        di_dim = self.group_dims[self.di_idx]
        self.di_proj = nn.Linear(di_dim * 4, d)        # per-patch mean/max/min/last
        self.di_pos = nn.Parameter(torch.zeros(1, self.n_patch, d))
        nn.init.trunc_normal_(self.di_pos, std=0.02)
        self.di_enc = nn.TransformerEncoder(_hbt_layer(d, DI_HEADS, DI_FFN, DI_DROPOUT), 1)
        self.di_pool = _AttnPool(d, DI_EVENT_TOKENS, DI_HEADS, DI_DROPOUT)

        # ----- auxiliary CAN encoders (ego / lead / road / imu) -----
        self.aux_norm = nn.ModuleList([nn.LayerNorm(self.group_dims[i]) for i in self.aux_idx])
        self.aux_tok = nn.ModuleList([
            nn.Conv1d(self.group_dims[i], d, kernel_size=DI_PATCH, stride=DI_PATCH)
            for i in self.aux_idx])
        self.aux_pos = nn.Parameter(torch.zeros(1, self.n_patch, d))
        nn.init.trunc_normal_(self.aux_pos, std=0.02)
        self.aux_type = nn.Embedding(len(self.aux_idx), d)
        self.aux_enc = nn.TransformerEncoder(_hbt_layer(d, DI_HEADS, DI_FFN, DI_DROPOUT), 1)
        self.aux_pool = _AttnPool(d, DI_AUX_TOKENS, DI_HEADS, DI_DROPOUT)

        # ----- video temporal encoders (frame + delta + late-pool tokens) -----
        def _video_mod():
            return nn.ModuleDict({
                "norm": nn.LayerNorm(vf_dim),
                "proj": nn.Linear(vf_dim, d),
                "enc": nn.TransformerEncoder(_hbt_layer(d, DI_HEADS, DI_FFN, DI_DROPOUT),
                                             DI_VIDEO_LAYERS),
                "head": nn.Linear(d, out_dim),
            })
        n_vtok = VIDEO_SEQ_LEN + (VIDEO_SEQ_LEN - 1) + 1   # frame + delta + late
        if use_front_video:
            self.fv = _video_mod()
            self.fv_pos = nn.Parameter(torch.zeros(1, n_vtok, d)); nn.init.trunc_normal_(self.fv_pos, std=0.02)
            self.x_front = _CrossAttn(d, DI_HEADS, DI_FFN, DI_DROPOUT)
        if use_cabin_video:
            self.cv = _video_mod()
            self.cv_pos = nn.Parameter(torch.zeros(1, n_vtok, d)); nn.init.trunc_normal_(self.cv_pos, std=0.02)
            self.x_cabin = _CrossAttn(d, DI_HEADS, DI_FFN, DI_DROPOUT)
        self.x_aux = _CrossAttn(d, DI_HEADS, DI_FFN, DI_DROPOUT)

        # ----- fusion + interaction head -----
        self.fusion = nn.TransformerEncoder(_hbt_layer(d, DI_HEADS, DI_FFN, DI_DROPOUT),
                                            DI_FUSION_LAYERS)
        uf, uc = int(use_front_video), int(use_cabin_video)
        n_vec = 3 + 3 * uf + 3 * uc            # base(2+uf+uc) + interaction(2uf+2uc+1)
        self.head = nn.Sequential(
            nn.Linear(n_vec * d, d), nn.GELU(), nn.Dropout(DI_DROPOUT), nn.Linear(d, out_dim))
        self.in_drop = nn.Dropout(DI_DROPOUT)

        # ----- DCNv2 tabular-cross head (wide path): rich window statistics over ALL
        #       leak-safe channels (+ driver-input last-0.5s/1s maxes) -> logit via a
        #       DCNv2 cross network, bypassing attention. Recovers the axis-aligned
        #       threshold + feature-cross power that gradient-boosted trees exploit;
        #       added as a residual to the multimodal fusion logit (wide & deep). -----
        di_dim = self.group_dims[self.di_idx]
        self.n_tab = 6 * struct_dim + 2 * di_dim   # mean/std/max/min/last/delta + 2 recent maxes
        self.tab_head = _DCNTabHead(self.n_tab, out_dim=out_dim)

    def _slice(self, struct, i):
        return struct[:, :, self.offsets[i]:self.offsets[i + 1]]

    def _tab_logit(self, struct):
        # window statistics over ALL channels + driver-input recent maxes
        di = self._slice(struct, self.di_idx)            # [B, 250, Cdi]
        feats = torch.cat([
            struct.mean(1), struct.std(1), struct.amax(1), struct.amin(1),
            struct[:, -1, :], struct[:, -1, :] - struct[:, 0, :],
            di[:, -25:, :].amax(1),                      # last 0.5 s max (25 @ 50Hz)
            di[:, -50:, :].amax(1),                      # last 1.0 s max
        ], dim=-1)
        return self.tab_head(feats)

    def _di_tokens(self, struct):
        x = self._slice(struct, self.di_idx)                 # [B, 250, Cdi]
        B, T, C = x.shape
        xp = x.view(B, self.n_patch, T // self.n_patch, C)
        feats = torch.cat([xp.mean(2), xp.amax(2), xp.amin(2), xp[:, :, -1, :]], dim=-1)
        h = self.di_proj(feats) + self.di_pos               # [B, 25, d]
        h = self.di_enc(h)
        return self.di_pool(h)                               # [B, K, d]

    def _aux_tokens(self, struct):
        toks = []
        for j, i in enumerate(self.aux_idx):
            xc = self.aux_norm[j](self._slice(struct, i)).permute(0, 2, 1)
            xc = self.aux_tok[j](xc).permute(0, 2, 1) + self.aux_pos + \
                self.aux_type(torch.tensor(j, device=struct.device)).view(1, 1, -1)
            xc = self.aux_enc(xc)
            toks.append(self.aux_pool(xc))                   # [B, 2, d]
        return torch.cat(toks, dim=1)

    def _video_tokens(self, x, mod, pos):
        f = mod["proj"](mod["norm"](x))                      # [B, 10, d]
        delta = f[:, 1:, :] - f[:, :-1, :]                   # [B, 9, d]
        late = f[:, -3:, :].mean(dim=1, keepdim=True)        # [B, 1, d]
        h = torch.cat([f, delta, late], dim=1) + pos
        return mod["enc"](h)                                 # [B, 20, d]

    def forward(self, struct=None, gps=None, front_video=None, cabin_video=None,
                return_aux=False):
        B = struct.size(0)
        q_di = self.in_drop(self._di_tokens(struct))         # [B, K, d]
        aux_tok = self._aux_tokens(struct)
        K = q_di.size(1)

        groups = [("di", q_di)]
        aux_logits = []
        if self.use_front_video:
            ft = self._video_tokens(front_video, self.fv, self.fv_pos)
            groups.append(("front", self.x_front(q_di, ft)))
            aux_logits.append(self.fv["head"](ft.mean(dim=1)))
        if self.use_cabin_video:
            ct = self._video_tokens(cabin_video, self.cv, self.cv_pos)
            groups.append(("cabin", self.x_cabin(q_di, ct)))
            aux_logits.append(self.cv["head"](ct.mean(dim=1)))
        groups.append(("aux", self.x_aux(q_di, aux_tok)))

        fused = self.fusion(torch.cat([g for _, g in groups], dim=1))  # [B, K*ngroups, d]
        pools = {name: fused[:, i * K:(i + 1) * K].mean(dim=1)
                 for i, (name, _) in enumerate(groups)}

        z_di = pools["di"]
        parts = [pools[n] for n, _ in groups]
        inter = []
        for n in ("front", "cabin"):
            if n in pools:
                inter += [z_di * pools[n], (z_di - pools[n]).abs()]
        inter += [z_di * pools["aux"]]
        logit = self.head(torch.cat(parts + inter, dim=-1)) + self._tab_logit(struct)
        return (logit, aux_logits) if return_aux else logit
