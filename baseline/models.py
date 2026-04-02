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
