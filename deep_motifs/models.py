from __future__ import annotations

import torch
import torch.nn as nn


class MetaMLP(nn.Module):
    """
    Projects the full meta feature vector through a 2-layer MLP into a
    single token of size token_dim.  Unlike GroupedTokenizer, this sees
    all features simultaneously, preserving pairwise interactions 鈥?the
    same interactions that make XGBoost strong on composite_table data.
    """
    def __init__(self, in_dim: int, token_dim: int, dropout: float) -> None:
        super().__init__()
        hidden = max(token_dim * 2, 64)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Returns [B, 1, token_dim] 鈥?one token representing all meta features
        return self.net(x).unsqueeze(1)


# ============================================================
# Grouped tokenizer (kept for STRING view)
# ============================================================

def build_group_sizes(total_dim: int, n_groups: int) -> list[int]:
    total_dim = int(max(total_dim, 1))
    n_groups  = int(max(min(n_groups, total_dim), 1))
    base, rem = divmod(total_dim, n_groups)
    out = [base + (1 if i < rem else 0) for i in range(n_groups)]
    return [v for v in out if v > 0]


class GroupedTokenizer(nn.Module):
    def __init__(self, in_dim: int, n_groups: int, token_dim: int, dropout: float):
        super().__init__()
        self.group_sizes = build_group_sizes(in_dim, n_groups)
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(g, token_dim),
                nn.LayerNorm(token_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for g in self.group_sizes
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chunks = torch.split(x, self.group_sizes, dim=1)
        return torch.stack(
            [proj(c) for proj, c in zip(self.projections, chunks)], dim=1
        )


# ============================================================
# BrainSpan temporal encoder
# ============================================================

class BrainSpanEncoder(nn.Module):
    """
    v2 upgrade: LSTM-augmented BrainSpan encoder.

    Pipeline:
      1. Reshape (B, 16*50) 鈫?(B*16, 1, 50)
      2. Per-region temporal Conv1d 鈫?pool 鈫?(B, 16, token_dim)
         Captures *within-region* developmental trajectories.
      3. Cross-region TransformerEncoderLayer over the 16 region tokens
         Captures *between-region* coordinated expression patterns 鈥?a signal
         that XGBoost cannot model because it treats all 800 BrainSpan columns
         as independent flat features.

    Output: (B, 16, token_dim)  鈥?same shape as before, drop-in replacement.
    """
    def __init__(
        self,
        total_bs_dim: int,
        n_regions: int,
        n_timepoints: int,
        token_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.n_regions    = n_regions
        self.n_timepoints = n_timepoints
        self.token_dim    = token_dim
        self.structured   = (total_bs_dim == n_regions * n_timepoints)

        if self.structured:
            # Step 1-2: per-region temporal encoding (unchanged from v4)
            self.temporal_conv = nn.Sequential(
                nn.Conv1d(1, token_dim // 2, kernel_size=5, padding=2),
                nn.GELU(),
                nn.Conv1d(token_dim // 2, token_dim, kernel_size=3, padding=1),
                nn.GELU(),
            )
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.norm = nn.LayerNorm(token_dim)

            # BiLSTM branch copied in spirit from lstm.py:
            # input sequence is timepoints, each step sees all regions.
            lstm_hidden = max(token_dim // 2, 16)
            self.bs_lstm = nn.LSTM(
                input_size=n_regions,
                hidden_size=lstm_hidden,
                num_layers=2,
                bidirectional=True,
                batch_first=True,
                dropout=dropout,
            )
            self.lstm_proj = nn.Sequential(
                nn.Linear(lstm_hidden * 2, token_dim),
                nn.LayerNorm(token_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.region_gate = nn.Sequential(
                nn.Linear(token_dim * 2, token_dim),
                nn.Sigmoid(),
            )
            self.region_fuse_norm = nn.LayerNorm(token_dim)

            # v4 simplification: remove the second, BrainSpan-only transformer.
            # Region tokens are passed to the single global multi-view transformer.
            self.region_attn = nn.Identity()
            self.drop = nn.Dropout(dropout)
        else:
            self.fallback = GroupedTokenizer(total_bs_dim, n_regions, token_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.structured:
            return self.fallback(x)
        B = x.shape[0]
        bs_region_major = x.contiguous().view(B, self.n_regions, self.n_timepoints)

        # Per-region temporal encoding keeps the original Deep-MOTIFs view.
        h_region = bs_region_major.view(B * self.n_regions, 1, self.n_timepoints)
        h_region = self.temporal_conv(h_region)
        h_region = self.pool(h_region).squeeze(-1).view(B, self.n_regions, self.token_dim)
        h_region = self.norm(h_region)

        # BiLSTM global developmental token, matching the time-major layout in lstm.py.
        bs_time_major = bs_region_major.transpose(1, 2)  # (B, timepoints, regions)
        _, (hn, _) = self.bs_lstm(bs_time_major)
        h_lstm = torch.cat([hn[-2], hn[-1]], dim=1)
        h_global = self.lstm_proj(h_lstm)  # (B, token_dim)

        # Gate global LSTM information into each region token.
        global_expanded = h_global.unsqueeze(1).expand(-1, self.n_regions, -1)
        gate = self.region_gate(torch.cat([h_region, global_expanded], dim=-1))
        h_region = self.region_fuse_norm(h_region + gate * global_expanded)

        # v4 keeps only light LSTM-conditioned region tokens here. Cross-view and
        # cross-region interactions are handled by the global transformer.
        h_region = self.drop(self.region_attn(h_region))
        return torch.cat([h_global.unsqueeze(1), h_region], dim=1)


class LSTMStyleSupervisedBranch(nn.Module):
    """LSTM-baseline style supervised branch sharing the same standardized views."""

    def __init__(
        self,
        meta_dim: int,
        bs_dim: int,
        str_dim: int,
        bs_n_regions: int,
        bs_n_timepoints: int,
        hidden_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.bs_dim = int(bs_dim)
        self.bs_n_regions = int(bs_n_regions)
        self.bs_n_timepoints = int(bs_n_timepoints)
        self.structured = self.bs_dim == self.bs_n_regions * self.bs_n_timepoints
        self.hidden_size = int(max(hidden_size, 16))

        if self.structured:
            self.bs_lstm = nn.LSTM(
                input_size=self.bs_n_regions,
                hidden_size=self.hidden_size,
                num_layers=2,
                bidirectional=True,
                batch_first=True,
                dropout=dropout,
            )
            bs_out_dim = self.hidden_size * 2
        else:
            bs_out_dim = self.hidden_size * 2
            self.bs_fallback = nn.Sequential(
                nn.Linear(max(self.bs_dim, 1), bs_out_dim),
                nn.LayerNorm(bs_out_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        meta_str_dim = int(meta_dim + str_dim)
        self.meta_str_mlp = nn.Sequential(
            nn.Linear(max(meta_str_dim, 1), 128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(bs_out_dim + 128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x_m: torch.Tensor, x_b: torch.Tensor, x_s: torch.Tensor) -> torch.Tensor:
        B = x_m.shape[0]
        if self.structured:
            bs_region_major = x_b.contiguous().view(B, self.bs_n_regions, self.bs_n_timepoints)
            bs_time_major = bs_region_major.transpose(1, 2)
            _, (hn, _) = self.bs_lstm(bs_time_major)
            bs_vec = torch.cat([hn[-2], hn[-1]], dim=1)
        else:
            if x_b.shape[1] == 0:
                x_b = x_b.new_zeros((B, 1))
            bs_vec = self.bs_fallback(x_b)

        meta_str = torch.cat([x_m, x_s], dim=1)
        if meta_str.shape[1] == 0:
            meta_str = meta_str.new_zeros((B, 1))
        meta_str_vec = self.meta_str_mlp(meta_str)
        return self.classifier(torch.cat([bs_vec, meta_str_vec], dim=1)).squeeze(1)


# ============================================================
# DeepMOTIFs (Deep Multi-Omics Transformer with Integrated Features and Scores)
# ============================================================

class DeepMOTIFs(nn.Module):
    """
    Changes vs previous version:
      FIX 1: meta_tok replaced by MetaMLP (single MLP, no group splitting)
      FIX 6: norm_first=False (restores PyTorch nested-tensor optimisation)
    """

    def __init__(
        self,
        meta_dim: int,
        bs_dim: int,
        str_dim: int,
        token_dim: int,
        bs_n_regions: int,
        bs_n_timepoints: int,
        str_token_count: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.token_dim = int(token_dim)

        # Full multi-view PU ranker.  The LSTM empirical prior guides PU weights
        # outside this encoder; raw BrainSpan/STRING views stay available here
        # so the PU branch can learn independent posterior ranking evidence.
        self.meta_tok = MetaMLP(meta_dim, self.token_dim, dropout)
        self.bs_tok   = BrainSpanEncoder(
            bs_dim, bs_n_regions, bs_n_timepoints, self.token_dim, dropout
        )
        self.str_tok  = GroupedTokenizer(str_dim, str_token_count, self.token_dim, dropout)
        self.sup_branch = LSTMStyleSupervisedBranch(
            meta_dim=meta_dim,
            bs_dim=bs_dim,
            str_dim=str_dim,
            bs_n_regions=bs_n_regions,
            bs_n_timepoints=bs_n_timepoints,
            hidden_size=max(self.token_dim, 64),
            dropout=dropout,
        )

        self.cls_token  = nn.Parameter(torch.zeros(1, 1, self.token_dim))
        self.type_embed = nn.Parameter(torch.zeros(1, 4, self.token_dim))

        # FIX 6: norm_first=False 鈥?restores nested-tensor speed optimisation
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.token_dim,
            nhead=int(max(n_heads, 1)),
            dim_feedforward=int(max(self.token_dim * 4, 64)),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(max(n_layers, 1)))
        self.norm     = nn.LayerNorm(self.token_dim)

        bottleneck = max(64, self.token_dim)
        self.shared_bottleneck = nn.Sequential(
            nn.Linear(self.token_dim, bottleneck),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pu_head   = nn.Linear(bottleneck, 1)
        self.rank_head = nn.Linear(bottleneck, 1)

        nn.init.normal_(self.cls_token,  std=0.02)
        nn.init.normal_(self.type_embed, std=0.02)

    def forward(
        self,
        x_m: torch.Tensor,
        x_b: torch.Tensor,
        x_s: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        B  = x_m.shape[0]
        tm = self.meta_tok(x_m)   # [B, 1, D]
        tb = self.bs_tok(x_b)     # [B, n_regions(+global), D]
        ts = self.str_tok(x_s)    # [B, str_token_count, D]
        cls = self.cls_token.expand(B, -1, -1)

        seq = torch.cat([cls, tm, tb, ts], dim=1)
        type_seq = torch.cat([
            self.type_embed[:, 0:1, :].expand(B, 1,           self.token_dim),
            self.type_embed[:, 1:2, :].expand(B, tm.shape[1], self.token_dim),
            self.type_embed[:, 2:3, :].expand(B, tb.shape[1], self.token_dim),
            self.type_embed[:, 3:4, :].expand(B, ts.shape[1], self.token_dim),
        ], dim=1)
        seq = seq + type_seq

        emb    = self.norm(self.encoder(seq)[:, 0, :])
        shared = self.shared_bottleneck(emb)
        return {
            "emb":        emb,
            "sup_logit":  self.sup_branch(x_m, x_b, x_s),
            "pu_logit":   self.pu_head(shared).squeeze(1),
            "rank_logit": self.rank_head(shared).squeeze(1),
        }
