# -*- coding: utf-8 -*-
"""
Train_ColdPU.py
"""

import argparse
import copy
import json
import os
import sys
import platform
import random
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv


device = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# Tee: dual stdout
# =========================================================
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, s):
        for f in self.files:
            try:
                f.write(s); f.flush()
            except Exception:
                pass
    def flush(self):
        for f in self.files:
            try: f.flush()
            except Exception: pass


# =========================================================
# Utils
# =========================================================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def generate_run_name(prefix: str = "single_omic_pu_gatv2_v2") -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def read_gene_ids(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def build_pu_masks(
    data: Data,
    positive_class: int = 1,
    train_ratio: float = 0.5,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Data:
    g = torch.Generator().manual_seed(seed)
    y = data.y
    pos_idx = (y == positive_class).nonzero(as_tuple=True)[0]
    n_pos = pos_idx.numel()
    if n_pos < 5:
        raise ValueError(f"Too few positives: {n_pos}")
    perm = pos_idx[torch.randperm(n_pos, generator=g)]
    n_train = max(1, int(n_pos * train_ratio))
    n_val = int(n_pos * val_ratio)
    if n_train + n_val >= n_pos:
        n_val = max(0, n_pos - n_train - 1)
    train_pos = perm[:n_train]
    val_pos = perm[n_train:n_train + n_val]
    test_pos = perm[n_train + n_val:]

    train_pos_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    val_pos_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_pos_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    train_pos_mask[train_pos] = True
    val_pos_mask[val_pos] = True
    test_pos_mask[test_pos] = True

    data.train_pos_mask = train_pos_mask
    data.val_pos_mask = val_pos_mask
    data.test_pos_mask = test_pos_mask
    data.unlabeled_mask = (y == 0)
    return data


# =========================================================
# Data loading
# =========================================================
def _load_edges(data_dir: Path, name_prefix: str) -> Tuple[np.ndarray, np.ndarray]:
    edge_path = data_dir / f"edges_{name_prefix}.npy"
    weight_path = data_dir / f"edges_{name_prefix}_weight.npy"
    if not edge_path.exists() or not weight_path.exists():
        return np.empty((2, 0), dtype=np.int64), np.empty((0,), dtype=np.float32)
    edge_index = np.load(edge_path).astype(np.int64)
    edge_weight = np.load(weight_path).astype(np.float32)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"{edge_path} should have shape [2, E]")
    if edge_weight.shape[0] != edge_index.shape[1]:
        raise ValueError(f"{weight_path} length mismatch with edges")
    return edge_index, edge_weight


def combine_edges(coexp_e, coexp_w, ppi_e, ppi_w,
                  coexp_ratio=0.5, ppi_ratio=0.5,
                  use_coexp=True, use_ppi=True):
    """Merge coexp / ppi edges into a single graph (for non-hetero mode)."""
    a, b = float(coexp_ratio) if use_coexp else 0.0, float(ppi_ratio) if use_ppi else 0.0
    s = a + b
    if s <= 0:
        raise ValueError("At least one graph must be enabled")
    a, b = a / s, b / s
    parts = []
    if use_coexp and coexp_e.shape[1] > 0:
        parts.append((coexp_e, coexp_w * a))
    if use_ppi and ppi_e.shape[1] > 0:
        parts.append((ppi_e, ppi_w * b))
    if len(parts) == 0:
        return np.empty((2, 0), dtype=np.int64), np.empty((0,), dtype=np.float32)
    edge_index = np.concatenate([p[0] for p in parts], axis=1)
    edge_weight = np.concatenate([p[1] for p in parts], axis=0)
    df = pd.DataFrame({"src": edge_index[0], "dst": edge_index[1], "weight": edge_weight})
    df = df.groupby(["src", "dst"], as_index=False)["weight"].sum()
    df["weight"] = df["weight"].clip(lower=0.0, upper=1.0)
    merged_e = df[["src", "dst"]].to_numpy(dtype=np.int64).T
    merged_w = df["weight"].to_numpy(dtype=np.float32)
    return merged_e, merged_w



def load_preprocessed_v4(data_dir: str):
    """
    Load preprocessed output:
      - X.npy / y.npy / gene_ids.txt
      - edges_coexp.npy / edges_coexp_weight.npy
      - edges_ppi.npy / edges_ppi_weight.npy
      - Optional cond_input.npy / other_input.npy
    """
    d = Path(data_dir)
    required = ["X.npy", "y.npy", "gene_ids.txt",
                "edges_coexp.npy", "edges_coexp_weight.npy",
                "edges_ppi.npy", "edges_ppi_weight.npy"]
    missing = [f for f in required if not (d / f).exists()]
    if missing:
        raise FileNotFoundError("Missing files:\n" + "\n".join(missing))

    X = np.load(d / "X.npy").astype(np.float32)
    y = np.load(d / "y.npy").astype(np.int64)
    nodes = read_gene_ids(d / "gene_ids.txt")

    fn_path = d / "feature_names.txt"
    feature_names = read_gene_ids(fn_path) if fn_path.exists() else [f"x_{i}" for i in range(X.shape[1])]

    coexp_e, coexp_w = _load_edges(d, "coexp")
    ppi_e, ppi_w = _load_edges(d, "ppi")

    cond_input = None
    other_input = None
    if (d / "cond_input.npy").exists():
        cond_input = np.load(d / "cond_input.npy").astype(np.float32)
    if (d / "other_input.npy").exists():
        other_input = np.load(d / "other_input.npy").astype(np.float32)

    print(f"[i] Loaded: X={X.shape}, y_pos={int((y==1).sum())}, "
          f"coexp_edges={coexp_e.shape[1]}, ppi_edges={ppi_e.shape[1]}")
    if cond_input is not None:
        print(f"[i] cond_input: {cond_input.shape}")
    if other_input is not None:
        print(f"[i] other_input: {other_input.shape}")

    return {
        "X": X, "y": y, "nodes": nodes, "feature_names": feature_names,
        "coexp_e": coexp_e, "coexp_w": coexp_w,
        "ppi_e": ppi_e, "ppi_w": ppi_w,
        "cond_input": cond_input, "other_input": other_input,
    }


def build_data_for_split(
    bundle: Dict,
    split_seed: int,
    train_ratio: float,
    val_ratio: float,
    use_coexp: bool,
    use_ppi: bool,
    coexp_ratio: float,
    ppi_ratio: float,
    use_hetero: bool,
) -> Data:
    """
    Build Data object for one split:
      1. Directly use saved PPI graph from preprocessing
      2. If use_hetero, store coexp and ppi separately in data.edge_index_coexp / data.edge_index_ppi
         else merge into single edge_index
    """
    X = bundle["X"]; y = bundle["y"]
    coexp_e, coexp_w = bundle["coexp_e"], bundle["coexp_w"]
    ppi_e, ppi_w = bundle["ppi_e"], bundle["ppi_w"]

    data = Data(
        x=torch.tensor(X, dtype=torch.float32),
        y=torch.tensor(y, dtype=torch.long),
        num_nodes=X.shape[0],
    )

    # Attach cond/other features to data for model access
    if bundle["cond_input"] is not None:
        data.cond_input = torch.tensor(bundle["cond_input"], dtype=torch.float32)
    if bundle["other_input"] is not None:
        data.other_input = torch.tensor(bundle["other_input"], dtype=torch.float32)

    data = build_pu_masks(data, positive_class=1,
                          train_ratio=train_ratio, val_ratio=val_ratio,
                          seed=split_seed)

    # Use preprocessed PPI graph directly
    ppi_use_e, ppi_use_w = ppi_e, ppi_w

    if use_hetero:
        # Heterogeneous: separate branches
        ce, cw = (coexp_e, coexp_w) if use_coexp else (np.empty((2,0), np.int64), np.empty((0,), np.float32))
        pe, pw = (ppi_use_e, ppi_use_w) if use_ppi else (np.empty((2,0), np.int64), np.empty((0,), np.float32))
        # Add self-loop for empty graph to avoid GAT errors
        if ce.shape[1] == 0:
            n = data.num_nodes
            ce = np.vstack([np.arange(n), np.arange(n)]).astype(np.int64)
            cw = np.ones(n, dtype=np.float32) * 1e-4
        if pe.shape[1] == 0:
            n = data.num_nodes
            pe = np.vstack([np.arange(n), np.arange(n)]).astype(np.int64)
            pw = np.ones(n, dtype=np.float32) * 1e-4
        data.edge_index_coexp = torch.tensor(ce, dtype=torch.long)
        data.edge_weight_coexp = torch.tensor(cw, dtype=torch.float32)
        data.edge_index_ppi = torch.tensor(pe, dtype=torch.long)
        data.edge_weight_ppi = torch.tensor(pw, dtype=torch.float32)
        # Also attach merged for feature-only compatibility
        data.edge_index = data.edge_index_coexp
        data.edge_weight = data.edge_weight_coexp
    else:
        merged_e, merged_w = combine_edges(coexp_e, coexp_w, ppi_use_e, ppi_use_w,
                                           coexp_ratio=coexp_ratio, ppi_ratio=ppi_ratio,
                                           use_coexp=use_coexp, use_ppi=use_ppi)
        if merged_e.shape[1] == 0:
            n = data.num_nodes
            merged_e = np.vstack([np.arange(n), np.arange(n)]).astype(np.int64)
            merged_w = np.ones(n, dtype=np.float32) * 1e-4
        data.edge_index = torch.tensor(merged_e, dtype=torch.long)
        data.edge_weight = torch.tensor(merged_w, dtype=torch.float32)

    return data


# =========================================================
# Class prior estimation
# =========================================================
def estimate_pi(method: str,
                n_pos: int, n_nodes: int,
                pi_override: float = -1.0,
                X: np.ndarray = None,
                pos_mask: np.ndarray = None) -> float:
    """
    method:
      fixed: n_pos/n_nodes (traditional, often biased small)
      manual: use pi_override (recommend 0.015, 0.02, 0.03, 0.05)
      tice: simple threshold-based estimation, clip to [0.005, 0.05]
    """
    if method == "fixed":
        return float(n_pos) / float(n_nodes)
    if method == "manual":
        if pi_override <= 0:
            raise ValueError("--pi_method manual requires --pi_override > 0")
        return float(pi_override)
    if method == "tice":
        # Simple approximation: use PCA 1st component as score
        if X is None or pos_mask is None:
            return float(n_pos) / float(n_nodes)
        from sklearn.decomposition import PCA
        pc1 = PCA(n_components=1).fit_transform(X).reshape(-1)
        pos_scores = pc1[pos_mask]
        all_scores = pc1
        # Estimate positive density in high-score region
        thr = np.quantile(pos_scores, 0.5)
        n_above_all = float((all_scores >= thr).sum())
        n_above_pos = float((pos_scores >= thr).sum())
        if n_above_all <= 0:
            return float(n_pos) / float(n_nodes)
        est = (n_pos * n_above_all) / (float(n_nodes) * n_above_pos + 1e-8)
        return float(np.clip(est, 0.005, 0.05))
    raise ValueError(f"Unknown pi method: {method}")


# =========================================================
# Model: hetero-aware + cross-stress contrastive
# =========================================================
class SingleOmic_PU_GATv2_v2(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hidden_dim: int = 128,
                 proj_dim: int = 64,
                 heads: int = 4,
                 dropout: float = 0.3,
                 K: int = 5,
                 alpha: float = 0.2,
                 use_hetero: bool = False):
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads")
        self.hidden_dim = hidden_dim
        self.proj_dim = proj_dim
        self.dropout_p = dropout
        self.use_hetero = bool(use_hetero)

        # Feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.BatchNorm1d(proj_dim), nn.ReLU(), nn.Dropout(dropout),
        )
        # Feature-only head
        self.feature_head = nn.Sequential(
            nn.Linear(proj_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        # Edge encoder (shared)
        self.edge_encoder = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 16),
        )

        # GNN branches
        if use_hetero:
            self.gnn_coexp = GATv2Conv(proj_dim, hidden_dim // heads,
                                       heads=heads, concat=True, dropout=dropout, edge_dim=16)
            self.gnn_ppi = GATv2Conv(proj_dim, hidden_dim // heads,
                                     heads=heads, concat=True, dropout=dropout, edge_dim=16)
            self.bn_coexp = nn.BatchNorm1d(hidden_dim)
            self.bn_ppi = nn.BatchNorm1d(hidden_dim)
            # Gated fusion for two relations
            self.rel_gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim // 2), nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
            )
        else:
            self.gnn1 = GATv2Conv(proj_dim, hidden_dim // heads,
                                  heads=heads, concat=True, dropout=dropout, edge_dim=16)
            self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.graph_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.rank_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.mix_gate = nn.Sequential(
            nn.Linear(proj_dim + hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1),
        )

    def _graph_branch(self, h0, data):
        if self.use_hetero:
            ew_c = data.edge_weight_coexp
            ea_c = self.edge_encoder(ew_c.view(-1, 1))
            h_c = self.gnn_coexp(h0, data.edge_index_coexp, edge_attr=ea_c)
            h_c = self.bn_coexp(h_c); h_c = F.relu(h_c)
            h_c = F.dropout(h_c, p=self.dropout_p, training=self.training)

            ew_p = data.edge_weight_ppi
            ea_p = self.edge_encoder(ew_p.view(-1, 1))
            h_p = self.gnn_ppi(h0, data.edge_index_ppi, edge_attr=ea_p)
            h_p = self.bn_ppi(h_p); h_p = F.relu(h_p)
            h_p = F.dropout(h_p, p=self.dropout_p, training=self.training)

            alpha = torch.sigmoid(self.rel_gate(torch.cat([h_c, h_p], dim=1)))
            h = alpha * h_c + (1.0 - alpha) * h_p

            return h
        else:
            ew = data.edge_weight
            ea = self.edge_encoder(ew.view(-1, 1))
            h = self.gnn1(h0, data.edge_index, edge_attr=ea)
            h = self.bn1(h); h = F.relu(h)
            h = F.dropout(h, p=self.dropout_p, training=self.training)

            return h

    def forward(self, data: Data, return_details: bool = False):
        h0 = self.feature_encoder(data.x)
        feature_logit = self.feature_head(h0).squeeze(-1)
        h_graph = self._graph_branch(h0, data)
        graph_logit = self.graph_head(h_graph).squeeze(-1)
        rank_logit = self.rank_head(h_graph).squeeze(-1)
        mix_alpha = torch.sigmoid(self.mix_gate(torch.cat([h0, h_graph], dim=1))).squeeze(-1)
        final_logit = mix_alpha * feature_logit + (1.0 - mix_alpha) * graph_logit

        out = {
            "feature_logit": feature_logit,
            "graph_logit": graph_logit,
            "final_logit": final_logit,
            "rank_logit": rank_logit,
            "mix_alpha": mix_alpha,
            "embedding": h_graph,
        }

        if return_details:
            return out
        return final_logit


# =========================================================
# Losses
# =========================================================
class PULossWithRanking(nn.Module):
    def __init__(self, pi, beta=0.0, rank_weight=1.0,
                 margin=0.3, num_rank_pairs=2048, hard_negative_ratio=0.5):
        super().__init__()
        self.pi = float(pi); self.beta = float(beta)
        self.rank_weight = float(rank_weight)
        self.margin = float(margin); self.num_rank_pairs = int(num_rank_pairs)
        self.hard_negative_ratio = float(hard_negative_ratio)

    def nnpu_part(self, logits, pos_mask, unlabeled_mask):
        l_pos = F.softplus(-logits); l_neg = F.softplus(logits)
        Rp_pos = l_pos[pos_mask].mean(); Rp_neg = l_neg[pos_mask].mean()
        eff_un = unlabeled_mask.clone()

        if eff_un.sum() > 0:
            Ru_neg = l_neg[eff_un].mean()
        else:
            Ru_neg = torch.tensor(0.0, device=logits.device)
        neg_risk_raw = Ru_neg - self.pi * Rp_neg
        neg_risk_used = (torch.clamp(neg_risk_raw, min=0.0) if self.beta == 0.0
                         else torch.clamp(neg_risk_raw, min=-self.beta))
        pu_loss = self.pi * Rp_pos + neg_risk_used

        stats = {"Rp_pos": Rp_pos.detach(), "Rp_neg": Rp_neg.detach(),
                 "Ru_neg": Ru_neg.detach(), "neg_risk_raw": neg_risk_raw.detach(),
                 "neg_risk_used": neg_risk_used.detach(),
                 "pi": torch.tensor(self.pi, device=logits.device)}
        return pu_loss, stats

    def ranking_part(self, rank_logits, pos_mask, unlabeled_mask):
        pos_idx = torch.where(pos_mask)[0]
        neg_mask = unlabeled_mask.clone()
        
        neg_idx = torch.where(neg_mask)[0]
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            return torch.tensor(0.0, device=rank_logits.device)
        pos_sample = pos_idx[torch.randint(0, len(pos_idx), (self.num_rank_pairs,),
                                           device=rank_logits.device)]
        neg_scores = rank_logits[neg_idx].detach()
        hard_k = max(1, min(int(len(neg_idx) * self.hard_negative_ratio), len(neg_idx)))
        _, order = torch.topk(neg_scores, k=hard_k, largest=True)
        hard_neg = neg_idx[order]
        neg_pool = hard_neg if len(hard_neg) > 0 else neg_idx
        neg_sample = neg_pool[torch.randint(0, len(neg_pool), (self.num_rank_pairs,),
                                            device=rank_logits.device)]
        return F.relu(self.margin - (rank_logits[pos_sample] - rank_logits[neg_sample])).mean()

    def forward(self, final_logit, rank_logit, pos_mask, unlabeled_mask):
        pu_loss, stats = self.nnpu_part(final_logit, pos_mask, unlabeled_mask)
        rank_loss = self.ranking_part(rank_logit, pos_mask, unlabeled_mask)
        total = pu_loss + self.rank_weight * rank_loss
        stats["rank_loss"] = rank_loss.detach()
        return total, stats



# =========================================================
# Metrics / prediction
# =========================================================
def compute_ranking_metrics(scores, train_pos_mask, test_pos_mask, Ks=(50, 100, 200)):
    train_pos = train_pos_mask.detach().cpu().numpy()
    test_pos = test_pos_mask.detach().cpu().numpy()
    exclude = train_pos.copy()
    cand = np.where(~exclude)[0]
    if len(cand) == 0:
        out = {"MRR": 0.0, "AUPRC": 0.0, "AUROC": 0.0}
        for K in Ks:
            out[f"Hits@{K}"] = 0
            out[f"Precision@{K}"] = 0.0
            out[f"Recall@{K}"] = 0.0
            out[f"nDCG@{K}"] = 0.0
        return out

    cs = scores[cand]
    order = np.argsort(-cs)
    ranked = cand[order]
    hidden = set(np.where(test_pos)[0].tolist())
    hits = [1 if n in hidden else 0 for n in ranked]
    mrr = 0.0
    for i, h in enumerate(hits, start=1):
        if h == 1:
            mrr = 1.0 / i
            break

    def dcg(hk):
        return sum(h / np.log2(i + 1) for i, h in enumerate(hk, start=1))

    m = {"MRR": float(mrr)}
    denom = max(1, len(hidden))
    for K in Ks:
        topk = hits[:K]
        hit_count = int(sum(topk))
        idcg_ideal = [1] * min(len(hidden), K) + [0] * max(0, K - min(len(hidden), K))
        idcg = dcg(idcg_ideal) if len(hidden) > 0 else 0.0
        ndcg = (dcg(topk) / idcg) if idcg > 0 else 0.0
        m[f"Hits@{K}"] = hit_count
        m[f"Precision@{K}"] = hit_count / K
        m[f"Recall@{K}"] = hit_count / denom
        m[f"nDCG@{K}"] = float(ndcg)

    from sklearn.metrics import average_precision_score, roc_auc_score
    y_true = test_pos[cand].astype(int)
    if y_true.sum() > 0:
        m["AUPRC"] = float(average_precision_score(y_true, cs))
    else:
        m["AUPRC"] = 0.0
    if 0 < y_true.sum() < len(y_true):
        m["AUROC"] = float(roc_auc_score(y_true, cs))
    else:
        m["AUROC"] = 0.0
    return m


def enable_dropout(model):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


@torch.no_grad()
def mc_dropout_predict(model, data, mc_times=20, uncertainty_penalty=0.10):
    model.eval()
    feats, graphs, finals, ranks, mixes = [], [], [], [], []
    for _ in range(mc_times):
        enable_dropout(model)
        out = model(data, return_details=True)
        feats.append(torch.sigmoid(out["feature_logit"]).detach().cpu())
        graphs.append(torch.sigmoid(out["graph_logit"]).detach().cpu())
        finals.append(torch.sigmoid(out["final_logit"]).detach().cpu())
        ranks.append(torch.sigmoid(out["rank_logit"]).detach().cpu())
        mixes.append(out["mix_alpha"].detach().cpu())
    feats = torch.stack(feats); graphs = torch.stack(graphs)
    finals = torch.stack(finals); ranks = torch.stack(ranks); mixes = torch.stack(mixes)
    final_mean = finals.mean(0).numpy(); final_std = finals.std(0).numpy()
    rank_mean = ranks.mean(0).numpy(); rank_std = ranks.std(0).numpy()
    return {
        "feature_score": feats.mean(0).numpy(),
        "graph_score": graphs.mean(0).numpy(),
        "pu_score": final_mean,
        "uncertainty": final_std,
        "confidence": 1.0 / (1.0 + final_std),
        "rank_head_mean": rank_mean,
        "rank_head_std": rank_std,
        "rank_score": rank_mean - uncertainty_penalty * rank_std,
        "mix_alpha": mixes.mean(0).numpy(),
    }



# =========================================================
# Training
# =========================================================
def train_one_stage(model, data, pi, stage_id=0,
                    epochs=300, lr=1e-3, weight_decay=5e-4,
                    patience=80, log_steps=10, Ks=(50, 100, 200),
                    beta=0.0, rank_weight=1.0,
                    save_dir=".", run_name="run", log_fn=None):
    def _log(s):
        print(s)
        if log_fn: log_fn(s)
    os.makedirs(save_dir, exist_ok=True)
    model = model.to(device); data = data.to(device)

    criterion = PULossWithRanking(pi=pi, beta=beta,
                                  rank_weight=rank_weight, margin=0.3,
                                  num_rank_pairs=2048, hard_negative_ratio=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=20, min_lr=1e-5)

    best_state, best_val_key, bad = None, -1e9, 0
    history = []
    _log(f"[+] Start Stage-{stage_id} | pi={pi:.5f}")

    for e in range(1, epochs + 1):
        model.train(); optimizer.zero_grad()
        out = model(data, return_details=True)
        loss_pu, stats = criterion(out["final_logit"], out["rank_logit"],
                                   data.train_pos_mask, data.unlabeled_mask)
        loss = loss_pu

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if e == 1 or e % log_steps == 0:
            model.eval()
            preds = mc_dropout_predict(model, data, mc_times=10)
            val_m = compute_ranking_metrics(preds["rank_score"],
                                            data.train_pos_mask, data.val_pos_mask,
                                            Ks=Ks)
            val_key = 0.6 * val_m.get("nDCG@100", 0.0) + 0.4 * val_m.get("Recall@100", 0.0)
            scheduler.step(val_key)
            rec = {"epoch": e, "stage": stage_id, "loss": float(loss.item()),
                   **{k: float(v.item()) for k, v in stats.items()},
                   **val_m, "val_key": float(val_key),
                   "lr": float(optimizer.param_groups[0]["lr"])}
            history.append(rec)
            _log(f"[S{stage_id}|{e:04d}] Loss:{rec['loss']:.4f} "
                 f"Rank:{rec.get('rank_loss',0):.3f} "
                 f"Rp+:{rec['Rp_pos']:.3f} NegRaw:{rec['neg_risk_raw']:.4f} "
                 f"R@100:{rec.get('Recall@100',0):.3f} nDCG@100:{rec.get('nDCG@100',0):.3f} "
                 f"MRR:{rec.get('MRR',0):.3f} LR:{rec['lr']:.5f}")
            if val_key > best_val_key:
                best_val_key = val_key
                best_state = copy.deepcopy(model.state_dict())
                torch.save(best_state, os.path.join(save_dir, f"{run_name}_best_stage_{stage_id}.pth"))
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    _log(f"[!] Stage-{stage_id} early stopping, best val_key={best_val_key:.4f}")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_val_key


def train_iterative(model, data, pi,
                    self_training_rounds=2, stage_epochs=300,
                    lr=1e-3, weight_decay=5e-4, patience=80, log_steps=10,
                    Ks=(50, 100, 200), beta=0.0,
                    rank_weight=1.0,
                    save_dir=".", run_name="run",
                    experiment_params=None, split_seed=None):
    os.makedirs(save_dir, exist_ok=True)

    # Save parameters
    run_params = {
        "meta": {"run_name": run_name, "split_seed": split_seed,
                 "device": device,
                 "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
        "training_params": {"pi": float(pi),
                            "self_training_rounds": int(self_training_rounds),
                            "stage_epochs": int(stage_epochs), "lr": float(lr),
                            "weight_decay": float(weight_decay), "patience": int(patience),
                            "beta": float(beta), 
                            "rank_weight": float(rank_weight)},
        "data_info": {"num_nodes": int(data.num_nodes),
                      "num_edges_main": int(data.edge_index.size(1)),
                      "train_pos": int(data.train_pos_mask.sum().item()),
                      "val_pos": int(data.val_pos_mask.sum().item()),
                      "test_pos": int(data.test_pos_mask.sum().item())},
        "model_info": {"model_class": model.__class__.__name__,
                       "use_hetero": getattr(model, "use_hetero", False)},
    }
    if experiment_params is not None:
        run_params["experiment_params"] = experiment_params
    with open(os.path.join(save_dir, f"{run_name}_params.json"), "w", encoding="utf-8") as f:
        json.dump(run_params, f, ensure_ascii=False, indent=2)

    total_log_fp = open(os.path.join(save_dir, f"{run_name}_total_train_log.txt"),
                        "w", encoding="utf-8")
    def log(s):
        print(s); total_log_fp.write(s + "\n"); total_log_fp.flush()

    model = model.to(device); data = data.to(device)

    all_history = []
    g_best_state, g_best_val, g_best_stage = None, -1e9, -1

    log("=" * 80)
    log("[+] Start training (multi-stage warm-restart)")
    log(json.dumps(run_params, ensure_ascii=False, indent=2))
    log("=" * 80)
    if self_training_rounds > 0:
        log(f"[i] Will run {self_training_rounds} warm-restart stage(s) after base stage.")

    # Base stage
    model, hist, bv = train_one_stage(
        model, data, pi=pi, stage_id=0,
        epochs=stage_epochs, lr=lr, weight_decay=weight_decay, patience=patience,
        log_steps=log_steps, Ks=Ks, beta=beta, rank_weight=rank_weight,
        save_dir=save_dir, run_name=run_name, log_fn=log)
    all_history.extend(hist)
    if bv > g_best_val:
        g_best_val = bv; g_best_stage = 0
        g_best_state = copy.deepcopy(model.state_dict())
        torch.save(g_best_state, os.path.join(save_dir, f"{run_name}_best_global.pth"))

    # Warm restart stages
    for stage in range(1, self_training_rounds + 1):
        log(f"\n[+] Warm-restart Stage-{stage}")
        model, hist, bv = train_one_stage(
            model, data, pi=pi, stage_id=stage,
            epochs=stage_epochs, lr=lr, weight_decay=weight_decay, patience=patience,
            log_steps=log_steps, Ks=Ks, beta=beta, 
            rank_weight=rank_weight, save_dir=save_dir, run_name=run_name, log_fn=log)
        all_history.extend(hist)
        if bv > g_best_val:
            g_best_val = bv; g_best_stage = stage
            g_best_state = copy.deepcopy(model.state_dict())
            torch.save(g_best_state, os.path.join(save_dir, f"{run_name}_best_global.pth"))

    if g_best_state is not None:
        model.load_state_dict(g_best_state)
        log(f"[+] Reloaded global best from Stage-{g_best_stage}, val_key={g_best_val:.4f}")

    df_h = pd.DataFrame(all_history)
    df_h.to_csv(os.path.join(save_dir, f"{run_name}_training_history.csv"), index=False)

    model.eval()
    final_preds = mc_dropout_predict(model, data, mc_times=30)
    test_m = compute_ranking_metrics(final_preds["rank_score"],
                                     data.train_pos_mask, data.test_pos_mask,
                                     Ks=Ks)
    log(f"[+] Final test metrics: {test_m}")
    total_log_fp.close()
    return model, all_history, test_m, final_preds


def summarize_metrics(metrics_list):
    keys = sorted(metrics_list[0].keys())
    return {k: {"mean": float(np.mean([m[k] for m in metrics_list])),
                "std": float(np.std([m[k] for m in metrics_list], ddof=1)) if len(metrics_list) > 1 else 0.0}
            for k in keys}


# =========================================================
# Main
# =========================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str,
                   default="...")
    p.add_argument("--save_root", type=str, default="")

    # Graph
    p.add_argument("--coexp_ratio", type=float, default=0.5)
    p.add_argument("--ppi_ratio", type=float, default=0.5)
    p.add_argument("--no_coexp", action="store_true")
    p.add_argument("--no_ppi", action="store_true")

    # New features
    p.add_argument("--use_hetero_graph", action="store_true",
                   help="Enable heterogeneous graph: separate GATv2 for coexp and ppi")
    p.add_argument("--pi_method", type=str, default="fixed",
                   choices=["fixed", "manual", "tice"],
                   help="Class prior estimation method")
    p.add_argument("--pi_override", type=float, default=-1.0,
                   help="Positive prior for manual mode (0.015/0.02/0.03/0.05)")

    # Model
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--proj_dim", type=int, default=64)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--K", type=int, default=5)
    p.add_argument("--alpha", type=float, default=0.2)

    # Training
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_repeats", type=int, default=5)
    p.add_argument("--train_ratio", type=float, default=0.5)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--self_training_rounds", type=int, default=4)
    p.add_argument("--stage_epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--patience", type=int, default=80)
    p.add_argument("--log_steps", type=int, default=10)
    p.add_argument("--beta", type=float, default=0.0)
    p.add_argument("--rank_weight", type=float, default=1.0)
    return p.parse_args()


def collect_env_info():
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    try:
        info["git_commit"] = subprocess.getoutput("git rev-parse HEAD").strip()
    except Exception:
        info["git_commit"] = "unknown"
    try:
        import torch_geometric
        info["torch_geometric"] = torch_geometric.__version__
    except Exception:
        info["torch_geometric"] = "unknown"
    return info


def main():
    args = parse_args()
    set_seed(args.seed)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.save_root:
        save_root = args.save_root
    else:
        save_root = f"..._{ts}"
    os.makedirs(save_root, exist_ok=True)

    # Global log
    global_log_fp = open(os.path.join(save_root, "global_run.log"), "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, global_log_fp)
    sys.stderr = Tee(sys.__stderr__, global_log_fp)
    with open(os.path.join(save_root, "env_info.json"), "w", encoding="utf-8") as f:
        json.dump(collect_env_info(), f, ensure_ascii=False, indent=2)

    print(f"[+] save_root: {save_root}")
    print(f"[+] args: {vars(args)}")

    bundle = load_preprocessed_v4(args.data_dir)
    n_pos = int((bundle["y"] == 1).sum()); n_nodes = bundle["y"].shape[0]
    pi = estimate_pi(args.pi_method, n_pos=n_pos, n_nodes=n_nodes,
                     pi_override=args.pi_override,
                     X=bundle["X"], pos_mask=(bundle["y"] == 1))
    print(f"[i] pi estimated by '{args.pi_method}': {pi:.5f} "
          f"(fixed would be {n_pos / n_nodes:.5f})")

    exp_params = vars(args).copy()
    exp_params["pi_used"] = pi
    exp_params["timestamp"] = ts
    with open(os.path.join(save_root, "experiment_config.json"), "w", encoding="utf-8") as f:
        json.dump(exp_params, f, ensure_ascii=False, indent=2)

    def model_builder(data: Data):
        return SingleOmic_PU_GATv2_v2(
            in_dim=data.x.size(1),
            hidden_dim=args.hidden_dim, proj_dim=args.proj_dim,
            heads=args.heads, dropout=args.dropout,
            K=args.K, alpha=args.alpha,
            use_hetero=args.use_hetero_graph,
        )

    all_metrics = []
    for r in range(args.num_repeats):
        split_seed = args.seed + r
        print("\n" + "-" * 100)
        print(f"[+] Repeat {r+1}/{args.num_repeats} | split_seed={split_seed}")
        print("-" * 100)

        data_r = build_data_for_split(
            bundle, split_seed=split_seed,
            train_ratio=args.train_ratio, val_ratio=args.val_ratio,
            use_coexp=not args.no_coexp, use_ppi=not args.no_ppi,
            coexp_ratio=args.coexp_ratio, ppi_ratio=args.ppi_ratio,
            use_hetero=args.use_hetero_graph,
        )

        run_name = f"repeat_seed_{split_seed}"
        sd = os.path.join(save_root, run_name); os.makedirs(sd, exist_ok=True)
        model = model_builder(data_r)
        _, _, test_m, final_preds = train_iterative(
            model=model, data=data_r, pi=pi,
            self_training_rounds=args.self_training_rounds,
            stage_epochs=args.stage_epochs, lr=args.lr,
            weight_decay=args.weight_decay, patience=args.patience,
            log_steps=args.log_steps, Ks=(50, 100, 200),
            beta=args.beta, 
            rank_weight=args.rank_weight,
            save_dir=sd, run_name=run_name,
            experiment_params=exp_params, split_seed=split_seed)

        # Save ranking
        rank_df = pd.DataFrame({
            "gene_id": bundle["nodes"],
            "rank_score": final_preds["rank_score"],
            "pu_score": final_preds["pu_score"],
            "uncertainty": final_preds["uncertainty"],
            "is_train_pos": data_r.train_pos_mask.detach().cpu().numpy(),
            "is_val_pos": data_r.val_pos_mask.detach().cpu().numpy(),
            "is_test_pos": data_r.test_pos_mask.detach().cpu().numpy()
        })
        rank_df = rank_df[~rank_df["is_train_pos"]].sort_values("rank_score", ascending=False)
        rank_df.to_csv(os.path.join(sd, f"{run_name}_ranking.csv"), index=False)

        all_metrics.append({**test_m, "split_seed": split_seed})
        print(f"[i] split_seed={split_seed} TEST: {test_m}")

    summary = summarize_metrics([{k: v for k, v in m.items() if k != "split_seed"}
                                 for m in all_metrics])

    print("\n" + "=" * 100)
    print("[+] Repeated split summary")
    for k, v in summary.items():
        print(f"{k}: {v['mean']:.4f} ± {v['std']:.4f}")
    print("=" * 100)

    out_path = os.path.join(save_root, "final_repeated_results.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("[+] Final repeated summary:\n")
        for k, v in summary.items():
            f.write(f"{k}: {v['mean']:.4f} ± {v['std']:.4f}\n")
        f.write("\n[+] Per-split metrics:\n")
        for i, m in enumerate(all_metrics, 1):
            f.write(f"Repeat-{i}: {m}\n")
    print(f"[+] Saved final results to: {out_path}")

    global_log_fp.close()


if __name__ == "__main__":
    main()
