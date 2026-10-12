from __future__ import annotations
import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from datasets import get_dataset_config
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Sequence, Tuple, Union
from utils import (
    classifier_topology_loss,
    covariance_loss,
    ema_update,
    prior_corrected_softmax,
    reliable_memory_loss,
    soft_cross_entropy,
    strong_hsi_augment,
    variance_loss,
    weak_hsi_augment,
)

try:
    import ot
except ImportError:
    ot = None


@dataclass
class TargetConfig:
    dataset: str = "pavia"
    num_classes: int = 7
    feature_dim: int = 32
    num_bands: int = None
    warmup_epochs: int = 2
    warmup_lr: float = 2e-05
    spectral_mask_ratio: float = None
    lambda_warm_cons: float = 1.0
    lambda_warm_recon: float = None
    lambda_warm_var: float = None
    lambda_warm_cov: float = None
    source_prototypes_per_class: int = None
    target_clusters_per_class: int = None
    knn_k: int = None
    stable_rounds: int = 2
    confidence_threshold: float = None
    initial_coverage: float = None
    final_coverage: float = None
    min_samples_per_class: int = None
    pseudo_temperature: float = None
    uot_reg: float = None
    uot_reg_m: float = None
    uot_iterations: int = None
    lambda_pl: float = 1.0
    lambda_cons: float = None
    lambda_src_proto: float = None
    lambda_memory: float = None
    lambda_topology: float = None
    teacher_temperature: float = None
    ema_momentum: float = None
    max_grad_norm: float = 5.0
    source_class_prior: Optional[torch.Tensor] = None
    source_prior_correction_tau: float = 0.0
    pseudo_balance_power: float = 0.5
    pseudo_balance_max_weight: float = 3.0
    memory_per_class: int = None
    memory_batch_size: int = None
    kd_temperature: float = 2.0
    random_state: int = 42

    def __post_init__(self):
        defaults = get_dataset_config(self.dataset)["target_config"]
        for name, value in defaults.items():
            if getattr(self, name) is None:
                setattr(self, name, value)


class UnlabeledTargetView(Dataset):
    # Target labels are exposed only to diagnostics, never to the optimizer.
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        sample = self.dataset[index]
        if len(sample) == 3:
            image, _, sample_id = sample
        elif len(sample) == 2:
            image, _ = sample
            sample_id = index
        else:
            raise ValueError("Target dataset must return (image,label[,id]).")
        return image, int(sample_id)


class SpectralReconstructionHead(nn.Module):
    def __init__(self, feature_dim: int, num_bands: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_bands),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.net(feature)


def build_target_optimizer(
    model: nn.Module,
    lr: float,
    reconstruction_head: Optional[nn.Module] = None,
    warmup: bool = False,
):
    groups = [
        {"params": model.backbone.parameters(), "lr": lr * 0.1},
        {"params": model.fc.parameters(), "lr": lr},
    ]
    if not warmup:
        groups.append({"params": model.classifier.parameters(), "lr": lr * 0.1})
    else:
        for p in model.classifier.parameters():
            p.requires_grad_(False)
    if reconstruction_head is not None:
        groups.append({"params": reconstruction_head.parameters(), "lr": lr})
    return torch.optim.SGD(groups, momentum=0.9, weight_decay=5e-4)


def label_free_warmup(
    student: nn.Module,
    ema_teacher: nn.Module,
    loader,
    cfg: TargetConfig,
    device: torch.device,
    logger=None,
) -> None:
    if cfg.warmup_epochs <= 0:
        return

    reconstruction_head = SpectralReconstructionHead(
        cfg.feature_dim, cfg.num_bands, hidden=48 if cfg.dataset == "houston" else 64
    ).to(device)
    optimizer = build_target_optimizer(
        student, cfg.warmup_lr, reconstruction_head, warmup=True
    )

    ema_teacher.eval()
    for p in ema_teacher.parameters():
        p.requires_grad_(False)

    for epoch in range(cfg.warmup_epochs):
        student.train()
        reconstruction_head.train()
        loss_sum = 0.0
        sample_count = 0

        for images, _ in loader:
            images = images.to(device, non_blocking=True).float()
            center_spectrum = images[:, :, images.size(-2) // 2, images.size(-1) // 2]
            weak = weak_hsi_augment(images)
            strong, band_mask = strong_hsi_augment(
                images, cfg.spectral_mask_ratio
            )

            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                _, teacher_feature = ema_teacher(weak)
                teacher_feature = F.normalize(teacher_feature, dim=1)

            _, student_feature = student(strong)
            normalized_student = F.normalize(student_feature, dim=1)
            reconstruction = reconstruction_head(student_feature)

            consistency = 2.0 - 2.0 * (
                normalized_student * teacher_feature
            ).sum(dim=1).mean()
            reconstruction_loss = F.smooth_l1_loss(
                reconstruction[band_mask], center_spectrum[band_mask]
            )
            loss = (
                cfg.lambda_warm_cons * consistency
                + cfg.lambda_warm_recon * reconstruction_loss
                + cfg.lambda_warm_var * variance_loss(student_feature)
                + cfg.lambda_warm_cov * covariance_loss(student_feature)
            )
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), cfg.max_grad_norm)
            optimizer.step()
            ema_update(ema_teacher, student, cfg.ema_momentum)

            loss_sum += float(loss.detach()) * len(images)
            sample_count += len(images)

        message = (
            f"[label-free warm-up] epoch={epoch + 1}/{cfg.warmup_epochs} "
            f"loss={loss_sum / max(1, sample_count):.6f}"
        )
        if logger is not None:
            logger.info(message)
        else:
            print(message)

    for p in student.classifier.parameters():
        p.requires_grad_(True)


@torch.no_grad()
def extract_stage_outputs(
    ema_teacher: nn.Module,
    stage_anchor: nn.Module,
    loader,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    ema_teacher.eval()
    stage_anchor.eval()
    ids, features, ema_logits, old_logits = [], [], [], []
    for images, sample_ids in loader:
        images = images.to(device, non_blocking=True).float()
        logits_ema, feature_ema = ema_teacher(images)
        logits_old, _ = stage_anchor(images)
        ids.append(torch.as_tensor(sample_ids).long().cpu())
        features.append(F.normalize(feature_ema, dim=1).cpu())
        ema_logits.append(logits_ema.cpu())
        old_logits.append(logits_old.cpu())

    ids = torch.cat(ids)
    order = torch.argsort(ids)
    return {
        "ids": ids[order],
        "features": torch.cat(features)[order],
        "ema_logits": torch.cat(ema_logits)[order],
        "old_logits": torch.cat(old_logits)[order],
    }


def source_prototype_logits(
    features: torch.Tensor,
    multi_prototypes: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    similarity = torch.einsum(
        "nd,ckd->nck",
        F.normalize(features, dim=1),
        F.normalize(multi_prototypes, dim=2),
    )
    return similarity.max(dim=2).values / temperature


def discover_active_classes(
    old_probs: torch.Tensor,
    prototype_probs: torch.Tensor,
    expected_class_count: int,
    excluded_classes: Optional[Sequence[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    class_score = (
        0.45 * old_probs.mean(dim=0)
        + 0.55 * prototype_probs.mean(dim=0)
    )
    ranking_score = class_score.clone()
    if excluded_classes:
        excluded = torch.tensor(
            sorted(set(int(c) for c in excluded_classes)), dtype=torch.long
        )
        ranking_score[excluded] = -float("inf")
    available = int(torch.isfinite(ranking_score).sum())
    count = min(expected_class_count, available)
    if count <= 0:
        raise RuntimeError("No candidate classes remain after excluding old classes.")
    active = ranking_score.topk(count).indices.sort().values
    return active, class_score


def _torch_unbalanced_sinkhorn(
    a: np.ndarray,
    b: np.ndarray,
    cost: np.ndarray,
    reg: float,
    reg_m: float,
    iterations: int,
) -> np.ndarray:
    dtype = torch.float64
    a_t = torch.as_tensor(a, dtype=dtype).clamp_min(1e-12)
    b_t = torch.as_tensor(b, dtype=dtype).clamp_min(1e-12)
    c_t = torch.as_tensor(cost, dtype=dtype)
    kernel = torch.exp(-c_t / max(reg, 1e-8)).clamp_min(1e-300)
    exponent = reg_m / (reg_m + reg)
    u = torch.ones_like(a_t)
    v = torch.ones_like(b_t)
    for _ in range(iterations):
        u_prev = u
        u = (a_t / (kernel @ v).clamp_min(1e-300)).pow(exponent)
        v = (b_t / (kernel.t() @ u).clamp_min(1e-300)).pow(exponent)
        if torch.max(torch.abs(u - u_prev)) < 1e-9:
            break
    plan = u[:, None] * kernel * v[None, :]
    return plan.cpu().numpy()


def solve_unbalanced_ot(
    a: np.ndarray,
    b: np.ndarray,
    cost: np.ndarray,
    cfg: TargetConfig,
) -> np.ndarray:
    if ot is not None:
        try:
            return np.asarray(
                ot.unbalanced.sinkhorn_knopp_unbalanced(
                    a,
                    b,
                    cost,
                    reg=cfg.uot_reg,
                    reg_m=cfg.uot_reg_m,
                    numItermax=cfg.uot_iterations,
                    stopThr=1e-8,
                )
            )
        except (AttributeError, TypeError):
            return np.asarray(
                ot.unbalanced.sinkhorn_unbalanced(
                    a,
                    b,
                    cost,
                    reg=cfg.uot_reg,
                    reg_m=cfg.uot_reg_m,
                    numItermax=cfg.uot_iterations,
                    stopThr=1e-8,
                )
            )
    return _torch_unbalanced_sinkhorn(
        a, b, cost, cfg.uot_reg, cfg.uot_reg_m, cfg.uot_iterations
    )


def target_cluster_uot(
    features: torch.Tensor,
    old_probs: torch.Tensor,
    source_multi_prototypes: torch.Tensor,
    active_classes: torch.Tensor,
    cfg: TargetConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    n = len(features)
    active_count = len(active_classes)
    cluster_count = min(
        n,
        max(active_count, active_count * cfg.target_clusters_per_class),
    )
    kmeans = MiniBatchKMeans(
        n_clusters=cluster_count,
        batch_size=min(2048, n),
        random_state=cfg.random_state,
        n_init=10,
    )
    cluster_ids_np = kmeans.fit_predict(features.numpy())
    cluster_ids = torch.from_numpy(cluster_ids_np).long()
    centers = F.normalize(
        torch.from_numpy(kmeans.cluster_centers_).float(), dim=1
    )

    active_prototypes = source_multi_prototypes[active_classes]
    proto_similarity = torch.einsum(
        "qd,akd->qak", centers, active_prototypes
    ).max(dim=2).values

    cluster_old_probs = []
    for q in range(cluster_count):
        mask = cluster_ids == q
        cluster_old_probs.append(
            old_probs[mask][:, active_classes].mean(dim=0)
        )
    cluster_old_probs = torch.stack(cluster_old_probs).clamp_min(1e-8)
    cluster_old_probs /= cluster_old_probs.sum(dim=1, keepdim=True)

    cost = 0.70 * (1.0 - proto_similarity) + 0.30 * (
        -cluster_old_probs.log()
    )
    cost_np = cost.numpy().astype(np.float64)
    cost_np -= cost_np.min()
    if cost_np.max() > 0:
        cost_np /= cost_np.max()

    counts = np.bincount(cluster_ids_np, minlength=cluster_count).astype(
        np.float64
    )
    cluster_mass = counts / counts.sum()

    model_prior = old_probs[:, active_classes].mean(dim=0).numpy().astype(np.float64)
    model_prior = np.clip(model_prior, 1e-5, None)
    model_prior /= model_prior.sum()
    uniform_prior = np.full(active_count, 1.0 / active_count, dtype=np.float64)
    class_prior = 0.80 * uniform_prior + 0.20 * model_prior if cfg.dataset == "houston" else model_prior

    plan = solve_unbalanced_ot(cluster_mass, class_prior, cost_np, cfg)
    cluster_probs = plan / np.clip(plan.sum(axis=1, keepdims=True), 1e-12, None)
    sample_active_probs = torch.from_numpy(cluster_probs[cluster_ids_np]).float()

    full_probs = torch.full(
        (n, cfg.num_classes), 1e-8, dtype=torch.float32
    )
    full_probs[:, active_classes] = sample_active_probs
    full_probs /= full_probs.sum(dim=1, keepdim=True)
    return full_probs, cluster_ids


def knn_probability_smoothing(
    features: torch.Tensor,
    probs: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    n = len(features)
    if n <= 1:
        return probs, torch.ones(n)
    neighbor_count = min(k + 1, n)
    index = NearestNeighbors(
        n_neighbors=neighbor_count, metric="cosine", n_jobs=1
    )
    index.fit(features.numpy())
    neighbors = index.kneighbors(return_distance=False)[:, 1:]
    neighbor_index = torch.from_numpy(neighbors).long()
    neighbor_probs = probs[neighbor_index].mean(dim=1)
    smoothed = 0.80 * probs + 0.20 * neighbor_probs
    smoothed /= smoothed.sum(dim=1, keepdim=True)
    label = smoothed.argmax(dim=1)
    agreement = (label[neighbor_index] == label[:, None]).float().mean(dim=1)
    return smoothed, agreement


class PseudoLabelBank:
    def __init__(self, num_classes: int, stable_rounds: int):
        self.num_classes = num_classes
        self.stable_rounds = stable_rounds
        self.ids: Optional[torch.Tensor] = None
        self.id_to_position: Dict[int, int] = {}
        self.soft: Optional[torch.Tensor] = None
        self.confidence: Optional[torch.Tensor] = None
        self.selected: Optional[torch.Tensor] = None
        self.stable: Optional[torch.Tensor] = None
        self.last_label: Optional[torch.Tensor] = None
        self.stable_count: Optional[torch.Tensor] = None

    def state_dict(self) -> Dict[str, object]:
        def clone_or_none(value):
            return None if value is None else value.detach().cpu().clone()

        return {
            "num_classes": int(self.num_classes),
            "stable_rounds": int(self.stable_rounds),
            "ids": clone_or_none(self.ids),
            "soft": clone_or_none(self.soft),
            "confidence": clone_or_none(self.confidence),
            "selected": clone_or_none(self.selected),
            "stable": clone_or_none(self.stable),
            "last_label": clone_or_none(self.last_label),
            "stable_count": clone_or_none(self.stable_count),
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        saved_num_classes = int(state.get("num_classes", self.num_classes))
        if saved_num_classes != self.num_classes:
            raise ValueError(
                f"Pseudo-label bank num_classes mismatch: "
                f"checkpoint={saved_num_classes}, current={self.num_classes}."
            )

        self.stable_rounds = int(
            state.get("stable_rounds", self.stable_rounds)
        )

        def restore(name):
            value = state.get(name)
            return None if value is None else torch.as_tensor(value).cpu().clone()

        self.ids = restore("ids")
        self.soft = restore("soft")
        self.confidence = restore("confidence")
        self.selected = restore("selected")
        self.stable = restore("stable")
        self.last_label = restore("last_label")
        self.stable_count = restore("stable_count")
        self.id_to_position = (
            {}
            if self.ids is None
            else {
                int(sample_id): i
                for i, sample_id in enumerate(self.ids.tolist())
            }
        )

    def update(
        self,
        ids: torch.Tensor,
        soft: torch.Tensor,
        confidence: torch.Tensor,
        selected: torch.Tensor,
    ) -> None:
        ids = ids.long().cpu()
        label = soft.argmax(dim=1).cpu()
        if self.ids is None or not torch.equal(ids, self.ids):
            self.ids = ids
            self.id_to_position = {
                int(sample_id): i for i, sample_id in enumerate(ids.tolist())
            }
            self.last_label = label.clone()
            self.stable_count = torch.ones_like(label)
        else:
            same = label == self.last_label
            self.stable_count = torch.where(
                same, self.stable_count + 1, torch.ones_like(self.stable_count)
            )
            self.last_label = label.clone()

        self.soft = soft.cpu()
        self.confidence = confidence.cpu()
        self.selected = selected.cpu()
        self.stable = selected.cpu() & (
            self.stable_count >= self.stable_rounds
        )

    def get(
        self, sample_ids: torch.Tensor, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.soft is None:
            raise RuntimeError("Pseudo-label bank is empty.")
        positions = torch.tensor(
            [
                self.id_to_position[int(sample_id)]
                for sample_id in torch.as_tensor(sample_ids).tolist()
            ],
            dtype=torch.long,
        )
        return (
            self.soft[positions].to(device),
            self.confidence[positions].to(device),
            self.selected[positions].to(device),
            self.stable[positions].to(device),
        )


def class_balanced_selection(
    probs: torch.Tensor,
    confidence: torch.Tensor,
    active_classes: torch.Tensor,
    coverage: float,
    threshold: float,
    min_per_class: int,
    dataset: str = "pavia",
) -> torch.Tensor:
    if dataset == "pavia":
        label = probs.argmax(dim=1)
        selected = torch.zeros(len(probs), dtype=torch.bool)
        for c in active_classes.tolist():
            candidates = torch.where(label == c)[0]
            if len(candidates) == 0:
                continue
            ranked = candidates[confidence[candidates].argsort(descending=True)]
            take = min(
                len(ranked),
                max(min_per_class, int(math.ceil(coverage * len(ranked)))),
            )
            chosen = ranked[:take]
            chosen_above_threshold = chosen[confidence[chosen] >= threshold]
            if len(chosen_above_threshold) == 0:

                chosen_above_threshold = ranked[: min(min_per_class, len(ranked))]
            selected[chosen_above_threshold] = True
        return selected
    selected = torch.zeros(len(probs), dtype=torch.bool)
    used = torch.zeros(len(probs), dtype=torch.bool)
    active_count = max(1, len(active_classes))
    per_class_budget = max(
        min_per_class, int(math.ceil(coverage * len(probs) / active_count))
    )
    for c in active_classes.tolist():
        class_score = probs[:, c] * (0.5 + 0.5 * confidence)
        ranked = class_score.argsort(descending=True)
        available = ranked[~used[ranked]]
        above = available[class_score[available] >= threshold]
        chosen = above[:per_class_budget]
        if len(chosen) < min_per_class:
            chosen = available[:min(min_per_class, len(available))]
        selected[chosen] = True
        used[chosen] = True
    return selected


def _entropy_confidence(
    active_probs: torch.Tensor,
) -> torch.Tensor:
    entropy = -(
        active_probs.clamp_min(1e-8) * active_probs.clamp_min(1e-8).log()
    ).sum(dim=1)
    denominator = math.log(max(2, active_probs.size(1)))
    return (1.0 - entropy / denominator).clamp(0.0, 1.0)


@torch.no_grad()
def refresh_global_pseudo_labels(
    ema_teacher: nn.Module,
    stage_anchor: nn.Module,
    loader,
    source_multi_prototypes: torch.Tensor,
    bank: PseudoLabelBank,
    expected_class_count: int,
    epoch: int,
    total_epochs: int,
    cfg: TargetConfig,
    device: torch.device,
    excluded_classes: Optional[Sequence[int]] = None,
    forced_active_classes: Optional[Sequence[int]] = None,
) -> Dict[str, torch.Tensor]:
    output = extract_stage_outputs(
        ema_teacher, stage_anchor, loader, device
    )
    features = output["features"]

    if (
        output["old_logits"].size(1) != cfg.num_classes
        or output["ema_logits"].size(1) != cfg.num_classes
    ):
        raise RuntimeError(
            "Model logits must match TargetConfig.num_classes "
            f"({cfg.num_classes})."
        )
    if source_multi_prototypes.size(0) != cfg.num_classes:
        raise RuntimeError(
            "The source prototype class count must equal TargetConfig.num_classes "
            f"({cfg.num_classes}); got {source_multi_prototypes.size(0)}."
        )
    old_probs = prior_corrected_softmax(output["old_logits"], cfg)
    ema_probs = prior_corrected_softmax(output["ema_logits"], cfg)

    proto_logits_raw = source_prototype_logits(
        features, source_multi_prototypes, cfg.pseudo_temperature
    )
    proto_probs = F.softmax(proto_logits_raw, dim=1)

    discovered_active_classes, class_score = discover_active_classes(
        old_probs,
        proto_probs,
        expected_class_count,
        excluded_classes,
    )
    if forced_active_classes is None:
        active_classes = discovered_active_classes
    else:
        forced = sorted({int(c) for c in forced_active_classes})
        if len(forced) != expected_class_count:
            raise ValueError(
                "forced_active_classes must contain exactly "
                f"{expected_class_count} unique classes, got {forced}."
            )
        invalid = [c for c in forced if c < 0 or c >= cfg.num_classes]
        if invalid:
            raise ValueError(f"Invalid forced active classes: {invalid}.")
        excluded = set(int(c) for c in (excluded_classes or []))
        overlap = sorted(excluded.intersection(forced))
        if overlap:
            raise ValueError(
                "Forced active classes overlap excluded old classes: "
                f"{overlap}."
            )
        active_classes = torch.tensor(forced, dtype=torch.long)

    uot_probs, _ = target_cluster_uot(
        features,
        old_probs,
        source_multi_prototypes,
        active_classes,
        cfg,
    )

    progress = epoch / max(1, total_epochs - 1)

    w_old = 0.20 - 0.05 * progress
    w_ema = 0.10 + 0.30 * progress
    w_proto = 0.35 - 0.15 * progress
    w_uot = 1.0 - w_old - w_ema - w_proto
    log_fused = (
        w_old * old_probs.clamp_min(1e-8).log()
        + w_ema * ema_probs.clamp_min(1e-8).log()
        + w_proto * proto_probs.clamp_min(1e-8).log()
        + w_uot * uot_probs.clamp_min(1e-8).log()
    )
    inactive = torch.ones(cfg.num_classes, dtype=torch.bool)
    inactive[active_classes] = False
    log_fused[:, inactive] = -30.0
    fused = F.softmax(log_fused, dim=1)
    fused, neighbor_agreement = knn_probability_smoothing(
        features, fused, cfg.knn_k
    )

    predicted = fused.argmax(dim=1)
    top2 = fused[:, active_classes].topk(
        k=min(2, len(active_classes)), dim=1
    ).values
    margin = (
        top2[:, 0]
        if top2.size(1) == 1
        else top2[:, 0] - top2[:, 1]
    )
    source_agreement = (
        (predicted == old_probs.argmax(dim=1)).float()
        + (predicted == proto_probs.argmax(dim=1)).float()
        + (predicted == uot_probs.argmax(dim=1)).float()
    ) / 3.0
    confidence = (
        0.35 * _entropy_confidence(fused[:, active_classes])
        + 0.25 * margin
        + 0.25 * neighbor_agreement
        + 0.15 * source_agreement
    ).clamp(0.0, 1.0)

    coverage = cfg.initial_coverage + progress * (
        cfg.final_coverage - cfg.initial_coverage
    )
    selected = class_balanced_selection(
        fused,
        confidence,
        active_classes,
        coverage,
        cfg.confidence_threshold,
        cfg.min_samples_per_class,
        dataset=cfg.dataset,
    )
    bank.update(output["ids"], fused, confidence, selected)

    active_distribution = fused[:, active_classes]
    mean_entropy = -(
        active_distribution.clamp_min(1e-8)
        * active_distribution.clamp_min(1e-8).log()
    ).sum(dim=1).mean()
    mean_prediction = active_distribution.mean(dim=0)
    diversity = -(
        mean_prediction.clamp_min(1e-8)
        * mean_prediction.clamp_min(1e-8).log()
    ).sum()
    selected_confidence = (
        confidence[selected].mean() if selected.any() else confidence.new_zeros(())
    )
    stable_ratio = bank.stable.float().mean()
    selection_score = (
        diversity - mean_entropy
        + 0.25 * selected_confidence
        + 0.05 * stable_ratio
    )

    return {
        "soft": bank.soft.clone(),
        "confidence": bank.confidence.clone(),
        "selected": bank.selected.clone(),
        "stable": bank.stable.clone(),
        "class_score": class_score,
        "active_classes": active_classes,
    }


@torch.no_grad()
def label_free_prediction_score(
    model: nn.Module,
    loader,
    active_classes: Sequence[int],
    device: torch.device,
) -> float:
    classifier_class_count = model.classifier.out_features
    valid_active = sorted({
        int(c) for c in active_classes
        if 0 <= int(c) < classifier_class_count
    })
    active = torch.tensor(valid_active, dtype=torch.long, device=device)
    if active.numel() == 0:
        return -float("inf")
    model.eval()
    probability_sum = torch.zeros(len(active), device=device)
    entropy_sum = 0.0
    count = 0
    for images, _ in loader:
        images = images.to(device, non_blocking=True).float()
        logits, _ = model(images)
        probs = F.softmax(logits[:, active], dim=1)
        entropy_sum += float(
            (-(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(dim=1)).sum()
        )
        probability_sum += probs.sum(dim=0)
        count += len(images)
    if count == 0:
        return -float("inf")
    mean_entropy = entropy_sum / count
    mean_probability = probability_sum / count
    diversity = float(
        -(mean_probability.clamp_min(1e-8) * mean_probability.clamp_min(1e-8).log()).sum()
    )
    return diversity - mean_entropy


class ReliableTargetMemory:
    def __init__(self, num_classes: int, memory_per_class: int):
        self.num_classes = num_classes
        self.memory_per_class = memory_per_class
        self.memory: Dict[int, List[Dict[str, Union[torch.Tensor, float]]]] = {
            c: [] for c in range(num_classes)
        }

    def class_counts(self) -> Dict[int, int]:
        return {c: len(self.memory[c]) for c in range(self.num_classes)}

    def state_dict(self):
        return {
            "num_classes": self.num_classes,
            "memory_per_class": self.memory_per_class,
            "memory": self.memory,
        }

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    @staticmethod
    def stable_available(bank: PseudoLabelBank) -> bool:
        return bank.stable is not None and bool(bank.stable.any())

    @torch.no_grad()
    def update_from_bank(
        self,
        model: nn.Module,
        loader,
        bank: PseudoLabelBank,
        current_classes: Sequence[int],
        device: torch.device,
    ) -> None:
        model.eval()
        candidates = {int(c): [] for c in current_classes}
        use_stable_only = bool(self.stable_available(bank))
        for images, sample_ids in loader:
            images_device = images.to(device).float()
            logits, features = model(images_device)
            soft, confidence, selected, stable = bank.get(sample_ids, device)
            reliable = stable if use_stable_only else selected
            label = soft.argmax(dim=1)
            for i in torch.where(reliable)[0].tolist():
                c = int(label[i])
                if c not in candidates:
                    continue
                candidates[c].append(
                    {
                        "score": float(confidence[i].cpu()),
                        "image": images[i].cpu().clone(),
                        "label": torch.tensor(c, dtype=torch.long),
                        "feature": features[i].cpu().clone(),
                        "logit": logits[i].cpu().clone(),
                    }
                )

        for c in current_classes:
            c = int(c)
            merged = self.memory[c] + candidates[c]
            merged.sort(key=lambda item: item["score"], reverse=True)
            self.memory[c] = merged[: self.memory_per_class]

    def sample(
        self,
        batch_size: int,
        classes: Sequence[int],
        device: torch.device,
    ):
        available = [int(c) for c in classes if len(self.memory[int(c)]) > 0]
        if not available:
            return None
        per_class = max(1, batch_size // len(available))
        selected = []
        for c in available:
            selected.extend(
                random.choice(self.memory[c]) for _ in range(per_class)
            )
        all_items = [item for c in available for item in self.memory[c]]
        while len(selected) < batch_size:
            selected.append(random.choice(all_items))
        selected = selected[:batch_size]
        return (
            torch.stack([item["image"] for item in selected]).to(device).float(),
            torch.stack([item["label"] for item in selected]).to(device).long(),
            torch.stack([item["feature"] for item in selected]).to(device).float(),
            torch.stack([item["logit"] for item in selected]).to(device).float(),
        )


def train_target_epoch(
    student: nn.Module,
    ema_teacher: nn.Module,
    stage_anchor: nn.Module,
    loader,
    optimizer,
    bank: PseudoLabelBank,
    source_multi_prototypes: torch.Tensor,
    memory: ReliableTargetMemory,
    old_classes: Sequence[int],
    cfg: TargetConfig,
    device: torch.device,
) -> Dict[str, float]:
    student.train()
    ema_teacher.eval()
    stage_anchor.eval()
    source_multi_prototypes = source_multi_prototypes.to(device)

    totals = {
        "loss": 0.0,
        "pl": 0.0,
        "cons": 0.0,
        "src_proto": 0.0,
        "memory": 0.0,
        "topology": 0.0,
        "selected": 0.0,
        "samples": 0.0,
    }

    pseudo_class_weights = torch.ones(
        cfg.num_classes, dtype=torch.float32, device=device
    )
    if bank.soft is not None and bank.selected is not None:
        bank_labels = bank.soft.argmax(dim=1)
        bank_selected = bank.selected.bool()
        if bank_selected.any():
            counts = torch.bincount(
                bank_labels[bank_selected], minlength=cfg.num_classes
            ).float()
            present = counts > 0
            reference = counts[present].mean()
            pseudo_class_weights[present] = (
                reference / counts[present]
            ).pow(float(cfg.pseudo_balance_power)).to(device)
            pseudo_class_weights[present] = pseudo_class_weights[present].clamp(
                max=float(cfg.pseudo_balance_max_weight)
            )
            pseudo_class_weights[present] /= pseudo_class_weights[present].mean().clamp_min(1e-8)

    for images, sample_ids in loader:
        images = images.to(device, non_blocking=True).float()
        soft, confidence, selected, stable = bank.get(sample_ids, device)
        weak = weak_hsi_augment(images)
        strong, _ = strong_hsi_augment(
            images, cfg.spectral_mask_ratio * 0.5
        )

        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            teacher_logits, _ = ema_teacher(weak)
            active_mask = soft.sum(dim=0) > 1e-6
            masked_teacher_logits = teacher_logits / cfg.teacher_temperature
            masked_teacher_logits[:, ~active_mask] = -30.0
            teacher_probs = F.softmax(masked_teacher_logits, dim=1)

        logits, features = student(strong)
        if logits.size(1) != cfg.num_classes or soft.size(1) != cfg.num_classes:
            raise RuntimeError("Classifier logits and pseudo-label dimensions disagree.")
        pseudo_label = soft.argmax(dim=1)

        if cfg.dataset == "houston":
            weight = confidence * selected.float()
            weight = weight * torch.where(
                stable, torch.ones_like(weight), torch.full_like(weight, 0.5)
            )
            sample_balance = pseudo_class_weights[pseudo_label]
            balanced_weight = weight * sample_balance
            active_idx = torch.where(soft.sum(dim=0) > 1e-6)[0]
            if selected.any() and active_idx.numel() > 0:
                local_target = soft[:, active_idx]
                local_target = local_target / local_target.sum(dim=1, keepdim=True).clamp_min(1e-8)
                pl_loss = soft_cross_entropy(logits[:, active_idx], local_target, balanced_weight)

                local_teacher = teacher_probs[:, active_idx]
                local_teacher = local_teacher / local_teacher.sum(dim=1, keepdim=True).clamp_min(1e-8)
                cons_per = F.kl_div(
                    F.log_softmax(logits[:, active_idx], dim=1), local_teacher,
                    reduction="none",
                ).sum(dim=1)
                cons_weight = (
                    confidence * selected.float() * sample_balance
                ).detach()
                consistency_loss = (cons_per * cons_weight).sum() / cons_weight.sum().clamp_min(1e-6)
            else:
                pl_loss = logits.new_zeros(())
                consistency_loss = logits.new_zeros(())

            proto_logits = source_prototype_logits(
                F.normalize(features, dim=1),
                source_multi_prototypes,
                cfg.pseudo_temperature,
            )
            if selected.any():
                source_proto_loss = soft_cross_entropy(
                    proto_logits,
                    soft,
                    weight,
                )
            else:
                source_proto_loss = logits.new_zeros(())
        else:
            sample_balance = pseudo_class_weights[pseudo_label]
            weight = confidence * selected.float() * sample_balance
            weight = weight * torch.where(
                stable, torch.ones_like(weight), torch.full_like(weight, 0.5)
            )
            pl_loss = (
                soft_cross_entropy(logits, soft, weight)
                if selected.any()
                else logits.new_zeros(())
            )

            per_sample_kl = F.kl_div(
                F.log_softmax(logits, dim=1),
                teacher_probs,
                reduction="none",
            ).sum(dim=1)

            consistency_weight = sample_balance
            consistency_loss = (per_sample_kl * consistency_weight).sum() / (
                consistency_weight.sum().clamp_min(1e-6)
            )

            proto_logits = source_prototype_logits(
                F.normalize(features, dim=1),
                source_multi_prototypes,
                cfg.pseudo_temperature,
            )
            source_proto_loss = (
                soft_cross_entropy(proto_logits, soft, weight)
                if selected.any()
                else logits.new_zeros(())
            )

        memory_batch = memory.sample(
            cfg.memory_batch_size, old_classes, device
        )
        memory_loss = reliable_memory_loss(
            student, memory_batch, old_classes, cfg.kd_temperature,
            kd_weight=0.0 if cfg.dataset == "houston" else 0.5
        )
        topology_loss = classifier_topology_loss(
            student, stage_anchor, old_classes
        )

        pl_loss = torch.nan_to_num(pl_loss, nan=0.0, posinf=10.0, neginf=0.0)
        consistency_loss = torch.nan_to_num(consistency_loss, nan=0.0, posinf=10.0, neginf=0.0)
        source_proto_loss = torch.nan_to_num(source_proto_loss, nan=0.0, posinf=10.0, neginf=0.0)
        memory_loss = torch.nan_to_num(memory_loss, nan=0.0, posinf=10.0, neginf=0.0)
        topology_loss = torch.nan_to_num(topology_loss, nan=0.0, posinf=10.0, neginf=0.0)

        loss = (
            cfg.lambda_pl * pl_loss
            + cfg.lambda_cons * consistency_loss
            + cfg.lambda_src_proto * source_proto_loss
            + cfg.lambda_memory * memory_loss
            + cfg.lambda_topology * topology_loss
        )

        loss = loss + 0.0 * logits.sum()
        loss.backward()
        for parameter in student.parameters():
            if parameter.grad is not None:
                parameter.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        nn.utils.clip_grad_norm_(student.parameters(), cfg.max_grad_norm)
        optimizer.step()
        ema_update(ema_teacher, student, cfg.ema_momentum)

        batch_size = len(images)
        totals["loss"] += float(loss.detach()) * batch_size
        totals["pl"] += float(pl_loss.detach()) * batch_size
        totals["cons"] += float(consistency_loss.detach()) * batch_size
        totals["src_proto"] += float(source_proto_loss.detach()) * batch_size
        totals["memory"] += float(memory_loss.detach()) * batch_size
        totals["topology"] += float(topology_loss.detach()) * batch_size
        totals["selected"] += int(selected.sum())
        totals["samples"] += batch_size

    denominator = max(1.0, totals["samples"])
    return {
        key: (value / denominator if key not in {"selected", "samples"} else value)
        for key, value in totals.items()
    }
