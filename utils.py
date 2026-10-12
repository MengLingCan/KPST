from __future__ import annotations
import logging
import math
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union


def setup_logger(save_path, file_name):
    os.makedirs(save_path, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(
        os.path.join(save_path, file_name), mode="w", encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def val_net(net, loader, device=None):
    net.eval()
    if device is None:
        device = next(net.parameters()).device
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True).float()
            labels = labels.to(device, non_blocking=True).long()
            outputs, _ = net(inputs)
            predicted = torch.argmax(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = correct / total if total > 0 else 0.0
    return round(acc, 4)


def val_per_label(net, single_label_loads):
    accs = []
    for i in range(len(single_label_loads)):
        accs.append(val_net(net, single_label_loads[i]))
    return accs


def compute_AA(net, single_label_loads):
    accs = val_per_label(net, single_label_loads)
    return round(sum(accs) / len(accs), 4) , accs


def compute_HAS(net, tar_loader, single_cls_tar_loader, single_cls_src_loader):
    overall_acc = val_net(net, tar_loader)
    average_acc, accs = compute_AA(net, single_cls_tar_loader)
    acc_curr = (2 * overall_acc * average_acc) / (overall_acc + average_acc)

    recall_past, _ = compute_AA(net, single_cls_src_loader)
    has = (2 * acc_curr * recall_past) / (acc_curr + recall_past)

    return round(has, 6), overall_acc, average_acc, accs


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_possible_class(model, loader, num_classes=7, top_n=2, top_k=100):
    model.eval()
    class_confidences = {i: [] for i in range(num_classes)}

    for img, *_ in loader:
        img = img.cuda()
        with torch.no_grad():
            output, _ = model(img)

        probs = F.softmax(output, dim=1)
        max_confs, preds = torch.max(probs, dim=1)

        for c in range(num_classes):
            mask = (preds == c)
            if mask.sum() > 0:
                confs = max_confs[mask].cpu().tolist()
                class_confidences[c].extend(confs)

    class_scores = torch.zeros(num_classes).cuda()
    for c in range(num_classes):
        confs_list = class_confidences[c]
        if len(confs_list) > 0:
            confs_list.sort(reverse=True)
            top_k_confs = confs_list[:top_k]
            class_scores[c] = sum(top_k_confs)

    sort_bank, sort_class_idx = torch.sort(class_scores, descending=True)
    possible_class = sort_class_idx[:top_n].tolist()

    max_score = sort_bank[0]
    min_score = sort_bank[-1]
    prediction_bank = class_scores.clone()
    if max_score > min_score:
        prediction_bank = (prediction_bank - min_score) / (max_score - min_score)

    return possible_class, prediction_bank


@torch.no_grad()
def get_pseudo_labels(
    teacher_model,
    images,
    possible_classes,
):
    teacher_model.eval()

    device = images.device
    possible_classes = torch.tensor(
        possible_classes,
        device=device,
        dtype=torch.long
    )

    logits, features = teacher_model(images)

    logits_sub = logits[:, possible_classes]
    probs_sub = F.softmax(logits_sub, dim=1)

    confidence, pred_local = probs_sub.max(dim=1)
    pseudo_labels = possible_classes[pred_local]

    entropy = -(probs_sub * torch.log(probs_sub + 1e-6)).sum(dim=1)

    return pseudo_labels, entropy, confidence


@torch.no_grad()
def compute_unsup_score(model, loader, seen_classes, device=None):
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    seen = torch.tensor(sorted(seen_classes), dtype=torch.long, device=device)

    ent_sum = 0.0
    n = 0
    p_bar = torch.zeros(len(seen), device=device)

    for batch in loader:
        images = batch[0].to(device).float()
        logits, _ = model(images)
        logits_sub = logits[:, seen]
        probs = F.softmax(logits_sub, dim=1)

        ent = -(probs * torch.log(probs + 1e-6)).sum(dim=1)
        ent_sum += float(ent.sum().item())
        p_bar += probs.sum(dim=0)
        n += probs.size(0)

    if n == 0:
        return 0.0

    mean_ent = ent_sum / n
    p_bar = p_bar / n
    div = float(-(p_bar * torch.log(p_bar + 1e-6)).sum().item())

    return div - mean_ent


@torch.no_grad()
def ema_update_teacher(teacher, student, momentum=0.99):
    for t_param, s_param in zip(teacher.parameters(), student.parameters()):
        t_param.data.mul_(momentum).add_(s_param.data, alpha=1.0 - momentum)

    for t_buf, s_buf in zip(teacher.buffers(), student.buffers()):
        t_buf.data.copy_(s_buf.data)


def InfoNCELoss(fea_a, fea_b, temperature=0.07):
    batch_size = fea_a.shape[0]

    fea_a = F.normalize(fea_a, p=2, dim=1)
    fea_b = F.normalize(fea_b, p=2, dim=1)
    similarity_matrix = torch.matmul(fea_a, fea_b.T) / temperature

    labels = torch.arange(batch_size, device=fea_a.device)
    loss = F.cross_entropy(similarity_matrix, labels)

    return loss


def _get_logits(outputs):
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    return outputs


def _confusion_from_loader(model, loader, num_classes, device=None):
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for batch in loader:
            images = batch[0]
            labels = batch[1]

            images = images.to(device).float()
            labels = labels.to(device).long()

            logits = _get_logits(model(images))
            preds = torch.argmax(logits, dim=1)

            labels_np = labels.detach().cpu().numpy()
            preds_np = preds.detach().cpu().numpy()

            for y_true, y_pred in zip(labels_np, preds_np):
                if 0 <= y_true < num_classes and 0 <= y_pred < num_classes:
                    confusion[y_true, y_pred] += 1

    return confusion


def evaluate(
    model,
    target_loader,
    num_classes,
    device=None,
    source_loader=None,
    source_baseline=None,
    best_target_per_class=None,
    seen_classes=None,
    current_classes=None,
    beta=0.0,
    gamma=0.05,
):
    target_confusion = _confusion_from_loader(
        model=model,
        loader=target_loader,
        num_classes=num_classes,
        device=device,
    )

    total = target_confusion.sum()
    correct = np.trace(target_confusion)

    target_oa = correct / total if total > 0 else 0.0

    row_sum = target_confusion.sum(axis=1)
    tp = np.diag(target_confusion)

    target_per_class_acc = np.full(num_classes, np.nan, dtype=np.float64)

    for c in range(num_classes):
        if row_sum[c] > 0:
            target_per_class_acc[c] = tp[c] / row_sum[c]

    if seen_classes is None:
        seen_classes = [
            c for c in range(num_classes)
            if not np.isnan(target_per_class_acc[c])
        ]

    if current_classes is None:
        current_classes = []

    seen_accs = [
        target_per_class_acc[c]
        for c in seen_classes
        if not np.isnan(target_per_class_acc[c])
    ]

    target_aa = float(np.mean(seen_accs)) if len(seen_accs) > 0 else 0.0

    old_classes = [
        c for c in seen_classes
        if c not in current_classes
    ]

    target_forgetting_list = []

    if best_target_per_class is not None:
        for c in old_classes:
            old_best = best_target_per_class[c]
            current_acc = target_per_class_acc[c]

            if not np.isnan(old_best) and not np.isnan(current_acc):
                target_forgetting_list.append(
                    max(0.0, float(old_best - current_acc))
                )

    target_forgetting = (
        float(np.mean(target_forgetting_list))
        if len(target_forgetting_list) > 0
        else 0.0
    )

    source_oa = None
    source_aa = None
    source_forgetting = 0.0

    if source_loader is not None:
        source_confusion = _confusion_from_loader(
            model=model,
            loader=source_loader,
            num_classes=num_classes,
            device=device,
        )

        source_total = source_confusion.sum()
        source_correct = np.trace(source_confusion)
        source_oa = source_correct / source_total if source_total > 0 else 0.0

        source_row_sum = source_confusion.sum(axis=1)
        source_tp = np.diag(source_confusion)

        source_per_class_acc = np.full(num_classes, np.nan, dtype=np.float64)

        for c in range(num_classes):
            if source_row_sum[c] > 0:
                source_per_class_acc[c] = source_tp[c] / source_row_sum[c]

        source_aa = (
            float(np.nanmean(source_per_class_acc))
            if np.any(~np.isnan(source_per_class_acc))
            else 0.0
        )

        if source_baseline is not None:
            if isinstance(source_baseline, dict):
                source_base_aa = float(source_baseline["AA"])
            else:
                source_base_aa = float(source_baseline)

            source_forgetting = max(0.0, source_base_aa - source_aa)

    save_score = target_aa - beta * target_forgetting - gamma * source_forgetting

    return {
        "save_score": float(save_score),
        "target_OA": float(target_oa),
        "target_AA": float(target_aa),
        "target_forgetting": float(target_forgetting),
        "source_OA": None if source_oa is None else float(source_oa),
        "source_AA": None if source_aa is None else float(source_aa),
        "source_forgetting": float(source_forgetting),
        "target_per_class_acc": target_per_class_acc,
        "target_confusion_matrix": target_confusion,
    }


class BaseAugmentation:
    def __init__(self, flip_prob=0.6, rot_prob=0.75, noise_prob=0.7, noise_std=0.01, band_drop_prob=0.7, drop_size=5):
        self.flip_prob = flip_prob
        self.rot_prob = rot_prob
        self.noise_prob = noise_prob
        self.noise_std = noise_std
        self.band_drop_prob = band_drop_prob
        self.drop_size = drop_size

    def __call__(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        if random.random() < self.flip_prob:
            dim = random.choice([1, 2])
            x = torch.flip(x, dims=[dim])

        if random.random() < self.rot_prob:
            k = random.choice([1, 2, 3])
            x = torch.rot90(x, k, [1, 2])

        if random.random() < self.noise_prob:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise

        if random.random() < self.band_drop_prob:
            c, h, w = x.shape
            if c > self.drop_size:
                start_band = random.randint(0, c - self.drop_size)
                x[start_band : start_band + self.drop_size, :, :] = 0

        return x


class CenterRepulsionLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, margin=20.0, device=torch.device('cuda')):
        super(CenterRepulsionLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.margin = margin
        self.device = device

        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(self.device))

    def forward(self, x, labels):
        batch_size = x.size(0)

        centers_batch = self.centers[labels]
        loss_center = (x - centers_batch).pow(2).sum() / 2.0 / batch_size

        centers_sq = torch.sum(self.centers.pow(2), dim=1, keepdim=True)
        dist_mat = centers_sq + centers_sq.t() - 2.0 * torch.matmul(self.centers, self.centers.t())

        dist_mat = torch.clamp(dist_mat, min=1e-8)

        repulsion_loss_mat = torch.clamp(self.margin - dist_mat, min=0.0)

        mask = torch.eye(self.num_classes, device=self.device)
        repulsion_loss_mat = repulsion_loss_mat * (1 - mask)

        num_pairs = self.num_classes * (self.num_classes - 1)
        loss_repulsion = repulsion_loss_mat.sum() / num_pairs

        return loss_center, loss_repulsion


class PrototypeManager:
    def __init__(self, num_classes, num_incre=3, feature_dim=32, alpha=0.3, device='cuda'):
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.alpha = alpha
        self.device = device
        self.num_incre = num_incre
        self.incre_idx = -1

        self.source_prototypes = torch.zeros(num_classes, feature_dim, device=device)
        self.target_prototypes = torch.zeros(num_classes, feature_dim, device=device)

    def init_source_prototypes(self, source_features, source_labels):
        for c in range(self.num_classes):
            mask = (source_labels == c)
            if mask.sum() > 0:
                class_features = source_features[mask]
                mean_feature = class_features.mean(dim=0)
                self.source_prototypes[c] = F.normalize(mean_feature.unsqueeze(0), p=2, dim=1).squeeze(0)

    def init_source_multi_prototypes(self, source_features, source_labels, prototypes_per_class=3, random_state=0):
        source_features = source_features.detach().cpu().float()
        source_labels = source_labels.detach().cpu().long()
        all_prototypes = []

        for c in range(self.num_classes):
            class_features = source_features[source_labels == c]
            if len(class_features) == 0:
                fallback = self.source_prototypes[c].detach().cpu().float()
                if torch.linalg.norm(fallback) < 1e-8:
                    fallback = torch.zeros(self.feature_dim)
                    fallback[0] = 1.0
                centers = fallback.unsqueeze(0).repeat(prototypes_per_class, 1)
            else:
                k = min(int(prototypes_per_class), len(class_features))
                if k == 1:
                    centers = class_features.mean(dim=0, keepdim=True)
                else:
                    kmeans = MiniBatchKMeans(
                        n_clusters=k,
                        batch_size=min(2048, len(class_features)),
                        random_state=random_state,
                        n_init=10,
                    )
                    centers = torch.from_numpy(
                        kmeans.fit(class_features.numpy()).cluster_centers_
                    ).float()
                if k < prototypes_per_class:
                    centers = torch.cat([
                        centers,
                        centers[:1].repeat(prototypes_per_class - k, 1),
                    ], dim=0)
            all_prototypes.append(F.normalize(centers, p=2, dim=1))

        self.source_multi_prototypes = torch.stack(all_prototypes, dim=0)
        return self.source_multi_prototypes

    def save_source_multi_prototypes(self, save_path, prototypes_per_class):
        if not hasattr(self, 'source_multi_prototypes'):
            raise RuntimeError('Call init_source_multi_prototypes before saving.')
        dir_name = os.path.dirname(save_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name)
        torch.save({
            'prototypes': self.source_multi_prototypes.cpu(),
            'num_classes': self.num_classes,
            'feature_dim': self.feature_dim,
            'prototypes_per_class': int(prototypes_per_class),
        }, save_path)

    def save_source_prototypes(self, save_path):
        dir_name = os.path.dirname(save_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name)

        state_dict = {
            'source_prototypes': self.source_prototypes.cpu(),
            'num_classes': self.num_classes,
            'feature_dim': self.feature_dim
        }
        torch.save(state_dict, save_path)

    def load_source_prototypes(self, load_path):
        state_dict = torch.load(load_path, map_location=self.device)

        assert state_dict['num_classes'] == self.num_classes, \
            f"class_num isn't matched! {self.num_classes} needed, but {state_dict['num_classes']} given"
        assert state_dict['feature_dim'] == self.feature_dim, \
            f"dim isn't matched! {self.feature_dim} needed, but {state_dict['feature_dim']} given"

        self.source_prototypes = state_dict['source_prototypes'].to(self.device)

    def init_target_prototypes(self, classifier_weights):
        self.incre_idx += 1
        self.target_prototypes = F.normalize(classifier_weights.detach(), p=2, dim=1).to(self.device)

    def update_target_prototypes(self, reliable_features, pseudo_labels, allowed_classes=None):
        reliable_features = reliable_features.detach()

        if allowed_classes is None:
            allowed_classes = list(range(self.num_classes))
        else:
            allowed_classes = [int(c) for c in allowed_classes]

        for c in allowed_classes:
            mask = (pseudo_labels == c)
            if mask.sum() > 0:
                class_features = reliable_features[mask]

                mean_feature = class_features.mean(dim=0)
                norm_feature = F.normalize(
                    mean_feature.unsqueeze(0),
                    p=2,
                    dim=1
                ).squeeze(0)

                self.target_prototypes[c] = (
                    self.alpha * self.target_prototypes[c]
                    + (1 - self.alpha) * norm_feature
                )

                self.target_prototypes[c] = F.normalize(
                    self.target_prototypes[c].unsqueeze(0),
                    p=2,
                    dim=1
                ).squeeze(0)

    def compute_ema_loss(self, features, pseudo_labels):
        norm_features = F.normalize(features, p=2, dim=1)
        norm_prototypes = F.normalize(self.target_prototypes, p=2, dim=1)

        logits = torch.matmul(norm_features, norm_prototypes.T)

        temperature = 10.0
        return F.cross_entropy(logits * temperature, pseudo_labels)

    def compute_src_loss(self, features, pseudo_labels, confidence=None,
                         temperature=0.5, conf_threshold=0.0):
        norm_features = F.normalize(features, p=2, dim=1)
        norm_protos = F.normalize(self.source_prototypes, p=2, dim=1)

        logits = torch.matmul(norm_features, norm_protos.T) / temperature

        loss = F.cross_entropy(logits, pseudo_labels, reduction='none')

        if confidence is not None:
            mask = (confidence >= conf_threshold).float()
            weight = confidence * mask
            denom = weight.sum().clamp_min(1e-6)
            return (loss * weight).sum() / denom

        return loss.mean()


class BalancedSoftmaxLoss(nn.Module):
    def __init__(
        self,
        class_counts: torch.Tensor,
        tau: float = 1.0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        counts = torch.as_tensor(class_counts, dtype=torch.float32).clamp_min(1.0)
        self.register_buffer("log_counts", counts.log())
        self.tau = float(tau)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        adjusted_logits = logits + self.tau * self.log_counts.unsqueeze(0)
        return F.cross_entropy(
            adjusted_logits,
            target,
            label_smoothing=self.label_smoothing,
        )


def safe_torch_load(path: Union[str, Path], device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _spatial_transform(x: torch.Tensor) -> torch.Tensor:
    if random.random() < 0.5:
        x = x.flip(-1)
    if random.random() < 0.5:
        x = x.flip(-2)
    return torch.rot90(x, random.randint(0, 3), dims=(-2, -1))


def weak_hsi_augment(x: torch.Tensor) -> torch.Tensor:
    x = _spatial_transform(x)
    n = x.size(0)
    gain = 1.0 + 0.015 * torch.randn(n, 1, 1, 1, device=x.device)
    bias = 0.005 * torch.randn(n, 1, 1, 1, device=x.device)
    return x * gain + bias + 0.003 * torch.randn_like(x)


def strong_hsi_augment(
    x: torch.Tensor,
    mask_ratio: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x = _spatial_transform(x)
    n, bands, _, _ = x.shape
    gain = 1.0 + 0.05 * torch.randn(n, 1, 1, 1, device=x.device)
    bias = 0.015 * torch.randn(n, 1, 1, 1, device=x.device)
    x = x * gain + bias + 0.01 * torch.randn_like(x)

    mask = torch.zeros(n, bands, dtype=torch.bool, device=x.device)
    width = max(1, min(bands, int(round(mask_ratio * bands))))
    for i in range(n):
        start = random.randint(0, bands - width)
        mask[i, start:start + width] = True
    return x.masked_fill(mask[:, :, None, None], 0.0), mask


@torch.no_grad()
def ema_update(teacher: nn.Module, student: nn.Module, momentum: float) -> None:
    for p_t, p_s in zip(teacher.parameters(), student.parameters()):
        p_t.data.mul_(momentum).add_(p_s.data, alpha=1.0 - momentum)
    for b_t, b_s in zip(teacher.buffers(), student.buffers()):
        b_t.copy_(b_s)


def soft_cross_entropy(
    logits: torch.Tensor,
    target_probs: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    per_sample = -(target_probs * F.log_softmax(logits, dim=1)).sum(dim=1)
    if weights is None:
        return per_sample.mean()
    return (per_sample * weights).sum() / weights.sum().clamp_min(1e-6)


def prior_corrected_softmax(
    logits: torch.Tensor,
    cfg,
) -> torch.Tensor:
    tau = float(cfg.source_prior_correction_tau)
    prior = cfg.source_class_prior
    if prior is None or tau <= 0.0:
        return F.softmax(logits, dim=1)

    prior = torch.as_tensor(prior, dtype=logits.dtype, device=logits.device)
    if prior.numel() != logits.size(1):
        raise ValueError(
            f"source_class_prior has {prior.numel()} entries but logits have "
            f"{logits.size(1)} classes."
        )
    prior = prior.clamp_min(1e-8)
    prior = prior / prior.sum()
    corrected_logits = logits - tau * prior.log().unsqueeze(0)
    return F.softmax(corrected_logits, dim=1)


def variance_loss(z: torch.Tensor) -> torch.Tensor:
    std = torch.sqrt(z.var(dim=0, unbiased=False) + 1e-4)
    return F.relu(1.0 - std).mean()


def covariance_loss(z: torch.Tensor) -> torch.Tensor:
    if z.size(0) < 2:
        return z.new_zeros(())
    z = z - z.mean(dim=0)
    cov = z.t() @ z / (z.size(0) - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    return off_diag.square().sum() / z.size(1)


def classifier_topology_loss(
    student: nn.Module,
    stage_anchor: nn.Module,
    old_classes: Sequence[int],
) -> torch.Tensor:
    valid_old_classes = [
        int(c) for c in old_classes
        if 0 <= int(c) < student.classifier.out_features
    ]
    if len(valid_old_classes) < 2:
        return student.classifier.weight.new_zeros(())
    idx = torch.tensor(valid_old_classes, device=student.classifier.weight.device)
    w_now = F.normalize(student.classifier.weight[idx], dim=1)
    w_old = F.normalize(stage_anchor.classifier.weight[idx], dim=1)
    return F.mse_loss(w_now @ w_now.t(), w_old @ w_old.t())


def reliable_memory_loss(
    model: nn.Module,
    memory_batch,
    old_classes: Sequence[int],
    temperature: float,
    kd_weight: float = 0.5,
) -> torch.Tensor:
    if memory_batch is None:
        return model.classifier.weight.new_zeros(())
    images, labels, old_features, old_logits = memory_batch
    logits, features = model(images)

    ce = F.cross_entropy(logits, labels)
    feature_distill = F.mse_loss(
        F.normalize(features, dim=1), F.normalize(old_features, dim=1)
    )
    valid_old_classes = [
        int(c) for c in old_classes if 0 <= int(c) < logits.size(1)
    ]
    if valid_old_classes:
        idx = torch.tensor(valid_old_classes, device=logits.device)
        kd = F.kl_div(
            F.log_softmax(logits[:, idx] / temperature, dim=1),
            F.softmax(old_logits[:, idx] / temperature, dim=1),
            reduction="batchmean",
        ) * temperature**2
    else:
        kd = logits.new_zeros(())
    return ce + feature_distill + kd_weight * kd


def _as_cpu_tensor(value, *, dtype=None) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor.detach().cpu()


def compute_pseudo_label_metrics(
    *,
    soft,
    selected,
    sample_ids,
    true_labels,
    num_classes: int,
    current_classes: Optional[Sequence[int]] = None,
    previous_per_class_accuracy: Optional[Sequence[float]] = None,
) -> Dict[str, Union[float, int, Sequence[float], Sequence[int]]]:
    if num_classes <= 0:
        raise ValueError("num_classes must be positive.")

    soft_tensor = _as_cpu_tensor(soft, dtype=torch.float32)
    selected_tensor = _as_cpu_tensor(selected, dtype=torch.bool).reshape(-1)
    id_tensor = _as_cpu_tensor(sample_ids, dtype=torch.long).reshape(-1)
    label_tensor = _as_cpu_tensor(true_labels, dtype=torch.long).reshape(-1)

    if soft_tensor.ndim != 2:
        raise ValueError("soft must be a two-dimensional [N, C] tensor.")
    if soft_tensor.size(1) != num_classes:
        raise ValueError(
            "soft class dimension does not match num_classes: "
            f"{soft_tensor.size(1)} != {num_classes}."
        )

    sample_count = soft_tensor.size(0)
    if len(selected_tensor) != sample_count or len(id_tensor) != sample_count:
        raise ValueError(
            "soft, selected, and sample_ids must contain the same number of samples."
        )
    if sample_count == 0:
        raise ValueError("The pseudo-label bank is empty.")
    if len(torch.unique(id_tensor)) != len(id_tensor):
        raise ValueError("sample_ids must be unique within the pseudo-label bank.")
    if bool((id_tensor < 0).any()) or bool((id_tensor >= len(label_tensor)).any()):
        raise ValueError("sample_ids contain an index outside true_labels.")

    aligned_labels = label_tensor[id_tensor]
    predicted = soft_tensor.argmax(dim=1)
    correct = predicted.eq(aligned_labels)

    selected_count = int(selected_tensor.sum().item())
    all_accuracy = float(correct.float().mean().item())
    selected_accuracy = (
        float(correct[selected_tensor].float().mean().item())
        if selected_count > 0
        else math.nan
    )

    per_class_accuracy = []
    per_class_count = []
    for class_id in range(num_classes):
        class_mask = selected_tensor & predicted.eq(class_id)
        class_count = int(class_mask.sum().item())
        per_class_count.append(class_count)
        per_class_accuracy.append(
            float(correct[class_mask].float().mean().item())
            if class_count > 0
            else math.nan
        )

    if current_classes is None:
        current_class_ids = list(range(num_classes))
    else:
        current_class_ids = [int(class_id) for class_id in current_classes]
    if any(class_id < 0 or class_id >= num_classes for class_id in current_class_ids):
        raise ValueError("current_classes contain a class outside num_classes.")

    if previous_per_class_accuracy is None:
        overall_per_class_accuracy = [0.0 for _ in range(num_classes)]
    else:
        if len(previous_per_class_accuracy) != num_classes:
            raise ValueError(
                "previous_per_class_accuracy must contain one value per class."
            )
        overall_per_class_accuracy = [
            float(value) for value in previous_per_class_accuracy
        ]

    for class_id in current_class_ids:
        current_accuracy = per_class_accuracy[class_id]
        overall_per_class_accuracy[class_id] = (
            0.0 if math.isnan(current_accuracy) else current_accuracy
        )
    overall_accuracy = sum(overall_per_class_accuracy) / num_classes

    return {
        "all_accuracy": all_accuracy,
        "selected_accuracy": selected_accuracy,
        "overall_accuracy": overall_accuracy,
        "coverage": selected_count / sample_count,
        "selected_count": selected_count,
        "total_count": sample_count,
        "per_class_accuracy": per_class_accuracy,
        "per_class_count": per_class_count,
        "overall_per_class_accuracy": overall_per_class_accuracy,
    }


def _format_float(value: float) -> str:
    return "nan" if math.isnan(float(value)) else f"{float(value):.6f}"


def format_pseudo_label_log(
    metrics: Mapping[str, object],
    *,
    phase: str,
    stage: int,
    epoch: str,
) -> str:
    per_class_accuracy = ",".join(
        _format_float(value) for value in metrics["per_class_accuracy"]
    )
    per_class_count = ",".join(
        str(int(value)) for value in metrics["per_class_count"]
    )
    overall_per_class_accuracy = ",".join(
        _format_float(value) for value in metrics["overall_per_class_accuracy"]
    )
    return (
        f"pseudo_label_diag | phase={phase} | stage={stage} | epoch={epoch} | "
        f"overall_acc={_format_float(metrics['overall_accuracy'])} | "
        f"all_acc={_format_float(metrics['all_accuracy'])} | "
        f"selected_acc={_format_float(metrics['selected_accuracy'])} | "
        f"coverage={_format_float(metrics['coverage'])} | "
        f"selected={int(metrics['selected_count'])}/{int(metrics['total_count'])} | "
        f"per_class_acc=[{per_class_accuracy}] | "
        f"per_class_count=[{per_class_count}] | "
        f"overall_per_class_acc=[{overall_per_class_accuracy}] | "
        "diagnostic_only=true"
    )
