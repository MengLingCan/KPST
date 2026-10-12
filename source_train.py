import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import DatasetArgumentParser, HSIDataset, get_dataset_config
from net import AttentionParallelCNN
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from tqdm import tqdm
from typing import Dict, List, Tuple
from utils import (
    BalancedSoftmaxLoss,
    CenterRepulsionLoss,
    PrototypeManager,
    safe_torch_load,
    set_seed,
    setup_logger,
)

CLASS_NUM = 7
FEATURE_DIM = 32


def build_parser():
    parser = DatasetArgumentParser("source_train", description="HSI source_train")
    parser.add_argument("--data_path", default="./Pavia")
    parser.add_argument("--save_path", default="./results")
    parser.add_argument("--source_domain", default=None)
    parser.add_argument("--patch_size", type=int, default=15)
    parser.add_argument("--gpu", default="5")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--max_epoch", type=int, default=15)
    parser.add_argument("--prototypes_per_class", type=int, default=3)

    parser.add_argument(
        "--loss_mode",
        choices=["balanced_softmax", "effective", "ce"],
        default="balanced_softmax",
        help="Classification loss used after deferred rebalancing starts",
    )
    parser.add_argument("--balanced_softmax_tau", type=float, default=1.0)
    parser.add_argument("--effective_beta", type=float, default=0.9999)
    parser.add_argument("--max_class_weight", type=float, default=5.0)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument(
        "--rebalance_start_ratio",
        type=float,
        default=0.5,
        help="Fraction of epochs using natural sampling before rebalancing",
    )
    parser.add_argument(
        "--sampler_power",
        type=float,
        default=0.75,
        help="Tempered inverse-frequency sampling exponent",
    )
    parser.add_argument(
        "--sampler_max_multiplier",
        type=float,
        default=30.0,
        help="Maximum minority-class sampling multiplier",
    )
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument(
        "--selection_metric",
        choices=["aa", "hybrid", "oa"],
        default="hybrid",
        help="hybrid uses 0.7*AA + 0.3*OA",
    )
    parser.add_argument("--seed", type=int, default=25)
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def stratified_split(
    labels: np.ndarray,
    val_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    labels = np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(seed)
    train_indices: List[int] = []
    val_indices: List[int] = []

    for class_id in np.unique(labels):
        class_indices = np.where(labels == class_id)[0]
        rng.shuffle(class_indices)
        if len(class_indices) == 1:
            train_indices.extend(class_indices.tolist())
            continue
        val_count = max(1, int(round(float(val_ratio) * len(class_indices))))
        val_count = min(val_count, len(class_indices) - 1)
        val_indices.extend(class_indices[:val_count].tolist())
        train_indices.extend(class_indices[val_count:].tolist())

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    if not train_indices or not val_indices:
        raise RuntimeError("Unable to construct a non-empty stratified split.")
    return train_indices, val_indices


def effective_number_weights(
    class_counts: np.ndarray,
    beta: float,
    max_weight: float,
) -> torch.Tensor:
    counts = np.asarray(class_counts, dtype=np.float64)
    present = counts > 0
    weights = np.ones_like(counts, dtype=np.float64)
    if np.any(present):
        effective_num = 1.0 - np.power(float(beta), counts[present])
        weights[present] = (1.0 - float(beta)) / np.maximum(effective_num, 1e-12)
        weights[present] /= weights[present].mean()
        weights[present] = np.minimum(weights[present], float(max_weight))
        weights[present] /= weights[present].mean()
    return torch.tensor(weights, dtype=torch.float32)


def build_rebalanced_sampler(
    train_labels: np.ndarray,
    class_counts: np.ndarray,
    power: float,
    max_multiplier: float,
    seed: int,
) -> WeightedRandomSampler:
    counts = np.asarray(class_counts, dtype=np.float64)
    present = counts > 0
    if not np.any(present):
        raise RuntimeError("No source samples are available for rebalancing.")

    reference_count = counts[present].max()
    class_multiplier = np.ones_like(counts, dtype=np.float64)
    class_multiplier[present] = np.power(
        reference_count / counts[present], float(power)
    )
    class_multiplier[present] = np.minimum(
        class_multiplier[present], float(max_multiplier)
    )
    sample_weights = class_multiplier[np.asarray(train_labels, dtype=np.int64)]
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(train_labels),
        replacement=True,
        generator=generator,
    )


@torch.no_grad()
def evaluate_source(
    model: nn.Module,
    loader: DataLoader,
    class_num: int,
    device: torch.device,
) -> Dict[str, object]:
    model.eval()
    confusion = np.zeros((class_num, class_num), dtype=np.int64)
    for images, labels in loader:
        images = images.to(device, non_blocking=True).float()
        labels = labels.to(device, non_blocking=True).long()
        logits, _ = model(images)
        predictions = logits.argmax(dim=1)
        for true_value, pred_value in zip(
            labels.cpu().numpy(), predictions.cpu().numpy()
        ):
            confusion[int(true_value), int(pred_value)] += 1

    support = confusion.sum(axis=1)
    correct = np.diag(confusion)
    per_class = np.full(class_num, np.nan, dtype=np.float64)
    valid = support > 0
    per_class[valid] = correct[valid] / support[valid]
    total = int(confusion.sum())
    oa = float(correct.sum() / total) if total else 0.0
    aa = float(np.nanmean(per_class)) if np.any(valid) else 0.0
    return {"OA": oa, "AA": aa, "per_class": per_class, "confusion": confusion}


def choose_checkpoint_score(metrics: Dict[str, object], mode: str) -> float:
    oa = float(metrics["OA"])
    aa = float(metrics["AA"])
    if mode == "aa":
        return aa
    if mode == "oa":
        return oa
    return 0.7 * aa + 0.3 * oa


def add_source_domain_metadata(path: str, source_domain: str, device: torch.device) -> None:
    state_dict = safe_torch_load(path, device)
    if isinstance(state_dict, dict):
        state_dict["source_domain"] = source_domain
        torch.save(state_dict, path)


def main():
    args = parse_args()
    if not 0.0 <= args.rebalance_start_ratio <= 1.0:
        raise ValueError("rebalance_start_ratio must be in [0, 1).")
    if not 0.0 < args.val_ratio < 0.5:
        raise ValueError("val_ratio must be in (0, 0.5).")

    class_num = CLASS_NUM
    in_channels = args.in_channels

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    os.makedirs(args.save_path, exist_ok=True)
    run_start = time.time()

    logger = setup_logger(args.save_path, file_name="src_log.txt")
    logger.info(
        f"run | phase=source_train | dataset={get_dataset_config(args.dataset)['name']} | "
        f"source={args.source_domain} | device={device}"
    )
    logger.info(
        f"params | lr={args.lr} | batch_size={args.batch_size} | "
        f"patch_size={args.patch_size} | max_epoch={args.max_epoch} | seed={args.seed}"
    )
    logger.info(
        f"class_balance | loss={args.loss_mode} | "
        f"rebalance_start={args.rebalance_start_ratio:.2f} | "
        f"sampler_power={args.sampler_power:.2f} | "
        f"sampler_cap={args.sampler_max_multiplier:.1f} | "
        f"selection_metric={args.selection_metric}"
    )

    train_dataset = HSIDataset(
        args.data_path,
        dataset=args.dataset,
        domain=args.source_domain,
        patch_size=args.patch_size,
        data_augment=True,
    )
    val_dataset = HSIDataset(
        args.data_path,
        dataset=args.dataset,
        domain=args.source_domain,
        patch_size=args.patch_size,
        data_augment=False,
    )
    prototype_dataset = HSIDataset(
        args.data_path,
        dataset=args.dataset,
        domain=args.source_domain,
        patch_size=args.patch_size,
        data_augment=False,
    )
    if len(train_dataset) < 2:
        raise RuntimeError("The source domain contains too few labeled samples.")

    train_indices, val_indices = stratified_split(
        train_dataset.labels, args.val_ratio, args.seed
    )
    train_set = Subset(train_dataset, train_indices)
    val_set = Subset(val_dataset, val_indices)
    train_labels = np.asarray(train_dataset.labels, dtype=np.int64)[train_indices]
    val_labels = np.asarray(val_dataset.labels, dtype=np.int64)[val_indices]
    train_counts = np.bincount(train_labels, minlength=class_num)
    val_counts = np.bincount(val_labels, minlength=class_num)
    class_priors = train_counts.astype(np.float64)
    class_priors /= class_priors.sum()
    rebalance_start_epoch = int(round(args.max_epoch * args.rebalance_start_ratio))
    rebalance_start_epoch = min(max(0, rebalance_start_epoch), args.max_epoch - 1)

    logger.info(
        f"train | source_samples={len(train_dataset)} | split_train={len(train_set)} | "
        f"split_validation={len(val_set)} | rebalance_start_epoch={rebalance_start_epoch + 1}"
    )
    logger.info(
        "train | class_counts="
        + str(train_counts.tolist())
        + " | priors="
        + str([round(float(value), 8) for value in class_priors.tolist()])
    )
    logger.info("validation | class_counts=" + str(val_counts.tolist()))

    pin_memory = torch.cuda.is_available()
    natural_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    rebalanced_sampler = build_rebalanced_sampler(
        train_labels=train_labels,
        class_counts=train_counts,
        power=args.sampler_power,
        max_multiplier=args.sampler_max_multiplier,
        seed=args.seed + 17,
    )
    rebalanced_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=rebalanced_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = AttentionParallelCNN(
        in_channels, class_num, args.patch_size, dataset=args.dataset
    ).to(device)
    optimizer = optim.SGD(
        model.parameters(), args.lr, momentum=0.9, weight_decay=5e-4
    )

    scheduler = (optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.max_epoch), eta_min=args.lr * 0.05)
                 if args.dataset == "houston" else None)

    natural_ce = nn.CrossEntropyLoss(
        label_smoothing=args.label_smoothing
    ).to(device)
    balanced_softmax = BalancedSoftmaxLoss(
        torch.as_tensor(train_counts, dtype=torch.float32),
        tau=args.balanced_softmax_tau,
        label_smoothing=args.label_smoothing,
    ).to(device)
    effective_weights = effective_number_weights(
        train_counts, args.effective_beta, args.max_class_weight
    ).to(device)
    effective_ce = nn.CrossEntropyLoss(
        weight=effective_weights,
        label_smoothing=args.label_smoothing,
    ).to(device)
    logger.info(
        "train | effective_number_weights="
        + str([round(float(value), 4) for value in effective_weights.cpu()])
    )

    criterion_ct = CenterRepulsionLoss(
        class_num, feat_dim=FEATURE_DIM, margin=20.0, device=device
    ).to(device)
    optimizer_ct = optim.SGD(criterion_ct.parameters(), lr=0.5)

    lambda_ce = 1.5
    lambda_center = 0.026
    lambda_repulsion = 0.032
    checkpoint_path = os.path.join(args.save_path, "source.pth")

    best_score = -float("inf")
    best_metrics = None
    logger.info("train | status=starting")

    for epoch in range(args.max_epoch):
        is_rebalanced = epoch >= rebalance_start_epoch
        train_loader = rebalanced_loader if is_rebalanced else natural_loader
        if not is_rebalanced or args.loss_mode == "ce":
            criterion_ce = natural_ce
            active_loss_name = "CE"
        elif args.loss_mode == "balanced_softmax":
            criterion_ce = balanced_softmax
            active_loss_name = "BalancedSoftmax"
        else:
            criterion_ce = effective_ce
            active_loss_name = "EffectiveNumberCE"

        model.train()
        losses = []
        start = time.time()
        for patches, labels in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{args.max_epoch}"
        ):
            patches = patches.to(device, non_blocking=True).float()
            labels = labels.to(device, non_blocking=True).long()

            optimizer.zero_grad(set_to_none=True)
            optimizer_ct.zero_grad(set_to_none=True)
            logits, features = model(patches)
            if logits.size(1) != class_num:
                raise RuntimeError(
                    f"Expected {class_num} logits, got {logits.size(1)}."
                )
            loss_ce = criterion_ce(logits, labels)
            loss_center, loss_repulsion = criterion_ct(features, labels)
            loss = (
                lambda_ce * loss_ce
                + lambda_center * loss_center
                + lambda_repulsion * loss_repulsion
            )
            loss.backward()
            if args.dataset == "houston":
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            optimizer_ct.step()
            losses.append(float(loss.detach()))

        if scheduler is not None:
            scheduler.step()
        metrics = evaluate_source(model, val_loader, class_num, device)
        score = choose_checkpoint_score(metrics, args.selection_metric)
        per_class_text = [
            None if np.isnan(value) else round(100.0 * float(value), 2)
            for value in metrics["per_class"]
        ]
        logger.info(
            f"epoch | current={epoch + 1} | total={args.max_epoch} | "
            f"loss={np.mean(losses):.4f} | "
            f"OA={100 * float(metrics['OA']):.2f}% | "
            f"AA={100 * float(metrics['AA']):.2f}% | score={score:.5f} | "
            f"phase={'rebalanced' if is_rebalanced else 'natural'} | "
            f"loss_type={active_loss_name} | per_class={per_class_text}"
        )
        logger.info(f"duration | epoch={epoch + 1} | seconds={time.time() - start:.1f}")
        if score > best_score:
            best_score = score
            best_metrics = metrics
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "class_counts": train_counts.tolist(),
                    "class_priors": class_priors.tolist(),
                    "loss_mode": args.loss_mode,
                    "balanced_softmax_tau": float(args.balanced_softmax_tau),
                    "sampler_power": float(args.sampler_power),
                    "sampler_max_multiplier": float(args.sampler_max_multiplier),
                    "rebalance_start_epoch": int(rebalance_start_epoch),
                    "selection_metric": args.selection_metric,
                    "class_num": class_num,
                    "dataset": args.dataset,
                    "in_channels": in_channels,
                    "patch_size": args.patch_size,
                    "source_domain": args.source_domain,
                },
                checkpoint_path,
            )
            logger.info(
                f"checkpoint | path={checkpoint_path} | score={best_score:.5f} | "
                f"OA={100 * float(metrics['OA']):.2f}% | "
                f"AA={100 * float(metrics['AA']):.2f}%"
            )

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Source checkpoint was not saved: {checkpoint_path}")
    checkpoint = safe_torch_load(checkpoint_path, device)
    state_dict = (
        checkpoint["state_dict"]
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint
        else checkpoint
    )
    model.load_state_dict(state_dict)
    model.eval()
    if best_metrics is not None:
        logger.info(
            f"validation | best_OA={100 * float(best_metrics['OA']):.2f}% | "
            f"best_AA={100 * float(best_metrics['AA']):.2f}%"
        )

    # Preserve each dataset's original source-prototype sampling scope.
    prototype_loader = DataLoader(
        Subset(prototype_dataset, train_indices) if args.dataset == "houston" else prototype_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    all_features = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(prototype_loader, desc="Computing source prototypes"):
            images = images.to(device, non_blocking=True).float()
            all_features.append(model.extract_features(images).cpu())
            all_labels.append(torch.as_tensor(labels).long().cpu())

    source_features = torch.cat(all_features, dim=0)
    source_labels = torch.cat(all_labels, dim=0)
    prototype_manager = PrototypeManager(
        class_num, feature_dim=FEATURE_DIM, device=torch.device("cpu")
    )
    prototype_manager.init_source_prototypes(source_features, source_labels)

    single_name = "protos.pt"
    multi_name = f"source_multi_prototypes_k{args.prototypes_per_class}.pt"
    single_path = os.path.join(args.save_path, single_name)
    prototype_manager.save_source_prototypes(single_path)
    add_source_domain_metadata(single_path, args.source_domain, device)
    multi_prototypes = prototype_manager.init_source_multi_prototypes(
        source_features,
        source_labels,
        prototypes_per_class=args.prototypes_per_class,
        random_state=args.seed,
    )
    multi_path = os.path.join(args.save_path, multi_name)
    prototype_manager.save_source_multi_prototypes(
        multi_path,
        prototypes_per_class=args.prototypes_per_class,
    )
    add_source_domain_metadata(multi_path, args.source_domain, device)
    logger.info(
        f"checkpoint | single={single_path} | multi={multi_path} | "
        f"shape={tuple(multi_prototypes.shape)}"
    )
    logger.info(f"duration | phase=source_train | seconds={time.time() - run_start:.1f}")

if __name__ == "__main__":
    main()
