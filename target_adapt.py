import copy
import numpy as np
import os
import time
import torch
from datasets import DatasetArgumentParser, HSIDataset, get_dataset_config, validate_domain_pair
from net import AttentionParallelCNN
from pathlib import Path
from target_ciuda import (
    PseudoLabelBank,
    ReliableTargetMemory,
    TargetConfig,
    UnlabeledTargetView,
    build_target_optimizer,
    label_free_prediction_score,
    label_free_warmup,
    refresh_global_pseudo_labels,
    train_target_epoch,
)
from torch.utils.data import DataLoader
from utils import (
    compute_pseudo_label_metrics,
    evaluate,
    format_pseudo_label_log,
    safe_torch_load,
    set_seed,
    setup_logger,
)


def build_parser():
    parser = DatasetArgumentParser("target_adapt", description="HSI target_adapt")
    parser.add_argument("--data_path", default="./Pavia")
    parser.add_argument("--save_path", default="./results")
    parser.add_argument("--model_path", default="./results")
    parser.add_argument("--source_domain", default=None)
    parser.add_argument("--target_domain", default=None)
    parser.add_argument("--patch_size", type=int, default=15)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--max_epoch", type=int, default=5)
    parser.add_argument("--warmup_epoch", type=int, default=2)
    parser.add_argument("--gpu", default="5")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=100)

    parser.add_argument("--source_prototypes_per_class", type=int, default=3)
    parser.add_argument("--target_clusters_per_class", type=int, default=4)
    parser.add_argument("--knn_k", type=int, default=2)
    parser.add_argument("--stable_rounds", type=int, default=0)
    parser.add_argument("--confidence_threshold", type=float, default=0.01)
    parser.add_argument("--initial_coverage", type=float, default=0.01)
    parser.add_argument("--final_coverage", type=float, default=0.02)
    parser.add_argument("--min_samples_per_class", type=int, default=0)

    parser.add_argument("--lambda_pl", type=float, default=1.0)
    parser.add_argument("--lambda_cons", type=float, default=0.15)
    parser.add_argument("--lambda_src_proto", type=float, default=0.25)
    parser.add_argument("--lambda_memory", type=float, default=0.7)
    parser.add_argument("--lambda_topology", type=float, default=0.0)
    parser.add_argument("--memory_per_class", type=int, default=300)
    parser.add_argument("--memory_batch_size", type=int, default=64)
    parser.add_argument("--ema_momentum", type=float, default=0.999)
    parser.add_argument("--teacher_temperature", type=float, default=0.7)

    parser.add_argument(
        "--source_prior_correction_tau",
        type=float,
        default=-1.0,
        help=(
            "Source-prior correction strength during target pseudo-labeling. "
            "-1 selects 0 for a Balanced-Softmax source checkpoint and 0.5 "
            "for a conventional CE checkpoint."
        ),
    )
    parser.add_argument("--pseudo_balance_power", type=float, default=0.5)
    parser.add_argument("--pseudo_balance_max_weight", type=float, default=3.0)
    parser.add_argument(
        "--log_pseudo_label_accuracy",
        action="store_true",
        help=(
            "Diagnostic only: compare pseudo-label-bank predictions with target "
            "ground truth in tar_log.txt. The metric never affects optimization, "
            "pseudo-label selection, or checkpoint selection."
        ),
    )
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def make_loader(dataset, batch_size, shuffle, num_workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def main():
    args = parse_args()
    validate_domain_pair(args.source_domain, args.target_domain)
    in_channels = args.in_channels
    class_num = 7
    feature_dim = 32

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    torch.multiprocessing.set_sharing_strategy("file_system")

    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(str(save_path), "tar_log.txt")

    cfg = TargetConfig(
        dataset=args.dataset,
        num_classes=class_num,
        feature_dim=feature_dim,
        num_bands=in_channels,
        warmup_epochs=args.warmup_epoch,
        warmup_lr=args.lr * 0.4,
        source_prototypes_per_class=args.source_prototypes_per_class,
        target_clusters_per_class=args.target_clusters_per_class,
        knn_k=args.knn_k,
        stable_rounds=args.stable_rounds,
        confidence_threshold=args.confidence_threshold,
        initial_coverage=args.initial_coverage,
        final_coverage=args.final_coverage,
        min_samples_per_class=args.min_samples_per_class,
        lambda_pl=args.lambda_pl,
        lambda_cons=args.lambda_cons,
        lambda_src_proto=args.lambda_src_proto,
        lambda_memory=args.lambda_memory,
        lambda_topology=args.lambda_topology,
        memory_per_class=args.memory_per_class,
        memory_batch_size=args.memory_batch_size,
        ema_momentum=args.ema_momentum,
        teacher_temperature=args.teacher_temperature,
        pseudo_balance_power=args.pseudo_balance_power,
        pseudo_balance_max_weight=args.pseudo_balance_max_weight,
        random_state=args.seed,
    )

    logger.info(
        f"run | phase=target_adapt | dataset={get_dataset_config(args.dataset)['name']} | "
        f"source={args.source_domain} | target={args.target_domain} | device={device}"
    )
    logger.info(
        f"params | lr={args.lr} | batch_size={args.batch_size} | "
        f"patch_size={args.patch_size} | warmup_epoch={args.warmup_epoch} | "
        f"max_epoch={args.max_epoch} | seed={args.seed}"
    )
    logger.info(
        "config | "
        f"source_prototypes_per_class={cfg.source_prototypes_per_class} | "
        f"target_clusters_per_class={cfg.target_clusters_per_class} | "
        f"knn_k={cfg.knn_k} | stable_rounds={cfg.stable_rounds} | "
        f"confidence_threshold={cfg.confidence_threshold} | "
        f"coverage={cfg.initial_coverage}->{cfg.final_coverage}"
    )
    logger.info(
        "pseudo_label_diag_config | "
        f"enabled={str(args.log_pseudo_label_accuracy).lower()} | "
        "diagnostic_only=true | affects_optimization=false | "
        "affects_checkpoint_selection=false"
    )
    if args.log_pseudo_label_accuracy:
        logger.warning(
            "Pseudo-label accuracy diagnostics read target ground-truth labels "
            "for logging only; do not use these values to tune or select checkpoints."
        )

    source_checkpoint = Path(args.model_path) / "source.pth"
    if not source_checkpoint.exists():
        raise FileNotFoundError(f"Source checkpoint not found: {source_checkpoint}")

    model = AttentionParallelCNN(
        in_channels, class_num, args.patch_size, dataset=args.dataset
    ).to(device)
    source_payload = safe_torch_load(source_checkpoint, device)
    source_metadata = source_payload if isinstance(source_payload, dict) else {}
    source_state = (
        source_payload["state_dict"]
        if isinstance(source_payload, dict) and "state_dict" in source_payload
        else source_payload
    )
    model.load_state_dict(source_state, strict=True)
    if model.classifier.out_features != class_num:
        raise RuntimeError(
            f"Expected a {class_num}-logit classifier, got "
            f"{model.classifier.out_features}."
        )

    multi_name = f"source_multi_prototypes_k{cfg.source_prototypes_per_class}.pt"
    multi_proto_path = Path(args.model_path) / multi_name
    if not multi_proto_path.exists():
        raise FileNotFoundError(
            f"Source multi-prototypes not found: {multi_proto_path}."
        )
    multi_state = safe_torch_load(multi_proto_path, torch.device("cpu"))
    source_multi_prototypes = (
        multi_state["prototypes"] if isinstance(multi_state, dict) else multi_state
    ).float()
    expected_shape = (
        class_num,
        cfg.source_prototypes_per_class,
        feature_dim,
    )
    if tuple(source_multi_prototypes.shape) != expected_shape:
        raise ValueError(
            f"Invalid source prototype shape {tuple(source_multi_prototypes.shape)}; "
            f"expected {expected_shape}."
        )
    logger.info(
        f"artifact | type=source_multi_prototypes | path={multi_proto_path} | "
        f"shape={tuple(source_multi_prototypes.shape)}"
    )

    source_dataset = HSIDataset(
        args.data_path,
        dataset=args.dataset,
        domain=args.source_domain,
        patch_size=args.patch_size,
    )
    source_loader = make_loader(
        source_dataset, args.batch_size, False, args.num_workers
    )

    stored_prior = source_metadata.get("class_priors")
    source_loss_mode = str(source_metadata.get("loss_mode", "ce"))
    if stored_prior is None:
        source_counts = np.bincount(
            np.asarray(source_dataset.labels, dtype=np.int64),
            minlength=class_num,
        )[:class_num]
        source_prior = source_counts.astype(np.float64)
    else:
        source_prior = np.asarray(stored_prior, dtype=np.float64)
        if len(source_prior) != class_num:
            raise ValueError(
                f"Stored source prior has {len(source_prior)} entries; "
                f"expected {class_num}."
            )
    if source_prior.sum() <= 0:
        raise RuntimeError("Invalid source class prior: the total count is zero.")
    source_prior /= source_prior.sum()

    if args.source_prior_correction_tau < 0:
        prior_tau = 0.0 if source_loss_mode == "balanced_softmax" else 0.5
    else:
        prior_tau = float(args.source_prior_correction_tau)
    cfg.source_class_prior = torch.tensor(source_prior, dtype=torch.float32)
    cfg.source_prior_correction_tau = prior_tau
    logger.info(
        "source_prior | "
        f"loss_mode={source_loss_mode} | tau={prior_tau:.3f} | "
        f"prior={[round(float(value), 8) for value in source_prior.tolist()]}"
    )
    logger.info(
        "pseudo_balance | "
        f"power={cfg.pseudo_balance_power:.3f} | "
        f"max_weight={cfg.pseudo_balance_max_weight:.3f}"
    )

    memory = ReliableTargetMemory(class_num, cfg.memory_per_class)
    discovered_classes = set()
    best_target_per_class = [0.0 for _ in range(class_num)]
    overall_pseudo_class_accuracy = [0.0 for _ in range(class_num)]
    source_baseline = {"OA": 0.9972, "AA": 0.9976}
    if args.dataset == "houston":
        baseline_classes = list(range(class_num))
        baseline_metrics = evaluate(model, source_loader, class_num, device,
                                    seen_classes=baseline_classes, current_classes=baseline_classes)
        source_baseline = {"OA": baseline_metrics["target_OA"], "AA": baseline_metrics["target_AA"]}
    stage_schedule = [[0, 1], [2, 3], [4, 5, 6]]

    for stage_idx, train_classes in enumerate(stage_schedule):
        stage_start = time.time()
        test_classes = list(range(max(train_classes) + 1))
        expected_class_count = len(train_classes)
        logger.info(
            f"stage | index={stage_idx} | train_classes={train_classes} | "
            f"test_classes={test_classes}"
        )

        if stage_idx > 0:
            previous_path = save_path / f"adapt-{stage_idx - 1}.pth"
            model.load_state_dict(safe_torch_load(previous_path, device), strict=True)

        stage_anchor = copy.deepcopy(model).to(device).eval()
        ema_teacher = copy.deepcopy(model).to(device).eval()
        for frozen_model in (stage_anchor, ema_teacher):
            for parameter in frozen_model.parameters():
                parameter.requires_grad_(False)

        target_phase_labeled = HSIDataset(
            args.data_path,
            dataset=args.dataset,
            domain=args.target_domain,
            patch_size=args.patch_size,
            filter_func=lambda label, allowed=set(train_classes): label in allowed,
            return_id=True,
        )

        target_phase_unlabeled = UnlabeledTargetView(target_phase_labeled)
        train_loader = make_loader(
            target_phase_unlabeled, args.batch_size, True, args.num_workers
        )
        global_loader = make_loader(
            target_phase_unlabeled, args.batch_size, False, args.num_workers
        )

        test_dataset = HSIDataset(
            args.data_path,
            dataset=args.dataset,
            domain=args.target_domain,
            patch_size=args.patch_size,
            filter_func=lambda label, allowed=set(test_classes): label in allowed,
        )
        test_loader = make_loader(
            test_dataset, args.batch_size, False, args.num_workers
        )

        label_free_warmup(
            model, ema_teacher, train_loader, cfg, device, logger
        )

        old_classes = sorted(discovered_classes)
        bank = PseudoLabelBank(class_num, cfg.stable_rounds)
        optimizer = build_target_optimizer(model, args.lr, warmup=False)
        best_score = -float("inf")
        best_active_classes = None

        for epoch in range(args.max_epoch):
            pseudo = refresh_global_pseudo_labels(
                ema_teacher=ema_teacher,
                stage_anchor=stage_anchor,
                loader=global_loader,
                source_multi_prototypes=source_multi_prototypes,
                bank=bank,
                expected_class_count=expected_class_count,
                epoch=epoch,
                excluded_classes=old_classes,
                forced_active_classes=train_classes,
                total_epochs=args.max_epoch,
                cfg=cfg,
                device=device,
            )
            active_classes = pseudo["active_classes"].tolist()
            selected_count = int(pseudo["selected"].sum())
            stable_count = int(pseudo["stable"].sum())
            mean_confidence = (
                float(pseudo["confidence"][pseudo["selected"]].mean())
                if selected_count > 0
                else 0.0
            )
            if args.log_pseudo_label_accuracy:
                pseudo_metrics = compute_pseudo_label_metrics(
                    soft=pseudo["soft"],
                    selected=pseudo["selected"],
                    sample_ids=bank.ids,
                    true_labels=target_phase_labeled.labels,
                    num_classes=class_num,
                    current_classes=train_classes,
                    previous_per_class_accuracy=overall_pseudo_class_accuracy,
                )
                overall_pseudo_class_accuracy = list(
                    pseudo_metrics["overall_per_class_accuracy"]
                )
                logger.info(
                    format_pseudo_label_log(
                        pseudo_metrics,
                        phase="train",
                        stage=stage_idx,
                        epoch=f"{epoch + 1}/{args.max_epoch}",
                    )
                )

            train_stats = train_target_epoch(
                student=model,
                ema_teacher=ema_teacher,
                stage_anchor=stage_anchor,
                loader=train_loader,
                optimizer=optimizer,
                bank=bank,
                source_multi_prototypes=source_multi_prototypes,
                memory=memory,
                old_classes=old_classes,
                cfg=cfg,
                device=device,
            )

            metrics = evaluate(
                model=model,
                target_loader=test_loader,
                num_classes=class_num,
                device=device,
                source_loader=source_loader,
                source_baseline=source_baseline,
                best_target_per_class=best_target_per_class,
                seen_classes=test_classes,
                current_classes=train_classes,
                beta=1.0,
                gamma=0.2,
            )
            post_update_im = label_free_prediction_score(
                ema_teacher, global_loader, active_classes, device
            )
            balance_penalty = 0.0
            if args.dataset == "houston":
                evidence = pseudo["class_score"][active_classes].float()
                evidence /= evidence.sum().clamp_min(1e-8)
                balance_penalty = float((evidence * (evidence.clamp_min(1e-8) * len(evidence)).log()).sum())
            score = (
                post_update_im
                + 0.25 * mean_confidence
                + (0.10 if args.dataset == "houston" else 0.05) * (stable_count / max(1, len(target_phase_unlabeled)))
                - 0.30 * balance_penalty
            )
            logger.info(
                f"epoch | stage={stage_idx} | epoch={epoch + 1}/{args.max_epoch} | "
                f"active_classes={active_classes} | selected={selected_count}/"
                f"{len(target_phase_unlabeled)} | stable={stable_count} | "
                f"mean_confidence={mean_confidence:.4f} | score={score:.6f}"
            )
            logger.info(
                "loss | "
                f"stage={stage_idx} | epoch={epoch + 1} | "
                f"total={train_stats['loss']:.6f} | pl={train_stats['pl']:.6f} | "
                f"cons={train_stats['cons']:.6f} | "
                f"src_proto={train_stats['src_proto']:.6f} | "
                f"memory={train_stats['memory']:.6f} | "
                f"topology={train_stats['topology']:.6f}"
            )
            logger.info(
                f"metric | stage={stage_idx} | epoch={epoch + 1} | "
                f"target_OA={100 * metrics['target_OA']:.2f}% | "
                f"target_AA={100 * metrics['target_AA']:.2f}% | "
                f"forgetting={100 * metrics['target_forgetting']:.2f}%"
            )

            checkpoint_eligible = args.dataset == "pavia" or (
                epoch + 1 >= max(3, cfg.stable_rounds + 1) and stable_count > 0)
            if args.dataset == "houston":
                torch.save(model.state_dict(), save_path / f"adapt-{stage_idx}-student-last.pth")
                torch.save(ema_teacher.state_dict(), save_path / f"adapt-{stage_idx}-ema-last.pth")
            if checkpoint_eligible and score > best_score:
                best_score = score
                best_active_classes = active_classes
                torch.save(
                    ema_teacher.state_dict(),
                    save_path / f"adapt-{stage_idx}.pth",
                )
                logger.info(
                    f"checkpoint | stage={stage_idx} | path=adapt-{stage_idx}.pth | "
                    f"score={best_score:.6f}"
                )

            for c in range(class_num):
                current = metrics["target_per_class_acc"][c]
                if not np.isnan(current):
                    best_target_per_class[c] = max(best_target_per_class[c], current)

        best_path = save_path / f"adapt-{stage_idx}.pth"
        if best_score == -float("inf"):
            torch.save(ema_teacher.state_dict(), best_path)
            logger.warning("No checkpoint met stability gating; saved EMA-last fallback.")
        model.load_state_dict(safe_torch_load(best_path, device), strict=True)

        memory_bank = PseudoLabelBank(class_num, cfg.stable_rounds)
        final_pseudo = None
        for _ in range(max(1, cfg.stable_rounds)):
            final_pseudo = refresh_global_pseudo_labels(
                ema_teacher=model,
                stage_anchor=stage_anchor,
                loader=global_loader,
                source_multi_prototypes=source_multi_prototypes,
                bank=memory_bank,
                expected_class_count=expected_class_count,
                epoch=max(0, args.max_epoch - 1),
                excluded_classes=old_classes,
                forced_active_classes=train_classes,
                total_epochs=max(1, args.max_epoch),
                cfg=cfg,
                device=device,
            )
        if args.log_pseudo_label_accuracy:
            final_pseudo_metrics = compute_pseudo_label_metrics(
                soft=final_pseudo["soft"],
                selected=final_pseudo["selected"],
                sample_ids=memory_bank.ids,
                true_labels=target_phase_labeled.labels,
                num_classes=class_num,
                current_classes=train_classes,
                previous_per_class_accuracy=overall_pseudo_class_accuracy,
            )
            overall_pseudo_class_accuracy = list(
                final_pseudo_metrics["overall_per_class_accuracy"]
            )
            logger.info(
                format_pseudo_label_log(
                    final_pseudo_metrics,
                    phase="stage_end",
                    stage=stage_idx,
                    epoch="best",
                )
            )
        best_active_classes = final_pseudo["active_classes"].tolist()
        memory.update_from_bank(
            model,
            global_loader,
            memory_bank,
            train_classes,
            device,
        )
        memory.save(save_path / "target_memory.pt")
        discovered_classes.update(best_active_classes)

        torch.save(
            {
                "dataset": args.dataset,
                "source_domain": args.source_domain,
                "target_domain": args.target_domain,
                "model_num_classes": 7,
                "active_classes": best_active_classes,
                "seen_classes": sorted(discovered_classes),
                "source_multi_prototypes": source_multi_prototypes,
                "best_unsup_score": best_score,
            },
            save_path / f"stage-{stage_idx}-state.pt",
        )
        logger.info(
            f"memory | stage={stage_idx} | active_classes={best_active_classes} | "
            f"seen_classes={sorted(discovered_classes)} | counts={memory.class_counts()}"
        )
        logger.info(
            f"duration | stage={stage_idx} | seconds={time.time() - stage_start:.1f}"
        )

if __name__ == "__main__":
    main()
