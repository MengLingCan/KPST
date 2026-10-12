import numpy as np
import os
import torch
from datasets import DatasetArgumentParser, HSIDataset, get_dataset_config, validate_domain_pair
from net import AttentionParallelCNN
from pathlib import Path
from torch.utils.data import DataLoader
from utils import safe_torch_load, set_seed

FIXED_CLASS_NUM = 7


def build_parser():
    parser = DatasetArgumentParser("eval", description="HSI eval")
    parser.add_argument("--data_path", default="./Pavia")
    parser.add_argument("--save_path", default="./results")
    parser.add_argument("--result_file", default="adapt-test.txt")
    parser.add_argument(
        "--eval_part", choices=["source", "target", "both"], default="target"
    )
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--src_model_path", default=None)
    parser.add_argument("--tar_model_path", default=None)
    parser.add_argument(
        "--source_domain", default=None
    )
    parser.add_argument(
        "--target_domain", default=None
    )
    parser.add_argument("--patch_size", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--gpu", default="5")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--in_channels", type=int, default=102)
    parser.add_argument("--class_num", type=int, default=7)
    parser.add_argument("--seed", type=int, default=25)
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def clean_state_dict(state):
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError("Checkpoint must contain a model state_dict.")
    return {
        (key.replace("module.", "", 1) if key.startswith("module.") else key): value
        for key, value in state.items()
    }


def load_model(model_path, in_channels, class_num, patch_size, device, dataset=None):
    if model_path is None:
        raise ValueError("model_path is required.")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    model = AttentionParallelCNN(in_channels, class_num, patch_size, dataset=dataset)
    state_dict = clean_state_dict(safe_torch_load(model_path, device))
    model.load_state_dict(state_dict, strict=True)
    if model.classifier.out_features != FIXED_CLASS_NUM:
        raise RuntimeError(
            f"Expected {FIXED_CLASS_NUM} logits, got {model.classifier.out_features}."
        )
    return model.to(device).eval()


def build_loader(args, domain):
    dataset = HSIDataset(
        args.data_path,
        dataset=args.dataset,
        domain=domain,
        patch_size=args.patch_size,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def compute_confusion_matrix(model, loader, class_num, device):
    confusion = np.zeros((class_num, class_num), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch[0].to(device, non_blocking=True).float()
            labels = batch[1].long()
            model_output = model(images)
            logits = model_output[0] if isinstance(model_output, (tuple, list)) else model_output
            if logits.size(1) != FIXED_CLASS_NUM:
                raise RuntimeError(
                    f"Expected {FIXED_CLASS_NUM} logits, got {logits.size(1)}."
                )
            predictions = logits.argmax(dim=1).cpu()
            for true_label, pred_label in zip(labels.reshape(-1), predictions.reshape(-1)):
                true_label = int(true_label)
                pred_label = int(pred_label)
                if 0 <= true_label < class_num and 0 <= pred_label < class_num:
                    confusion[true_label, pred_label] += 1
    return confusion


def compute_metrics_from_confusion(confusion):
    class_num = confusion.shape[0]
    total = confusion.sum()
    correct = np.trace(confusion)
    row_sum = confusion.sum(axis=1)
    col_sum = confusion.sum(axis=0)
    tp = np.diag(confusion)
    oa = correct / total if total > 0 else 0.0

    per_class_acc = np.full(class_num, np.nan, dtype=np.float64)
    per_class_precision = np.full(class_num, np.nan, dtype=np.float64)
    per_class_recall = np.full(class_num, np.nan, dtype=np.float64)
    per_class_f1 = np.full(class_num, np.nan, dtype=np.float64)
    per_class_iou = np.full(class_num, np.nan, dtype=np.float64)

    for class_index in range(class_num):
        if row_sum[class_index] > 0:
            per_class_acc[class_index] = tp[class_index] / row_sum[class_index]
            per_class_recall[class_index] = tp[class_index] / row_sum[class_index]
        if col_sum[class_index] > 0:
            per_class_precision[class_index] = tp[class_index] / col_sum[class_index]
        precision = per_class_precision[class_index]
        recall = per_class_recall[class_index]
        if not np.isnan(precision) and not np.isnan(recall) and precision + recall > 0:
            per_class_f1[class_index] = 2 * precision * recall / (precision + recall)
        union = row_sum[class_index] + col_sum[class_index] - tp[class_index]
        if union > 0:
            per_class_iou[class_index] = tp[class_index] / union

    aa = np.nanmean(per_class_acc) if np.any(~np.isnan(per_class_acc)) else 0.0
    macro_precision = (
        np.nanmean(per_class_precision)
        if np.any(~np.isnan(per_class_precision))
        else 0.0
    )
    macro_recall = (
        np.nanmean(per_class_recall)
        if np.any(~np.isnan(per_class_recall))
        else 0.0
    )
    macro_f1 = np.nanmean(per_class_f1) if np.any(~np.isnan(per_class_f1)) else 0.0
    miou = np.nanmean(per_class_iou) if np.any(~np.isnan(per_class_iou)) else 0.0

    valid_f1 = ~np.isnan(per_class_f1)
    valid_f1_support = row_sum[valid_f1].sum()
    weighted_f1 = (
        np.sum(per_class_f1[valid_f1] * row_sum[valid_f1]) / valid_f1_support
        if valid_f1_support > 0
        else 0.0
    )
    if total > 0:
        expected_agreement = np.sum(row_sum * col_sum) / total**2
        kappa = (
            (oa - expected_agreement) / (1 - expected_agreement)
            if abs(1 - expected_agreement) > 1e-12
            else 0.0
        )
    else:
        kappa = 0.0

    return {
        "total": int(total),
        "correct": int(correct),
        "OA": float(oa),
        "AA": float(aa),
        "Kappa": float(kappa),
        "Macro_Precision": float(macro_precision),
        "Macro_Recall": float(macro_recall),
        "Macro_F1": float(macro_f1),
        "Weighted_F1": float(weighted_f1),
        "mIoU": float(miou),
        "support": row_sum.astype(np.int64),
        "per_class_acc": per_class_acc,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "per_class_f1": per_class_f1,
        "per_class_iou": per_class_iou,
        "confusion_matrix": confusion,
    }


def evaluate_one(args, model_path, domain, device):
    model = load_model(
        model_path,
        args.in_channels,
        args.class_num,
        args.patch_size,
        device,
        dataset=args.dataset,
    )
    loader = build_loader(args, domain)
    confusion = compute_confusion_matrix(model, loader, args.class_num, device)
    metrics = compute_metrics_from_confusion(confusion)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def format_percent(value):
    return "     nan" if value is None or np.isnan(value) else f"{100 * value:8.2f}%"


def format_confusion_matrix(confusion):
    labels = [str(index + 1) for index in range(confusion.shape[0])]
    width = 10
    header = "True\\Pred |" + "".join(f"{label:>{width}}" for label in labels)
    lines = [header, "-" * len(header)]
    for index, label in enumerate(labels):
        row = f"{label:>9} |" + "".join(
            f"{int(confusion[index, column]):>{width}}"
            for column in range(confusion.shape[1])
        )
        lines.append(row)
    return "\n".join(lines)


def format_configuration(args, device):
    return "\n".join(
        [
            "=" * 96,
            "Evaluation Configuration",
            "=" * 96,
            f"Dataset              : {get_dataset_config(args.dataset)['name']}",
            f"Source Domain        : {args.source_domain}",
            f"Target Domain        : {args.target_domain}",
            f"Device               : {device}",
            f"Patch Size           : {args.patch_size}",
            f"Seed                 : {args.seed}",
            "",
        ]
    )


def format_result(title, model_path, domain, metrics):
    lines = ["=" * 96, title, "=" * 96]
    lines.extend(
        [
            f"Model Path           : {model_path}",
            f"Physical Domain      : {domain}",
            f"OA                   : {format_percent(metrics['OA'])}",
            f"AA                   : {format_percent(metrics['AA'])}",
            f"Kappa                : {metrics['Kappa']:.6f}",
            "",
            "Per-class Recall / Precision / F1",
            f"{'Class':>8} | {'Support':>8} | {'Recall':>10} | {'Precision':>10} | {'F1':>10}",
            "-" * 64,
        ]
    )
    for class_index in range(len(metrics["support"])):
        lines.append(
            f"{class_index + 1:>8} | {int(metrics['support'][class_index]):>8} | "
            f"{format_percent(metrics['per_class_recall'][class_index]):>10} | "
            f"{format_percent(metrics['per_class_precision'][class_index]):>10} | "
            f"{format_percent(metrics['per_class_f1'][class_index]):>10}"
        )
    lines.extend(
        [
            "",
            "Confusion Matrix",
            "Rows = true labels, columns = predicted labels",
            format_confusion_matrix(metrics["confusion_matrix"]),
            "",
        ]
    )
    return "\n".join(lines)


def format_comparison(source_metrics, target_metrics):
    lines = ["=" * 96, "Source/Target Comparison", "=" * 96]
    lines.append(f"{'Metric':<20} | {'Source':>18} | {'Target':>18}")
    lines.append("-" * 64)
    for metric_name in ("OA", "AA", "Kappa"):
        if metric_name == "Kappa":
            source_value = f"{source_metrics[metric_name]:.6f}"
            target_value = f"{target_metrics[metric_name]:.6f}"
        else:
            source_value = format_percent(source_metrics[metric_name])
            target_value = format_percent(target_metrics[metric_name])
        lines.append(
            f"{metric_name:<20} | {source_value:>18} | {target_value:>18}"
        )
    lines.append("")
    return "\n".join(lines)


def default_model_path(args, part):
    filename = "source.pth" if part == "source" else "adapt-2.pth"
    return str(Path(args.save_path) / filename)


def resolve_model_path(args, part):
    explicit_path = args.src_model_path if part == "source" else args.tar_model_path
    if explicit_path is not None:
        return explicit_path
    if args.eval_part == part and args.model_path is not None:
        return args.model_path
    return default_model_path(args, part)


def resolve_result_path(args):
    result_path = Path(args.result_file)
    return result_path if result_path.is_absolute() else Path(args.save_path) / result_path


def main():
    args = parse_args()
    validate_domain_pair(args.source_domain, args.target_domain)
    if args.class_num != FIXED_CLASS_NUM:
        raise ValueError(f"class_num must be {FIXED_CLASS_NUM} for closed-set evaluation.")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_path, exist_ok=True)

    results = {}
    if args.eval_part in {"source", "both"}:
        source_path = resolve_model_path(args, "source")
        results["source"] = {
            "path": source_path,
            "domain": args.source_domain,
            "metrics": evaluate_one(args, source_path, args.source_domain, device),
        }
    if args.eval_part in {"target", "both"}:
        target_path = resolve_model_path(args, "target")
        results["target"] = {
            "path": target_path,
            "domain": args.target_domain,
            "metrics": evaluate_one(args, target_path, args.target_domain, device),
        }

    sections = [format_configuration(args, device)]
    if "source" in results:
        source = results["source"]
        sections.append(
            format_result(
                "Source Result", source["path"], source["domain"], source["metrics"]
            )
        )
    if "target" in results:
        target = results["target"]
        sections.append(
            format_result(
                "Target Result", target["path"], target["domain"], target["metrics"]
            )
        )
    if "source" in results and "target" in results:
        sections.append(
            format_comparison(results["source"]["metrics"], results["target"]["metrics"])
        )

    text = "\n".join(sections)
    result_path = resolve_result_path(args)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as file:
        file.write(text)
    print(text)
    print(f"Result saved to: {result_path}")

if __name__ == "__main__":
    main()
