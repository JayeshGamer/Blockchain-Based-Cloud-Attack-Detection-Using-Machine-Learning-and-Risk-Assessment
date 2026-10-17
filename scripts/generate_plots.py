"""
Genuine Plot and Figure Generation Suite for Adaptive ML-DST (V4_REVIEW).
Strictly complies with Phase 19 Requirements:
- ZERO synthetic or random numbers (no np.random.randn).
- Every point, bar, curve, and alarm is loaded directly from saved experiment outputs:
    reports/experiments/protocol_a_results.json
    reports/experiments/protocol_a_predictions.json.gz
    reports/experiments/protocol_b_results.json
    reports/experiments/protocol_b_predictions.json.gz
    reports/experiments/protocol_c_results.json
    reports/experiments/ablation_study_results.json
    reports/dataset_audit/class_distribution.csv
    reports/dataset_audit/corrected_audit.json
"""

import os
import sys
import json
import gzip
import logging
import argparse
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Aesthetic styling
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})


def save_fig(fig, filename: str, output_dir: str = "reports/figures"):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved figure: {out_path}")


# 1. Figure 1: Dataset Class Distribution and Deduplication Summary
def plot_dataset_distribution_and_deduplication(
    audit_dir: str = "reports/dataset_audit",
    figures_dir: str = "reports/figures",
):
    dist_csv = os.path.join(audit_dir, "class_distribution.csv")
    audit_json = os.path.join(audit_dir, "corrected_audit.json")

    if not os.path.exists(dist_csv) or not os.path.exists(audit_json):
        logger.warning("Dataset audit files not found. Skipping Figure 1.")
        return

    df_dist = pd.read_csv(dist_csv)
    with open(audit_json, "r", encoding="utf-8") as f:
        audit_data = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # (a) Attack Family Deduplicated Counts
    fam_counts = df_dist[df_dist["Class"] != "BENIGN"].groupby("Attack_Family")["Deduped_Count"].sum().sort_values(ascending=True)
    y_pos = np.arange(len(fam_counts))
    ax1.barh(y_pos, fam_counts.values, color="#3b82f6", edgecolor="black", alpha=0.85)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(fam_counts.index)
    ax1.set_xscale("log")
    ax1.set_xlabel("Unique Deduplicated Flows (Log Scale)")
    ax1.set_title("(a) Attack Family Distribution (Deduplicated)")
    ax1.grid(True, axis="x")

    # (b) Duplicate Removal Rate across Files
    files_df = pd.DataFrame(audit_data["per_file_breakdown"])
    ax2.bar(
        range(len(files_df)),
        files_df["duplicate_rate_pct"],
        color="#ef4444",
        edgecolor="black",
        alpha=0.85,
    )
    ax2.set_xticks(range(len(files_df)))
    ax2.set_xticklabels(files_df["source_day"], rotation=35, ha="right")
    ax2.set_ylabel("Duplicate Removal Rate (%)")
    ax2.set_title("(b) Exact Duplicate Rate across Chronological Files")
    ax2.grid(True, axis="y")

    fig.tight_layout()
    save_fig(fig, "fig1_dataset_distribution_and_deduplication.png", output_dir=figures_dir)


# 2. Figure 2: Protocol A ROC and PR Curves
def plot_protocol_a_curves(
    experiments_dir: str = "reports/experiments",
    figures_dir: str = "reports/figures",
):
    pred_path = os.path.join(experiments_dir, "protocol_a_predictions.json.gz")
    if not os.path.exists(pred_path):
        logger.warning(f"Protocol A predictions not found at {pred_path}. Skipping Figure 2.")
        return

    with gzip.open(pred_path, "rt", encoding="utf-8") as f:
        data = json.load(f)

    y_true = np.array(data["predictions"]["y_true"])
    scores_dict = data["scores"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    palette = {
        "Random Forest": "#10b981",
        "HistGradientBoosting": "#06b6d4",
        "Static SGD": "#6366f1",
        "Logistic Regression": "#f59e0b",
        "Isolation Forest (Unsupervised)": "#8b5cf6",
        "Static DST (Base Heuristic)": "#ec4899",
        "Simplified Static Heuristic-DST (paper-inspired; not an exact reproduction)": "#ec4899",
        "Adaptive ML-DST (Proposed Batch)": "#ef4444",
    }

    for name, scores in scores_dict.items():
        if name not in palette:
            continue
        s_arr = np.array(scores)
        fpr, tpr, _ = roc_curve(y_true, s_arr)
        prec, rec, _ = precision_recall_curve(y_true, s_arr)

        label_name = "Simplified Static Heuristic-DST" if "Simplified Static" in name or "Static DST" in name else name
        lw = 2.2 if "Proposed" in name else 1.2
        ax1.plot(fpr, tpr, label=label_name, color=palette.get(name, "#64748b"), linewidth=lw)
        ax2.plot(rec, prec, label=label_name, color=palette.get(name, "#64748b"), linewidth=lw)

    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate (Recall)")
    ax1.set_title("(a) Conventional ROC Curves (Protocol A)")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.grid(True)

    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("(b) Precision-Recall Curves (Protocol A)")
    ax2.legend(loc="lower left", fontsize=8)
    ax2.grid(True)

    fig.tight_layout()
    save_fig(fig, "fig2_protocol_a_roc_pr_curves.png", output_dir=figures_dir)


# 3. Figure 3: Protocol A Confusion Matrices
def plot_protocol_a_confusion_matrices(
    experiments_dir: str = "reports/experiments",
    figures_dir: str = "reports/figures",
):
    pred_path = os.path.join(experiments_dir, "protocol_a_predictions.json.gz")
    if not os.path.exists(pred_path):
        return

    with gzip.open(pred_path, "rt", encoding="utf-8") as f:
        data = json.load(f)

    y_true = np.array(data["predictions"]["y_true"])
    preds = data["predictions"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    dst_key = "Simplified Static Heuristic-DST (paper-inspired; not an exact reproduction)"
    if dst_key not in preds:
        dst_key = "Static DST (Base Heuristic)"

    models_to_plot = [
        ("Random Forest", "Random Forest (Static)"),
        (dst_key, "Simplified Static Heuristic-DST"),
        ("Adaptive ML-DST (Proposed Batch)", "Adaptive ML-DST (Proposed)"),
    ]

    for ax, (key, title) in zip(axes, models_to_plot):
        if key not in preds:
            continue
        p = np.array(preds[key])
        cm = confusion_matrix(y_true, p)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            cbar=False,
            xticklabels=["Benign", "Attack"],
            yticklabels=["Benign", "Attack"],
        )
        ax.set_title(title)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

    fig.tight_layout()
    save_fig(fig, "fig3_protocol_a_confusion_matrices.png", output_dir=figures_dir)


# 4. Figure 4: Real Chronological Streaming Risk Trajectory & Adaptive Threshold
def plot_real_chronological_trajectory(
    experiments_dir: str = "reports/experiments",
    figures_dir: str = "reports/figures",
):
    pred_path_l = os.path.join(experiments_dir, "protocol_b_predictions.jsonl.gz")
    pred_path_legacy = os.path.join(experiments_dir, "protocol_b_predictions.json.gz")
    pred_path = pred_path_l if os.path.exists(pred_path_l) else pred_path_legacy
    res_path = os.path.join(experiments_dir, "protocol_b_results.json")
    if not os.path.exists(pred_path) or not os.path.exists(res_path):
        logger.warning("Protocol B logs not found. Skipping Figure 4.")
        return

    from src.evaluation.streaming_core import read_prediction_records
    y_true_list = []
    risks_list = []
    for rec in read_prediction_records(pred_path):
        y_true_list.append(rec["y_true"])
        risks_list.append(rec["scores"].get("8_Full_Adaptive_ML_DST", 0.0))
        if len(y_true_list) >= 1500:
            break

    with open(res_path, "r", encoding="utf-8") as f:
        res_data = json.load(f)

    y_true = np.array(y_true_list)
    risks = np.array(risks_list)
    drift_events = res_data.get("drift_events", [])

    fig, ax = plt.subplots(figsize=(12, 4.2))
    steps = np.arange(len(risks))

    ax.plot(steps, risks, color="#38bdf8", alpha=0.6, linewidth=1.0, label="Pignistic Risk Score BetP(Attack)")

    attack_mask = y_true == 1
    if np.any(attack_mask):
        ax.scatter(steps[attack_mask], risks[attack_mask], color="#ef4444", s=10, alpha=0.7, label="True Attack Flow")

    for ev in drift_events:
        s_idx = ev.get("step_index", 0)
        if s_idx < len(risks):
            ax.axvline(s_idx, color="#ec4899", linestyle="--", linewidth=1.5, alpha=0.8)
            ax.text(s_idx + 5, 0.90, f"ADWIN Alarm\n({ev.get('action')})", color="#ec4899", fontsize=8)

    ax.set_xlabel("Replay Flow Sequence Index (Ordered Ingress)")
    ax.set_ylabel("Risk Score / Decision Boundary")
    ax.set_title("Figure 4: Real Chronological Risk Trajectory & Genuine ADWIN Concept-Drift Alarms")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True)

    fig.tight_layout()
    save_fig(fig, "fig4_chronological_streaming_trajectory.png", output_dir=figures_dir)


# 5. Figure 5: Daily Recall and False Negative Rate Progression
def plot_daily_fnr_recall(
    tables_dir: str = "reports/tables",
    figures_dir: str = "reports/figures",
):
    t3_path = os.path.join(tables_dir, "table3_chronological_daily.csv")
    if not os.path.exists(t3_path):
        return

    df = pd.read_csv(t3_path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    days = df["Day"].unique()
    models = df["Model"].unique()
    palette = {
        "1_Frozen_Static": "#94a3b8",
        "2_Online_SGD_NoDrift": "#3b82f6",
        "8_Full_Adaptive_ML_DST": "#10b981",
    }
    model_labels = {
        "1_Frozen_Static": "Frozen Static (Monday)",
        "2_Online_SGD_NoDrift": "Online SGD (No Drift)",
        "8_Full_Adaptive_ML_DST": "Adaptive ML-DST (Proposed)",
    }

    for m in models:
        sub = df[df["Model"] == m]
        ax1.plot(sub["Day"], sub["Recall (ADR)"], marker="o", label=model_labels.get(m, m), color=palette.get(m, "#000"))
        ax2.plot(sub["Day"], sub["FNR"], marker="s", label=model_labels.get(m, m), color=palette.get(m, "#000"))

    ax1.set_ylabel("Attack Recall (ADR)")
    ax1.set_title("(a) Attack Recall Across Chronological Days")
    ax1.set_xticklabels(days, rotation=30, ha="right")
    ax1.legend(loc="lower left")
    ax1.grid(True)

    ax2.set_ylabel("False Negative Rate (FNR)")
    ax2.set_title("(b) False Negative Rate Across Chronological Days")
    ax2.set_xticklabels(days, rotation=30, ha="right")
    ax2.legend(loc="upper left")
    ax2.grid(True)

    fig.tight_layout()
    save_fig(fig, "fig5_daily_fnr_and_recall.png", output_dir=figures_dir)


# 6. Figure 6: Phase 17 Fair Ablation Study Breakdown
def plot_ablation_study(
    tables_dir: str = "reports/tables",
    figures_dir: str = "reports/figures",
):
    t2_path = os.path.join(tables_dir, "table2_ablation_study.csv")
    if not os.path.exists(t2_path):
        return

    df = pd.read_csv(t2_path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    short_labels = [f"Var {chr(65+i)}" for i in range(len(df))]

    # Macro-F1
    col_f1 = "Macro_F1" if "Macro_F1" in df.columns else "F1_Score"
    ax1.bar(short_labels, df[col_f1], color="#3b82f6", edgecolor="black", alpha=0.85)
    ax1.set_ylabel("Macro-F1 Score")
    ax1.set_title("(a) Macro-F1 across Ablation Variants A to I")
    ax1.set_ylim(0.4, 1.0)
    ax1.grid(True, axis="y")

    # Recall vs FNR
    x = np.arange(len(df))
    w = 0.35
    ax2.bar(x - w/2, df["Recall (ADR)"], width=w, label="Recall", color="#10b981", edgecolor="black", alpha=0.85)
    ax2.bar(x + w/2, df["FNR"], width=w, label="FNR", color="#ef4444", edgecolor="black", alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_labels)
    ax2.set_ylabel("Rate")
    ax2.set_title("(b) Attack Recall vs False Negative Rate (FNR)")
    ax2.legend(loc="upper right")
    ax2.grid(True, axis="y")

    fig.tight_layout()
    save_fig(fig, "fig6_ablation_study_breakdown.png", output_dir=figures_dir)


# 7. Figure 7: Controlled Concept-Drift Benchmark
def plot_controlled_drift_benchmark(
    experiments_dir: str = "reports/experiments",
    figures_dir: str = "reports/figures",
):
    c_path = os.path.join(experiments_dir, "protocol_c_results.json")
    if not os.path.exists(c_path):
        return

    with open(c_path, "r", encoding="utf-8") as f:
        c_data = json.load(f)

    seg_dict = c_data.get("segment_metrics", {})
    if not seg_dict:
        return

    fig, ax = plt.subplots(figsize=(10, 4.0))
    seg_names = [k.replace("Segment ", "S") for k in seg_dict.keys()]
    f1s = [m.get("macro_f1", 0.0) for m in seg_dict.values()]
    recalls = [m.get("recall", 0.0) for m in seg_dict.values()]

    x = np.arange(len(seg_names))
    w = 0.35
    ax.bar(x - w/2, f1s, width=w, label="Macro-F1", color="#3b82f6", edgecolor="black", alpha=0.85)
    ax.bar(x + w/2, recalls, width=w, label="Recall", color="#10b981", edgecolor="black", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(seg_names, rotation=25, ha="right")
    ax.set_ylabel("Metric Score")
    ax.set_title("Figure 7: Protocol C Controlled Drift Benchmark - Segment Performance & Recovery")
    ax.legend(loc="lower right")
    ax.grid(True, axis="y")

    fig.tight_layout()
    save_fig(fig, "fig7_controlled_drift_recovery.png", output_dir=figures_dir)


# 8. Figure 8: Isolated Latency and Throughput Benchmarks
def plot_isolated_performance_benchmarks(
    tables_dir: str = "reports/tables",
    figures_dir: str = "reports/figures",
):
    t1_path = os.path.join(tables_dir, "table1_comparative_models.csv")
    if not os.path.exists(t1_path):
        return

    df = pd.read_csv(t1_path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    models = df["Model"].tolist()
    short_names = [
        m.replace(" (Unsupervised)", "")
         .replace(" (Base Heuristic)", "")
         .replace("Simplified Static Heuristic-DST (paper-inspired; not an exact reproduction)", "Heuristic-DST*")
         .replace(" (Proposed Batch)", "*")
        for m in models
    ]

    ax1.barh(short_names, df["Median_Latency_ms"], color="#6366f1", edgecolor="black", alpha=0.85)
    ax1.set_xlabel("Median Inference Latency (ms / flow)")
    ax1.set_title("(a) Isolated Model Latency")
    ax1.grid(True, axis="x")

    ax2.barh(short_names, df["Throughput_FPS"], color="#06b6d4", edgecolor="black", alpha=0.85)
    ax2.set_xlabel("Throughput (Flows per Second)")
    ax2.set_title("(b) Isolated Processing Throughput")
    ax2.set_xscale("log")
    ax2.grid(True, axis="x")

    fig.tight_layout()
    save_fig(fig, "fig8_isolated_latency_and_throughput.png", output_dir=figures_dir)


def generate_all_figures(
    experiments_dir: str = "reports/experiments",
    tables_dir: str = "reports/tables",
    figures_dir: str = "reports/figures",
    audit_dir: str = "reports/dataset_audit",
):
    """Generates all verified experimental figures from persisted JSON and CSV outputs."""
    logger.info(f"=== Generating Figures (Experiments: {experiments_dir}, Tables: {tables_dir}) ===")
    plot_dataset_distribution_and_deduplication(audit_dir=audit_dir, figures_dir=figures_dir)
    plot_protocol_a_curves(experiments_dir=experiments_dir, figures_dir=figures_dir)
    plot_protocol_a_confusion_matrices(experiments_dir=experiments_dir, figures_dir=figures_dir)
    plot_real_chronological_trajectory(experiments_dir=experiments_dir, figures_dir=figures_dir)
    plot_daily_fnr_recall(tables_dir=tables_dir, figures_dir=figures_dir)
    plot_ablation_study(tables_dir=tables_dir, figures_dir=figures_dir)
    plot_controlled_drift_benchmark(experiments_dir=experiments_dir, figures_dir=figures_dir)
    plot_isolated_performance_benchmarks(tables_dir=tables_dir, figures_dir=figures_dir)
    logger.info("All experimental figures successfully generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Experimental Figures")
    parser.add_argument("--experiments-dir", type=str, default="reports/experiments")
    parser.add_argument("--tables-dir", type=str, default="reports/tables")
    parser.add_argument("--figures-dir", type=str, default="reports/figures")
    parser.add_argument("--audit-dir", type=str, default="reports/dataset_audit")
    args = parser.parse_args()

    generate_all_figures(
        experiments_dir=args.experiments_dir,
        tables_dir=args.tables_dir,
        figures_dir=args.figures_dir,
        audit_dir=args.audit_dir,
    )
