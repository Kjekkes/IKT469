"""Render PNG charts from the ablation CSVs.

Reads:
    report/figures/ablate_retrieval_<dataset>.csv  (from script 06)
    report/figures/ablate_generation_<dataset>.csv (from script 07)

Writes (any subset that has input data):
    report/figures/plot_chunk_size.png
    report/figures/plot_reranker.png
    report/figures/plot_topk_faithfulness.png
    report/figures/plot_topk_latency.png
    report/figures/plot_prompt_variant.png

Run:
    python scripts/08_make_plots.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import load_config


# Generic style: clean and report-friendly. Avoid emoji / unicode glyphs.
plt.rcParams.update({
    "figure.figsize": (7, 4.2),
    "figure.dpi": 110,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.alpha": 0.25,
})


def _load_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        print(f"  (skip) missing {path.name}")
        return None
    return pd.read_csv(path)


def _norm_rerank(series: pd.Series) -> pd.Series:
    """Normalize a rerank column to lowercase strings 'true'/'false'.

    The ablation runner writes `str(bool).lower()`, which gives "true" or
    "false" in the CSV. Depending on pandas version and column composition
    the value can come back as a Python bool, a numpy bool_, or a string of
    any case — so we coerce defensively before comparing.
    """
    return series.astype(str).str.strip().str.lower()


# --- Retrieval plots ---------------------------------------------------


def plot_chunk_size(df: pd.DataFrame, out_dir: Path) -> None:
    """Bar chart: NDCG@10 and Recall@10 vs chunk size, no reranker.

    Filters to the default embedder so the fine-tuned row doesn't pollute
    this chart -- it gets its own plot in `plot_finetune`.
    """
    work = df.copy()
    work["rerank_n"] = _norm_rerank(work["rerank"])
    embed_col = work["embedding_model"] if "embedding_model" in work.columns \
        else pd.Series(["default"] * len(work))
    sub = work[(work["rerank_n"] == "false")
               & (embed_col == "default")
               & (work["metric"].isin(["NDCG@10", "Recall@10"]))]
    if sub.empty:
        print(f"  (skip) plot_chunk_size: no matching rows. "
              f"unique rerank values: {work['rerank'].unique().tolist()}, "
              f"unique metrics: {work['metric'].unique().tolist()}")
        return

    pivot = sub.pivot_table(index="chunk_size", columns="metric",
                            values="value", aggfunc="first")
    pivot.index = pivot.index.astype(str)
    order = [c for c in ["none", "128", "256", "512"] if c in pivot.index]
    pivot = pivot.reindex(order)

    ax = pivot.plot(kind="bar", rot=0, edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Chunk size (tokens; 'none' = one chunk per passage)")
    ax.set_ylabel("Score")
    ax.set_title("Retrieval quality vs chunk size")
    ax.set_ylim(0, max(pivot.values.max() * 1.15, 0.1))
    ax.legend(title="", loc="upper right")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=2, fontsize=8)
    plt.savefig(out_dir / "plot_chunk_size.png")
    plt.close()
    print("  wrote plot_chunk_size.png")


def plot_finetune(df: pd.DataFrame, out_dir: Path) -> None:
    """Bar chart: pretrained MiniLM vs contrastively fine-tuned MiniLM.

    Compares only baseline retrieval rows (no chunk-split, no reranker)
    so the encoder is the only thing that changes.
    """
    if "embedding_model" not in df.columns:
        print("  (skip) plot_finetune: CSV has no embedding_model column "
              "(re-run scripts/06_ablate_retrieval.py)")
        return

    work = df.copy()
    work["rerank_n"] = _norm_rerank(work["rerank"])
    sub = work[(work["rerank_n"] == "false")
               & (work["chunk_size"].astype(str) == "none")
               & (work["metric"].isin(["NDCG@10", "Recall@10", "MRR@10"]))]
    if sub["embedding_model"].nunique() < 2:
        print("  (skip) plot_finetune: only one embedder in results "
              "(run scripts/10_finetune_embedder.py first)")
        return

    label_map = {"default": "MiniLM (pretrained)"}
    sub = sub.assign(
        encoder=sub["embedding_model"].map(
            lambda v: label_map.get(v, "MiniLM (fine-tuned)")
        )
    )
    pivot = sub.pivot_table(index="encoder", columns="metric",
                            values="value", aggfunc="first")
    encoder_order = ["MiniLM (pretrained)", "MiniLM (fine-tuned)"]
    pivot = pivot.reindex([e for e in encoder_order if e in pivot.index])

    ax = pivot.plot(kind="bar", rot=0, edgecolor="black", linewidth=0.4)
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_title("Effect of contrastive fine-tuning on the bi-encoder")
    ax.set_ylim(0, max(pivot.values.max() * 1.2, 0.1))
    ax.legend(title="", loc="upper right", fontsize=8)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=2, fontsize=8)
    plt.savefig(out_dir / "plot_finetune.png")
    plt.close()
    print("  wrote plot_finetune.png")


def plot_reranker(df: pd.DataFrame, out_dir: Path) -> None:
    """Grouped bars: with vs without reranker, for NDCG@10 and Recall@10."""
    sub = df[df["metric"].isin(["NDCG@10", "Recall@10"])].copy()
    if sub.empty:
        print("  (skip) plot_reranker: no rows match")
        return

    sub["chunk_label"] = sub["chunk_size"].astype(str)
    sub["rerank_n"] = _norm_rerank(sub["rerank"])
    sub["rerank_label"] = sub["rerank_n"].map({"true": "rerank", "false": "no rerank"})

    # Show the chunk sizes that have BOTH a rerank and no-rerank row.
    have_both_series = sub.groupby("chunk_label")["rerank_n"].nunique()
    have_both = have_both_series[have_both_series == 2].index.tolist()
    if not have_both:
        print("  (skip) plot_reranker: no chunk size has both rerank and no-rerank rows")
        return

    chunk_order = [c for c in ["none", "128", "256", "512"] if c in have_both]
    sub = sub[sub["chunk_label"].isin(chunk_order)]

    # Single-level row index (chunk_label), MultiIndex columns
    # (metric, rerank_label) -- pandas' .plot(kind="bar") groups them
    # cleanly into 4 bars per chunk size.
    pivot = sub.pivot_table(
        index="chunk_label",
        columns=["metric", "rerank_label"],
        values="value",
        aggfunc="first",
    ).reindex(chunk_order)

    ax = pivot.plot(kind="bar", rot=0, edgecolor="black", linewidth=0.4,
                    figsize=(8, 4.4))
    ax.set_xlabel("Chunk size")
    ax.set_ylabel("Score")
    ax.set_title("Effect of cross-encoder reranking")
    ax.set_ylim(0, max(pivot.values.max() * 1.2, 0.1))
    # Compact legend labels: "NDCG@10 / rerank" instead of "(NDCG@10, rerank)".
    ax.legend([" / ".join(map(str, c)) for c in pivot.columns],
              loc="upper right", ncol=2, fontsize=7.5)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=2, fontsize=7)
    plt.savefig(out_dir / "plot_reranker.png")
    plt.close()
    print("  wrote plot_reranker.png")


# --- Generation plots --------------------------------------------------


def plot_topk_faithfulness(df: pd.DataFrame, out_dir: Path) -> None:
    sub = df[df["prompt_variant"] == "grounded"].copy()
    if sub.empty:
        print("  (skip) plot_topk_faithfulness: no grounded rows")
        return
    sub = sub.sort_values("top_k")

    fig, ax = plt.subplots()
    ax.plot(sub["top_k"], sub["avg_faithfulness"], marker="o", linewidth=2)
    ax.set_xlabel("top_k passages in prompt")
    ax.set_ylabel("Mean LLM-judge faithfulness (0-2)")
    ax.set_title("Faithfulness vs context size (grounded prompt)")
    ax.set_xticks(sub["top_k"].tolist())
    ax.set_ylim(0, 2.05)
    for x, y in zip(sub["top_k"], sub["avg_faithfulness"]):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=8)
    plt.savefig(out_dir / "plot_topk_faithfulness.png")
    plt.close()
    print("  wrote plot_topk_faithfulness.png")


def plot_topk_latency(df: pd.DataFrame, out_dir: Path) -> None:
    sub = df[df["prompt_variant"] == "grounded"].copy()
    if sub.empty:
        print("  (skip) plot_topk_latency: no grounded rows")
        return
    sub = sub.sort_values("top_k")

    fig, ax = plt.subplots()
    ax.plot(sub["top_k"], sub["avg_latency_s"], marker="s", linewidth=2, color="#c0392b")
    ax.set_xlabel("top_k passages in prompt")
    ax.set_ylabel("Mean answer latency (s)")
    ax.set_title("Generator latency vs context size (grounded prompt)")
    ax.set_xticks(sub["top_k"].tolist())
    ax.set_ylim(0, sub["avg_latency_s"].max() * 1.2)
    for x, y in zip(sub["top_k"], sub["avg_latency_s"]):
        ax.annotate(f"{y:.1f}s", (x, y), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=8)
    plt.savefig(out_dir / "plot_topk_latency.png")
    plt.close()
    print("  wrote plot_topk_latency.png")


def plot_prompt_variant(df: pd.DataFrame, out_dir: Path) -> None:
    """Compare grounded vs vanilla at the same top_k (default 5)."""
    common_k = sorted(set(df[df["prompt_variant"] == "grounded"]["top_k"]) &
                      set(df[df["prompt_variant"] == "vanilla"]["top_k"]))
    if not common_k:
        print("  (skip) plot_prompt_variant: no top_k value has both variants")
        return
    k = common_k[len(common_k) // 2]  # median match
    sub = df[df["top_k"] == k]

    fig, ax = plt.subplots()
    ax.bar(sub["prompt_variant"], sub["avg_faithfulness"],
           color=["#2c7fb8", "#cc4c02"], edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Mean LLM-judge faithfulness (0-2)")
    ax.set_title(f"Prompt template effect (top_k={k})")
    ax.set_ylim(0, 2.05)
    for i, v in enumerate(sub["avg_faithfulness"]):
        ax.text(i, v + 0.04, f"{v:.2f}", ha="center", fontsize=9)
    plt.savefig(out_dir / "plot_prompt_variant.png")
    plt.close()
    print("  wrote plot_prompt_variant.png")


# --- Driver ------------------------------------------------------------


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config = load_config(repo_root / "config.yaml")
    dataset = config["beir"]["dataset"]
    fig_dir = repo_root / "report" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("Retrieval plots:")
    retr = _load_csv(fig_dir / f"ablate_retrieval_{dataset}.csv")
    if retr is not None:
        plot_chunk_size(retr, fig_dir)
        plot_reranker(retr, fig_dir)
        plot_finetune(retr, fig_dir)

    print("Generation plots:")
    gen = _load_csv(fig_dir / f"ablate_generation_{dataset}.csv")
    if gen is not None:
        plot_topk_faithfulness(gen, fig_dir)
        plot_topk_latency(gen, fig_dir)
        plot_prompt_variant(gen, fig_dir)

    print(f"\nDone. PNGs in {fig_dir}")


if __name__ == "__main__":
    main()
