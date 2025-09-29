#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dataset × (Kendall τ, Spearman ρ) 分组柱状图（马卡龙配色 + IEEE TRANS 风格）
- 自动识别 CSV（有/无表头均可）
- 自动从一组 Times 兼容衬线体里择优选择可用字体
- 导出 PNG(600 dpi) 与矢量 PDF

用法：
  python plot_corr_bars.py --csv kt.csv
  # 自定义字体（若你刚安装了 Times）
  python plot_corr_bars.py --csv kt.csv --font_family "Times New Roman"
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

FONT_CANDIDATES = [
    "Times New Roman",     # Windows / macOS 常有
    "Times",               # macOS 别名
    "TeX Gyre Termes",     # Linux 建议安装
    "Nimbus Roman No9 L",  # Linux 常见
    "Liberation Serif",    # Linux 常见
    "STIX Two Text",
    "STIXGeneral",
    "DejaVu Serif",        # 保底
]

def pick_font(preferred: str | None = None) -> str:
    if preferred:
        return preferred
    # 系统字体名集合（不含路径）
    names = set(f.name for f in fm.fontManager.ttflist)
    for cand in FONT_CANDIDATES:
        if cand in names:
            print(f"[font] Using: {cand}")
            return cand
    # 最后兜底
    print("[font] No candidate found; fallback to default serif.")
    return "DejaVu Serif"

def load_corr_csv(path: str) -> pd.DataFrame:
    def _norm(df):
        cols = [c.strip().lower() for c in df.columns]
        df.columns = cols
        alias = {
            "dataset": ["dataset", "data", "name"],
            "kendall": ["kendall", "kendall_tau", "kendall tau", "tau"],
            "spearman": ["spearman", "spearman_rho", "spearman rho", "rho"],
        }
        pick = {}
        for k, cands in alias.items():
            for c in cands:
                if c in df.columns:
                    pick[k] = c
                    break
        if len(pick) != 3:
            raise KeyError("Missing required columns")
        out = df[[pick["dataset"], pick["kendall"], pick["spearman"]]].copy()
        out.columns = ["dataset", "kendall", "spearman"]
        out["kendall"]  = pd.to_numeric(out["kendall"],  errors="coerce")
        out["spearman"] = pd.to_numeric(out["spearman"], errors="coerce")
        return out

    # 尝试按表头读取
    try:
        df = pd.read_csv(path)
        try:
            return _norm(df)
        except Exception:
            pass
    except Exception:
        pass
    # 无表头兜底
    df = pd.read_csv(path, header=None, names=["dataset","kendall","spearman"])
    df["kendall"]  = pd.to_numeric(df["kendall"],  errors="coerce")
    df["spearman"] = pd.to_numeric(df["spearman"], errors="coerce")
    return df

def plot_corr_bars(
    df: pd.DataFrame,
    out_base: str = "kt_barchart_macaron",
    width: float = 7.2,
    height: float = 3.8,
    font_family: str | None = None,
    font_size: int = 10,
    tick_size: int = 9,
    legend_size: int = 9,
    decimals: int = 3,
):
    font_family = pick_font(font_family)

    matplotlib.rcParams.update({
        "font.family": font_family,
        "font.size": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": tick_size,
        "ytick.labelsize": tick_size,
        "legend.fontsize": legend_size,
        "pdf.fonttype": 42,  # 嵌入 TrueType，便于后续在 AI/Word 中编辑
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
    })

    datasets = df["dataset"].astype(str).tolist()
    kendall  = df["kendall"].astype(float).values
    spearman = df["spearman"].astype(float).values

    fig, ax = plt.subplots(figsize=(width, height), dpi=150)

    for side in ["left", "bottom", "right", "top"]:
        ax.spines[side].set_linewidth(0.5)  # 线宽：0.3~0.8 都可以
        ax.spines[side].set_edgecolor((0, 0, 0, 0.6))
    x = np.arange(len(datasets))
    bar_width = 0.36
    offset = bar_width / 2

    # 马卡龙配色
    macaron_kendall  = "#BDE0FE"  # pastel blue
    macaron_spearman = "#FFD1DC"  # pastel pink

    bars1 = ax.bar(x - offset, kendall,  width=bar_width, label="Kendall τ",
                   color=macaron_kendall, edgecolor="none")
    bars2 = ax.bar(x + offset, spearman, width=bar_width, label="Spearman ρ",
                   color=macaron_spearman, edgecolor="none")

    ax.set_xlabel("Dataset")
    ax.set_ylabel("Correlation coefficient")

    ymin = min(0.0, float(min(np.nanmin(kendall), np.nanmin(spearman)) - 0.05))
    ymax = min(1.05, float(max(np.nanmax(kendall), np.nanmax(spearman)) + 0.08))
    if ymax - ymin < 0.3:
        ymax = ymin + 0.3
    ax.set_ylim(ymin, ymax)

    ax.set_xticks(x, datasets, rotation=0)
    ax.grid(axis="y", linestyle="--", alpha=0.25, linewidth=0.6)
    leg = ax.legend(
        ncols=2,  # 两个条目横向并排；想竖排就改成 1
        loc="lower right",
        bbox_to_anchor=(0.98, 0.02),  # 右下角稍微内缩
        frameon=True,  # 开启边框
        fancybox=True,  # 圆角
        framealpha=0.85,  # 边框/底色整体透明度（0~1）
        borderaxespad=0.6,
        labelspacing=0.4,
        handlelength=1.6,
        handletextpad=0.6
    )

    def annotate(bars):
        for b in bars:
            h = b.get_height()
            if np.isnan(h):
                continue
            ax.text(b.get_x() + b.get_width()/2,
                    h + 0.02*(1 if h >= 0 else -1),
                    f"{h:.{decimals}f}",
                    ha="center",
                    va=("bottom" if h >= 0 else "top"))

    annotate(bars1)
    annotate(bars2)

    fig.tight_layout()
    png_path = f"{out_base}.png"
    pdf_path = f"{out_base}.pdf"
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"[OK] Saved: {os.path.abspath(png_path)}")
    print(f"[OK] Saved: {os.path.abspath(pdf_path)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV path: dataset,kendall,spearman")
    ap.add_argument("--out", default="kt_barchart_macaron")
    ap.add_argument("--width", type=float, default=7.2)
    ap.add_argument("--height", type=float, default=3.8)
    ap.add_argument("--font_family", default=None, help="Force a font family (e.g., 'Times New Roman').")
    ap.add_argument("--font_size", type=int, default=10)
    ap.add_argument("--tick_size", type=int, default=9)
    ap.add_argument("--legend_size", type=int, default=9)
    ap.add_argument("--decimals", type=int, default=3)
    args = ap.parse_args()

    df = load_corr_csv(args.csv)
    plot_corr_bars(
        df,
        out_base=args.out,
        width=args.width,
        height=args.height,
        font_family=args.font_family,
        font_size=args.font_size,
        tick_size=args.tick_size,
        legend_size=args.legend_size,
        decimals=args.decimals,
    )

if __name__ == "__main__":
    main()


# python KT.py --csv ./results/kt.csv