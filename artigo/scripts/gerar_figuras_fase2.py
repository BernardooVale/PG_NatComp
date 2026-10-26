"""Gera as figuras comparativas Fase 1 x Fase 2 usadas no artigo.

Uso (a partir da raiz do repositório):
    .\\.venv\\Scripts\\python.exe artigo/scripts/gerar_figuras_fase2.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
P2 = RES / "phase2_audit"
OUT = ROOT / "artigo" / "figuras"
OUT.mkdir(parents=True, exist_ok=True)

DATASETS = [("synth1", "synth1"), ("synth2", "synth2"), ("concrete1_norm", "concrete1")]
PHASE1 = {
    "synth1": RES / "exp11_1_baseline",
    "synth2": RES / "exp11_8_replication_synth2",
    "concrete1_norm": RES / "exp11_8_replication_concrete1_norm",
}
BEST = {
    "synth1": "StackA2_H3_H5_H2_ops",
    "synth2": "StackB3_B2_plus_H2_ops",
    "concrete1_norm": "StackB4_Full_Nonlinear_AQ",
}
LABEL_BEST = {"synth1": "Stack A2", "synth2": "Stack B3", "concrete1_norm": "Stack B4"}

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})


def summary(ds, cfg):
    return pd.read_csv(P2 / f"{ds}_{cfg}" / "summary.csv")


def history(ds, cfg):
    return pd.read_csv(P2 / f"{ds}_{cfg}" / "history.csv")


# ---------------------------------------------------------------- Boxplots
fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
for ax, (ds, name) in zip(axes, DATASETS):
    data = [
        pd.read_csv(PHASE1[ds] / "summary.csv").test_rmse.values,
        summary(ds, "S0_estado_atual_lexicase").test_rmse.values,
        summary(ds, "StackA2_H3_H5_H2_ops").test_rmse.values,
        summary(ds, "StackB3_B2_plus_H2_ops").test_rmse.values,
        summary(ds, "StackB4_Full_Nonlinear_AQ").test_rmse.values,
    ]
    labels = ["F1\nTorneio", "S0\nLexicase", "Stack\nA2", "Stack\nB3", "Stack\nB4"]
    bp = ax.boxplot(data, tick_labels=labels, showfliers=True, patch_artist=True)
    colors = ["#bbbbbb", "#9ecae1", "#a1d99b", "#fdae6b", "#fc9272"]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
    ax.set_yscale("log")
    ax.set_title(name)
    ax.set_ylabel("RMSE de teste (log)")
fig.tight_layout()
fig.savefig(OUT / "f2_boxplot_bases.png", dpi=200)
plt.close(fig)

# ------------------------------------------------- Heatmap das hipóteses isoladas
hyps = [
    ("H1_analytic_quotient", "H1 (AQ)"),
    ("H2_op_rebalance_mixed_mut", "H2 (Operadores)"),
    ("H3_dedup_diversity", "H3 (Dedup)"),
    ("H4_island_model", "H4 (Ilhas)"),
    ("H5_parsimony_pressure", "H5 (Parcimônia)"),
    ("H6a_linear_scaling", "H6a (Scaling)"),
    ("H6b_nonlinear_plus_scaling", "H6b (Não-lin.+Scaling)"),
]
M = np.zeros((len(hyps), 3))
for j, (ds, _) in enumerate(DATASETS):
    s0 = summary(ds, "S0_estado_atual_lexicase").test_rmse.median()
    for i, (cfg, _) in enumerate(hyps):
        M[i, j] = summary(ds, cfg).test_rmse.median() / s0
fig, ax = plt.subplots(figsize=(5.2, 3.6))
L = np.log2(M)
im = ax.imshow(L, cmap="RdYlGn_r", vmin=-2.5, vmax=2.5, aspect="auto")
ax.set_xticks(range(3), [n for _, n in DATASETS])
ax.set_yticks(range(len(hyps)), [n for _, n in hyps])
ax.grid(False)
for i in range(len(hyps)):
    for j in range(3):
        ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8)
cb = fig.colorbar(im, ax=ax)
cb.set_label(r"$\log_2$(mediana / mediana S0)")
ax.set_title("Mediana do RMSE de teste relativa a S0")
fig.tight_layout()
fig.savefig(OUT / "f2_heatmap_hipoteses.png", dpi=200)
plt.close(fig)

# --------------------------------------------------- Dinâmica (convergência, diversidade, tamanho)
fig, axes = plt.subplots(3, 3, figsize=(10, 7.5), sharex=True)
for j, (ds, name) in enumerate(DATASETS):
    series = [
        ("Fase 1 (Torneio)", pd.read_csv(PHASE1[ds] / "history.csv"), "#555555"),
        ("S0 (Lexicase)", history(ds, "S0_estado_atual_lexicase"), "#3182bd"),
        (LABEL_BEST[ds] + " (melhor)", history(ds, BEST[ds]), "#e6550d"),
    ]
    for lab, h, c in series:
        g = h.groupby("generation").median(numeric_only=True)
        axes[0, j].plot(g.index, g.best_fitness, label=lab, color=c)
        axes[1, j].plot(g.index, h.groupby("generation").n_repeated.mean(), color=c)
        axes[2, j].plot(g.index, h.groupby("generation").mean_size.mean(), color=c)
    axes[0, j].set_title(name)
    axes[0, j].set_yscale("log")
    axes[2, j].set_xlabel("Geração")
axes[0, 0].set_ylabel("Melhor RMSE treino\n(mediana, log)")
axes[1, 0].set_ylabel("Indivíduos repetidos\n(média)")
axes[2, 0].set_ylabel("Tamanho médio da\npopulação (nós)")
axes[0, 0].legend(fontsize=7)
for j, (ds, _) in enumerate(DATASETS):
    axes[0, j].legend([l.get_label() for l in axes[0, j].lines], fontsize=6, loc="upper right")
fig.tight_layout()
fig.savefig(OUT / "f2_dinamica_comparada.png", dpi=200)
plt.close(fig)

print("Figuras salvas em", OUT)
