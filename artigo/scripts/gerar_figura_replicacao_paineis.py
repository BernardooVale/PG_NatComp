"""Gera a figura de replicação em dois painéis (com vs sem outliers extremos / zoom).

Painel (a): Escala completa exibindo os outliers catastróficos no synth2.
Painel (b): Detalhe da distribuição (zoom / sem outliers extremos) evidenciando as medianas e IQR.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
OUT = ROOT / "artigo" / "figuras"
OUT.mkdir(parents=True, exist_ok=True)

s1 = pd.read_csv(RES / "exp11_1_baseline" / "summary.csv").test_rmse.values
s2 = pd.read_csv(RES / "exp11_8_replication_synth2" / "summary.csv").test_rmse.values
cr = pd.read_csv(RES / "exp11_8_replication_concrete1_raw" / "summary.csv").test_rmse.values
cn = pd.read_csv(RES / "exp11_8_replication_concrete1_norm" / "summary.csv").test_rmse.values

data = [s1, s2, cr, cn]
labels = ["synth1\n(Baseline)", "synth2\n(Replicação)", "concrete1\n(Bruto)", "concrete1\n(Normalizado)"]
colors = ["#a6cee3", "#fb9a99", "#fdbf6f", "#b2df8a"]

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

# Painel (a): Escala Completa
bp1 = ax1.boxplot(data, tick_labels=labels, patch_artist=True, showmeans=True,
                  meanprops=dict(marker='^', markerfacecolor='green', markeredgecolor='green', markersize=7))
for patch, color in zip(bp1['boxes'], colors):
    patch.set_facecolor(color)
ax1.set_title("(a) Escala Completa (com outliers extremos)", fontsize=11, fontweight="bold")
ax1.set_ylabel("RMSE de Teste", fontsize=10)
# Anotação do outlier
ax1.annotate("Outlier catastrófico:\nSeed 20 (415.26)", xy=(2, 415.26), xytext=(2.2, 340),
             arrowprops=dict(facecolor='red', arrowstyle="->", lw=1.2),
             fontsize=9, color="darkred", fontweight="semibold")

# Painel (b): Detalhe / Zoom (sem outliers extremos)
bp2 = ax2.boxplot(data, tick_labels=labels, patch_artist=True, showmeans=True, showfliers=False,
                  meanprops=dict(marker='^', markerfacecolor='green', markeredgecolor='green', markersize=7))
for patch, color in zip(bp2['boxes'], colors):
    patch.set_facecolor(color)
ax2.set_title("(b) Detalhe das Distribuições (sem outliers extremos)", fontsize=11, fontweight="bold")
ax2.set_ylabel("RMSE de Teste", fontsize=10)
ax2.set_ylim(-0.5, 18.5)

# Legenda para a média
ax2.plot([], [], '^', color='green', label='Média')
ax2.plot([], [], '-', color='orange', label='Mediana')
ax2.legend(loc="upper left", fontsize=9)

fig.tight_layout()
fig.savefig(OUT / "replication_all_datasets_two_panels.png", dpi=250)
plt.close(fig)

print("Figura gerada com sucesso em:", OUT / "replication_all_datasets_two_panels.png")
