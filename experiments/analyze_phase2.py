from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from experiments.run_audit_phase2 import compare_paired

plt.switch_backend("Agg")


def summarize_exp(df: pd.DataFrame, df_s0: pd.DataFrame, exp_name: str, label: str) -> Dict[str, Any]:
    comp = compare_paired(df_s0, df, "test_rmse")
    tr_m, tr_s = df["train_rmse"].mean(), df["train_rmse"].std()
    te_m, te_s = comp["new_mean"], comp["new_std"]
    te_med, te_iqr = comp["new_median"], comp["new_iqr"]
    te_min = float(df["test_rmse"].min())
    te_max = comp["new_max"]
    sz_m, sz_s = df["size"].mean(), df["size"].std()
    t_m = df["time_sec"].mean()

    return {
        "exp_id": exp_name,
        "Configuration": label,
        "Train RMSE (Mean±Std)": f"{tr_m:.4f} ± {tr_s:.4f}",
        "Test RMSE (Mean±Std)": f"{te_m:.4f} ± {te_s:.4f}",
        "Test Median (IQR)": f"{te_med:.4f} ({te_iqr:.4f})",
        "Test Min / Max": f"{te_min:.3f} / {te_max:.2f}",
        "Size (Mean±Std)": f"{sz_m:.1f} ± {sz_s:.1f}",
        "W/L vs S0": f"{comp['wins']}W / {comp['losses']}L",
        "p (Wilcoxon)": f"{comp['p_wilcoxon']:.2e}",
        "p (Mann-Whitney)": f"{comp['p_mannwhitney']:.2e}",
        "test_mean_raw": te_m,
        "test_std_raw": te_s,
        "test_median_raw": te_med,
        "test_iqr_raw": te_iqr,
        "test_max_raw": te_max,
        "size_mean_raw": sz_m,
        "p_wilc_raw": comp["p_wilcoxon"],
        "p_mw_raw": comp["p_mannwhitney"],
        "wins": comp["wins"],
        "losses": comp["losses"],
    }


def generate_plots(out_dir: Path, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Stacking Ladder Progression (3 subplots: synth1, synth2, concrete1_norm)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))

    # synth1 ladder
    s1_steps = [
        ("Fase 1\n(Torneio)", Path("results/exp11_1_baseline")),
        ("Estado Atual\n(S0 Lexicase)", out_dir / "synth1_S0_estado_atual_lexicase"),
        ("+H3 (Dedup)\n[Isolado]", out_dir / "synth1_H3_dedup_diversity"),
        ("+H5 (Parcim.)\n[Isolado]", out_dir / "synth1_H5_parsimony_pressure"),
        ("Stack A1\n(S0+H3+H5)", out_dir / "synth1_StackA1_H3_dedup_H5_parsimony"),
        ("Stack A2\n(+H2 Oper.)", out_dir / "synth1_StackA2_H3_H5_H2_ops"),
    ]
    # synth2 ladder
    s2_steps = [
        ("Fase 1\n(Torneio)", Path("results/exp11_8_replication_synth2")),
        ("Estado Atual\n(S0 Lexicase)", out_dir / "synth2_S0_estado_atual_lexicase"),
        ("+H1 (AQ)\n[Isolado]", out_dir / "synth2_H1_analytic_quotient"),
        ("Stack B1\n(S0+H6a+H1)", out_dir / "synth2_StackB1_H6a_scaling_H1_aq"),
        ("Stack B2\n(+H3+H5)", out_dir / "synth2_StackB2_B1_plus_H3_H5"),
        ("Novo Estado\n(B3: +H2 Ops)", out_dir / "synth2_StackB3_B2_plus_H2_ops"),
    ]
    # concrete1_norm ladder
    c1_steps = [
        ("Fase 1\n(Torneio)", Path("results/exp11_8_replication_concrete1_norm")),
        ("Estado Atual\n(S0 Lexicase)", out_dir / "concrete1_norm_S0_estado_atual_lexicase"),
        ("Stack B1\n(S0+H6a+H1)", out_dir / "concrete1_norm_StackB1_H6a_scaling_H1_aq"),
        ("Stack B2\n(+H3+H5)", out_dir / "concrete1_norm_StackB2_B1_plus_H3_H5"),
        ("Stack B3\n(+H2 Ops)", out_dir / "concrete1_norm_StackB3_B2_plus_H2_ops"),
        ("Novo Estado\n(B4: +Não-Lin)", out_dir / "concrete1_norm_StackB4_Full_Nonlinear_AQ"),
    ]

    for ax, (title, steps) in zip(
        axes,
        [
            ("synth1: Escada Auditável (RMSE Teste)", s1_steps),
            ("synth2: Escada Auditável (RMSE Teste)", s2_steps),
            ("concrete1: Escada Auditável (RMSE Teste)", c1_steps),
        ],
    ):
        labels = []
        means = []
        medians = []
        q25s = []
        q75s = []
        for lbl, p in steps:
            df = pd.read_csv(p / "summary.csv")
            vals = df["test_rmse"].to_numpy()
            labels.append(lbl)
            means.append(np.mean(vals))
            medians.append(np.median(vals))
            q25s.append(np.percentile(vals, 25))
            q75s.append(np.percentile(vals, 75))

        x = np.arange(len(labels))
        ax.plot(x, means, "o--", color="#d95f02", label="Média RMSE Teste", linewidth=2, markersize=7)
        ax.plot(x, medians, "s-", color="#1b9e77", label="Mediana RMSE Teste", linewidth=2.2, markersize=7)
        ax.fill_between(x, q25s, q75s, color="#1b9e77", alpha=0.18, label="IQR (25%-75%)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel("Test RMSE (Menor = Melhor)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(fig_dir / "stacking_ladder_progression.png", dpi=300)
    plt.close()

    # 2. Convergence Curves: Fase 1 vs Estado Atual (S0) vs Novo Estado Final
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    conv_configs = [
        (
            "synth1: Convergência de Treino",
            [
                ("Fase 1 (Torneio k=2)", Path("results/exp11_1_baseline")),
                ("Estado Atual (S0 Lexicase)", out_dir / "synth1_S0_estado_atual_lexicase"),
                ("Novo Estado A1 (S0+H3+H5)", out_dir / "synth1_StackA1_H3_dedup_H5_parsimony"),
                ("Novo Estado A2 (S0+H3+H5+H2)", out_dir / "synth1_StackA2_H3_H5_H2_ops"),
            ],
        ),
        (
            "synth2: Convergência de Treino",
            [
                ("Fase 1 (Torneio k=2)", Path("results/exp11_8_replication_synth2")),
                ("Estado Atual (S0 Lexicase)", out_dir / "synth2_S0_estado_atual_lexicase"),
                ("H1 Isolado (Analytic Quotient)", out_dir / "synth2_H1_analytic_quotient"),
                ("Novo Estado B3 (H6a+H1+H3+H5+H2)", out_dir / "synth2_StackB3_B2_plus_H2_ops"),
            ],
        ),
        (
            "concrete1: Convergência de Treino",
            [
                ("Fase 1 (Torneio k=2)", Path("results/exp11_8_replication_concrete1_norm")),
                ("Estado Atual (S0 Lexicase)", out_dir / "concrete1_norm_S0_estado_atual_lexicase"),
                ("Stack B1 (S0+H6a+H1)", out_dir / "concrete1_norm_StackB1_H6a_scaling_H1_aq"),
                ("Novo Estado B4 (Stack Completo)", out_dir / "concrete1_norm_StackB4_Full_Nonlinear_AQ"),
            ],
        ),
    ]

    for ax, (title, runs) in zip(axes, conv_configs):
        for lbl, p in runs:
            hdf = pd.read_csv(p / "history.csv")
            grp = hdf.groupby("generation")["best_fitness"].agg(["mean", "std"]).reset_index()
            ax.plot(grp["generation"], grp["mean"], label=lbl, linewidth=1.9)
            ax.fill_between(
                grp["generation"],
                np.maximum(0, grp["mean"] - grp["std"]),
                grp["mean"] + grp["std"],
                alpha=0.12,
            )
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Geração")
        ax.set_ylabel("Melhor RMSE de Treino")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(fig_dir / "convergence_s0_vs_novo_estado.png", dpi=300)
    plt.close()


def main():
    out_dir = Path("results/phase2_audit")
    fig_dir = out_dir / "figures"
    generate_plots(out_dir, fig_dir)

    datasets = ["synth1", "synth2", "concrete1_norm"]
    exp_labels = [
        ("S0_estado_atual_lexicase", "S0: Estado Atual (ε-Lexicase)"),
        ("H1_analytic_quotient", "H1 [Polos]: Analytic Quotient (AQ)"),
        ("H2_op_rebalance_mixed_mut", "H2 [Rugosidade]: Pc=0.7, Pm=0.15 + Mut. Mista"),
        ("H3_dedup_diversity", "H3 [Diversidade]: Rejeição de Duplicatas"),
        ("H4_island_model", "H4 [Bacias]: Modelo de Ilhas (4x25)"),
        ("H5_parsimony_pressure", "H5 [Redes Neutras]: Pressão de Parcimônia"),
        ("H6a_linear_scaling", "H6a [Escala]: Linear Scaling (OLS)"),
        ("H6b_nonlinear_plus_scaling", "H6b [Expressividade]: Não-Lineares + Scaling"),
        ("StackA1_H3_dedup_H5_parsimony", "Stack A1: S0 + H3 (Dedup) + H5 (Parcimônia)"),
        ("StackA2_H3_H5_H2_ops", "Stack A2: Stack A1 + H2 (Operadores)"),
        ("StackB1_H6a_scaling_H1_aq", "Stack B1: S0 + H6a (Scaling) + H1 (AQ)"),
        ("StackB2_B1_plus_H3_H5", "Stack B2: Stack B1 + H3 (Dedup) + H5 (Parcim.)"),
        ("StackB3_B2_plus_H2_ops", "Stack B3: Stack B2 + H2 (Operadores)"),
        ("StackB4_Full_Nonlinear_AQ", "Stack B4: Stack B3 + H6b (Não-Lineares)"),
    ]

    all_rows = []
    for ds in datasets:
        df_s0 = pd.read_csv(out_dir / f"{ds}_S0_estado_atual_lexicase" / "summary.csv")
        for suffix, label in exp_labels:
            p = out_dir / f"{ds}_{suffix}" / "summary.csv"
            if p.exists():
                df = pd.read_csv(p)
                row = summarize_exp(df, df_s0, f"{ds}_{suffix}", label)
                row["Dataset"] = ds
                all_rows.append(row)

    cons_df = pd.DataFrame(all_rows)
    cons_df.to_csv(out_dir / "phase2_consolidated.csv", index=False)
    print("Saved phase2_consolidated.csv and figures successfully.")


if __name__ == "__main__":
    main()
