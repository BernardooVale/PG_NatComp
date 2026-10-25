from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def summarize_basic(exp_dir: Path, label: str) -> dict:
    sdf = pd.read_csv(exp_dir / "summary.csv")
    hdf = pd.read_csv(exp_dir / "history.csv")

    g0 = hdf[hdf["generation"] == 0]
    g_last = hdf[hdf["generation"] == hdf["generation"].max()]
    g_mid = hdf[hdf["generation"] == min(20, int(hdf["generation"].max()))]

    grp_best = hdf.groupby("generation")["best_fitness"].mean()
    final_best = grp_best.iloc[-1]
    sat_gen = int(grp_best[grp_best <= final_best * 1.05].index[0])

    return {
        "Configuração": label,
        "Train RMSE (Média ± Std)": f"{sdf['train_rmse'].mean():.4f} ± {sdf['train_rmse'].std():.4f}",
        "Test RMSE (Média ± Std)": f"{sdf['test_rmse'].mean():.4f} ± {sdf['test_rmse'].std():.4f}",
        "Test RMSE Mediana (IQR)": f"{sdf['test_rmse'].median():.4f} ({sdf['test_rmse'].quantile(0.75) - sdf['test_rmse'].quantile(0.25):.4f})",
        "Test Min / Max": f"{sdf['test_rmse'].min():.4f} / {sdf['test_rmse'].max():.2f}",
        "Geração de Saturação (~5%)": sat_gen,
        "Repetidos (Gen 0 -> Gen 20 -> Final)": f"{g0['n_repeated'].mean():.1f} -> {g_mid['n_repeated'].mean():.1f} -> {g_last['n_repeated'].mean():.1f}",
        "Tamanho Médio Pop (Gen 0 -> Final)": f"{g0['mean_size'].mean():.1f} -> {g_last['mean_size'].mean():.1f}",
        "Tamanho Melhor Indiv. (Média ± Std)": f"{sdf['size'].mean():.1f} ± {sdf['size'].std():.1f}",
        "Tempo Médio (s)": f"{sdf['time_sec'].mean():.2f}",
    }


def summarize_operators(exp_dir: Path, label: str) -> dict:
    sdf = pd.read_csv(exp_dir / "summary.csv")
    hdf = pd.read_csv(exp_dir / "history.csv")
    h_pos = hdf[hdf["generation"] > 0]

    h_early = hdf[(hdf["generation"] >= 1) & (hdf["generation"] <= 20)]
    h_late = hdf[hdf["generation"] >= 81]

    per_run = h_pos.groupby("run")[
        [
            "cx_better_than_mean_parents",
            "cx_worse_than_mean_parents",
            "cx_better_than_best_parent",
            "cx_worse_than_best_parent",
            "mut_better",
            "mut_worse",
        ]
    ].sum()

    cx_bm = per_run["cx_better_than_mean_parents"].mean()
    cx_wm = per_run["cx_worse_than_mean_parents"].mean()
    cx_bb = per_run["cx_better_than_best_parent"].mean()
    cx_wb = per_run["cx_worse_than_best_parent"].mean()
    mb = per_run["mut_better"].mean()
    mw = per_run["mut_worse"].mean()

    cx_rate_mean = 100.0 * cx_bm / max(1e-9, cx_bm + cx_wm)
    cx_rate_best = 100.0 * cx_bb / max(1e-9, cx_bb + cx_wb)
    mut_rate = 100.0 * mb / max(1e-9, mb + mw)

    early_cx_bb = h_early["cx_better_than_best_parent"].mean()
    late_cx_bb = h_late["cx_better_than_best_parent"].mean()

    return {
        "Configuração": label,
        "Cx > Média Pais (Acum/Run)": f"{cx_bm:.1f} vs {cx_wm:.1f} ({cx_rate_mean:.1f}%)",
        "Cx > Melhor Pai (Acum/Run)": f"{cx_bb:.1f} vs {cx_wb:.1f} ({cx_rate_best:.1f}%)",
        "Cx > Melhor Pai/Gen (G1-20 -> G81-100)": f"{early_cx_bb:.2f} -> {late_cx_bb:.2f}",
        "Mut > Pai (Acum/Run)": f"{mb:.1f} vs {mw:.1f} ({mut_rate:.1f}%)",
        "Train RMSE (Média ± Std)": f"{sdf['train_rmse'].mean():.4f} ± {sdf['train_rmse'].std():.4f}",
        "Test RMSE (Média ± Std)": f"{sdf['test_rmse'].mean():.4f} ± {sdf['test_rmse'].std():.4f}",
        "Test Mediana (IQR)": f"{sdf['test_rmse'].median():.4f} ({sdf['test_rmse'].quantile(0.75) - sdf['test_rmse'].quantile(0.25):.4f})",
        "Tamanho (Nós)": f"{sdf['size'].mean():.1f} ± {sdf['size'].std():.1f}",
    }


def main():
    rdir = Path("results")
    p2dir = rdir / "phase2_audit"
    out = rdir / "relatorio_final_dados"
    out.mkdir(parents=True, exist_ok=True)

    pop_gen_rows = [
        summarize_basic(rdir / "exp11_2_pop50_gen100", "Pop=50, Gen=100 (5.000 av.)"),
        summarize_basic(rdir / "exp11_2_pop100_gen50", "Pop=100, Gen=50 (5.000 av.)"),
        summarize_basic(rdir / "exp11_2_pop100_gen100", "Pop=100, Gen=100 [Baseline] (10.000 av.)"),
        summarize_basic(rdir / "exp11_2_pop500_gen100", "Pop=500, Gen=100 (50.000 av.)"),
        summarize_basic(rdir / "exp11_2_pop100_gen500", "Pop=100, Gen=500 (50.000 av.)"),
    ]
    df_pg = pd.DataFrame(pop_gen_rows)
    df_pg.to_csv(out / "01_populacao_e_geracoes.csv", index=False)

    sel_rows = [
        summarize_basic(rdir / "exp11_3_sel_roulette", "synth1: Roleta"),
        summarize_basic(rdir / "exp11_3_sel_tournament_k2", "synth1: Torneio (k=2)"),
        summarize_basic(rdir / "exp11_3_sel_tournament_k3", "synth1: Torneio (k=3)"),
        summarize_basic(rdir / "exp11_3_sel_lexicase", "synth1: ε-Lexicase (MAD)"),
        summarize_basic(rdir / "exp11_8_replication_synth2", "synth2: Torneio (k=2) [Fase 1]"),
        summarize_basic(p2dir / "synth2_S0_estado_atual_lexicase", "synth2: ε-Lexicase (MAD) [Fase 2 S0]"),
        summarize_basic(rdir / "exp11_8_replication_concrete1_norm", "concrete1 (Norm): Torneio (k=2) [Fase 1]"),
        summarize_basic(p2dir / "concrete1_norm_S0_estado_atual_lexicase", "concrete1 (Norm): ε-Lexicase (MAD) [Fase 2 S0]"),
    ]
    df_sel = pd.DataFrame(sel_rows)
    df_sel.to_csv(out / "02_metodos_de_selecao.csv", index=False)

    op_rows = [
        summarize_operators(rdir / "exp11_4_config_A", "Fase 1 Config A (pc=0.90, pm=0.05, Subtree, Torneio)"),
        summarize_operators(rdir / "exp11_4_config_B", "Fase 1 Config B (pc=0.60, pm=0.30, Subtree, Torneio)"),
        summarize_operators(rdir / "exp11_4_sweep_pc_0.5", "Fase 1 Sweep (pc=0.50, pm=0.05, Subtree, Torneio)"),
        summarize_operators(rdir / "exp11_4_sweep_pc_0.7", "Fase 1 Sweep (pc=0.70, pm=0.05, Subtree, Torneio)"),
        summarize_operators(rdir / "exp11_4_sweep_pm_0.01", "Fase 1 Sweep (pc=0.90, pm=0.01, Subtree, Torneio)"),
        summarize_operators(rdir / "exp11_4_sweep_pm_0.1", "Fase 1 Sweep (pc=0.90, pm=0.10, Subtree, Torneio)"),
        summarize_operators(rdir / "exp11_4_sweep_pm_0.2", "Fase 1 Sweep (pc=0.90, pm=0.20, Subtree, Torneio)"),
        summarize_operators(p2dir / "synth1_S0_estado_atual_lexicase", "Fase 2 S0 (pc=0.90, pm=0.05, Subtree, Lexicase)"),
        summarize_operators(p2dir / "synth1_H2_op_rebalance_mixed_mut", "Fase 2 H2 (pc=0.70, pm=0.15, Mista, Lexicase)"),
        summarize_operators(p2dir / "synth1_StackA2_H3_H5_H2_ops", "Fase 2 Stack A2 (pc=0.70, pm=0.15, Mista+Dedup+Parcim.)"),
    ]
    df_op = pd.DataFrame(op_rows)
    df_op.to_csv(out / "03_crossover_e_mutacao.csv", index=False)

    el_rows = [
        summarize_basic(rdir / "exp11_1_baseline", "Com Elitismo (elite_size=1)"),
        summarize_basic(rdir / "exp11_5_elitism_false", "Sem Elitismo (elite_size=0)"),
    ]
    df_el = pd.DataFrame(el_rows)
    df_el.to_csv(out / "04_elitismo.csv", index=False)

    bl_rows = [
        summarize_basic(rdir / "exp11_1_baseline", "Fase 1: Max Depth = 7 (Torneio k=2)"),
        summarize_basic(rdir / "exp11_6_bloating_maxsize10", "Fase 1: Max Depth = 10 (Torneio k=2)"),
        summarize_basic(rdir / "exp11_2_pop100_gen500", "Fase 1: Max Depth = 7, 500 Gerações (Torneio k=2)"),
        summarize_basic(p2dir / "synth1_S0_estado_atual_lexicase", "Fase 2 S0: Max Depth = 7 (ε-Lexicase sem Parcimônia)"),
        summarize_basic(p2dir / "synth1_H5_parsimony_pressure", "Fase 2 H5: Max Depth = 7 (ε-Lexicase + Parcimônia)"),
        summarize_basic(p2dir / "synth1_StackA2_H3_H5_H2_ops", "Fase 2 Stack A2: ε-Lexicase + Parcimônia + Dedup + H2"),
        summarize_basic(rdir / "exp11_8_replication_concrete1_norm", "concrete1 Fase 1: Max Depth = 7 (Torneio k=2)"),
        summarize_basic(p2dir / "concrete1_norm_StackB4_Full_Nonlinear_AQ", "concrete1 Fase 2 Stack B4: Parcimônia + Scaling + Não-Lin"),
    ]
    df_bl = pd.DataFrame(bl_rows)
    df_bl.to_csv(out / "05_bloating_e_tamanho.csv", index=False)

    for name, df in [
        ("01_populacao_e_geracoes", df_pg),
        ("02_metodos_de_selecao", df_sel),
        ("03_crossover_e_mutacao", df_op),
        ("04_elitismo", df_el),
        ("05_bloating_e_tamanho", df_bl),
    ]:
        (out / f"{name}.md").write_text(df.to_markdown(index=False), encoding="utf-8")
        print(f"=== {name} ===")
        print(df.to_markdown(index=False))


if __name__ == "__main__":
    main()
