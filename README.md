# Framework de Programação Genética (GP) para Regressão Simbólica

Implementação em Python de um algoritmo completo e extensível de Programação Genética (GP) com indivíduos representados em árvore para problemas de Regressão Simbólica, desenvolvido para a disciplina de Computação Natural.

O projeto inclui o estudo experimental canônico inicial (**Fase 1 — Seção 11**) e um estudo de melhorias auditáveis guiado por análise da *fitness landscape* (**Fase 2**), cobrindo 30 repetições com sementes independentes em três bases de dados (`synth1`, `synth2` e `concrete1`).

---

## 1. Inspeção dos Datasets

Os dados estão localizados no diretório `datasets/`. Todos os arquivos foram inspecionados quanto à presença de cabeçalho, tipos, valores nulos e dimensões:

| Dataset | Arquivo | Amostras | Atributos ($X$) | Saída ($y$) | Separador | Header | NaNs | Tipos |
|---|---|---|---|---|---|---|---|---|
| **synth1** | `synth1-train.csv` | 60 | 2 (`x0`, `x1`) | 1 (`y`) | Vírgula (`,`) | Não | 0 | `float64` |
| **synth1** | `synth1-test.csv` | 600 | 2 (`x0`, `x1`) | 1 (`y`) | Vírgula (`,`) | Não | 0 | `float64` |
| **synth2** | `synth2-train.csv` | 300 | 2 (`x0`, `x1`) | 1 (`y`) | Vírgula (`,`) | Não | 0 | `float64` |
| **synth2** | `synth2-test.csv` | 1000 | 2 (`x0`, `x1`) | 1 (`y`) | Vírgula (`,`) | Não | 0 | `float64` |
| **concrete1** | `concrete-train.csv` | 824 | 8 (`x0..x7`) | 1 (`y`) | Vírgula (`,`) | Não | 0 | `float64` / `int64` |
| **concrete1** | `concrete-test.csv` | 206 | 8 (`x0..x7`) | 1 (`y`) | Vírgula (`,`) | Não | 0 | `float64` / `int64` |

---

## 2. Decisões de Implementação e Arquitetura

1. **Genótipo e Avaliação Vetorizada** ([`gp/tree.py`](gp/tree.py)):
   - Implementado na classe `Node`. Suporta nós de função, variáveis (`x0, x1, ...`) e constantes efêmeras (ERC).
   - Avaliação totalmente vetorizada via NumPy sobre a matriz $X \in \mathbb{R}^{N \times d}$.
   - Classe `Individual` encapsula raiz, metadados de origem, cache de avaliação e parâmetros de escalonamento linear.

2. **Divisão Protegida, Quociente Analítico e Primitivas** ([`gp/primitives.py`](gp/primitives.py)):
   - **Divisão Protegida (Koza)**: Retorna `1.0` se $|b| < 10^{-6}$.
   - **Quociente Analítico (Analytic Quotient - AQ)**: $\text{aq}(a, b) = a / \sqrt{1 + b^2}$, contínuo e diferenciável em todo o domínio, eliminando polos e assíntotas verticais.
   - Funções protegidas adicionais: `exp` (com *clipping* em $[-50, 50]$), `log` protegido ($\log(|a| + 10^{-6})$) e `sqrt` protegido ($\sqrt{|a|}$).
   - Tratamento de `NaN`/`inf`: Penalidade estrita de aptidão (`1e12`).

3. **Escalonamento Linear Ótimo (Linear Scaling)** ([`gp/fitness.py`](gp/fitness.py)):
   - Ajuste analítico em $\mathcal{O}(N)$ dos coeficientes $(\alpha, \beta)$ por Mínimos Quadrados Ordinários (OLS) no conjunto de treino: $\hat{y}_{\text{scaled}} = \alpha + \beta \hat{y}$.
   - Desacopla a busca da forma funcional da escala/deslocamento numérico, crucial para dados reais de engenharia.

4. **Inicialização e Limites de Tamanho** ([`gp/init.py`](gp/init.py)):
   - Suporta os métodos `grow`, `full` e `ramped_half_and_half` (padrão).
   - Modos de tamanho `depth` (limite padrão de profundidade 7) e `nodes`.

5. **Operadores Genéticos e Mutação Mista** ([`gp/operators.py`](gp/operators.py)):
   - **Crossover de Subárvore**: Seleção enviesada para nós internos, preservando estritamente a imutabilidade dos pais.
   - **Mutação de Subárvore**: Substituição por subárvore gerada via `grow`.
   - **Mutação Pontual**: Substituição conservativa de primitivas mantendo a aridade.
   - **Mutação Mista (`mixed`)**: Combina perturbação Gaussiana $\mathcal{N}(0, \sigma^2)$ em constantes ERC ($35\%$), mutação pontual ($35\%$) e micro-subárvores ($30\%$), suavizando a navegação na paisagem rugosa.

6. **Métodos de Seleção e Parcimônia** ([`gp/selection.py`](gp/selection.py)):
   - **Torneio**: Seleção clássica por tamanho de torneio $k$.
   - **Roleta**: Mapeamento $1 / (1 + \text{RMSE})$.
   - **$\varepsilon$-Lexicase (MAD)**: Filtragem caso a caso com limiar adaptativo de desvio absoluto mediano ($\text{MAD}$).
   - **Pressão de Parcimônia Lexicográfica**: Desempate por menor tamanho de árvore entre os finalistas do $\varepsilon$-Lexicase dentro de $5\%$ de tolerância do melhor erro.

7. **Preservação de Diversidade e Topologia** ([`gp/engine.py`](gp/engine.py)):
   - **Rejeição de Duplicatas (`dedup_offspring`)**: Mutação de clones canônicos antes da inserção na nova geração.
   - **Modelo de Ilhas (`n_islands`)**: Suporte a subpopulações semi-isoladas com migração em anel periódica.
   - **Elitismo**: Preserva o melhor indivíduo absoluto de treino com desempate parcimonioso.

8. **Configuração e Reprodutibilidade** ([`gp/config.py`](gp/config.py)):
   - Centralizado na dataclass `GPConfig`, validada, carregável via YAML e sobrescrevível por CLI.
   - Determinismo estrito via gerador de números aleatórios NumPy (`numpy.random.default_rng(seed)`).

---

## 3. Resumo dos Resultados: Fase 1 vs. Fase 2

O projeto comparou a evolução metodológica completa em **30 repetições independentes** (`seeds 0..29`) por configuração:

| Base | Fase 1 Baseline (Torneio $k=2$) | Fase 2 Estado Atual (`S0` Lexicase) | Fase 2 Novo Estado (Metodologia Ótima) | Melhoria no Teste (Fase 1 $\to$ Novo Estado) |
|---|:---:|:---:|:---:|:---:|
| **`synth1`** | $5.5209 \pm 4.8934$ (Med: $5.895$) | $1.6172 \pm 3.3648$ (Med: $0.676$) | **$0.6875 \pm 1.8073$ (Mediana: $0.0933$)** | **$-87.5\%$** no erro médio (Mediana caiu de $5.89 \to \mathbf{0.093}$, erro mín $= \mathbf{0.0006}$) |
| **`synth2`** | $24.9658 \pm 80.3860$ (Máx: $415.26$) | $13.1473 \pm 17.4791$ (Máx: $86.55$) | **$1.5208 \pm 0.2558$ (Máx: $2.32$)** | **$-93.9\%$** no erro médio (eliminação de $100\%$ das explosões assintóticas de divisão) |
| **`concrete1`** | $11.9689 \pm 1.5085$ ($R^2 = 0.44$, 74 nós) | $12.1875 \pm 1.5881$ (60 nós) | **$7.7186 \pm 0.6438$ ($R^2 = \mathbf{0.78}$, $\mathbf{43}$ nós)** | **$-35.5\%$** no erro de teste, $R^2$ saltou de $0.44 \to \mathbf{0.78}$ com árvores **$41.6\%$ menores** |

---

## 4. Como Executar

Certifique-se de utilizar o Python do ambiente virtual `.venv`:

### 4.1 Executar os Testes Unitários (19 testes cobrindo Fase 1 e Fase 2)
```bash
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

### 4.2 Executar uma Execução Individual (CLI)
```bash
.\.venv\Scripts\python.exe experiments/run_single.py --dataset synth1 --seed 42 --exp_name single_test --verbose
```

### 4.3 Executar um Lote em Paralelo
```bash
.\.venv\Scripts\python.exe experiments/run_batch.py --dataset synth1 --runs 30 --n_jobs -1 --exp_name batch_baseline
```

### 4.4 Reproduzir o Estudo Inicial da Fase 1 (Seção 11)
```bash
.\.venv\Scripts\python.exe experiments/run_study.py --runs 30 --n_jobs -1
```

### 4.5 Reproduzir a Auditoria e a Escada da Fase 2
```bash
# Executa a ablação isolada das hipóteses H1 a H6b nas 3 bases (30 seeds cada):
.\.venv\Scripts\python.exe experiments/run_audit_phase2.py

# Executa a escada de empilhamento sequencial até o Novo Estado:
.\.venv\Scripts\python.exe experiments/run_stacking_phase2.py

# Gera as tabelas consolidadas e gráficos comparativos da Fase 2:
.\.venv\Scripts\python.exe experiments/analyze_phase2.py
```

### 4.6 Exportar as Tabelas Tratadas para o Relatório Final
```bash
.\.venv\Scripts\python.exe experiments/export_report_tables.py
```

---

## 5. Estrutura do Repositório

```
.
├── configs/
│   └── default.yaml                        # Configuração padrão em YAML
├── datasets/
│   ├── concrete/                           # concrete-train.csv e concrete-test.csv
│   ├── synth1/                             # synth1-train.csv e synth1-test.csv
│   └── synth2/                             # synth2-train.csv e synth2-test.csv
├── gp/
│   ├── __init__.py
│   ├── config.py                           # GPConfig com suporte a H1-H6b e CLI
│   ├── data.py                             # Carga e normalização z-score de dados
│   ├── engine.py                           # Laço evolutivo, ilhas, dedup e métricas
│   ├── fitness.py                          # RMSE e Linear Scaling via OLS
│   ├── init.py                             # Métodos full, grow e ramped_half_and_half
│   ├── operators.py                        # Crossover, mutação subárvore, pontual e mista
│   ├── primitives.py                       # Conjunto funcional, div protegida e AQ
│   ├── selection.py                        # Torneio, Roleta, ε-Lexicase e parcimônia
│   ├── stats.py                            # Coleta geracional de erros, operadores e bloat
│   └── tree.py                             # Estrutura Node, Individual e avaliação NumPy
├── experiments/
│   ├── analyze.py                          # Análise e gráficos da Fase 1
│   ├── analyze_phase2.py                   # Análise, gráficos e tabelas da Fase 2
│   ├── export_report_tables.py             # Exportador de tabelas temáticas do relatório
│   ├── run_audit_phase2.py                 # Runner da bateria de ablação isolada
│   ├── run_batch.py                        # Execução paralela em lote com N seeds
│   ├── run_single.py                       # Execução única via CLI
│   ├── run_stacking_phase2.py              # Runner da escada de empilhamento cumulativo
│   └── run_study.py                        # Roteiro experimental da Seção 11 (Fase 1)
├── results/
│   ├── figures/                            # Gráficos da Fase 1 (convergência, boxplots, bloat)
│   ├── phase2_audit/                       # Logs, summaries e gráficos da Fase 2
│   │   ├── figures/                        # Gráficos da escada e convergência da Fase 2
│   │   └── phase2_consolidated.csv         # Tabela consolidada da Fase 2
│   ├── relatorio_final_dados/              # Dados tratados separados por tema do relatório
│   │   ├── 01_populacao_e_geracoes.md/.csv
│   │   ├── 02_metodos_de_selecao.md/.csv
│   │   ├── 03_crossover_e_mutacao.md/.csv
│   │   ├── 04_elitismo.md/.csv
│   │   ├── 05_bloating_e_tamanho.md/.csv
│   │   └── DOSSIE_RELATORIO_FINAL.md       # Dossiê completo integrado
│   ├── AUDIT_PHASE2.md                     # Relatório completo de auditoria da Fase 2
│   ├── experiments_consolidated.md/.csv    # Tabela consolidada da Fase 1
│   └── SUMMARY.md                          # Relatório executivo da Fase 1
├── tests/
│   ├── test_engine.py
│   ├── test_init.py
│   ├── test_operators.py
│   ├── test_phase2_improvements.py         # Testes de AQ, mutação mista e linear scaling
│   ├── test_primitives.py
│   ├── test_selection.py
│   └── test_tree.py
├── requirements.txt
└── README.md
```

---

## 6. Documentação e Relatórios Gerados

* **Dossiê para o Relatório Final:** [`results/relatorio_final_dados/DOSSIE_RELATORIO_FINAL.md`](results/relatorio_final_dados/DOSSIE_RELATORIO_FINAL.md) (consolida todas as respostas e tabelas exigidas pelo enunciado do trabalho).
* **Relatório de Auditoria da Fase 2:** [`results/AUDIT_PHASE2.md`](results/AUDIT_PHASE2.md) (detalha a cadeia *Dado Avaliado $\to$ Hipótese da Landscape $\to$ Intervenção $\to$ Resultado* para as 6 hipóteses).
* **Relatório Executivo da Fase 1:** [`results/SUMMARY.md`](results/SUMMARY.md) (resultados consolidados da Seção 11 original).

## 7. Repositório

Repositório completo [aqui](https://github.com/BernardooVale/PG_NatComp), caso queira baixar o restante dos arquivos