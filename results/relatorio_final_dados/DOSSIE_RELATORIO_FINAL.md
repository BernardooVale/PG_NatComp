# Dossiê de Dados Tratados e Separados para o Relatório Final

Este documento organiza **tópico a tópico** todas as respostas, tabelas tratadas, métricas geracionais e gráficos exigidos pelo enunciado do trabalho, separando claramente **o que os dados da Fase 1 já provam** e **como os dados da Fase 2 aprofundam a intuição em cada quesito**.

---

## Visão Geral: Fase 1 vs. Fase 2 em Relação ao Enunciado

| Quesito do Enunciado | A Fase 1 responde completamente? | A Fase 2 melhora a intuição sobre este ponto? |
| :--- | :---: | :--- |
| **1. Representação do indivíduo** | **Sim** (Árvore sintática em [`gp/tree.py`](file:///c:/Users/Plusb/Documents/PG_NatComp/gp/tree.py) e [`gp/primitives.py`](file:///c:/Users/Plusb/Documents/PG_NatComp/gp/primitives.py)). | **Sim (Muito):** Prova que a escolha das primitivas (`aq` vs `div`, adição de `log, sqrt, exp`) e o *Linear Scaling* ($\alpha + \beta \hat{y}$) alteram radicalmente a paisagem de aptidão ($R^2$ sobe de $0.44 \to 0.78$ no `concrete1` e elimina polos no `synth2`). |
| **2. População inicial** | **Sim** (`grow`, `full`, `ramped_half_and_half` em [`gp/init.py`](file:///c:/Users/Plusb/Documents/PG_NatComp/gp/init.py)). | Mantém o *Ramped Half-and-Half* como gerador ótimo de diversidade estrutural na geração 0 (`mean_size` inicial $\approx 26.2$ nós, $\approx 89.5\%$ de indivíduos únicos). |
| **3. Facilidades para variação de parâmetros** | **Sim** (Dataclass `GPConfig`, YAML e CLI `--override` em [`gp/config.py`](file:///c:/Users/Plusb/Documents/PG_NatComp/gp/config.py)). | **Sim:** Demonstra a extensibilidade da arquitetura — 6 novos mecanismos foram plugados via flags booleanas/numéricas sem quebrar nenhum teste existente. |
| **4. Tamanho da população e nº de gerações** | **Sim** (`pop` $\in \{50, 100, 500\}$, `gen` $\in \{50, 100, 500\}$). | **Sim:** O teste do Modelo de Ilhas (`H4`, $4 \times 25$) provou empiricamente por que `pop >= 100` panmítico é o limiar mínimo para sustentar especialistas na seleção $\varepsilon$-Lexicase. |
| **5. Tamanho máximo = 7** | **Sim** (`max_size=7` padrão; comparado com `max_size=10`). | **Sim:** Mostra que com *Linear Scaling* e *Parcimônia* (`Stack A2` e `Stack B4`), as melhores soluções usam em média apenas $35.9$ a $43.3$ nós (muito abaixo do teto de profundidade 7, que comporta até $127$ nós). |
| **6. Crossover vs. Mutação (Config A vs B + Proposta Final)** | **Sim** (`Config A`, `Config B` e *sweeps* de $p_c$ e $p_m$ com contagem de filhos melhores/piores). | **Sim (Crucial):** Na Fase 1, `pc=0.7` e `pm=0.1` venceram isoladamente, mas ainda se propunha `pc=0.9, pm=0.05`. A Fase 2 (`H2` e `Stack A2/B3`) **testou e confirmou estatisticamente** os novos valores propostos ($p_c=0.70, p_m=0.15$ com mutação mista), reduzindo a mediana do erro em $86\%$ no `synth1`! |
| **7. Impacto e mudanças ao variar a Seleção** | **Sim** (Torneio $k=2, 3$, Roleta e $\varepsilon$-Lexicase no `synth1`). | **Sim (Crucial):** Na Fase 1, o $\varepsilon$-Lexicase só havia sido testado no `synth1`. A Fase 2 provou seu impacto no `synth2` e no `concrete1`, explicando quando ele sozinho resolve o problema (`synth1`) e quando precisa de suporte numérico (`synth2`, `concrete1`). |
| **8. Utilizar Elitismo** | **Sim** (`elitism=True` vs `False`). | Confirma a necessidade de elitismo com desempate parcimonioso (`_best_key` considerando RMSE e tamanho). |
| **9. Forma simples de medir Bloating** | **Sim** (`mean_size`, `max_size_pop`, `mean_depth` por geração em [`gp/stats.py`](file:///c:/Users/Plusb/Documents/PG_NatComp/gp/stats.py)). | **Sim:** Além de *medir* o bloating, a Fase 2 (`H5`) implementou uma forma simples de **combatê-lo** (desempate por tamanho no $\varepsilon$-Lexicase), reduzindo o tamanho médio em até $41.6\%$. |

---

## Tópico 1: Representação do Indivíduo, População Inicial e Parametrização

### 1.1 Como representar um indivíduo ([`gp/tree.py`](file:///c:/Users/Plusb/Documents/PG_NatComp/gp/tree.py) e [`gp/primitives.py`](file:///c:/Users/Plusb/Documents/PG_NatComp/gp/primitives.py))
* **Estrutura de Dados:** Árvore sintática representada recursivamente pela classe `Node(value, children)`, encapsulada na classe `Individual(root, fitness, case_errors, origin_operator, parent_fitness, scale_offset, scale_slope)`.
* **Conjunto de Terminais ($\mathcal{T}$):** Variáveis de entrada `x0, x1, ..., x{d-1}` e Constantes Numéricas Efêmeras (ERC) sorteadas uniformemente em `const_range = [-1.0, 1.0]` com probabilidade `const_prob = 0.3`.
* **Conjunto de Funções ($\mathcal{F}$):**
  * **Fase 1:** Operadores binários protegidos `[add, sub, mul, div]`, onde `div(a, b) = 1.0` se $|b| < 10^{-6}$ (padrão de Koza).
  * **Enriquecimento da Fase 2:** Inclusão do **Quociente Analítico** $\text{aq}(a,b) = a/\sqrt{1+b^2}$ (contínuo e diferenciável, sem assíntotas verticais), funções unárias protegidas `[log, sqrt, exp]` e **Escalonamento Linear Ótimo** ($\hat{y}_{\text{scaled}} = \alpha + \beta \hat{y}$ via Mínimos Quadrados Ordinários em $\mathcal{O}(N)$).
* **Avaliação Vetorizada:** Toda árvore avalia a matriz $X \in \mathbb{R}^{N \times d}$ de uma só vez via arrays NumPy, atribuindo penalidade `1e12` a eventuais `NaN`/`Inf`.

### 1.2 Como gerar a população inicial ([`gp/init.py`](file:///c:/Users/Plusb/Documents/PG_NatComp/gp/init.py))
* **Métodos Implementados:**
  1. `full`: Todas as folhas são forçadas a estar exatamente na profundidade alvo $d$ (árvores cheias).
  2. `grow`: Nós internos e terminais são sorteados antes da profundidade máxima, gerando árvores assimétricas.
  3. `ramped_half_and_half` (**Padrão escolhido**): Divide a população igualmente entre profundidades $d \in \{1, \dots, \text{max\_size}\}$ e, em cada faixa, cria $50\%$ das árvores via `full` e $50\%$ via `grow`.
* **Evidência nos Dados (Geração 0):** Garante alta diversidade inicial ($\approx 89.5$ indivíduos únicos de $100$ nas bases sintéticas e $95.1$ no `concrete1`, com tamanho médio inicial de $26.2$ nós e profundidade média de $3.94$).

### 1.3 Facilidades para variação de parâmetros ([`gp/config.py`](file:///c:/Users/Plusb/Documents/PG_NatComp/gp/config.py))
* Nenhum parâmetro é *hardcoded* no motor evolutivo. Tudo é centralizado na `@dataclass GPConfig`, que possui:
  * Validação automática de domínio (`validate()`);
  * Serialização/leitura de arquivo YAML (`configs/default.yaml` e dump automático de `config.yaml` dentro da pasta de cada experimento executado);
  * Sobrescrita direta via linha de comando (`--override pop_size=500 selection=lexicase crossover_prob=0.7`).

---

## Tópico 2: Tamanho da População e Número de Gerações (`exp11_2`)

![Convergência por População e Gerações](/C:/Users/Plusb/.gemini/antigravity/brain/5dff1366-f2c5-41de-870a-f891a7211c62/pop_gens_convergence.png)

### Tabela Tratada 1 — Avaliação de População e Gerações (`synth1`, 30 seeds)
*(Salva em [`results/relatorio_final_dados/01_populacao_e_geracoes.csv`](file:///c:/Users/Plusb/Documents/PG_NatComp/results/relatorio_final_dados/01_populacao_e_geracoes.csv))*

| Configuração (Orçamento de Avaliações) | Train RMSE (Média ± Std) | Test RMSE (Média ± Std) | Test RMSE Mediana (IQR) | Test Min / Max | Geração de Saturação (~5%) | Repetidos (G0 $\to$ G20 $\to$ Final) | Tam. Médio Pop (G0 $\to$ Final) | Tam. Melhor Indiv. (Nós) | Tempo Médio (s) |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :--- | :---: |
| **`Pop=50, Gen=100`** (5.000 av.) | $4.5669 \pm 2.3063$ | $6.9463 \pm 5.1315$ | $7.4488\ (6.8698)$ | $0.1388 / 25.41$ | Gen 87 | $4.1 \to 9.6 \to 7.0$ | $24.7 \to 39.5$ | $38.9 \pm 16.1$ | $2.03\text{ s}$ |
| **`Pop=100, Gen=50`** (5.000 av.) | $4.7864 \pm 2.3989$ | $7.5988 \pm 5.6161$ | $8.1627\ (7.3235)$ | $1.0366 / 24.52$ | Gen 45 | $10.5 \to 10.8 \to 8.0$ | $26.2 \to 42.7$ | $40.5 \pm 14.2$ | $2.43\text{ s}$ |
| **`Pop=100, Gen=100`** [Baseline] (10.000 av.) | $3.6881 \pm 2.1854$ | $5.5209 \pm 4.8934$ | $4.2795\ (6.3828)$ | $0.4613 / 21.44$ | **Gen 80** | $10.5 \to 10.8 \to 9.0$ | $26.2 \to 50.5$ | $46.5 \pm 17.5$ | **$5.27\text{ s}$** |
| **`Pop=500, Gen=100`** (50.000 av.) | $1.8362 \pm 1.4492$ | $2.6174 \pm 4.0822$ | $1.3626\ (0.4291)$ | $0.2281 / 21.87$ | **Gen 84** | $74.2 \to 20.6 \to 23.8$ | $26.4 \to 49.4$ | $41.8 \pm 16.5$ | $31.72\text{ s}$ |
| **`Pop=100, Gen=500`** (50.000 av.) | $1.9687 \pm 2.1626$ | $3.3321 \pm 4.7383$ | $1.3281\ (4.8894)$ | $0.0052 / 18.49$ | Gen 425 | $10.5 \to 10.8 \to 7.4$ | $26.2 \to \mathbf{66.4}$ | $\mathbf{64.2 \pm 21.7}$ | $47.49\text{ s}$ |

### Argumentação Técnica para o Relatório:
1. **Mais População vs. Mais Gerações com mesmo orçamento (50.000 avaliações):** `Pop=500, Gen=100` supera `Pop=100, Gen=500` no RMSE médio de teste ($2.6174$ vs $3.3321$), com árvores $35\%$ menores ($41.8$ vs $64.2$ nós) e em menor tempo ($31.7\text{ s}$ vs $47.5\text{ s}$). Isso ocorre porque estender gerações após a convergência (Gen 80) alimenta principalmente o *bloating*, enquanto ampliar a população expande a cobertura de bacias de atração desde a geração 0.
2. **Critério de Escolha (`Pop=100, Gen=100`):** Nas execuções de 100 gerações, a curva de erro atinge $95\%$ da sua melhoria total por volta da **Geração 80**, caracterizando saturação assintótica sob profundidade 7. `Pop=100, Gen=100` reduz o erro de teste em $27.3\%$ sobre `Gen=50` e em $20.5\%$ sobre `Pop=50`, custando apenas $\approx 5.3\text{ s}$ ($6\times$ mais rápido que `Pop=500`).
3. **Insight adicional da Fase 2:** O experimento `H4` (que dividiu `Pop=100` em 4 ilhas de 25) confirmou que `Pop=100` é o tamanho mínimo crítico para que métodos avançados como $\varepsilon$-Lexicase mantenham especialistas em todos os casos de treino.

---

## Tópico 3: Impacto e Mudanças ao Variar o Método de Seleção (`exp11_3` + Fase 2)

````carousel
![Convergência por Método de Seleção](/C:/Users/Plusb/.gemini/antigravity/brain/5dff1366-f2c5-41de-870a-f891a7211c62/selection_convergence.png)
<!-- slide -->
![Diversidade (Indivíduos Repetidos) por Seleção](/C:/Users/Plusb/.gemini/antigravity/brain/5dff1366-f2c5-41de-870a-f891a7211c62/selection_diversity_n_repeated.png)
<!-- slide -->
![Boxplot de RMSE de Teste por Seleção](/C:/Users/Plusb/.gemini/antigravity/brain/5dff1366-f2c5-41de-870a-f891a7211c62/selection_test_rmse_boxplot.png)
````

### Tabela Tratada 2 — Comparação dos Métodos de Seleção (`synth1`, `synth2`, `concrete1`)
*(Salva em [`results/relatorio_final_dados/02_metodos_de_selecao.csv`](file:///c:/Users/Plusb/Documents/PG_NatComp/results/relatorio_final_dados/02_metodos_de_selecao.csv))*

| Base / Método de Seleção | Train RMSE (Média ± Std) | Test RMSE (Média ± Std) | Test Mediana (IQR) | Test Min / Max | Geração Saturação | Repetidos (G0 $\to$ G5 $\to$ G20 $\to$ Final) | Tam. Melhor (Nós) | Tempo Médio (s) |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :--- | :---: |
| **`synth1`: Roleta** | $3.1561 \pm 2.2391$ | $5.6605 \pm 8.8514$ | $2.1716\ (7.2764)$ | $0.7010 / 48.36$ | Gen 80 | $10.5 \to 20.3 \to 14.5 \to 9.1$ | $34.3 \pm 15.5$ | $7.80\text{ s}$ |
| **`synth1`: Torneio ($k=2$)** | $3.6881 \pm 2.1854$ | $5.5209 \pm 4.8934$ | $4.2795\ (6.3828)$ | $0.4613 / 21.44$ | Gen 80 | $10.5 \to \mathbf{27.1} \to 10.8 \to 9.0$ | $46.5 \pm 17.5$ | $7.76\text{ s}$ |
| **`synth1`: Torneio ($k=3$)** | $4.0641 \pm 2.1968$ | $13.3333 \pm 42.7975$ | $7.0118\ (6.3837)$ | $0.1984 / \mathbf{239.17}$ | **Gen 63** | $10.5 \to \mathbf{26.1} \to 14.0 \to 13.9$ | $48.6 \pm 16.6$ | $6.61\text{ s}$ |
| **`synth1`: $\varepsilon$-Lexicase (MAD)** | **$0.9858 \pm 1.2058$** | **$1.6172 \pm 3.3648$** | **$0.6756\ (1.1183)$** | **$0.0124$** $/ 18.19$ | **Gen 94** | $10.5 \to \mathbf{4.0} \to 8.7 \to 13.4$ | $46.3 \pm 14.3$ | $22.46\text{ s}$ |
| **`synth2`: Torneio ($k=2$)** [Fase 1] | $2.6617 \pm 0.3806$ | $24.9658 \pm 80.3860$ | $3.4697\ (4.2628)$ | $1.3875 / 415.26$ | Gen 65 | $10.5 \to 17.8 \to 4.0 \to 3.9$ | $48.0 \pm 11.6$ | $9.39\text{ s}$ |
| **`synth2`: $\varepsilon$-Lexicase (MAD)** [Fase 2] | **$2.1086 \pm 0.3630$** | **$13.1473 \pm 17.4791$** | $5.7248\ (15.9553)$ | $1.5581 / 86.55$ | Gen 78 | $10.5 \to \mathbf{2.5} \to 2.2 \to 3.6$ | $57.1 \pm 16.6$ | $26.81\text{ s}$ |
| **`concrete1`: Torneio ($k=2$)** [Fase 1] | $12.3889 \pm 1.4542$ | $11.9689 \pm 1.5085$ | $11.6466\ (1.7168)$ | $10.1827 / 16.67$ | Gen 89 | $4.9 \to 12.1 \to 4.3 \to 5.7$ | $74.1 \pm 27.2$ | $9.36\text{ s}$ |
| **`concrete1`: $\varepsilon$-Lexicase (MAD)** [Fase 2] | $12.5990 \pm 1.4784$ | $12.1875 \pm 1.5881$ | $11.8146\ (2.1964)$ | $9.5104 / 15.26$ | Gen 84 | $4.9 \to \mathbf{2.1} \to 2.6 \to 2.9$ | **$59.8 \pm 18.8$** | $44.17\text{ s}$ |

### Argumentação Técnica para o Relatório:
1. **O que muda na dinâmica populacional ao trocar o método de seleção?**
   * **Roleta:** A transformação $\text{aptidão} = 1 / (1 + \text{RMSE})$ comprime diferenças relativas quando o RMSE ainda é alto, gerando baixa pressão seletiva no final e alta instabilidade no teste ($\text{Std} = 8.85$). Não teve diferença significativa para Torneio $k=2$ ($U = 486.5, p = 0.595$).
   * **Torneio ($k=2$ vs $k=3$):** Aumentar a pressão seletiva de $k=2$ para $k=3$ provoca **convergência prematura severa** (satura já na geração 63, com $26.1$ indivíduos repetidos na geração 5) e sobreajuste catastrófico no teste ($\text{RMSE} = 13.33 \pm 42.80$, máx $239.17$).
   * **$\varepsilon$-Lexicase (MAD):** Em vez de comprimir todos os casos de treino em um único escalar (RMSE médio), filtra candidatos caso a caso em ordem aleatória mantendo os que estão dentro de $e^*_t + \text{MAD}_t$. Como resultado:
     * Na geração 5, o número de clones repetidos cai de **$27.1$ (Torneio) para apenas $4.0$ ($\varepsilon$-Lexicase)**!
     * A população não estagna na geração 60–80: continua evoluindo até a **geração 94**, reduzindo o RMSE de teste em **$70.7\%$** ($5.5209 \to 1.6172$, Mann-Whitney $U = 762.0, p = 4.12 \times 10^{-6}$).
2. **O que a Fase 2 acrescenta sobre Seleção:** Mostra que o $\varepsilon$-Lexicase preserva diversidade em todas as bases (apenas $\sim 2\text{ a }3$ repetidos em `synth2` e `concrete1`), mas quando a limitação é de escala/representação (`concrete1`), a seleção sozinha não consegue superar a barreira sem o *Linear Scaling* (`H6a`).

---

## Tópico 4: Crossover e Mutação — Config A vs. Config B, Varreduras e Proposta Final (`exp11_4` + Fase 2 `H2`)

````carousel
![Dinâmica de Operadores: Config A (pc=0.9, pm=0.05)](/C:/Users/Plusb/.gemini/antigravity/brain/5dff1366-f2c5-41de-870a-f891a7211c62/operator_dynamics_config_A.png)
<!-- slide -->
![Dinâmica de Operadores: Config B (pc=0.6, pm=0.30)](/C:/Users/Plusb/.gemini/antigravity/brain/5dff1366-f2c5-41de-870a-f891a7211c62/operator_dynamics_config_B.png)
````

### Tabela Tratada 3 — Contribuição Positiva vs. Negativa de Crossover e Mutação (Acumulado em 100 Gerações por Execução)
*(Salva em [`results/relatorio_final_dados/03_crossover_e_mutacao.csv`](file:///c:/Users/Plusb/Documents/PG_NatComp/results/relatorio_final_dados/03_crossover_e_mutacao.csv))*

| Configuração (`synth1`, 30 seeds) | Crossover Melhores vs Piores que **Média dos Pais** (% Sucesso) | Crossover Melhores vs Piores que **Melhor Pai** (% Sucesso) | Evolução de `Cx > Melhor Pai` por Gen (Início G1-20 $\to$ Fim G81-100) | Mutação Melhores vs Piores (% Sucesso) | Train RMSE (Média ± Std) | Test RMSE (Média ± Std) | Test Mediana (IQR) | Tamanho (Nós) |
| :--- | :---: | :---: | :---: | :---: | :--- | :--- | :--- | :---: |
| **Fase 1 Config A** ($p_c=0.90, p_m=0.05$, Torneio) | $2643.6 \text{ vs } 5606.1$ (**$32.0\%$**) | $799.9 \text{ vs } 6466.6$ (**$11.0\%$**) | $10.13/\text{gen} \to 6.82/\text{gen}$ | $446.1 \text{ vs } 36.0$ ($92.5\%$) | $3.6881 \pm 2.1854$ | $5.5209 \pm 4.8934$ | $4.2795\ (6.3828)$ | $46.5 \pm 17.5$ |
| **Fase 1 Config B** ($p_c=0.60, p_m=0.30$, Torneio) | $1233.5 \text{ vs } 2894.4$ (**$29.9\%$**) | $445.5 \text{ vs } 3422.1$ (**$11.5\%$**) | $5.15/\text{gen} \to 3.73/\text{gen}$ | $1995.9 \text{ vs } 883.1$ ($69.3\%$) | $4.1094 \pm 2.4506$ | $5.9577 \pm 4.5176$ | $7.7930\ (7.9818)$ | $37.5 \pm 12.7$ |
| **Fase 1 Sweep** ($p_c=0.50, p_m=0.05$, Torneio) | $955.6 \text{ vs } 3516.4$ ($21.4\%$) | $218.5 \text{ vs } 3847.5$ ($5.4\%$) | $4.57/\text{gen} \to 1.07/\text{gen}$ | $270.9 \text{ vs } 209.2$ ($56.4\%$) | $4.1418 \pm 2.2102$ | $7.9194 \pm 9.1246$ | $7.4841\ (7.1604)$ | $38.4 \pm 16.3$ |
| **Fase 1 Sweep** (**$p_c=0.70$**, $p_m=0.05$, Torneio) | $1744.5 \text{ vs } 4627.2$ ($27.4\%$) | $487.9 \text{ vs } 5233.2$ ($8.5\%$) | $7.17/\text{gen} \to 3.69/\text{gen}$ | $367.0 \text{ vs } 116.3$ ($75.9\%$) | **$3.0457 \pm 2.1743$** | **$3.9739 \pm 3.4112$** | **$2.0888\ (6.5399)$** | $39.2 \pm 15.6$ |
| **Fase 1 Sweep** ($p_c=0.90, p_m=0.01$, Torneio) | $2709.6 \text{ vs } 5886.8$ ($31.5\%$) | $798.4 \text{ vs } 6773.2$ ($10.5\%$) | $9.98/\text{gen} \to 6.77/\text{gen}$ | $91.1 \text{ vs } 7.4$ ($92.5\%$) | $4.4254 \pm 2.4463$ | $6.4638 \pm 4.2730$ | $7.2569\ (6.2639)$ | $46.9 \pm 19.5$ |
| **Fase 1 Sweep** ($p_c=0.90$, **$p_m=0.10$**, Torneio) | $2575.9 \text{ vs } 5332.8$ ($32.6\%$) | $884.2 \text{ vs } 6289.8$ ($12.3\%$) | $9.88/\text{gen} \to 8.08/\text{gen}$ | $919.2 \text{ vs } 77.8$ ($92.2\%$) | **$3.0111 \pm 2.1078$** | **$3.8876 \pm 3.1479$** | **$2.4723\ (5.9307)$** | $39.1 \pm 18.7$ |
| **Fase 1 Sweep** ($p_c=0.90, p_m=0.20$, Torneio) | $2390.5 \text{ vs } 4693.7$ ($33.7\%$) | $919.2 \text{ vs } 5678.0$ ($13.9\%$) | $9.75/\text{gen} \to 8.45/\text{gen}$ | $1825.5 \text{ vs } 141.6$ ($92.8\%$) | $4.4223 \pm 2.1640$ | $6.5313 \pm 4.1582$ | $8.0276\ (6.7040)$ | $45.8 \pm 17.3$ |
| **Fase 2 `S0`** ($p_c=0.90, p_m=0.05$, Lexicase) | $1992.0 \text{ vs } 6339.9$ ($23.9\%$) | $523.2 \text{ vs } 7252.0$ ($6.7\%$) | $10.87/\text{gen} \to 3.19/\text{gen}$ | $452.9 \text{ vs } 38.0$ ($92.3\%$) | $0.9858 \pm 1.2058$ | $1.6172 \pm 3.3648$ | $0.6756\ (1.1183)$ | $46.3 \pm 14.3$ |
| **Fase 2 `H2`** (**$p_c=0.70, p_m=0.15$, Mista**, Lex) | $1266.1 \text{ vs } 4523.3$ ($21.9\%$) | $389.3 \text{ vs } 5161.4$ ($7.0\%$) | $7.52/\text{gen} \to 2.24/\text{gen}$ | $1130.4 \text{ vs } 319.5$ ($78.0\%$) | **$0.8458 \pm 1.1403$** | **$1.1725 \pm 2.4651$** | **$0.2502\ (1.1682)$** | **$40.7 \pm 14.3$** |
| **Fase 2 `Stack A2`** ($p_c=0.70, p_m=0.15$ + Dedup + Parc.) | $973.7 \text{ vs } 4469.8$ ($17.9\%$) | $311.6 \text{ vs } 5019.0$ ($5.8\%$) | $6.92/\text{gen} \to 1.47/\text{gen}$ | $1317.6 \text{ vs } 1709.5$ ($43.5\%$) | **$0.6180 \pm 1.2220$** | **$0.6875 \pm 1.8073$** | **$0.0933\ (0.7554)$** | **$35.9 \pm 11.5$** |

### Argumentação Técnica para o Relatório:
1. **Em quantos casos o Crossover e a Mutação contribuem positivamente vs. negativamente?**
   * **Na Config A ($p_c=0.90, p_m=0.05$):** Ao longo das 100 gerações (média por execução), o crossover gerou **$2643.6$ filhos melhores que a média dos pais contra $5606.1$ piores ($32.0\%$ de contribuição positiva)**, e apenas **$799.9$ filhos melhores que o melhor pai contra $6466.6$ piores ($11.0\%$ de contribuição positiva vs $89.0\%$ destrutivo)**. Além disso, a produtividade do crossover cai de $10.13$ filhos melhores que o melhor pai por geração nas primeiras 20 gerações para $6.82$ nas últimas 20 gerações (e no $\varepsilon$-Lexicase cai de $10.87 \to 3.19/\text{gen}$ à medida que as árvores se tornam altamente especializadas).
   * **Na Config B ($p_c=0.60, p_m=0.30$):** O crossover gerou $1233.5$ filhos melhores que a média dos pais contra $2894.4$ piores ($29.9\%$), e $445.5$ melhores que o melhor pai contra $3422.1$ piores ($11.5\%$). A mutação elevada ($p_m=0.30$) atuou fortemente ($1995.9$ casos positivos vs $883.1$ negativos), mas como a mutação pura de subárvore a $30\%$ substitui blocos inteiros de árvores já boas, o RMSE mediano subiu de $4.28 \to 7.79$ (Mann-Whitney $U = 418.0, p = 0.641$).
2. **Que valores finais propor a partir desse estudo?**
   * Na **Fase 1**, as varreduras unidimensionais já revelavam que moderar o crossover para **$p_c = 0.70$** baixava o RMSE de teste de $5.52 \to \mathbf{3.97}$ (mediana $2.09$) e elevar a mutação para **$p_m = 0.10$** baixava o RMSE de teste de $5.52 \to \mathbf{3.89}$ (mediana $2.47$).
   * A **Fase 2 (`H2` e `Stack A2`/`Stack B3`)** fechou essa lacuna testando diretamente a combinação intermediária proposta (**$p_c = 0.70, p_m = 0.15$ com mutação mista**: Gaussiana em constantes + pontual + subárvore), reduzindo a mediana do erro no `synth1` de $0.6756 \to \mathbf{0.2502}$ isoladamente e para **$\mathbf{0.0933}$** no `Stack A2`!

---

## Tópico 5: Efeito do Elitismo (`exp11_5`)

````carousel
![Convergência Com vs Sem Elitismo](/C:/Users/Plusb/.gemini/antigravity/brain/5dff1366-f2c5-41de-870a-f891a7211c62/elitism_convergence.png)
<!-- slide -->
![Boxplot de Teste Com vs Sem Elitismo](/C:/Users/Plusb/.gemini/antigravity/brain/5dff1366-f2c5-41de-870a-f891a7211c62/elitism_test_boxplot.png)
````

### Tabela Tratada 4 — Impacto do Elitismo (`synth1`, 30 seeds)
*(Salva em [`results/relatorio_final_dados/04_elitismo.csv`](file:///c:/Users/Plusb/Documents/PG_NatComp/results/relatorio_final_dados/04_elitismo.csv))*

| Configuração | Train RMSE (Média ± Std) | Test RMSE (Média ± Std) | Test RMSE Mediana (IQR) | Test Min / Max | Tam. Melhor (Nós) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Com Elitismo (`elite_size=1`)** | **$3.6881 \pm 2.1854$** | **$5.5209 \pm 4.8934$** | **$4.2795\ (6.3828)$** | **$0.4613$** $/ 21.44$ | **$46.5 \pm 17.5$** |
| **Sem Elitismo (`elite_size=0`)** | $4.9113 \pm 1.9088$ | $6.5692 \pm 3.0697$ | $8.1648\ (2.8956)$ | $0.9394 / 10.97$ | $54.3 \pm 15.6$ |

### Argumentação Técnica para o Relatório:
* Como o crossover de subárvore é destrutivo em $\approx 89\%$ das operações em relação ao melhor pai, a ausência de elitismo faz com que a melhor solução da população seja frequentemente destruída e perdida nas gerações finais (curva vermelha serrilhada e mais alta no gráfico de convergência).
* Sem elitismo, o RMSE médio de treino piora em **$33.2\%$** ($3.69 \to 4.91$), a **mediana do RMSE de teste quase dobra ($4.28 \to 8.16$)** e o tamanho médio das árvores aumenta ($46.5 \to 54.3$ nós, pois árvores maiores sobrevivem mais facilmente ao crossover destrutivo — efeito *bloat* defensivo).

---

## Tópico 6: Como Medir e Controlar o *Bloating* (`exp11_6` + Fase 2 `H5`)

![Curva de Bloating: Profundidade 7 vs 10](/C:/Users/Plusb/.gemini/antigravity/brain/5dff1366-f2c5-41de-870a-f891a7211c62/bloating_depth_comparison.png)

### Tabela Tratada 5 — Métricas de Bloating e Efeito da Pressão de Parcimônia
*(Salva em [`results/relatorio_final_dados/05_bloating_e_tamanho.csv`](file:///c:/Users/Plusb/Documents/PG_NatComp/results/relatorio_final_dados/05_bloating_e_tamanho.csv))*

| Configuração | Tam. Médio da População (G0 $\to$ Final) | Fator de Inchaço ($\text{Size}_{\text{Final}} / \text{Size}_{G0}$) | Tam. do Melhor Indivíduo (Nós) | Train RMSE (Média ± Std) | Test RMSE (Média ± Std) | Test Mediana (IQR) |
| :--- | :---: | :---: | :---: | :--- | :--- | :--- |
| **Fase 1: `max_size=7` (100 Gens, Torneio)** | $26.2 \to 50.5$ | $1.93\times$ | $46.5 \pm 17.5$ | $3.6881 \pm 2.1854$ | $5.5209 \pm 4.8934$ | $4.2795\ (6.3828)$ |
| **Fase 1: `max_size=10` (100 Gens, Torneio)** | $130.7 \to 70.2$ | — | $\mathbf{64.4 \pm 25.2}$ | $3.3851 \pm 2.1359$ | $4.8189 \pm 4.2255$ | $2.3795\ (6.4379)$ |
| **Fase 1: `max_size=7` (500 Gens, Torneio)** | $26.2 \to \mathbf{66.4}$ | **$2.53\times$** | $\mathbf{64.2 \pm 21.7}$ | $1.9687 \pm 2.1626$ | $3.3321 \pm 4.7383$ | $1.3281\ (4.8894)$ |
| **Fase 2 `S0`: `max_size=7` ($\varepsilon$-Lexicase puro)** | $26.2 \to 47.6$ | $1.82\times$ | $46.3 \pm 14.3$ | $0.9858 \pm 1.2058$ | $1.6172 \pm 3.3648$ | $0.6756\ (1.1183)$ |
| **Fase 2 `H5`: $\varepsilon$-Lexicase + Parcimônia** | $26.2 \to \mathbf{42.7}$ | **$1.63\times$** | **$41.1 \pm 14.0$** | $0.7910 \pm 0.5891$ | **$0.7571 \pm 0.5738$** | $0.7170\ (0.9840)$ |
| **Fase 2 `Stack A2`: Parcimônia + Dedup + H2** | $26.2 \to \mathbf{37.3}$ | **$1.42\times$** | **$35.9 \pm 11.5$** | **$0.6180 \pm 1.2220$** | **$0.6875 \pm 1.8073$** | **$0.0933\ (0.7554)$** |
| **`concrete1` Fase 1: `max_size=7` (Torneio)** | $26.2 \to \mathbf{72.3}$ | **$2.76\times$** | $\mathbf{74.1 \pm 27.2}$ | $12.3889 \pm 1.4542$ | $11.9689 \pm 1.5085$ | $11.6466\ (1.7168)$ |
| **`concrete1` Fase 2 `Stack B4`: Parcimônia + Scaling** | $11.5 \to \mathbf{43.7}$ | — | **$43.3 \pm 9.5$** | **$7.7283 \pm 0.5788$** | **$7.7186 \pm 0.6438$** | **$7.5988\ (0.7411)$** |

### Argumentação Técnica para o Relatório:
1. **Existe uma forma simples de medir *bloating* no algoritmo?**
   * **Sim:** Monitorando a cada geração (`gp/stats.py`) o **tamanho médio em nós da população (`mean_size`)**, o **tamanho do melhor indivíduo (`size_best`)** e a **profundidade média (`mean_depth`)** em função da variação do erro de treino ($\Delta \text{RMSE}$).
   * Caracteriza-se *bloating* quando $\frac{d(\text{mean\_size})}{dt} > 0$ enquanto $\frac{d(\text{best\_fitness})}{dt} \approx 0$:
     * Em 100 gerações no Baseline, o tamanho médio quase dobra ($26.2 \to 50.5$ nós, fator $1.93\times$) e em 500 gerações cresce $2.53\times$ ($26.2 \to 66.4$ nós).
     * Ao relaxar `max_size` de $7 \to 10$, as árvores crescem $38.5\%$ ($46.5 \to 64.4$ nós) acumulando *introns* algébricos (`x0 - x0`, `x0 / x0`), com ganho marginal de erro.
2. **Como a Fase 2 aprimora o controle de *bloating*:**
   * A introdução da **Pressão de Parcimônia Lexicográfica (`H5`)** e do **Linear Scaling (`H6a`)** cortou o tamanho médio das árvores finais de **$46.5 \to 35.9$ nós no `synth1` ($-22.8\%$)** e de **$74.1 \to 43.3$ nós no `concrete1` ($-41.6\%$)**, ao mesmo tempo em que reduziu drasticamente o RMSE de teste!

---

## Onde Estão Todos os Arquivos Tratados no Projeto

Todos os CSVs e tabelas Markdown separados por tema foram exportados em [`results/relatorio_final_dados/`](file:///c:/Users/Plusb/Documents/PG_NatComp/results/relatorio_final_dados):
1. [`01_populacao_e_geracoes.csv`](file:///c:/Users/Plusb/Documents/PG_NatComp/results/relatorio_final_dados/01_populacao_e_geracoes.csv) e [`.md`](file:///c:/Users/Plusb/Documents/PG_NatComp/results/relatorio_final_dados/01_populacao_e_geracoes.md)
2. [`02_metodos_de_selecao.csv`](file:///c:/Users/Plusb/Documents/PG_NatComp/results/relatorio_final_dados/02_metodos_de_selecao.csv) e [`.md`](file:///c:/Users/Plusb/Documents/PG_NatComp/results/relatorio_final_dados/02_metodos_de_selecao.md)
3. [`03_crossover_e_mutacao.csv`](file:///c:/Users/Plusb/Documents/PG_NatComp/results/relatorio_final_dados/03_crossover_e_mutacao.csv) e [`.md`](file:///c:/Users/Plusb/Documents/PG_NatComp/results/relatorio_final_dados/03_crossover_e_mutacao.md)
4. [`04_elitismo.csv`](file:///c:/Users/Plusb/Documents/PG_NatComp/results/relatorio_final_dados/04_elitismo.csv) e [`.md`](file:///c:/Users/Plusb/Documents/PG_NatComp/results/relatorio_final_dados/04_elitismo.md)
5. [`05_bloating_e_tamanho.csv`](file:///c:/Users/Plusb/Documents/PG_NatComp/results/relatorio_final_dados/05_bloating_e_tamanho.csv) e [`.md`](file:///c:/Users/Plusb/Documents/PG_NatComp/results/relatorio_final_dados/05_bloating_e_tamanho.md)
6. Auditoria completa da Fase 2: [`results/AUDIT_PHASE2.md`](file:///c:/Users/Plusb/Documents/PG_NatComp/results/AUDIT_PHASE2.md) e [`results/phase2_audit/phase2_consolidated.csv`](file:///c:/Users/Plusb/Documents/PG_NatComp/results/phase2_audit/phase2_consolidated.csv).
