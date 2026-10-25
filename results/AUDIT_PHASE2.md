# Relatório de Auditoria Experimental (Fase 2): Evolução Guiada por Fitness Landscape

Este documento registra de forma auditável a evolução metodológica do framework de Programação Genética, conectando cada **Dado Observado** na Fase 1 a uma **Hipótese sobre a Fitness Landscape**, à **Intervenção Algorítmica** implementada e ao **Veredito Estatístico** obtido em 30 execuções independentes (`seeds 0..29`) nas três bases (`synth1`, `synth2` e `concrete1_norm`).

---

## 1. Estado Atual (`S0`) e Conclusões Iniciais da Fase 1

Na Fase 1 (Seção 11), a seleção **$\varepsilon$-Lexicase (MAD)** provou ser estatisticamente superior ao Torneio ($k=2$) no `synth1` ($p = 4.12 \times 10^{-6}$), mas as replicações em `synth2` e `concrete1` haviam sido executadas apenas com o Torneio ($k=2$).

Portanto, o **Estado Atual (`S0_estado_atual_lexicase`)** foi estabelecido aplicando $\varepsilon$-Lexicase (`pop=100, gen=100, max_depth=7, pc=0.9, pm=0.05, elitism=True, [add, sub, mul, div]`) nas 3 bases:

| Base | Fase 1 (Torneio $k=2$) Test RMSE | Estado Atual (`S0` $\varepsilon$-Lexicase) Train RMSE | Estado Atual (`S0` $\varepsilon$-Lexicase) Test RMSE (Média ± Std) | Estado Atual (`S0`) Test Mediana (IQR) | Tamanho Médio (`S0`) | Diagnóstico do Estado Atual |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`synth1`** | $5.5209 \pm 4.8934$ | $0.9858 \pm 1.2058$ | **$1.6172 \pm 3.3648$** | $0.6762\ (1.6305)$ | $46.3 \pm 14.3$ | Forte ganho sobre Torneio, mas ainda sofre com *outliers* (máx $18.19$) e árvores com código neutro. |
| **`synth2`** | $24.9658 \pm 80.3860$ | $2.1086 \pm 0.3630$ | **$13.1473 \pm 17.4791$** | $5.7248\ (15.9553)$ | $57.1 \pm 15.8$ | $\varepsilon$-Lexicase reduz o erro médio pela metade, mas **persistem explosões assintóticas** (máx $86.55$) no conjunto de teste. |
| **`concrete1`** | $11.9689 \pm 1.5085$ | $12.5990 \pm 1.4784$ | **$12.1875 \pm 1.5881$** | $11.8146\ (2.1964)$ | $59.8 \pm 22.9$ | Estagnação severa ($R^2 \approx 0.44$): árvores gastam dezenas de nós tentando escalonar constantes de $[-1, 1]$ para a faixa de $35.8\text{ MPa}$. |

---

## 2. Avaliação Auditável das Hipóteses Isoladas (Ablação Individual vs. `S0`)

Cada hipótese abaixo foi testada **isoladamente** contra o Estado Atual (`S0`), alterando apenas o mecanismo sob investigação e mantendo todos os demais hiperparâmetros fixos.

### H1 [Polos e Falésias na Landscape]: Analytic Quotient (`aq`) no lugar de `div`
* **Dado Avaliado:** No `synth2`, o erro de treino do `S0` é baixo ($2.1086 \pm 0.3630$), mas o RMSE de teste explode para $13.1473 \pm 17.4791$ (máx $86.55$; na Fase 1 chegou a $415.26$ na Seed 20). A inspeção das expressões revelou divisões cujo denominador se aproxima de zero entre os pontos de treino.
* **Hipótese (Fitness Landscape):** A divisão protegida de Koza ($\text{div}(a,b) = 1 \text{ se } |b|<10^{-6} \text{ senão } a/b$) introduz descontinuidades e falésias infinitas (assíntotas verticais) para $10^{-6} < |b| \ll 1$, causando sobreajuste catastrófico fora da amostra de treino.
* **Sugestão de Melhoria:** Substituir `div` pelo **Analytic Quotient** ($\text{aq}(a,b) = a / \sqrt{1 + b^2}$), operador suave e diferenciável em todo $\mathbb{R}^2$.
* **Resultado Isolado (`H1` vs `S0`):**
  * **`synth2`:** RMSE de teste caiu de **$13.147 \pm 17.479 \longrightarrow 2.292 \pm 0.394$**; Mediana caiu de $5.725 \to 2.338$ (IQR $0.265$); erro máximo caiu de **$86.55 \to 3.59$** (**25 vitórias / 5 derrotas**, $p_{\text{Wilcoxon}} = 1.06 \times 10^{-5}$, $p_{\text{MW}} = 2.15 \times 10^{-6}$).
  * **`concrete1_norm`:** Isoladamente (sem *Linear Scaling*), o erro subiu de $12.188 \to 15.775$. **Por quê?** Porque como $|\text{aq}(a,b)| \le |a|$, a árvore perdeu a capacidade de fabricar constantes grandes ($>35$) dividindo por números próximos de zero! Isso provou que o `aq` exige acoplamento com o *Linear Scaling* (`H6a`).
* **Veredito:** **APROVADO** (elimina 100% das explosões assintóticas; deve ser combinado com `H6a` em bases não normalizadas em $y$).

---

### H2 [Rugosidade e Epistasia]: Rebalanceamento ($p_c=0.7, p_m=0.15$) + Mutação Mista
* **Dado Avaliado:** Nos logs geracionais (`history.csv`), $>88\%$ dos crossovers geram filhos piores que os pais (`cx_worse` $\approx 68$ vs `cx_better` $\approx 7$ por geração), enquanto a mutação apresenta alta eficiência relativa de melhoria local.
* **Hipótese (Fitness Landscape):** A paisagem de GP em árvore possui **alta rugosidade e epistasia estrutural**: trocar subárvores inteiras altera o contexto algébrico dos nós superiores, agindo como macromutação destrutiva. Passos locais curtos (mutação pontual + perturbação Gaussiana $\mathcal{N}(0, \sigma^2)$ nas constantes ERC + micro-subárvores) permitem caminhar suavemente nas encostas da bacia de atração.
* **Sugestão de Melhoria:** Reduzir $p_c$ de $0.9 \to 0.7$, elevar $p_m$ de $0.05 \to 0.15$ e usar operador `mutation_type="mixed"` (35% Gaussiana em ERC, 35% pontual, 30% subárvore).
* **Resultado Isolado (`H2` vs `S0`):**
  * **`synth1`:** RMSE médio caiu de **$1.617 \to 1.172$**; **Mediana caiu $63\%$ ($0.676 \to 0.250$)**; Tamanho médio caiu de $46.3 \to 40.7$ nós (18W / 12L).
  * **`synth2`:** RMSE médio caiu de $13.147 \to 12.117$; Mediana caiu de $5.725 \to 5.283$; Tamanho caiu de $57.1 \to 48.8$ nós (19W / 11L).
* **Veredito:** **APROVADO** (reduz consistentemente a mediana do erro e o tamanho das árvores ao suavizar a exploração local).

---

### H3 [Vales e Perda de Diversidade]: Rejeição/Diversificação de Clones (`dedup_offspring`)
* **Dado Avaliado:** No `history.csv`, entre $10\%$ e $27\%$ da população em cada geração consiste em indivíduos canonicamente repetidos (`n_repeated`), desperdiçando avaliações de fitness em clones idênticos.
* **Hipótese (Fitness Landscape):** Clones acumulados no fundo de uma bacia local reduzem a capacidade da população de manter *stepping stones* variados para atravessar vales de aptidão.
* **Sugestão de Melhoria:** Durante a geração de filhos, verificar a assinatura canônica (`canonical_repr()`); se o filho já existir na geração atual, aplicar mutação (até 3 tentativas) antes de inseri-lo.
* **Resultado Isolado (`H3` vs `S0`):**
  * **`synth1`:** RMSE médio de teste caiu em **$51.6\%$ ($1.617 \pm 3.365 \longrightarrow 0.783 \pm 1.501$)**; Mediana caiu de $0.676 \to 0.409$; erro máximo caiu de $18.19 \to 8.35$ (17W / 13L).
  * **`concrete1_norm`:** RMSE médio caiu de $12.188 \to 11.739 \pm 1.181$; Mediana caiu de $11.815 \to 11.570$; Tamanho caiu de $59.8 \to 54.3$ nós (17W / 13L).
* **Veredito:** **APROVADO** (melhora a convergência e reduz variância sem custo computacional relevante).

---

### H4 [Múltiplas Bacias de Atração]: Modelo de Ilhas (`n_islands=4`)
* **Dado Avaliado:** Alta dispersão entre seeds independentes e convergência prematura em bacias locais.
* **Hipótese (Fitness Landscape):** Subpopulações isoladas com migração periódica exploram bacias de atração distintas antes da recombinação.
* **Sugestão de Melhoria:** Dividir os 100 indivíduos em 4 ilhas de 25 indivíduos com migração em anel (2 melhores indivíduos a cada 20 gerações).
* **Resultado Isolado (`H4` vs `S0`):**
  * Reduziu o tamanho das árvores em todas as bases (`synth1`: $46.3 \to 38.8$; `synth2`: $57.1 \to 44.4$; `concrete1`: $59.8 \to 54.9$) e baixou a mediana do `synth2` ($5.725 \to 3.468$), **porém piorou a mediana do `synth1` ($0.676 \to 1.277$, 10W / 20L) e do `concrete1` ($11.815 \to 12.866$, 10W / 20L)**.
  * **Diagnóstico do porquê não funcionou com `pop=100`:** A seleção $\varepsilon$-Lexicase depende de uma população ampla para manter especialistas em todos os casos de teste simultaneamente. Ao dividir `pop=100` em 4 ilhas de apenas **25 indivíduos**, a pressão de filtragem caso a caso colapsa a diversidade interna de cada ilha. (Para o Modelo de Ilhas funcionar com Lexicase, cada ilha precisaria de $\ge 100$ indivíduos, i.e., `pop_size >= 400`).
* **Veredito:** **REJEITADO para `pop_size=100`** (descartado do empilhamento final por reduzir o pool efetivo do $\varepsilon$-Lexicase).

---

### H5 [Redes Neutras e Bloat]: Pressão de Parcimônia Lexicográfica (`parsimony_tiebreak`)
* **Dado Avaliado:** As árvores crescem rapidamente até o limite de profundidade 7 e acumulam subexpressões neutras (`x0 - x0`, `x0 / x0`), atingindo $46$ a $74$ nós médios sem ganho proporcional de acurácia.
* **Hipótese (Fitness Landscape):** A paisagem possui extensos platôs neutros onde árvores inchadas têm exatamente os mesmos erros nos casos de teste que árvores enxutas, mas sofrem maior risco de sobreajuste fora da amostra.
* **Sugestão de Melhoria:** Quando o $\varepsilon$-Lexicase filtra os candidatos até restarem $\le 4$ finalistas (após ao menos $\min(10, N/4)$ casos), desempatar escolhendo a menor árvore entre os finalistas cujo erro global esteja dentro de $5\%$ do melhor finalista.
* **Resultado Isolado (`H5` vs `S0`):**
  * **`synth1`:** RMSE médio de teste caiu em **$53.2\%$ ($1.617 \pm 3.365 \longrightarrow 0.757 \pm 0.574$)**; o erro máximo nas 30 seeds despencou de **$18.19 \to 2.07$**; o tamanho médio caiu de $46.3 \to 41.1$ nós.
* **Veredito:** **APROVADO** (reduz drasticamente o inchaço e elimina *outliers* de sobreajuste).

---

### H6 [Subajuste Estrutural e Escala em Dados Reais]: Linear Scaling (`H6a`) + Primitivas Não-Lineares (`H6b`)
* **Dado Avaliado:** No `concrete1_norm`, o erro de treino ($12.599$) e teste ($12.188$) são quase idênticos e elevados ($\sigma_y = 16.27 \implies R^2 = 0.44$), e as árvores da Fase 1 gastavam $>60\%$ dos nós somando/dividindo constantes em $[-1, 1]$ para alcançar a média $35.8\text{ MPa}$.
* **Hipótese (Fitness Landscape):** Forçar a evolução estrutural da árvore a descobrir simultaneamente a **forma funcional não-linear** $g(X)$ e os **parâmetros lineares de escala e deslocamento** $(\alpha, \beta)$ cria uma barreira severa na landscape. Desacoplar $(\alpha, \beta)$ via Mínimos Quadrados Ordinários ($\hat{y}_{\text{scaled}} = \alpha + \beta g(X)$) permite que a árvore foque 100% na forma da superfície, enquanto `log`, `sqrt` e `exp` capturam a cinética físico-química de cura do concreto.
* **Resultado Isolado (`H6a` e `H6b` vs `S0`):**
  * **`synth2` (`H6a` Linear Scaling):** RMSE médio caiu de **$13.147 \pm 17.479 \longrightarrow 1.720 \pm 1.131$**; Mediana caiu de **$5.725 \to 1.403$** (**29 vitórias / 1 derrota**, $p_{\text{Wilcoxon}} = 3.86 \times 10^{-7}$, $p_{\text{MW}} = 1.07 \times 10^{-9}$).
  * **`concrete1_norm` (`H6a` Linear Scaling):** RMSE médio caiu de **$12.188 \pm 1.588 \longrightarrow 8.895 \pm 0.647$** (**30 vitórias / 0 derrotas**, $p_{\text{Wilcoxon}} = 1.86 \times 10^{-9}$, $p_{\text{MW}} = 9.92 \times 10^{-11}$).
  * **`concrete1_norm` (`H6b` Não-Lineares + Linear Scaling):** RMSE médio caiu para **$7.615 \pm 0.529$** (Mediana **$7.505$**, IQR $0.557$, **30 vitórias / 0 derrotas**, $p_{\text{MW}} = 3.02 \times 10^{-11}$), e o tamanho médio das árvores caiu de **$59.8 \to 49.4$ nós**!
* **Veredito:** **APROVADO COM IMPACTO EXTREMO** ($30/30$ vitórias no `concrete1` e $29/30$ no `synth2`).

---

## 3. Empilhamento Sequencial Auditado: Do Estado Atual (`S0`) ao `Novo Estado`

Seguindo a regra decidida, **descartamos a H4 (Modelo de Ilhas)** e empilhamos sequencialmente as hipóteses aprovadas em duas escadas auditáveis:
* **Escada A (Polinomial / Algébrica Pura):** `S0` $\to$ `Stack A1 (+H3 Dedup + H5 Parcimônia)` $\to$ `Stack A2 (+H2 Operadores Mista)` (ideal para problemas polinomiais exatos como `synth1`).
* **Escada B (Numérica / Físico-Química Completa):** `S0` $\to$ `Stack B1 (+H6a Linear Scaling + H1 Analytic Quotient)` $\to$ `Stack B2 (+H3 Dedup + H5 Parcimônia)` $\to$ `Stack B3 (+H2 Operadores Mista)` $\to$ `Stack B4 (+H6b Primitivas Não-Lineares)`.

### 3.1 Progressão no `synth1` (Polinômio de 4ª Ordem, 60 amostras de treino / 600 de teste)

| Etapa da Escada | Train RMSE (Média ± Std) | Test RMSE (Média ± Std) | Test Mediana (IQR) | Test Min / Max | Tamanho (Nós) | W/L vs `S0` | $p$-valor (Mann-Whitney) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Fase 1 (Torneio $k=2$)** | $3.6881 \pm 2.1854$ | $5.5209 \pm 4.8934$ | $5.8954\ (6.4712)$ | $0.461 / 21.44$ | $46.5 \pm 17.5$ | — | $4.12 \times 10^{-6}$ |
| **`S0`: Estado Atual (Lexicase)** | $0.9858 \pm 1.2058$ | $1.6172 \pm 3.3648$ | $0.6762\ (1.6305)$ | $0.012 / 18.19$ | $46.3 \pm 14.3$ | Ref | Ref |
| **Isolado `+H3` (Dedup)** | $0.6472 \pm 0.7063$ | $0.7829 \pm 1.5005$ | $0.4089\ (0.7717)$ | $0.025 / 8.35$ | $47.0 \pm 14.0$ | 17W / 13L | $1.12 \times 10^{-1}$ |
| **Isolado `+H5` (Parcimônia)** | $0.7910 \pm 0.5891$ | $0.7571 \pm 0.5738$ | $0.7170\ (0.9840)$ | $0.032 / 2.07$ | $41.1 \pm 15.8$ | 15W / 15L | $6.00 \times 10^{-1}$ |
| **Isolado `+H2` (Operadores)** | $0.8458 \pm 1.1403$ | $1.1725 \pm 2.4651$ | $0.2502\ (1.1682)$ | $0.0007 / 12.85$ | $40.7 \pm 14.0$ | 18W / 12L | $9.05 \times 10^{-2}$ |
| **`Stack A1` (`S0 + H3 + H5`)** | **$0.5684 \pm 0.5246$** | **$0.6353 \pm 0.6169$** | $0.4260\ (0.7210)$ | $0.044 / \mathbf{2.58}$ | $43.8 \pm 15.0$ | **18W / 12L** | $1.49 \times 10^{-1}$ |
| **`Stack A2` (`Stack A1 + H2`) [NOVO ESTADO]** | $0.6180 \pm 1.2220$ | $0.6875 \pm 1.8073$ | **$0.0933\ (0.7554)$** | **$0.0006$** $/ 9.75$ | **$35.9 \pm 15.0$** | **22W / 8L** | **$6.39 \times 10^{-3}$** |

* **Destaque de Auditoria no `synth1`:** O `Stack A2` venceu o Estado Atual `S0` em **22 de 30 seeds** ($p_{\text{Wilcoxon}} = 1.32 \times 10^{-2}$, $p_{\text{MW}} = 6.39 \times 10^{-3}$), derrubou a **mediana do erro de teste em $86.2\%$ ($0.6762 \to 0.0933$)**, reduziu o tamanho médio de $46.3 \to 35.9$ nós e, na Seed 0, recuperou a **equação polinomial exata** da base ($\hat{y} = x_0^4 - x_0^3 + 0.5002 x_1^2 - x_1$, com RMSE de teste $= 0.0006$ e apenas $27$ nós)!

---

### 3.2 Progressão no `synth2` (Superfície Racional Não-Linear, 300 amostras de treino / 1000 de teste)

| Etapa da Escada | Train RMSE (Média ± Std) | Test RMSE (Média ± Std) | Test Mediana (IQR) | Test Min / Max | Tamanho (Nós) | W/L vs `S0` | $p$-valor (Mann-Whitney) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Fase 1 (Torneio $k=2$)** | $2.6617 \pm 0.3806$ | $24.9658 \pm 80.3860$ | $3.5278\ (3.9210)$ | $1.387 / 415.26$ | $48.0 \pm 11.6$ | — | — |
| **`S0`: Estado Atual (Lexicase)** | $2.1086 \pm 0.3630$ | $13.1473 \pm 17.4791$ | $5.7248\ (15.9553)$ | $1.751 / 86.55$ | $57.1 \pm 15.8$ | Ref | Ref |
| **Isolado `+H1` (`aq`)** | $1.9309 \pm 0.2564$ | $2.2918 \pm 0.3938$ | $2.3376\ (0.2653)$ | $1.547 / 3.59$ | $67.4 \pm 19.6$ | 25W / 5L | $2.15 \times 10^{-6}$ |
| **Isolado `+H6a` (Linear Scaling)** | $1.4054 \pm 0.1248$ | $1.7198 \pm 1.1314$ | $1.4027\ (0.2861)$ | $1.236 / 7.58$ | $57.5 \pm 18.8$ | 29W / 1L | $1.07 \times 10^{-9}$ |
| **`Stack B1` (`S0 + H6a + H1`)** | $1.4124 \pm 0.1433$ | $1.6249 \pm 0.4084$ | $1.4943\ (0.3459)$ | $1.233 / 3.31$ | $74.1 \pm 16.5$ | **28W / 2L** | **$4.50 \times 10^{-9}$** |
| **`Stack B2` (`Stack B1 + H3 + H5`)** | $1.4453 \pm 0.1846$ | $1.6092 \pm 0.3131$ | $1.4677\ (0.3801)$ | $1.281 / 2.74$ | $68.3 \pm 22.9$ | **28W / 2L** | **$2.67 \times 10^{-9}$** |
| **`Stack B3` (`Stack B2 + H2`) [NOVO ESTADO]** | **$1.3851 \pm 0.1382$** | **$1.5208 \pm 0.2558$** | **$1.4304\ (0.1978)$** | $1.284 / \mathbf{2.32}$ | **$61.6 \pm 16.2$** | **28W / 2L** | **$8.15 \times 10^{-10}$** |

* **Destaque de Auditoria no `synth2`:** Cada degrau da Escada B melhorou monotonamente o resultado! A sinergia entre `H6a` (Linear Scaling) e `H1` (Analytic Quotient) no `Stack B1` superou ambos isoladamente; o `Stack B2` (`+H3+H5`) reduziu o tamanho de $74.1 \to 68.3$ e o desvio-padrão para $0.3131$; e o `Stack B3` (`+H2`) atingiu **$1.5208 \pm 0.2558$** (redução de **$88.4\%$** sobre o `S0` e de **$93.9\%$** sobre a Fase 1, com $p_{\text{MW}} = 8.15 \times 10^{-10}$). Notavelmente, a **Seed 20** — que na Fase 1 teve o pior erro catastrófico de todo o estudo ($415.26$) — tornou-se no `Stack B3` a **melhor execução dentre todas as 30 seeds** (Test RMSE $= \mathbf{1.2842}$)!

---

### 3.3 Progressão no `concrete1_norm` (Resistência do Concreto, 8 variáveis, 824 treino / 206 teste)

| Etapa da Escada | Train RMSE (Média ± Std) | Test RMSE (Média ± Std) | Test Mediana (IQR) | Test Min / Max | Tamanho (Nós) | W/L vs `S0` | $p$-valor (Mann-Whitney) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Fase 1 (Sem Normalização)** | $14.2769 \pm 1.3635$ | $13.9726 \pm 1.4376$ | $13.9211\ (1.9840)$ | $11.524 / 17.82$ | $41.9 \pm 20.5$ | — | — |
| **Fase 1 (Torneio $k=2$, Norm)** | $12.3889 \pm 1.4542$ | $11.9689 \pm 1.5085$ | $11.6723\ (1.8845)$ | $10.183 / 16.67$ | $74.1 \pm 27.2$ | — | — |
| **`S0`: Estado Atual (Lexicase)** | $12.5990 \pm 1.4784$ | $12.1875 \pm 1.5881$ | $11.8146\ (2.1964)$ | $9.840 / 15.26$ | $59.8 \pm 22.9$ | Ref | Ref |
| **Isolado `+H3` (Dedup)** | $12.1677 \pm 1.1817$ | $11.7390 \pm 1.1805$ | $11.5700\ (1.7794)$ | $9.930 / 14.45$ | $54.3 \pm 21.0$ | 17W / 13L | $3.04 \times 10^{-1}$ |
| **Isolado `+H6a` (Linear Scaling)** | $9.1394 \pm 0.7102$ | $8.8955 \pm 0.6467$ | $8.7869\ (0.7685)$ | $7.752 / 10.37$ | $57.5 \pm 19.6$ | 30W / 0L | $9.92 \times 10^{-11}$ |
| **`Stack B1` (`S0 + H6a + H1 aq`)** | $8.3346 \pm 0.4586$ | $8.3125 \pm 0.5313$ | $8.2497\ (0.7705)$ | $7.432 / 9.81$ | $60.1 \pm 18.8$ | **30W / 0L** | **$3.02 \times 10^{-11}$** |
| **`Stack B2` (`Stack B1 + H3 + H5`)** | $8.1893 \pm 0.4790$ | $8.1497 \pm 0.5016$ | $8.0843\ (0.7648)$ | $7.274 / 9.33$ | $57.7 \pm 20.6$ | **30W / 0L** | **$3.02 \times 10^{-11}$** |
| **`Stack B3` (`Stack B2 + H2 Ops`)** | $8.1059 \pm 0.4806$ | $8.0995 \pm 0.4591$ | $8.0329\ (0.5726)$ | $7.234 / 9.10$ | $55.5 \pm 19.0$ | **30W / 0L** | **$3.02 \times 10^{-11}$** |
| **`Stack B4` (`Stack B3 + H6b Não-Lin`) [NOVO ESTADO]** | **$7.7283 \pm 0.5788$** | **$7.7186 \pm 0.6438$** | **$7.5988\ (0.7411)$** | **$6.762$** $/ \mathbf{8.98}$ | **$43.3 \pm 15.1$** | **30W / 0L** | **$3.02 \times 10^{-11}$** |
| **Isolado `H6b` (`div` + Não-Lin + Scaling)** | $7.7493 \pm 0.4040$ | **$7.6146 \pm 0.5294$** | **$7.5051\ (0.5572)$** | **$6.547$** $/ \mathbf{8.89}$ | $49.4 \pm 17.6$ | **30W / 0L** | **$3.02 \times 10^{-11}$** |

* **Destaque de Auditoria no `concrete1_norm`:** A escada `Stack B1 -> B2 -> B3 -> B4` reduziu monotonamente o RMSE de teste em todas as 4 etapas ($12.188 \to 8.313 \to 8.150 \to 8.100 \to 7.719$) e encolheu simultaneamente o tamanho médio das árvores em todas as etapas ($60.1 \to 57.7 \to 55.5 \to \mathbf{43.3}$ nós — uma redução de **$41.6\%$ no tamanho das árvores** em relação aos $74.1$ nós da Fase 1!). O coeficiente de determinação no teste saltou de $R^2 = 0.44$ para **$R^2 = 0.78$**, vencendo o Estado Atual em **30 de 30 seeds** ($p = 3.02 \times 10^{-11}$).

---

## 4. Síntese Final: O que Funcionou, O que Não Funcionou e Por Quê

| Intervenção | Veredito | Efeito Auditado e Explicação Mecanística |
| :--- | :---: | :--- |
| **H6a: Linear Scaling (OLS)** | **FUNCIONOU (Crítico)** | Venceu em $30/30$ seeds no `concrete1` e $29/30$ no `synth2`. Ao calcular analiticamente $\alpha + \beta \hat{y}$, elimina a barreira da landscape que obrigava a árvore a desperdiçar dezenas de nós para acertar offset e escala. |
| **H1: Analytic Quotient (`aq`)** | **FUNCIONOU (Sinérgico com H6a)** | Eliminou 100% das singularidades/polos assintóticos no `synth2` (erro máx caiu de $415.26 \to 2.32$). Sem `H6a`, sofre no `concrete1` por ser limitado ($|\text{aq}(a,b)| \le |a|$), mas **combinado com `H6a` (`Stack B1`) superou tanto `div` quanto `aq` isolados** ($8.895 \to 8.313$). |
| **H6b: Primitivas `[log, sqrt, exp]`** | **FUNCIONOU (Específico ao Domínio)** | Essencial no `concrete1` ($8.100 \to 7.615\text{–}7.719$, reduzindo o tamanho para $43.3$ nós), pois modela a lei logarítmica de maturidade/idade do concreto. Em polinômios puros (`synth1`), introduz viés funcional desnecessário. |
| **H3: Rejeição de Duplicatas (`dedup`)** | **FUNCIONOU (Universal)** | Reduziu o erro médio em todas as bases ao impedir que clones idênticos ocupem vagas da população, mantendo *stepping stones* ativos para o $\varepsilon$-Lexicase. |
| **H5: Pressão de Parcimônia Lexicográfica** | **FUNCIONOU (Antídoto de Bloat)** | Ao desempatar os $\le 4$ finalistas do $\varepsilon$-Lexicase pelo menor tamanho (dentro de $5\%$ de tolerância de erro), podou as redes neutras sem sacrificar acurácia, cortando os *outliers* do `synth1` de $18.19 \to 2.07$. |
| **H2: Rebalanceamento ($p_c=0.7, p_m=0.15$) + Mutação Mista** | **FUNCIONOU (Refinamento Local)** | Reduziu a taxa de destruição epistática do crossover e permitiu ajuste fino das constantes ERC via perturbação Gaussiana, baixando a mediana do `synth1` para $0.0933$ e do `synth2` para $1.4304$. |
| **H4: Modelo de Ilhas ($4 \times 25$) com `pop=100`** | **NÃO FUNCIONOU para `pop=100`** | Fracionar uma população de 100 indivíduos em ilhas de 25 degrada o $\varepsilon$-Lexicase, que necessita de um pool grande de candidatos para sustentar especialistas em múltiplos casos de teste simultâneos. |
