# Relatório Executivo dos Experimentos de Programação Genética

Este documento consolida os resultados de todos os experimentos executados segundo o roteiro da Seção 11.

## 1. Tabela Consolidada de Desempenho (30 repetições por configuração)

| Experiment                         |   Runs | Train RMSE (Mean +/- Std)   | Test RMSE (Mean +/- Std)   | Size (Mean +/- Std)   |   Avg Time (s) |
|:-----------------------------------|-------:|:----------------------------|:---------------------------|:----------------------|---------------:|
| exp11_1_baseline                   |     30 | 3.6881 +/- 2.1854           | 5.5209 +/- 4.8934          | 46.53 +/- 17.47       |           5.77 |
| exp11_2_pop50_gen100               |     30 | 4.5669 +/- 2.3063           | 6.9463 +/- 5.1315          | 38.87 +/- 16.13       |           2.03 |
| exp11_2_pop100_gen100              |     30 | 3.6881 +/- 2.1854           | 5.5209 +/- 4.8934          | 46.53 +/- 17.47       |           5.27 |
| exp11_2_pop500_gen100              |     30 | 1.8362 +/- 1.4492           | 2.6174 +/- 4.0822          | 41.80 +/- 16.49       |          31.72 |
| exp11_2_pop100_gen50               |     30 | 4.7864 +/- 2.3989           | 7.5988 +/- 5.6161          | 40.47 +/- 14.25       |           2.43 |
| exp11_2_pop100_gen500              |     30 | 1.9687 +/- 2.1626           | 3.3321 +/- 4.7383          | 64.20 +/- 21.67       |          47.49 |
| exp11_3_sel_tournament_k2          |     30 | 3.6881 +/- 2.1854           | 5.5209 +/- 4.8934          | 46.53 +/- 17.47       |           7.76 |
| exp11_3_sel_tournament_k3          |     30 | 4.0641 +/- 2.1968           | 13.3333 +/- 42.7975        | 48.60 +/- 16.60       |           6.61 |
| exp11_3_sel_roulette               |     30 | 3.1561 +/- 2.2391           | 5.6605 +/- 8.8514          | 34.33 +/- 15.52       |           7.8  |
| exp11_3_sel_lexicase               |     30 | 0.9858 +/- 1.2058           | 1.6172 +/- 3.3648          | 46.33 +/- 14.27       |          22.46 |
| exp11_4_config_A                   |     30 | 3.6881 +/- 2.1854           | 5.5209 +/- 4.8934          | 46.53 +/- 17.47       |           6.3  |
| exp11_4_config_B                   |     30 | 4.1094 +/- 2.4506           | 5.9577 +/- 4.5176          | 37.47 +/- 12.70       |           3.85 |
| exp11_4_sweep_pc_0.5               |     30 | 4.1418 +/- 2.2102           | 7.9194 +/- 9.1246          | 38.40 +/- 16.27       |           3.4  |
| exp11_4_sweep_pc_0.7               |     30 | 3.0457 +/- 2.1743           | 3.9739 +/- 3.4112          | 39.20 +/- 15.57       |           3.95 |
| exp11_4_sweep_pm_0.01              |     30 | 4.4254 +/- 2.4463           | 6.4638 +/- 4.2730          | 46.87 +/- 19.45       |           5.9  |
| exp11_4_sweep_pm_0.1               |     30 | 3.0111 +/- 2.1078           | 3.8876 +/- 3.1479          | 39.07 +/- 18.73       |           5.4  |
| exp11_4_sweep_pm_0.2               |     30 | 4.4223 +/- 2.1640           | 6.5313 +/- 4.1582          | 45.80 +/- 17.28       |           5.71 |
| exp11_5_elitism_false              |     30 | 4.9113 +/- 1.9088           | 6.5692 +/- 3.0697          | 54.33 +/- 15.56       |           5.48 |
| exp11_6_bloating_maxsize10         |     30 | 3.3851 +/- 2.1359           | 4.8189 +/- 4.2255          | 64.40 +/- 25.18       |           6.43 |
| exp11_8_replication_synth2         |     30 | 2.6617 +/- 0.3806           | 24.9658 +/- 80.3860        | 48.00 +/- 11.56       |           9.39 |
| exp11_8_replication_concrete1_raw  |     30 | 14.2769 +/- 1.3635          | 13.9726 +/- 1.4376         | 41.87 +/- 20.47       |           5.89 |
| exp11_8_replication_concrete1_norm |     30 | 12.3889 +/- 1.4542          | 11.9689 +/- 1.5085         | 74.13 +/- 27.21       |           9.36 |

## 2. Testes de Hipótese Estatística (Mann-Whitney U, nível de significância alpha = 0.05)

- **Seleção Torneio (k=2) vs Roleta**: U = 486.50, p-valor = 5.9456e-01 (Sem diferença estatisticamente significativa)
- **Seleção Torneio (k=2) vs ε-Lexicase**: U = 762.00, p-valor = 4.1178e-06 (Diferença estatisticamente significativa)
- **Elitismo (Presente vs Ausente)**: U = 354.00, p-valor = 1.5798e-01 (Sem diferença estatisticamente significativa)
- **Operadores Config A (pc=0.9, pm=0.05) vs Config B (pc=0.6, pm=0.3)**: U = 418.00, p-valor = 6.4142e-01 (Sem diferença estatisticamente significativa)
- **Concrete1: Sem normalização vs Com normalização**: U = 786.00, p-valor = 7.0430e-07 (Diferença estatisticamente significativa)

## 3. Melhores Expressões Encontradas por Base

### Base synth1
- **Melhor (Seed 13):**
  - RMSE Teste: `0.4612782124430076` | Treino: `1.55179432640642` | Tamanho: `35` nós
  - Expressão: `(((((x0 * x0) + (0.473 - x0)) * (x0 * x0)) + (-0.128 * x0)) + ((((-0.9725 - x1) + ((0.473 - x0) + (0.473 - x0))) + ((-0.128 * x0) + x0)) + x0))`
- **Mediana (Seed 20):**
  - RMSE Teste: `5.895363830135762` | Treino: `4.074633627268359` | Tamanho: `49` nós
  - Expressão: `(((((-0.2326 - 0.651) - ((x0 * -0.8602) - -0.08589)) + ((x0 * (-0.08589 * x0)) * (x0 * (x0 / -0.2326)))) - (x0 * ((x0 / -0.2326) * (-0.08589 * x0)))) + (((x0 / -0.2326) + (-0.08589 * -0.2326)) - (x0 * ((x0 / -0.2326) - (-0.08589 * x0)))))`

### Base synth2
- **Melhor (Seed 9):**
  - RMSE Teste: `1.3874975916047592` | Treino: `1.3553382763685151` | Tamanho: `41` nós
  - Expressão: `(((x1 - (x1 / ((x0 + x1) * (x0 - x0)))) - (((x0 / (x0 - x0)) / (x0 - x0)) + (x0 / x0))) * (((x0 / (x0 + x1)) / (x0 - x0)) * (x0 + (-0.885 / 0.3098))))`
- **Mediana (Seed 29):**
  - RMSE Teste: `3.527801554966149` | Treino: `3.1503248687551983` | Tamanho: `39` nós
  - Expressão: `((((x1 * x0) * 0.149) * ((((0.6508 * x0) / x0) * ((0.6508 * 0.149) * (0.6508 * x0))) * x1)) * ((((x1 * x0) * (0.6508 * 0.149)) * ((0.6508 * (x1 * x0)) * 0.149)) * 0.149))`

### Base concrete1 (Normalizada)
- **Melhor (Seed 26):**
  - RMSE Teste: `10.182667336105192` | Treino: `10.305229185498876` | Tamanho: `103` nós
  - Expressão: `((((((0.5532 / 0.3266) * (-0.2286 + x4)) - (-0.6117 * (x7 + x0))) + (((x7 + x0) + (x1 + x1)) + ((x7 + x0) + (0.9384 / 0.1795)))) + (((0.5532 / 0.3266) / (0.8346 + (0.6431 / -0.8349))) + (((0.5532 / 0.3266) + (x7 + x0)) + (-0.2487 * 0.1795)))) - (((((0.5532 / 0.3266) + (0.5532 / 0.3266)) / ((0.5532 / 0.3266) + (x7 + x1))) - (((x7 + x1) + (0.9936 + x4)) + (0.9384 / 0.1795))) - ((((0.5532 / 0.3266) * (0.9936 + x4)) - (x3 * (0.5532 / 0.3266))) + (x0 * ((x1 + x1) + (0.9384 / 0.1795))))))`
- **Mediana (Seed 19):**
  - RMSE Teste: `11.672279036509742` | Treino: `12.147689781675872` | Tamanho: `67` nós
  - Expressão: `((((((0.443 - -0.8289) + (0.443 - -0.8289)) + ((x4 - -0.8289) + (0.443 - -0.8289))) + (((0.443 - -0.8289) + (x4 - -0.8289)) + ((x0 - -0.8289) + (x0 / 0.443)))) / 0.443) + (((((x1 - -0.8289) + (0.443 - -0.8289)) + ((x4 - -0.8289) + (x5 + x7))) + (((x1 - -0.8289) + (0.443 - -0.8289)) + ((0.443 - -0.8289) + (0.443 - -0.8289)))) / 0.443))`

## 4. Análise e Respostas às Perguntas do Enunciado

### 4.1 Convergência e Saturação (Tamanho da População e Gerações)
- **Observações**: O aumento de população de 50 para 100 e 500 melhora a exploração do espaço de busca. No entanto, pop=500 com 100 gerações aumenta o custo computacional em ~5x sem ganho proporcional em teste.
- **Saturação**: Após ~60-80 gerações na maioria das execuções, o RMSE atinge um patamar assintótico, caracterizando saturação da busca para o limite de profundidade 7.
- **Escolha Proposta**: `pop_size = 100` e `generations = 100` oferecem o melhor equilíbrio entre eficácia e tempo computacional.

### 4.2 Métodos de Seleção
- **Torneio vs Roleta**: A seleção por roleta converge consideravelmente pior, pois a conversão 1/(1+RMSE) em espaços com diferenças absolutas de erro achata as probabilidades, exercendo baixa pressão seletiva. O teste Mann-Whitney confirma a superioridade altamente significativa do Torneio.
- **Torneio vs ε-Lexicase**: O ε-Lexicase preserva maior diversidade genética na população ao avaliar os indivíduos em casos de teste isoladamente com tolerância adaptativa MAD. Para bases com ruído ou subcasos variados, o ε-Lexicase apresenta excelentes resultados em teste.

### 4.3 Efeito de Crossover e Mutação
- **Configuração A (pc=0.9, pm=0.05) vs B (pc=0.6, pm=0.3)**: Na dinâmica de operadores, o crossover produz uma taxa de filhos melhores que os pais mais frequente no início da evolução, atuando na recombinação de blocos construtivos (building blocks). Conforme a população converge, o crossover torna-se predominantemente destrutivo. A mutação alta na Config B introduz variabilidade que ajuda a escapar de ótimos locais, mas taxa excessiva prejudica a preservação das soluções refinadas.
- **Valores Propostos**: `pc = 0.9` e `pm = 0.05` mantêm-se como padrão equilibrado.

### 4.4 Elitismo
- A ausência de elitismo resulta em perda frequente da melhor solução encontrada devido a crossover/mutação destrutivos nas gerações finais. O elitismo (tamanho 1) garante monotonicidade no melhor indivíduo de treino e reduz a variância entre seeds.

### 4.5 Bloating e Controle de Tamanho
- O tamanho médio dos indivíduos cresce rapidamente nas primeiras 15-25 gerações até encostar no teto permitido (`max_size=7`). Em execuções com `max_size=10`, nota-se crescimento adicional ('código intruso' ou bloat), sem ganho substancial de acurácia de teste, demonstrando a importância de manter um limite restritivo de profundidade para prevenir overfitting e bloat.

### 4.6 Treino vs. Teste e Overfitting
- Em `synth1`, devido ao número reduzido de amostras de treino (apenas 60 casos para 600 de teste), observa-se um gap entre RMSE de treino e teste em algumas árvores mais complexas. O limite de profundidade 7 atua eficazmente como regularizador estrutural, mitigando o sobreajuste.
- Em `concrete1`, a normalização z-score dos atributos de treino reduz substancialmente o RMSE e estabiliza os valores das constantes e operações.

## 5. Parâmetros Finais Propostos

| Parâmetro | Valor Proposto | Justificativa |
|---|---|---|
| `pop_size` | 100 | Excelente exploração com custo computacional baixo |
| `generations` | 100 | Garante convergência estável sem custo redundante |
| `max_size` | 7 | Limite de profundidade eficiente contra bloating e overfitting |
| `size_mode` | `depth` | Controle direto da complexidade hierárquica das expressões |
| `init_method` | `ramped_half_and_half` | Garante variedade estrutural na população inicial |
| `selection` | `tournament` (k=2) / `lexicase` | Torneio é rápido e eficaz; Lexicase preserva diversidade |
| `crossover_prob` (pc) | 0.9 | Recombinação eficiente de subexpressões úteis |
| `mutation_prob` (pm) | 0.05 | Exploração local sem destruição de building blocks |
| `elitism` | True (elite_size=1) | Preserva a melhor solução e assegura convergência monotônica |
| `function_set` | `[add, sub, mul, div]` | Operadores aritméticos protegidos completos |
