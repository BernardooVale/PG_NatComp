# PLANO DE IMPLEMENTAÇÃO — GP para Regressão Simbólica

Leia este arquivo inteiro antes de começar. Execute as tarefas em ordem.
Ao concluir cada tarefa, troque `[]` por `[x]`. Não pule etapas. Não deixe parâmetros hardcoded.

## 0. Contexto

Framework de Programação Genética (GP) com indivíduos em árvore para regressão simbólica.
Dados em `datasets/`, três bases, cada uma com dois arquivos:

- `synth1-train.csv`, `synth1-test.csv` (3 atributos, 60 treino / 600 teste, sintética)
- `synth2-train.csv`, `synth2-test.csv` (3 atributos, 300 treino / 1000 teste, sintética)
- `concrete1-train.csv`, `concrete1-test.csv` (9 atributos, 824 treino / 206 teste, real)

Última coluna de cada CSV = saída desejada `y`. As demais = atributos `X`.
Fitness = RMSE (minimização): `sqrt(mean((Eval(ind, X) - y)^2))`.

Linguagem: Python 3.10+. Libs: `numpy`, `pandas`, `matplotlib`, `scipy`, `pyyaml`, `tqdm`. Evitar loops Python sobre amostras: avaliação vetorizada com numpy.

## 1. Estrutura do projeto

[x] 1.1 Criar a estrutura:

```
.
├── datasets/                 # já existe, não modificar
├── configs/
│   └── default.yaml
├── gp/
│   ├── __init__.py
│   ├── data.py               # carga dos CSVs
│   ├── tree.py               # nós, árvore, avaliação, utilitários
│   ├── primitives.py         # funções e terminais
│   ├── init.py               # geração da população inicial
│   ├── operators.py          # crossover e mutação
│   ├── selection.py          # roleta, torneio, ε-lexicase
│   ├── fitness.py            # RMSE
│   ├── stats.py              # estatísticas por geração
│   ├── engine.py             # laço evolutivo
│   └── config.py             # dataclass de parâmetros
├── experiments/
│   ├── run_single.py         # 1 execução (CLI)
│   ├── run_batch.py          # N repetições com seeds, paralelo
│   ├── run_study.py          # roteiro completo dos experimentos
│   └── analyze.py            # tabelas, gráficos, testes estatísticos
├── results/                  # saídas (CSV/JSON/PNG), gerado
├── tests/
├── requirements.txt
└── README.md
```

[x] 1.2 Criar `requirements.txt` e ambiente virtual. Verificar instalação.
[x] 1.3 Inspecionar os 6 CSVs em `datasets/`: presença de header, separador, tipos, NaN, shapes. Registrar no README. Conferir shapes contra a Tabela 1 do enunciado.

## 2. Configuração (parâmetros variáveis)

[x] 2.1 Criar `gp/config.py` com dataclass `GPConfig`, carregável de YAML e sobrescrevível por CLI. Parâmetros mínimos:

| Parâmetro | Default | Observação |
|---|---|---|
| `dataset` | `synth1` | nome da base |
| `pop_size` | 100 | testar 50, 100, 500 |
| `generations` | 100 | testar 50, 100, 500 |
| `max_size` | 7 | limite de tamanho do indivíduo |
| `size_mode` | `depth` | `depth` ou `nodes`. Enunciado ambíguo; implementar ambos |
| `init_method` | `ramped_half_and_half` | `grow`, `full`, `ramped_half_and_half` |
| `init_min_depth` | 1 | |
| `selection` | `tournament` | `roulette`, `tournament`, `lexicase` |
| `tournament_size` | 2 | |
| `lexicase_epsilon` | `mad` | ε-lexicase com MAD; `none` = lexicase clássico |
| `crossover_prob` (pc) | 0.9 | |
| `mutation_prob` (pm) | 0.05 | |
| `mutation_type` | `subtree` | `subtree` ou `point` (opcional) |
| `elitism` | `true` | |
| `elite_size` | 1 | |
| `function_set` | `[add, sub, mul, div]` | ver seção 3 |
| `const_range` | `[-1, 1]` | faixa das constantes efêmeras (ERC) |
| `const_prob` | 0.3 | prob. de terminal ser constante |
| `seed` | 0 | |
| `n_runs` | 30 | |
| `n_jobs` | `-1` | paralelismo |

[x] 2.2 Validar config (pc, pm em [0,1]; pop_size > elite_size, etc.).

## 3. Representação e primitivas

[x] 3.1 `primitives.py`: conjunto de funções com aridade e versão vetorizada **protegida**:
  - `add`, `sub`, `mul`
  - `div` protegida: se `|b| < 1e-6` retorna 1.0 (ou `a`; documentar a escolha)
  - Opcionais via config: `sin`, `cos`, `exp` (com clip), `log` protegido (`log(|x|+eps)`), `sqrt` protegido (`sqrt(|x|)`)
[x] 3.2 Terminais: variáveis `x0..x{n-1}` (n lido do dataset) e constantes efêmeras (ERC) sorteadas uniformemente em `const_range`.
[x] 3.3 `tree.py`: classe `Node` (valor, filhos). Funções: `evaluate(tree, X) -> ndarray`, `size`, `depth`, `copy`, `to_string` (infix legível), `get_all_nodes` com caminho/pai para seleção de pontos de corte, `replace_subtree`, `hash/key` canônico (string) para contar repetidos.
[x] 3.4 Tratar saídas `NaN`/`inf`: substituir por penalidade (ex.: fitness = 1e12). Nunca deixar quebrar a execução.
[x] 3.5 Testes unitários: avaliação correta de árvores conhecidas, divisão protegida, cópia profunda, tamanho/profundidade.

## 4. Dados e fitness

[x] 4.1 `data.py`: `load_dataset(name) -> (X_train, y_train, X_test, y_test)`. Sem normalização por default; flag opcional de normalização (z-score calculado SÓ no treino) para `concrete1`.
[x] 4.2 `fitness.py`: `rmse(y_pred, y)`. Também função que retorna o **vetor de erros por caso** `|pred - y|` (ou erro quadrático), necessário para lexicase.
[x] 4.3 Cache no indivíduo: guardar `fitness` e `case_errors` para não reavaliar.

## 5. População inicial

[x] 5.1 `init.py`: implementar `grow`, `full` e `ramped_half_and_half` respeitando `max_size`/`size_mode`.
[x] 5.2 Garantir que nenhum indivíduo inicial viole o limite de tamanho.
[x] 5.3 Testes: distribuição de profundidades e ausência de violações.

## 6. Operadores genéticos

[x] 6.1 Crossover de subárvore: escolher ponto de corte aleatório em cada pai (viés opcional para nós internos), trocar subárvores, gerar 2 filhos.
[x] 6.2 Mutação de subárvore: escolher nó, substituir por subárvore nova (grow) com profundidade limitada.
[x] 6.3 (Opcional) Mutação pontual: trocar função por outra de mesma aridade, terminal por outro terminal, perturbar constante.
[x] 6.4 Controle de tamanho: se filho excede `max_size`, rejeitar e tentar novamente (N tentativas); se falhar, copiar o pai.
[x] 6.5 Retornar, para cada filho, metadados de origem: operador usado e referência às fitness dos pais (necessário para estatísticas da seção 8).
[x] 6.6 Testes: filhos nunca excedem o limite; pais não são modificados in-place.

## 7. Seleção

Todas recebem a população já avaliada e retornam 1 pai (ou k pais).

[x] 7.1 **Torneio**: amostrar `tournament_size` indivíduos, retornar o de menor RMSE.
[x] 7.2 **Roleta**: como o problema é de minimização, converter para aptidão `1 / (1 + rmse)` (documentar). Tratar fitness penalizada (≈0 probabilidade). Amostragem proporcional vetorizada.
[x] 7.3 **ε-Lexicase**, conforme enunciado:
  1. candidatos = toda a população
  2. embaralhar índices dos casos de treino
  3. para o próximo caso `t`: calcular `e*_t = min(e_t)` entre candidatos e `λ(e_t) = MAD = median(|e_t - median(e_t)|)` calculada sobre os **candidatos atuais**; manter candidatos com `e_t < e*_t + λ`
  4. se restar 1 candidato, retorna-o. Se acabarem os casos, sorteia um candidato aleatório. Caso contrário passa ao próximo caso
  - Usar a matriz de erros `(pop_size × n_casos)` pré-computada em numpy
  - Atenção: se `λ = 0`, manter apenas os candidatos com erro mínimo (`<=`), para nunca eliminar todos
  - Variante `lexicase_epsilon: none` = lexicase clássico (`e_t == e*_t`)
[x] 7.4 Testes: torneio com k=pop retorna o melhor; lexicase nunca retorna vazio; roleta com fitness iguais é uniforme.

## 8. Estatísticas por geração

[x] 8.1 `stats.py`: coletar a cada geração (incluindo geração 0) e salvar em CSV por execução:
  - `best_fitness`, `worst_fitness`, `mean_fitness` (treino)
  - `n_repeated`: nº de indivíduos repetidos (população − nº de indivíduos únicos, via chave canônica da árvore)
  - `cx_better_than_mean_parents`, `cx_worse_than_mean_parents`: filhos de crossover com fitness melhor/pior que a **média da fitness dos pais**
  - `cx_better_than_best_parent`, `cx_worse_than_best_parent` (extra, útil para análise do efeito do crossover)
  - `mut_better`, `mut_worse`: filhos de mutação melhores/piores que o pai
  - Indicadores de bloating: `mean_size`, `max_size_pop`, `mean_depth`, `size_best`
  - (Opcional) `n_unique`, diversidade fenotípica
[x] 8.2 Ao final da execução, avaliar o **melhor indivíduo** (menor RMSE de treino) no conjunto de teste. Salvar `train_rmse`, `test_rmse`, expressão em string, tamanho.
[x] 8.3 Definir e documentar a medida de bloating: tamanho médio da população (e do melhor) ao longo das gerações; correlação entre crescimento de tamanho e melhora de fitness.

## 9. Motor evolutivo

[x] 9.1 `engine.py`: laço principal:
  1. gerar e avaliar população inicial
  2. para cada geração: elitismo (copiar `elite_size` melhores) → gerar filhos até completar a população: com prob. `pc` aplicar crossover em 2 pais selecionados; senão copiar pais; aplicar mutação em cada filho com prob. `pm`; avaliar filhos; coletar estatísticas
  3. substituição geracional
  4. retornar histórico + melhor indivíduo
[x] 9.2 Semente controlada via `numpy.random.default_rng(seed)`; resultados reprodutíveis dado o seed.
[x] 9.3 Elitismo desligável (`elitism: false`) sem alterar mais nada.
[x] 9.4 Avaliação eficiente: reavaliar só indivíduos novos; usar cache.
[x] 9.5 Teste de sanidade: em função simples (ex.: `y = x0*x0 + x1`) com pop 200 / 50 gerações, melhor RMSE deve cair significativamente.

## 10. Execução de experimentos

[x] 10.1 `run_single.py`: CLI `--config configs/default.yaml --dataset synth1 --seed 3 [--override chave=valor]`. Salva histórico por geração e resumo final em `results/<experimento>/`.
[x] 10.2 `run_batch.py`: executa `n_runs` (default 30) com seeds distintas (`seed = 0..29`), em paralelo (`multiprocessing`/`joblib`). Salva um CSV consolidado: uma linha por execução (`seed, train_rmse, test_rmse, size, expr, ...`) e um CSV de histórico por geração com coluna `run`.
[x] 10.3 `run_study.py`: roteiro completo da seção 11, com nomes de experimento organizados, sem sobrescrever resultados antigos (ou com flag `--force`).
[x] 10.4 Registrar tempo de execução por experimento.

## 11. Roteiro experimental (seguir esta ordem)

Regra: ao variar um parâmetro, manter TODOS os outros fixos. Cada configuração = 30 repetições. Reportar média ± desvio-padrão de RMSE de treino e teste (do melhor indivíduo final). Se o desvio-padrão for alto, aumentar repetições.

Escolher UMA base sintética (`synth1` ou `synth2`) para o ajuste de parâmetros. Depois reutilizar os melhores parâmetros na outra sintética e em `concrete1`.

[x] 11.1 **Baseline**: tamanho máx. 7, elitismo, torneio k=2, pc=0.9, pm=0.05.
[x] 11.2 **Tamanho da população e nº de gerações**: testar `pop ∈ {50, 100, 500}` e `gens ∈ {50, 100, 500}` (grade 3×3 ou varrer um por vez). Avaliar: melhora do erro, saturação, convergência da população (curvas de best/mean/worst), custo computacional. Escolher valores e justificar.
[x] 11.3 **Método de seleção**: com pop/gens escolhidos, comparar `tournament` (k=2; opcionalmente k=3,5), `roulette` e `lexicase` (ε-lexicase/MAD). Comparar RMSE treino/teste, curvas de convergência, diversidade (`n_repeated`), tempo. Escolher o mais apropriado.
[x] 11.4 **Crossover e mutação**: com a seleção escolhida:
  - Config A: pc=0.9, pm=0.05
  - Config B: pc=0.6, pm=0.3
  - Para ambas: coletar e comparar nº de filhos melhores/piores que os pais (crossover e mutação), ao longo das gerações e acumulado.
  - Varrer `pc` (ex.: 0.5, 0.6, 0.7, 0.8, 0.9) mantendo pm fixo; depois varrer `pm` (ex.: 0.01, 0.05, 0.1, 0.2, 0.3) mantendo pc fixo.
  - Responder: os parâmetros anteriores continuam apropriados? Que valores finais propor?
[x] 11.5 **Elitismo**: com os melhores parâmetros, rodar com e sem elitismo. Comparar resultados.
[x] 11.6 **Bloating**: medir tamanho médio ao longo das gerações para as principais configurações. Discutir se há bloating, e o efeito do limite de tamanho. (Opcional: testar `max_size` diferente de 7 apenas como análise extra.)
[x] 11.7 (Opcional) Mecanismos de diversidade (ex.: rejeitar duplicatas, torneio com penalização por similaridade). Comparar.
[x] 11.8 **Replicar** a melhor configuração na outra base sintética e em `concrete1`, 30 execuções cada. Para `concrete1`, considerar normalização e conjunto de funções mais rico se necessário; documentar.
[x] 11.9 **Treino vs. teste**: para toda configuração final, comparar RMSE de treino e teste. Apontar sinais de overfitting (gap grande, especialmente em `synth1`, com apenas 60 amostras de treino).

## 12. Análise e relatório de resultados

[x] 12.1 `analyze.py`: gerar tabelas (CSV + Markdown) com média ± desvio-padrão para cada experimento.
[x] 12.2 Gráficos (PNG em `results/figures/`):
  - Curvas de convergência (best, mean, worst por geração; média entre execuções, com banda de desvio-padrão)
  - Boxplots de RMSE final (treino e teste) por configuração
  - Comparação entre métodos de seleção
  - Filhos melhores/piores por operador ao longo das gerações
  - Nº de repetidos por geração
  - Tamanho médio por geração (bloating)
[x] 12.3 Testes estatísticos entre configurações (Mann-Whitney U ou Wilcoxon, com α=0.05) nas comparações principais (seleção, elitismo, config A vs. B).
[x] 12.4 Listar, para cada base, as melhores expressões encontradas (execução de menor RMSE de teste e mediana), simplificadas se possível (ex.: `sympy`, opcional).
[x] 12.5 Gerar `results/SUMMARY.md` com: parâmetros finais propostos + justificativa; respostas às perguntas do enunciado (convergência, efeito de crossover/mutação, seleção, elitismo, bloating, overfitting).

## 13. Finalização

[x] 13.1 `README.md`: como instalar, rodar uma execução, rodar o lote, reproduzir todo o estudo (`python experiments/run_study.py`), descrição das decisões de implementação (genótipo, inicialização, operadores, tratamento de divisão por zero, interpretação de "tamanho máximo 7").
[x] 13.2 Rodar todos os testes em `tests/` (`pytest`). Todos devem passar.
[x] 13.3 Conferir reprodutibilidade: mesma seed → mesmos resultados.
[x] 13.4 Conferir que nenhum parâmetro está hardcoded fora de `GPConfig`/YAML.

## Critérios de aceite

- 3 bases rodam de ponta a ponta com 30 repetições cada.
- As 3 seleções (roleta, torneio, ε-lexicase) implementadas e comparadas.
- Estatísticas da seção 8 salvas para todas as gerações.
- Resultados reportados como média ± desvio-padrão, treino e teste.
- Respostas às perguntas do enunciado documentadas em `results/SUMMARY.md`.