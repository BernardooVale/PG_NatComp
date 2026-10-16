# Framework de Programação Genética (GP) para Regressão Simbólica

Implementação em Python de um algoritmo completo de Programação Genética (GP) com indivíduos representados em árvore para problemas de Regressão Simbólica, desenvolvido para a disciplina de Computação Natural.

---

## 1. Inspeção dos Datasets

Os dados estão localizados no diretório `datasets/`. Todos os arquivos foram inspecionados quanto à presença de cabeçalho, tipos, valores nulos e dimensões:

| Dataset | Arquivo | Amostras | Atributos (X) | Saída (y) | Separador | Header | NaNs | Tipos |
|---|---|---|---|---|---|---|---|---|
| **synth1** | `synth1-train.csv` | 60 | 2 (`x0`, `x1`) | 1 (`y`) | Vírgula (`,`) | Não | 0 | `float64` |
| **synth1** | `synth1-test.csv` | 600 | 2 (`x0`, `x1`) | 1 (`y`) | Vírgula (`,`) | Não | 0 | `float64` |
| **synth2** | `synth2-train.csv` | 300 | 2 (`x0`, `x1`) | 1 (`y`) | Vírgula (`,`) | Não | 0 | `float64` |
| **synth2** | `synth2-test.csv` | 1000 | 2 (`x0`, `x1`) | 1 (`y`) | Vírgula (`,`) | Não | 0 | `float64` |
| **concrete1** | `concrete-train.csv` | 824 | 8 (`x0..x7`) | 1 (`y`) | Vírgula (`,`) | Não | 0 | `float64` / `int64` |
| **concrete1** | `concrete-test.csv` | 206 | 8 (`x0..x7`) | 1 (`y`) | Vírgula (`,`) | Não | 0 | `float64` / `int64` |

> As dimensões conferem com a especificação do enunciado.

---

## 2. Decisões de Implementação e Arquitetura

1. **Genótipo (Estrutura de Árvore)**:
   - Implementado na classe `Node` (`gp/tree.py`). Cada nó pode ser uma primitiva funcional, uma variável (`x0`, `x1`, ...) ou uma constante numérica efêmera (ERC).
   - Avaliação totalmente vetorizada via NumPy sobre a matriz de entrada `X` de dimensões `(N, d)`.
   - `Individual` encapsula a raiz da árvore, metadados de origem e cache de avaliação (`fitness` e `case_errors`).

2. **Divisão Protegida e Primitivas Numéricas** (`gp/primitives.py`):
   - **Divisão Protegida (Koza)**: Se `|b| < 1e-6`, retorna `1.0`.
   - Funções adicionais protegidas: `exp` (com `np.clip` para evitar overflow), `log` protegido (`log(|x| + 1e-6)`) e `sqrt` protegido (`sqrt(|x|)`).
   - Tratamento de `NaN`/`inf`: Na ocorrência de valores indefinidos ou divergentes, o indivíduo recebe uma fitness penalizada (`1e12`).

3. **Interpretação de "Tamanho Máximo 7" e Modos de Tamanho** (`gp/config.py`):
   - O framework suporta tanto `size_mode = "depth"` quanto `size_mode = "nodes"`.
   - No modo padrão `depth`, folhas têm profundidade 1 e o limite máximo de profundidade é 7.
   - Qualquer operador ou inicialização que produza indivíduos violando `max_size` passa por tentativa de repetição com fallback seguro.

4. **Inicialização da População** (`gp/init.py`):
   - Implementa os métodos `grow`, `full` e `ramped_half_and_half`.
   - Gera constantes no intervalo configurável `const_range` (padrão `[-1.0, 1.0]`) com probabilidade `const_prob`.

5. **Operadores Genéticos e Não Modificação in-place** (`gp/operators.py`):
   - **Crossover de Subárvore**: Seleção de nós com viés para nós internos (funções). Faz cópia profunda para garantir que os pais nunca sejam alterados.
   - **Mutação de Subárvore**: Substituição de subárvore gerada por `grow`.
   - **Mutação Pontual**: Substituição conservativa de primitivas mantendo a aridade.
   - Rastreamento de metadados: cada filho registra a fitness dos pais e o operador gerador.

6. **Métodos de Seleção** (`gp/selection.py`):
   - **Torneio**: Seleciona `tournament_size` indivíduos aleatórios e escolhe o de menor RMSE.
   - **Roleta**: Mapeia minimização para aptidão positiva via `1 / (1 + RMSE)`. Indivíduos com fitness penalizada recebem probabilidade nula.
   - **ε-Lexicase**: Avaliação caso a caso em ordem aleatória, calculando o limiar de tolerância $\lambda = \text{MAD} = \text{median}(|e_t - \text{median}(e_t)|)$ entre os candidatos ativos.

7. **Estatísticas Geracionais e Bloating** (`gp/stats.py`):
   - Coleta por geração: `best_fitness`, `worst_fitness`, `mean_fitness`, `std_fitness`.
   - Diversidade: contagem de repetidos `n_repeated = pop_size - n_unique` usando representação canônica.
   - Dinâmica de operadores: contagem de filhos melhores/piores que os pais (`cx_better_than_mean_parents`, `mut_better`, etc.).
   - Indicadores de bloating: `mean_size`, `max_size_pop`, `mean_depth`.

8. **Reprodutibilidade**:
   - Totalmente determinístico para uma dada semente (`seed`) via `numpy.random.default_rng(seed)`.

---

## 3. Como Executar

Certifique-se de utilizar o Python do ambiente virtual `.venv`:

### Executar os Testes Unitários
```bash
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

### Executar uma Execução Individual (CLI)
```bash
.\.venv\Scripts\python.exe experiments/run_single.py --dataset synth1 --seed 42 --exp_name single_test --verbose
```

### Executar um Lote com Múltiplas Repetições em Paralelo
```bash
.\.venv\Scripts\python.exe experiments/run_batch.py --dataset synth1 --runs 30 --n_jobs -1 --exp_name batch_baseline
```

### Reproduzir o Estudo Experimental Completo (Seção 11)
```bash
.\.venv\Scripts\python.exe experiments/run_study.py --runs 30 --n_jobs -1
```

---

## 4. Estrutura do Repositório

```
.
├── configs/
│   └── default.yaml
├── datasets/
│   ├── concrete/
│   ├── synth1/
│   └── synth2/
├── gp/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── engine.py
│   ├── fitness.py
│   ├── init.py
│   ├── operators.py
│   ├── primitives.py
│   ├── selection.py
│   ├── stats.py
│   └── tree.py
├── experiments/
│   ├── analyze.py
│   ├── run_batch.py
│   ├── run_single.py
│   └── run_study.py
├── results/
│   ├── figures/
│   ├── experiments_consolidated.csv
│   ├── experiments_consolidated.md
│   └── SUMMARY.md
├── tests/
├── requirements.txt
└── README.md
```
