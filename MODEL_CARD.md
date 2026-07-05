# Model Card -- Recomendador Neural E-commerce

Documento de identidade do modelo, seguindo o padrao Model Card (Mitchell et al., 2019). Descreve o que o modelo faz, como foi treinado, seu desempenho, limitacoes e consideracoes eticas.

---

## 1. Detalhes do modelo

| Atributo | Valor |
|----------|-------|
| Nome | recomendador-neural |
| Versao em producao | v2 |
| Tipo | Rede neural de recomendacao baseada em embeddings |
| Framework | PyTorch 2.12 (CPU) |
| Arquitetura | Embeddings de usuario e item + MLP (Linear -> ReLU -> Linear) |
| Dimensao dos embeddings | 64 |
| Funcao de perda | MSE (Mean Squared Error) |
| Otimizador | Adam (learning rate 0.001) |
| Epocas de treino | 5 |
| Semente (reprodutibilidade) | 42 |
| Registro | MLflow Model Registry (alias "production") |
| Autor | Henrique Silva |
| Data | Julho de 2026 |

### Arquitetura

Cada usuario e cada item sao representados por um vetor aprendido de 64 dimensoes (embedding). A afinidade entre um par usuario-item e estimada concatenando os dois vetores e passando o resultado por uma MLP de duas camadas.

```
user_id --> [ user embedding ] --+
                                 +--> concat --> Linear(128->64) --> ReLU --> Linear(64->1) --> score
item_id --> [ item embedding ] --+
```

---

## 2. Uso pretendido

### Uso primario

Prever a afinidade de um usuario por um item (produto) em um contexto de e-commerce, a partir de historico de interacoes (visualizacoes, adicoes ao carrinho e compras). A pontuacao pode ser usada para ordenar recomendacoes de produtos.

### Usuarios pretendidos

Equipes de dados e engenharia de plataformas de e-commerce que desejam gerar recomendacoes personalizadas.

### Fora de escopo

- Nao deve ser usado como unico criterio para decisoes automatizadas de alto impacto sem supervisao humana.
- Nao foi projetado para recomendar itens sensiveis (saude, financas) que exijam garantias regulatorias.
- Nao generaliza para usuarios ou itens nunca vistos no treino (problema de cold start).

---

## 3. Dados de treino

| Atributo | Valor |
|----------|-------|
| Fonte | RetailRocket E-commerce Dataset |
| Tipo de sinal | Implicito (eventos de navegacao) |
| Eventos brutos | 2.756.101 |
| Interacoes apos preprocessamento | 172.471 |
| Usuarios (apos k-core) | 15.353 |
| Itens (apos k-core) | 13.031 |
| Divisao treino/teste | 80% / 20% |

### Traducao de sinais implicitos

Os eventos de navegacao foram convertidos em uma escala numerica de interesse:

| Evento | Peso | Interpretacao |
|--------|------|---------------|
| view | 1.0 | Interesse fraco (visualizacao) |
| addtocart | 2.0 | Interesse medio (carrinho) |
| transaction | 3.0 | Interesse forte (compra) |

### Preprocessamento

1. **k-core filtering iterativo (k=5):** remove usuarios e itens com menos de 5 interacoes, reduzindo esparsidade e ruido (reducao de 71,6% no volume).
2. **Amostragem:** subconjunto de usuarios para viabilizar o treino.
3. **Normalizacao:** rating escalado para [0, 1] (min-max).
4. **Reindexacao:** IDs originais mapeados para indices contiguos (exigencia dos embeddings).

---

## 4. Metricas de avaliacao

Comparacao entre as versoes treinadas e um baseline (previsao pela media global), no conjunto de teste:

| Versao | embedding_dim | epochs | RMSE | MAE |
|--------|--------------|--------|------|-----|
| v1 | 32 | 5 | 0.1851 | 0.1017 |
| **v2 (producao)** | **64** | **5** | **0.1843** | **0.101** |
| v3 | 32 | 10 | 0.1887 | 0.104 |
| Baseline (media) | -- | -- | 0.1893 | 0.0986 |

### Metrica de decisao

A versao promovida a producao foi selecionada automaticamente pelo menor **RMSE** (Root Mean Squared Error), metrica que penaliza mais os erros grandes. A v2 (embedding 64) obteve o melhor resultado.

### Interpretacao

O modelo neural supera o baseline no RMSE, confirmando que aprende padroes alem da simples media. A margem e pequena porque o sinal e dominado por visualizacoes (ver limitacoes).

---

## 5. Limitacoes

- **Sinal desbalanceado:** 96,7% das interacoes sao visualizacoes (view), tornando o sinal de preferencia fraco. Isso limita a margem de melhoria sobre o baseline.
- **Cold start:** o modelo nao consegue recomendar para usuarios ou itens ausentes no treino, pois nao possuem embedding aprendido.
- **Problema tratado como regressao:** o modelo preve um valor de rating (MSE), quando o problema real e mais proximo de ranking implicito. Tratar como ranking (ex: BPR loss) seria um refino natural.
- **Amostragem:** o treino usou um subconjunto dos dados para agilidade; o modelo final pode ser retreinado com o volume completo.
- **Metricas de erro apenas:** RMSE e MAE medem erro de predicao, nao qualidade de ranking (metricas como NDCG ou Recall@k seriam complementares).

---

## 6. Vieses e consideracoes eticas

- **Vies de popularidade:** recomendadores tendem a favorecer itens populares, podendo reduzir a diversidade e reforcar a "camara de eco" de produtos ja conhecidos.
- **Dados de comportamento:** o modelo aprende de comportamento passado, podendo perpetuar padroes historicos (ex: sub-representacao de produtos de nicho).
- **Privacidade:** o dataset usa IDs anonimizados. Em producao real, o tratamento de dados de navegacao deve respeitar LGPD/GDPR (consentimento, minimizacao, direito ao esquecimento).
- **Supervisao humana:** recomendacoes devem ser tratadas como sugestoes, nao decisoes automaticas irreversiveis.

---

## 7. Governanca e manutencao

- **Versionamento:** todas as versoes ficam registradas no MLflow Model Registry; a de producao e identificada pelo alias "production".
- **Promocao:** automatizada pelo script `promote_model.py`, que seleciona a versao de menor RMSE de forma rastreavel.
- **Reprodutibilidade:** dados versionados por DVC, ambiente por uv/lock file, pipeline por `dvc repro` e semente fixa (42).
- **Retreino recomendado:** periodicamente ou quando houver drift nos dados (mudanca no comportamento dos usuarios).

---

## Referencias

- Mitchell, M. et al. (2019). *Model Cards for Model Reporting*. FAT*.
- Koren, Y., Bell, R. & Volinsky, C. (2009). *Matrix Factorization Techniques for Recommender Systems*. IEEE Computer.
- He, X. et al. (2017). *Neural Collaborative Filtering*. WWW.
- Dataset: [RetailRocket E-commerce Dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)
