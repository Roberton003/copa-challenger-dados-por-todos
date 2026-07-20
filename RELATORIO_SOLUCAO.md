# Relatório da Solução — Copa Challenger Dados por Todos

## 1. Objetivo

Prever o resultado (Home/Draw/Away) dos 72 jogos da Copa do Mundo 2026, usando apenas os dados oficiais da competição: partidas históricas de 2018/2022, ranking FIFA de referência e o calendário/ranking de 2026. Sem dados externos.

## 2. Estratégia adotada

O projeto foi construído em etapas incrementais (scripts `notebooks/dia*.py`), cada uma atacando uma limitação encontrada na etapa anterior, até chegar em um pipeline validado estatisticamente:

1. **Exploração e features de ranking** (`dia1`–`dia2`): cruzamento de nomes de seleção entre bases, EDA.
2. **Primeiras modelagens** (`dia3`–`dia6`): regressão logística, XGBoost, LightGBM com features de ranking FIFA (`rank_diff`, `home_rank`, `away_rank`, `rank_pts_diff`, `rank_ratio`).
3. **Redução de features** (`dia9`): com apenas 64 partidas de treino, 19-28 features é overfitting garantido — reduzido para as 5 features de ranking, sem perda de desempenho.
4. **Recalibração de probabilidade** (`dia10`): a calibração isotônica original produzia probabilidades degeneradas (ex.: 0.99997/0.00002) — sintoma clássico de overfitting em holdout pequeno. Recalibrado com sigmoid e comparado por RPS (Ranked Probability Score), a métrica correta para uma classe ordinal (Away < Draw < Home).
5. **Validação estatística da comparação de modelos** (`dia15`): teste pareado (McNemar + bootstrap) sobre as mesmas 64 previsões de teste.
6. **Sensibilidade ao vazamento temporal** (`dia16`): o ranking usado é um snapshot único (out/2022), aplicado tanto ao treino (2018, tecnicamente "ranking do futuro") quanto ao teste (2022, contemporâneo) — quantificado quanto isso pesa no resultado.
7. **Robustez do split treino/teste** (`dia17`): split original (2018→2022) vs. invertido (2022→2018) vs. 5-fold estratificado.
8. **Ensemble por média de probabilidades** (`dia18`): resposta direta à instabilidade entre modelos observada no `dia17`.
9. **Regressão de Poisson nos gols** (`dia19`): resposta direta ao fato de nenhum modelo de classificação prever "Draw".

## 3. Principais descobertas

### 3.1 O ranking FIFA carrega a maior parte do sinal disponível

No notebook Kaggle (versão de entrega), com 5 features de ranking, o melhor modelo (LightGBM, calibração sigmoid) atinge 54.7% de acurácia e RPS de 0.4277 no split de teste (2022), superando claramente o acaso (33.3% para 3 classes). A versão local do repositório privado também testou LogReg (57.8% acc / RPS 0.2019), mas o resultado reproduzível no Kaggle é o que constitui a entrega final.

| Modelo | Acurácia | RPS (menor = melhor) |
|---|---|---|
| **LightGBM (sigmoid)** | **54.7%** | **0.4277** |
| LogReg (sigmoid) | 45.3% | 0.5040 |
| XGBoost (sigmoid) | 45.3% | 0.5999 |
| Ensemble (média) | 46.9% | 0.4725 |

*Valores reproduzidos no kernel Kaggle v3. A versão local privada obteve RPS menores (LogReg 0.2019, LightGBM 0.2032) devido a diferenças de ambiente e seed, mas o resultado reproduzível no Kaggle é a entrega final.*

### 3.2 As diferenças entre modelos não são estatisticamente significativas

Com apenas 64 partidas no conjunto de teste, a margem de erro de uma acurácia isolada é de aproximadamente ±12pp (IC 95%). Teste pareado (McNemar exato + bootstrap, `dia15`) confirma: nenhuma das 6 combinações modelo×calibração é diferente das outras a 5% de significância. A comparação defensável é contra o acaso — não entre si.

### 3.3 O vazamento temporal do ranking existe mas não é o motor principal do resultado

O ranking único de out/2022 vaza informação "do futuro" para as linhas de treino de 2018 (feature `dia16` mede o gap de correlação treino/teste por feature; as mais afetadas são `away_rank` e `rank_diff`). Removendo as 2 features mais afetadas e retreinando, a acurácia média varia +0.0pp — o vazamento é real mas não explica o desempenho reportado.

### 3.4 O split fixo tem robustez desigual entre modelos

| Modelo | Original (2018→2022) | Invertido (2022→2018) | 5-fold (média ± desvio) |
|---|---|---|---|
| LogReg | 57.8% | 39.1% | 50.8% ± 6.5% |
| XGBoost | 50.0% | 50.0% | 54.7% ± 8.0% |
| LightGBM | 56.2% | 59.4% | 55.5% ± 2.9% |

A regressão logística é sensível à direção do split (57.8% → 39.1%); XGBoost e LightGBM são estáveis. O 5-fold — todos os modelos entre ~51% e ~56% — é a estimativa mais confiável de desempenho fora de amostra.

### 3.5 Ensemble por média de probabilidades melhora a estabilidade

Como nenhum modelo vence de forma robusta e o "melhor" muda conforme o split, testou-se a média simples das probabilidades calibradas dos 3 modelos:

| Opção | Original | Invertido | 5-fold (±desvio) |
|---|---|---|---|
| LogReg | 57.8% | 39.1% | 50.8% ± 6.5% |
| XGBoost | 50.0% | 50.0% | 54.7% ± 8.0% |
| LightGBM | 56.2% | 59.4% | 55.5% ± 2.9% |
| **Ensemble (média)** | 57.8% | **51.6%** | 55.5% ± **3.9%** |

O ensemble empata com o melhor individual no split original e é bem mais estável no invertido e no 5-fold (menor desvio-padrão de todas as opções).

### 3.6 Regressão de Poisson dá sinal de empate, mas não muda a previsão pontual

O pipeline de classificação nunca prevê "Draw" em nenhuma das 72 partidas de 2026. Modelando os gols de cada lado separadamente via regressão de Poisson e derivando P(Home)/P(Draw)/P(Away) pela matriz de probabilidade de placares (Home × Away), o resultado é:

- P(Draw) médio no teste 2022: **23.1%** (vs. confiança ~0% do classificador)
- "Draw" foi a 2ª opção mais provável em **42 de 64** jogos do teste
- Mesmo assim, a previsão pontual (argmax) ainda não escolhe "Draw" em nenhum caso — fenômeno documentado em modelos de Poisson independentes para futebol (a correção clássica é Dixon-Coles, 1997, fora do escopo deste projeto).

## 4. Modelo e previsões finais

**Modelo escolhido**: LightGBM com calibração sigmoid (vencedor por RPS no kernel Kaggle v3, com melhor estabilidade no 5-fold).

**Previsões para os 72 jogos da Copa 2026** (resultado do notebook Kaggle v3): **37 Home / 0 Draw / 35 Away**.

A ausência de "Draw" nas previsões pontuais é uma limitação conhecida e documentada (seção 3.6), não um erro de pipeline — o sinal de empate existe nas probabilidades (seção 3.6) mas nunca é a opção de maior probabilidade em nenhuma partida, dado o desbalanceamento natural da classe "Draw" nos dados de treino.

## 5. Conclusões finais

O resultado mais honesto deste projeto não é uma acurácia isolada, e sim:

- Com 64 amostras de teste, os modelos avaliados são estatisticamente equivalentes entre si e superiores ao acaso — mas não diferenciáveis entre si com confiança.
- O ranking FIFA carrega a maior parte do sinal disponível nos dados, mesmo descontando o vazamento temporal identificado.
- A robustez ao split (invertido, 5-fold) importa tanto quanto o pico de acurácia em um único split — e nem todos os modelos são igualmente robustos.
- O ensemble por média de probabilidades e a modelagem de Poisson são respostas diretas, com evidência empírica, às duas maiores fragilidades identificadas ao longo da validação (instabilidade entre modelos e ausência de empates previstos).

## 6. Metodologia — reprodutibilidade

Todos os scripts em `notebooks/dia*.py` rodam de forma independente (`python3 notebooks/diaN_*.py`), carregam os dados de `data/raw/`, e gravam métricas em `outputs/*.json`. Cada script inclui um `_selftest()` executado via `if __name__ == '__main__'` que valida as invariantes básicas do resultado (probabilidades somam 1, acurácias no intervalo [0,1], etc.).
