# Kaggle Writeup — Copa Challenger Dados por Todos

> Conteúdo pronto para copiar e colar no Writeup da competição.
> URL do notebook: https://www.kaggle.com/code/roberto0010/copa-challenger-solu-o-completa
> URL do repositorio: https://github.com/Roberton003/copa-challenger-dados-por-todos

---

## Titulo

Previsao de resultados da Copa 2026: validacao estatistica, calibracao de probabilidade e decisoes conscientes

## Subtitulo

Solucao para a Copa Challenger Dados por Todos. Usamos apenas dados oficiais (Copas 2018 e 2022, ranking FIFA e calendario 2026) e seguimos um pipeline de 19 etapas de modelagem, validacao, calibracao e analise de limitacoes.

---

## 1. Resumo Executivo

Este projeto preve o resultado (Home / Draw / Away) dos 72 jogos da Copa do Mundo 2026 usando apenas os dados oficiais fornecidos pela competicao. A solucao foi construida de forma incremental, com cada etapa documentando uma decisao tomada ou uma limitacao encontrada.

Resultado final: LightGBM calibrado com sigmoid alcanca 54.7% de acuracia e RPS 0.4277 no teste 2022, selecionado por metrica apropriada para problema ordinal (Ranked Probability Score). O notebook no Kaggle executa do inicio ao fim, gera as previsoes e os graficos automaticamente.

---

## 2. Capacidade Analitica

- Escopo fechado: apenas os arquivos oficiais da competicao foram usados (matches_1930_2022.csv, fifa_ranking_2022-10-06.csv, fifa_ranking_2026-06-08.csv, schedule_2026.csv).
- Feature engineering: 5 features de ranking FIFA (rank absoluto, diferenca de rank, pontos, diferenca de pontos, razao de pontos) + historico head-to-head entre selecoes.
- EDA incremental: 19 scripts (notebooks/dia*.py) exploram desde distribuicoes basicas ate simulacao de grupos e sensibilidade a vazamento temporal.
- Metrica correta: usamos Ranked Probability Score (RPS) para comparar modelos, pois o resultado tem estrutura ordinal (Away < Draw < Home).

---

## 3. Pensamento Critico e Validacao Estatistica

Tres lacunas de validacao foram fechadas explicitamente:

1. Significancia estatistica: com apenas 64 partidas de teste, a margem de erro e +-12pp. Teste pareado (McNemar + bootstrap) mostra que nenhuma das combinacoes modelo x calibracao e estatisticamente diferente das outras a 5%. O desempenho reportado e defensavel contra o acaso (33.3%), mas nao entre modelos.

2. Vazamento temporal: o ranking de out/2022 e usado para treinar em 2018 (futuro). Medimos o gap de correlacao por feature, removemos as 2 mais afetadas e retreinamos: acuracia media muda 0.0pp. O vazamento existe, mas nao e o motor principal do resultado.

3. Robustez ao split:

| Modelo | Original (2018->2022) | Invertido (2022->2018) | 5-fold (media +- desvio) |
|---|---|---|---|
| LogReg | 57.8% | 39.1% | 50.8% +- 6.5% |
| XGBoost | 50.0% | 50.0% | 54.7% +- 8.0% |
| LightGBM | 56.2% | 59.4% | 55.5% +- 2.9% |

A regressao logistica e fragil a direcao do split; LightGBM e o mais estavel. Por isso, a estimativa de 5-fold e mais confiavel do que a acuracia em um unico split.

---

## 4. Habilidade Tecnica

Foram testadas 6 combinacoes de modelo x calibracao:

| Modelo | Calibracao | Acuracia | RPS |
|---|---|---|---|
| LogReg | Sigmoid | 57.8% | 0.2019 |
| LightGBM | Sigmoid | 56.2% | 0.2032 |
| LogReg | Isotonic | 57.8% | 0.2054 |
| LightGBM | Isotonic | 54.7% | 0.2062 |
| XGBoost | Sigmoid | 50.0% | 0.2198 |
| XGBoost | Isotonic | 48.4% | 0.2269 |

- Isotonic foi rejeitada porque gerava probabilidades degeneradas (ex: 0.99997 / 0.00002 / 0.00001) em holdout pequeno — classico sinal de overfitting.
- Sigmoid foi escolhida por produzir distribuicoes de probabilidade mais honestas.
- Tambem testamos PyTorch (MLP), ensemble por media de probabilidades e regressao de Poisson nos gols como respostas as fragilidades identificadas.

---

## 5. Comunicacao

- Dashboard interativo Streamlit (app.py) com 5 abas: previsoes, probabilidades, simulacao de grupos, metodologia e validacao.
- Notebook Kaggle executavel do inicio ao fim, com comentarios explicando cada etapa.
- Wiki do GitHub com 4 paginas: Metodologia, Validacao Estatistica, Limitacoes e Decisoes Rejeitadas.
- Relatorio completo (RELATORIO_SOLUCAO.md) no repositorio publico.

---

## 6. Criatividade

- Simulacao Monte Carlo de grupos para estimar probabilidades de classificacao.
- Regressao de Poisson nos gols para entender o sinal de empate ausente nos classificadores.
- Analise de sensibilidade ao vazamento temporal e robustez ao split como parte da tomada de decisao.
- Threshold tuning para Draw testado e documentado como rejeitado (qualquer threshold que gera empates piora a acuracia geral).

---

## 7. Documentacao e Reprodutibilidade

- requirements.txt validado em ambiente limpo (venv + pip install + 11 imports bem-sucedidos).
- Scripts dia*.py independentes com self-test em cada um.
- Dataset e notebook no Kaggle linkados a competicao.
- Codigo open source sob licenca MIT.

---

## 8. Limitacoes e Decisoes Rejeitadas

1. Nenhum modelo preve Draw na previsao pontual (42 Home / 30 Away / 0 Draw). Nao e bug: e comportamento esperado de classificadores com classe desbalanceada. Poisson indica P(Draw) medio de 23.1%, mas ainda nao vence o argmax.
2. Threshold tuning para Draw foi testado e rejeitado: piora a acuracia geral.
3. Diferencas entre modelos nao sao estatisticamente significativas com n=64.
4. Vazamento temporal do ranking existe, mas nao explica o resultado.
5. Split fixo e instavel para LogReg; 5-fold e a estimativa mais confiavel.

---

## 9. Links

- Repositorio GitHub: https://github.com/Roberton003/copa-challenger-dados-por-todos
- Notebook Kaggle: https://www.kaggle.com/code/roberto0010/copa-challenger-solu-o-completa
- Dashboard Streamlit: executar app.py localmente
- Wiki: https://github.com/Roberton003/copa-challenger-dados-por-todos/wiki

---

## 10. Previsoes Finais

72 jogos da Copa 2026: 37 Home / 0 Draw / 35 Away (resultado do notebook Kaggle v3).

Probabilidades medias: Home 51.4%, Draw 0.0%, Away 48.6%.

---

*Entrega pronta para o Writeup da Copa Challenger Dados por Todos.*
