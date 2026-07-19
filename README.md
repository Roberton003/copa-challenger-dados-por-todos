# Copa Challenger — Dados por Todos

Predição de resultados (Home/Draw/Away) para os jogos da Copa do Mundo 2026, treinado com dados históricos de 2018 e 2022 e ranking FIFA. Projeto desenvolvido para a competição **Copa Challenger — Comunidade Dados por Todos**.

## Escopo de dados

Somente os arquivos oficiais da competição, sem dados externos:

- `data/raw/matches_1930_2022.csv` — partidas históricas (usa-se apenas 2018 e 2022)
- `data/raw/fifa_ranking_2022-10-06.csv` — ranking FIFA de referência para treino/teste
- `data/raw/schedule_2026.csv` — calendário da Copa 2026
- `data/raw/fifa_ranking_2026-06-08.csv` — ranking FIFA para as previsões de 2026

## Estrutura

```
notebooks/    scripts numerados por etapa (exploração → modelagem → validação estatística)
outputs/      métricas, previsões e visualizações geradas pelos scripts
app.py        dashboard Streamlit
```

Os scripts em `notebooks/` são incrementais: cada um parte do resultado do anterior e ataca uma limitação específica (excesso de features, calibração de probabilidade, vazamento temporal, robustez do split, indecisão entre modelos, previsão de empates). `RELATORIO_SOLUCAO.md` traz a leitura consolidada.

## Como rodar

```bash
pip install -r requirements.txt
python3 notebooks/dia10_recalibracao_rps.py   # pipeline principal (modelo final)
streamlit run app.py                           # dashboard, porta 8501
```

Todos os scripts resolvem os caminhos de dados/outputs relativos à raiz do repositório — rodam de qualquer máquina sem edição.

## Modelo final

Regressão logística com calibração sigmoid sobre 5 features de ranking FIFA (`rank_diff`, `home_rank`, `rank_ratio`, `rank_pts_diff`, `away_rank`), escolhida por RPS (Ranked Probability Score) entre 6 combinações de modelo × método de calibração testadas. Detalhes de metodologia, validação e limitações em `RELATORIO_SOLUCAO.md`.
