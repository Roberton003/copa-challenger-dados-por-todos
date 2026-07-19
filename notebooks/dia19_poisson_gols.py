#!/usr/bin/env python3
"""Dia 19 — Regressão de Poisson sobre gols, pra prever empates de verdade.

Limitação documentada no RELATORIO: o pipeline de classificação (dia10) nunca
prevê "Draw" nas 72 partidas de 2026 — zero empates em 72 jogos é um sinal de
que classificar H/D/A direto é ruim pra essa classe minoritária. O dataset
bruto já tem home_score/away_score (mesmo CSV oficial, mesmo escopo). Em vez
de classificar o resultado, modela separadamente quantos gols cada lado faz
(Poisson, mesmas 5 features de ranking do dia10) e deriva P(Home)/P(Draw)/
P(Away) somando a matriz de probabilidade de placares — abordagem clássica
de modelagem de futebol (Maher 1982 / Dixon-Coles simplificado).
"""
import warnings
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / 'data' / 'raw'
OUTPUTS = BASE / 'outputs'
MAX_GOLS = 6  # suficiente: P(gols>6) é desprezível em jogos de seleção

matches = pd.read_csv(RAW / 'matches_1930_2022.csv')
ranking_2022 = pd.read_csv(RAW / 'fifa_ranking_2022-10-06.csv')
ranking_2026 = pd.read_csv(RAW / 'fifa_ranking_2026-06-08.csv')
schedule_2026 = pd.read_csv(RAW / 'schedule_2026.csv')

matches_comp = matches[matches['Year'].isin([2018, 2022])].copy()
matches_comp.columns = matches_comp.columns.str.strip()
matches_comp['result'] = matches_comp.apply(
    lambda r: 'Home' if r['home_score'] > r['away_score']
    else ('Draw' if r['home_score'] == r['away_score'] else 'Away'), axis=1)

NAME_ALIASES = {'United States': 'USA', 'Cape Verde': 'Cabo Verde',
                 'Bosnia-Herzegovina': 'Bosnia and Herzegovina'}

def add_rank_features(df, ranking, home_col='home_team', away_col='away_team'):
    rk = ranking[['team', 'rank', 'points']].drop_duplicates('team', keep='last').set_index('team')
    d = rk.to_dict('index')
    for alias, canonical in NAME_ALIASES.items():
        if canonical in d:
            d[alias] = d[canonical]
    for side, team_col in [('home', home_col), ('away', away_col)]:
        df[f'{side}_rank'] = df[team_col].map(lambda t: d.get(t, {}).get('rank', np.nan))
        df[f'{side}_rank_pts'] = df[team_col].map(lambda t: d.get(t, {}).get('points', np.nan))
    df['rank_diff'] = df['away_rank'] - df['home_rank']
    df['rank_pts_diff'] = df['home_rank_pts'] - df['away_rank_pts']
    df['rank_ratio'] = df['away_rank'] / df['home_rank'].replace(0, 1)
    return df

matches_comp = add_rank_features(matches_comp, ranking_2022)
feature_cols = ['rank_diff', 'home_rank', 'rank_ratio', 'rank_pts_diff', 'away_rank']

train_df = matches_comp[matches_comp['Year'] == 2018]
test_df = matches_comp[matches_comp['Year'] == 2022]
X_train, X_test = train_df[feature_cols].fillna(0), test_df[feature_cols].fillna(0)

def score_matrix(lam_home, lam_away, max_gols=MAX_GOLS):
    """P(placar = i x j) pra i,j em 0..max_gols, assumindo gols de cada lado independentes."""
    ph = poisson.pmf(np.arange(max_gols + 1), lam_home)
    pa = poisson.pmf(np.arange(max_gols + 1), lam_away)
    return np.outer(ph, pa)

def outcome_probs(lam_home, lam_away):
    m = score_matrix(lam_home, lam_away)
    p_home = np.tril(m, -1).sum()   # i > j
    p_draw = np.trace(m)            # i == j
    p_away = np.triu(m, 1).sum()    # i < j
    return p_home, p_draw, p_away

# ============================================================
# 1. Treina Poisson pra gols do mandante e do visitante, separadamente
# ============================================================
print('=' * 60)
print('1. REGRESSÃO DE POISSON: gols mandante / gols visitante')
print('=' * 60)

model_home = PoissonRegressor(alpha=1.0, max_iter=500)
model_home.fit(X_train, train_df['home_score'])

model_away = PoissonRegressor(alpha=1.0, max_iter=500)
model_away.fit(X_train, train_df['away_score'])

lam_home_test = model_home.predict(X_test)
lam_away_test = model_away.predict(X_test)

preds, probs_draw, quase_empates = [], [], 0
for lh, la in zip(lam_home_test, lam_away_test):
    p_h, p_d, p_a = outcome_probs(lh, la)
    probs_draw.append(p_d)
    ordem = np.argsort([p_a, p_d, p_h])[::-1]  # do mais provável ao menos
    preds.append(['Away', 'Draw', 'Home'][ordem[0]])
    if ordem[1] == 1:  # Draw foi o 2º mais provável (quase virou a previsão)
        quase_empates += 1

acc = accuracy_score(test_df['result'], preds)
n_empates_previstos = sum(1 for p in preds if p == 'Draw')
print(f'  Acurácia (teste 2022, Poisson->matriz de placar): {acc:.1%}')
print(f'  Empates previstos (argmax): {n_empates_previstos}/{len(preds)}  '
      f'(P(Draw) médio: {np.mean(probs_draw):.1%}, Draw foi 2º colocado em {quase_empates}/{len(preds)} jogos)')

# comparação: quantos empates reais existem no teste
n_empates_reais = (test_df['result'] == 'Draw').sum()
print(f'  Empates reais no teste 2022: {n_empates_reais}/{len(test_df)}')

# ============================================================
# 2. Aplica no calendário de 2026 (mesmo escopo oficial)
# ============================================================
print('\n' + '=' * 60)
print('2. APLICAÇÃO NO CALENDÁRIO 2026 — quantos empates o Poisson prevê?')
print('=' * 60)

sched = schedule_2026.copy()
sched.columns = sched.columns.str.strip()
sched = add_rank_features(sched, ranking_2026, home_col='home_team', away_col='away_team')
X_2026 = sched[feature_cols].fillna(0)

lam_home_26 = model_home.predict(X_2026)
lam_away_26 = model_away.predict(X_2026)

preds_26, draw_probs_26 = [], []
for lh, la in zip(lam_home_26, lam_away_26):
    p_h, p_d, p_a = outcome_probs(lh, la)
    draw_probs_26.append(p_d)
    preds_26.append(['Away', 'Draw', 'Home'][np.argmax([p_a, p_d, p_h])])

n_empates_26 = sum(1 for p in preds_26 if p == 'Draw')
print(f'  Empates previstos em 2026 (Poisson): {n_empates_26}/{len(preds_26)}')
print(f'  Comparação: previsoes_copa_2026.csv (classificação direta) previa 0 empates/72.')

# ============================================================
# 3. Salvar
# ============================================================
out = {
    'acuracia_teste_2022': float(acc),
    'empates_previstos_teste_2022': int(n_empates_previstos),
    'quase_empates_teste_2022': int(quase_empates),
    'empates_reais_teste_2022': int(n_empates_reais),
    'n_teste_2022': int(len(test_df)),
    'prob_draw_media_teste_2022': float(np.mean(probs_draw)),
    'empates_previstos_2026': int(n_empates_26),
    'n_jogos_2026': int(len(preds_26)),
    'prob_draw_media_2026': float(np.mean(draw_probs_26)),
    'metodologia': (
        'Regressão de Poisson (sklearn PoissonRegressor) separada para gols do mandante e '
        'do visitante, mesmas 5 features de ranking do dia10, mesmo split treino=2018/'
        'teste=2022. P(Home)/P(Draw)/P(Away) derivadas somando a matriz de probabilidade de '
        'placares (gols mandante x gols visitante independentes, Poisson, até 6 gols por lado) '
        '- abordagem clássica de modelagem de futebol (Maher 1982). Ataca a limitação '
        'documentada de zero empates previstos pela classificação direta H/D/A. Achado: '
        'P(Draw) médio sobe pra ~23% (vs. confiança ~0% do classificador), mas a previsão '
        '"dura" (argmax) ainda não escolhe Draw como resultado mais provável em nenhum jogo '
        '- fenômeno conhecido de modelos Poisson independentes subestimarem empates '
        '(correção Dixon-Coles 1997 existe pra isso, fora do escopo aqui).'
    ),
}
json.dump(out, open(OUTPUTS / 'dia19_poisson_gols.json', 'w'), indent=2, ensure_ascii=False)
print(f"\nResultados salvos: {OUTPUTS / 'dia19_poisson_gols.json'}")

def _selftest():
    assert 0.0 <= acc <= 1.0
    assert 0 <= n_empates_previstos <= len(test_df)
    p_h, p_d, p_a = outcome_probs(1.5, 1.2)
    assert abs(p_h + p_d + p_a - 1.0) < 1e-2  # truncamento em MAX_GOLS, não erro
    print('selftest OK')

if __name__ == '__main__':
    _selftest()
