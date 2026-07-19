#!/usr/bin/env python3
"""Dia 15 — Teste estatístico pareado entre modelos.

A comparação de modelos até aqui usa só IC 95% sobreposto "a olho", sem
teste pareado sobre as MESMAS 64 previsões (erros dos modelos são
correlacionados — testados no mesmo conjunto). Fecha essa lacuna com:
  - McNemar exato, pareado, para cada combo vs. o melhor (XGBoost sigmoid).
  - Bootstrap pareado (10000 reamostras) do delta de acurácia, com IC 95%.

Reusa exatamente a pipeline do dia10 (mesmas features/split/6 combos
modelo×calibração) para garantir que os 64 exemplos de teste e sua ordem
sejam idênticos entre os modelos — condição necessária pro pareamento.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import binomtest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / 'data' / 'raw'
OUTPUTS = BASE / 'outputs'

matches = pd.read_csv(RAW / 'matches_1930_2022.csv')
ranking_2022 = pd.read_csv(RAW / 'fifa_ranking_2022-10-06.csv')

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

le = LabelEncoder()
train = matches_comp[matches_comp['Year'] == 2018]
test = matches_comp[matches_comp['Year'] == 2022]
X_train, X_test = train[feature_cols].fillna(0), test[feature_cols].fillna(0)
y_train, y_test = le.fit_transform(train['result']), le.transform(test['result'])

base_models = {
    'LogReg': lambda seed: LogisticRegression(max_iter=1000, random_state=seed, class_weight='balanced'),
    'XGBoost': lambda seed: XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                                          random_state=seed, eval_metric='mlogloss'),
    'LightGBM': lambda seed: LGBMClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                            subsample=0.8, colsample_bytree=0.8,
                                            random_state=seed, verbose=-1),
}

# ============================================================
# 1. Gerar vetor de acerto/erro por exemplo (64 previsões) para cada combo
# ============================================================
correct = {}  # nome -> array bool (len 64), mesma ordem de X_test para todos
for name, make in base_models.items():
    for method in ['isotonic', 'sigmoid']:
        cal = CalibratedClassifierCV(make(42), method=method, cv=3)
        cal.fit(X_train, y_train)
        pred = cal.predict(X_test)
        correct[f'{name} ({method})'] = (pred == y_test)

accs = {k: v.mean() for k, v in correct.items()}
best_name = max(accs, key=accs.get)
print(f'Melhor por acurácia: {best_name} ({accs[best_name]:.1%}) — referência para os pares')

# ============================================================
# 2. McNemar exato (pareado) — melhor vs. cada outro combo
# ============================================================
print('\nMcNemar exato (H0: mesmo erro; testa discordâncias pareadas):')
mcnemar_results = {}
c_best = correct[best_name]
for name, c in correct.items():
    if name == best_name:
        continue
    # b = best acerta e outro erra / c_ = best erra e outro acerta
    b = int(np.sum(c_best & ~c))
    c_ = int(np.sum(~c_best & c))
    n_disc = b + c_
    if n_disc == 0:
        p = 1.0
    else:
        p = binomtest(min(b, c_), n_disc, 0.5, alternative='two-sided').pvalue
    mcnemar_results[name] = {'discordantes': n_disc, 'b_best_ganha': b, 'c_outro_ganha': c_, 'p_value': float(p)}
    print(f'  {best_name} vs {name:<22} discordantes={n_disc:>2}  p={p:.3f}  {"(sig. 5%)" if p < 0.05 else "(não sig.)"}')

# ============================================================
# 3. Bootstrap pareado do delta de acurácia (IC 95%)
# ============================================================
print('\nBootstrap pareado (10000 reamostras) do delta de acurácia vs. melhor modelo:')
rng = np.random.default_rng(42)
n = len(y_test)
bootstrap_results = {}
for name, c in correct.items():
    if name == best_name:
        continue
    deltas = []
    for _ in range(10000):
        idx = rng.integers(0, n, n)
        deltas.append(c_best[idx].mean() - c[idx].mean())
    deltas = np.array(deltas)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    contains_zero = lo <= 0 <= hi
    bootstrap_results[name] = {'delta_mean': float(deltas.mean()), 'ci_95_low': float(lo), 'ci_95_high': float(hi),
                                'ci_contains_zero': bool(contains_zero)}
    print(f'  {best_name} - {name:<22} delta={deltas.mean():+.1%}  IC95%=[{lo:+.1%}, {hi:+.1%}]  '
          f'{"IC cobre 0 -> não significativo" if contains_zero else "IC não cobre 0 -> significativo"}')

# ============================================================
# 4. Salvar
# ============================================================
out = {
    'best_model': best_name,
    'accuracies': {k: float(v) for k, v in accs.items()},
    'n_teste': n,
    'mcnemar': mcnemar_results,
    'bootstrap_delta_acc': bootstrap_results,
    'metodologia': (
        'McNemar exato (binomial nas discordâncias pareadas) e bootstrap pareado '
        '(10000 reamostras) do delta de acurácia, ambos sobre as MESMAS 64 previsões '
        'de teste (2022) por combo modelo×calibração — corrige a limitação de comparar '
        'só ICs univariados sobrepostos (dia10/RELATORIO_FINAL §9.1).'
    ),
}
json.dump(out, open(OUTPUTS / 'dia15_teste_pareado.json', 'w'), indent=2, ensure_ascii=False)
print(f"\nResultados salvos: {OUTPUTS / 'dia15_teste_pareado.json'}")

def _selftest():
    assert all(0.0 <= v <= 1.0 for v in accs.values())
    assert all(0.0 <= r['p_value'] <= 1.0 for r in mcnemar_results.values())
    assert len(correct) == 6
    print('selftest OK')

if __name__ == '__main__':
    _selftest()
