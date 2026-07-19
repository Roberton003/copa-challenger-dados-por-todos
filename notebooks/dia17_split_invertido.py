#!/usr/bin/env python3
"""Dia 17 — Variação do split treino/teste.

Todo o pipeline (dias 3-16) usa um único split fixo: treino=2018, teste=2022.
Com só 128 partidas totais, isso é uma amostra de validação; nunca foi
testado se o sinal (rank_diff dominando a importância) é genuíno ou um
artefato desse split específico. Dois experimentos complementares, mesma
pipeline/features do dia10:

  1. Split invertido: treino=2022, teste=2018.
  2. 5-fold estratificado sobre as 128 partidas combinadas (2018+2022) —
     estimativa mais robusta que um único split de 64/64.
"""
import warnings
from pathlib import Path
import json

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
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
y_all = le.fit_transform(matches_comp['result'])
X_all = matches_comp[feature_cols].fillna(0)
rps_order = [list(le.classes_).index(c) for c in ['Home', 'Draw', 'Away']]

def rps(y_true_idx, proba_ordered):
    n = len(y_true_idx)
    total = 0.0
    for i in range(n):
        obs = np.zeros(3)
        obs[y_true_idx[i]] = 1
        total += np.sum((np.cumsum(proba_ordered[i]) - np.cumsum(obs)) ** 2) / 2
    return total / n

base_models = {
    'LogReg': lambda seed: LogisticRegression(max_iter=1000, random_state=seed, class_weight='balanced'),
    'XGBoost': lambda seed: XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                                          random_state=seed, eval_metric='mlogloss'),
    'LightGBM': lambda seed: LGBMClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                            subsample=0.8, colsample_bytree=0.8,
                                            random_state=seed, verbose=-1),
}

def fit_eval(X_train, y_train, X_test, y_test, method):
    results = {}
    for name, make in base_models.items():
        cal = CalibratedClassifierCV(make(42), method=method, cv=3)
        cal.fit(X_train, y_train)
        proba = cal.predict_proba(X_test)
        acc = accuracy_score(y_test, cal.predict(X_test))
        y_test_ord = np.array([rps_order.index(y) for y in y_test])
        score = rps(y_test_ord, proba[:, rps_order])
        results[name] = {'accuracy': float(acc), 'rps': float(score)}
    return results

# ============================================================
# 1. Split original (2018->2022) vs invertido (2022->2018), método sigmoid
#    (sigmoid venceu por RPS no dia10 -> usa o mesmo método pra comparar limpo)
# ============================================================
print('=' * 60)
print('1. SPLIT ORIGINAL vs INVERTIDO (treino/teste trocados)')
print('=' * 60)

mask_2018 = matches_comp['Year'] == 2018
mask_2022 = matches_comp['Year'] == 2022

original = fit_eval(X_all[mask_2018], y_all[mask_2018], X_all[mask_2022], y_all[mask_2022], 'sigmoid')
invertido = fit_eval(X_all[mask_2022], y_all[mask_2022], X_all[mask_2018], y_all[mask_2018], 'sigmoid')

for name in base_models:
    print(f'  {name:<10} original(treino=2018,teste=2022): acc={original[name]["accuracy"]:.1%} rps={original[name]["rps"]:.4f}  '
          f'| invertido(treino=2022,teste=2018): acc={invertido[name]["accuracy"]:.1%} rps={invertido[name]["rps"]:.4f}')

# ============================================================
# 2. 5-fold estratificado sobre as 128 partidas combinadas
# ============================================================
print('\n' + '=' * 60)
print('2. 5-FOLD ESTRATIFICADO (128 partidas combinadas 2018+2022)')
print('=' * 60)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
kfold_results = {name: [] for name in base_models}
for fold_i, (tr_idx, te_idx) in enumerate(skf.split(X_all, y_all)):
    fold_res = fit_eval(X_all.iloc[tr_idx], y_all[tr_idx], X_all.iloc[te_idx], y_all[te_idx], 'sigmoid')
    for name in base_models:
        kfold_results[name].append(fold_res[name]['accuracy'])

kfold_summary = {}
for name, accs in kfold_results.items():
    kfold_summary[name] = {'accuracy_mean': float(np.mean(accs)), 'accuracy_std': float(np.std(accs)),
                            'accuracies_por_fold': [float(a) for a in accs]}
    print(f'  {name:<10} 5-fold: {np.mean(accs):.1%} ± {np.std(accs):.1%}  (folds: {[f"{a:.1%}" for a in accs]})')

# ============================================================
# 3. Salvar
# ============================================================
out = {
    'split_original_treino2018_teste2022': original,
    'split_invertido_treino2022_teste2018': invertido,
    'kfold_5_estratificado': kfold_summary,
    'metodologia': (
        'Split original (treino=2018,teste=2022) vs invertido (treino=2022,teste=2018) e '
        '5-fold estratificado sobre as 128 partidas combinadas — mesmo feature set/calibração '
        '(sigmoid) do dia10, para testar se o sinal de rank_diff/ranking generaliza nas duas '
        'direções ou é artefato do split fixo usado em todo o resto do projeto.'
    ),
}
json.dump(out, open(OUTPUTS / 'dia17_split_invertido.json', 'w'), indent=2, ensure_ascii=False)
print(f"\nResultados salvos: {OUTPUTS / 'dia17_split_invertido.json'}")

def _selftest():
    assert set(original.keys()) == set(base_models.keys())
    assert all(0.0 <= r['accuracy'] <= 1.0 for r in original.values())
    assert all(0.0 <= r['accuracy'] <= 1.0 for r in invertido.values())
    assert all(len(v['accuracies_por_fold']) == 5 for v in kfold_summary.values())
    print('selftest OK')

if __name__ == '__main__':
    _selftest()
