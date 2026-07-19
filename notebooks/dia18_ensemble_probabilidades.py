#!/usr/bin/env python3
"""Dia 18 — Ensemble por média de probabilidades (Kaggle Book, cap. 9).

Os dias 15-17 mostraram que nenhum dos 3 modelos (LogReg/XGBoost/LightGBM)
vence os outros de forma estatisticamente robusta — McNemar/bootstrap não
acham diferença significativa (dia15), e o "melhor" muda com a direção do
split (LogReg 57.8%->39.1% invertido, dia17). A recomendação padrão nesse
cenário (Kaggle Book, "Averaging models into an ensemble") é não escolher
um: tirar a média das probabilidades calibradas dos três. Testa isso nas
mesmas duas direções de split + 5-fold do dia17, pra comparabilidade direta.
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

base_models = {
    'LogReg': lambda seed: LogisticRegression(max_iter=1000, random_state=seed, class_weight='balanced'),
    'XGBoost': lambda seed: XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                                          random_state=seed, eval_metric='mlogloss'),
    'LightGBM': lambda seed: LGBMClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                            subsample=0.8, colsample_bytree=0.8,
                                            random_state=seed, verbose=-1),
}

def rps(y_true_idx, proba_ordered):
    n = len(y_true_idx)
    total = 0.0
    for i in range(n):
        obs = np.zeros(3)
        obs[y_true_idx[i]] = 1
        total += np.sum((np.cumsum(proba_ordered[i]) - np.cumsum(obs)) ** 2) / 2
    return total / n

def fit_eval_ensemble(X_train, y_train, X_test, y_test):
    """Treina os 3 modelos (sigmoid), tira a média das probabilidades, avalia individual + ensemble."""
    probas = {}
    for name, make in base_models.items():
        cal = CalibratedClassifierCV(make(42), method='sigmoid', cv=3)
        cal.fit(X_train, y_train)
        probas[name] = cal.predict_proba(X_test)

    y_test_ord = np.array([rps_order.index(y) for y in y_test])
    results = {}
    for name, proba in probas.items():
        acc = float((proba.argmax(axis=1) == y_test).mean())
        score = rps(y_test_ord, proba[:, rps_order])
        results[name] = {'accuracy': acc, 'rps': score}

    proba_ens = np.mean(list(probas.values()), axis=0)
    acc_ens = float((proba_ens.argmax(axis=1) == y_test).mean())
    score_ens = rps(y_test_ord, proba_ens[:, rps_order])
    results['Ensemble (média)'] = {'accuracy': acc_ens, 'rps': score_ens}
    return results

# ============================================================
# 1. Split original vs invertido (mesma comparação do dia17, + ensemble)
# ============================================================
print('=' * 60)
print('1. ENSEMBLE vs INDIVIDUAIS — split original e invertido')
print('=' * 60)

mask_2018 = matches_comp['Year'] == 2018
mask_2022 = matches_comp['Year'] == 2022

original = fit_eval_ensemble(X_all[mask_2018], y_all[mask_2018], X_all[mask_2022], y_all[mask_2022])
invertido = fit_eval_ensemble(X_all[mask_2022], y_all[mask_2022], X_all[mask_2018], y_all[mask_2018])

for name in list(base_models) + ['Ensemble (média)']:
    print(f'  {name:<20} original: acc={original[name]["accuracy"]:.1%} rps={original[name]["rps"]:.4f}  '
          f'| invertido: acc={invertido[name]["accuracy"]:.1%} rps={invertido[name]["rps"]:.4f}')

# ============================================================
# 2. 5-fold estratificado
# ============================================================
print('\n' + '=' * 60)
print('2. 5-FOLD ESTRATIFICADO — ENSEMBLE vs INDIVIDUAIS')
print('=' * 60)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
kfold_acc = {name: [] for name in list(base_models) + ['Ensemble (média)']}
for tr_idx, te_idx in skf.split(X_all, y_all):
    fold_res = fit_eval_ensemble(X_all.iloc[tr_idx], y_all[tr_idx], X_all.iloc[te_idx], y_all[te_idx])
    for name in kfold_acc:
        kfold_acc[name].append(fold_res[name]['accuracy'])

kfold_summary = {}
for name, accs in kfold_acc.items():
    kfold_summary[name] = {'accuracy_mean': float(np.mean(accs)), 'accuracy_std': float(np.std(accs))}
    print(f'  {name:<20} 5-fold: {np.mean(accs):.1%} ± {np.std(accs):.1%}')

# ============================================================
# 3. Salvar
# ============================================================
out = {
    'split_original': original,
    'split_invertido': invertido,
    'kfold_5_estratificado': kfold_summary,
    'metodologia': (
        'Média simples das probabilidades calibradas (sigmoid) de LogReg/XGBoost/LightGBM '
        '(Kaggle Book cap.9, "Averaging models into an ensemble"). Avaliado nas mesmas duas '
        'direções de split + 5-fold do dia17, para comparabilidade direta com os modelos '
        'individuais.'
    ),
}
json.dump(out, open(OUTPUTS / 'dia18_ensemble_probabilidades.json', 'w'), indent=2, ensure_ascii=False)
print(f"\nResultados salvos: {OUTPUTS / 'dia18_ensemble_probabilidades.json'}")

def _selftest():
    assert 'Ensemble (média)' in original and 'Ensemble (média)' in invertido
    assert all(0.0 <= r['accuracy'] <= 1.0 for r in original.values())
    assert all(0.0 <= v['accuracy_mean'] <= 1.0 for v in kfold_summary.values())
    print('selftest OK')

if __name__ == '__main__':
    _selftest()
