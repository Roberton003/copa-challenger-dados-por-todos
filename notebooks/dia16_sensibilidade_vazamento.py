#!/usr/bin/env python3
"""Dia 16 — Sensibilidade ao vazamento temporal do ranking.

O relatório documenta o vazamento (ranking único de out/2022 usado em treino
2018 e teste 2022) mas até aqui nunca testou o quanto do desempenho depende
dele — ficou só registrado como limitação. Este script fecha essa lacuna:

  1. Mede o gap de correlação treino(2018)/teste(2022) para CADA uma das 5
     features (todas derivadas do mesmo snapshot de ranking) — não só
     rank_diff, a única analisada até então.
  2. Remove as 2 features com maior gap (mais "contaminadas" pelo vazamento)
     e retreina o mesmo pipeline do dia10 (6 combos modelo×calibração).
  3. Compara acurácia/RPS com e sem essas features -> quantifica quanto do
     desempenho reportado depende do sinal potencialmente inflado.

Mesma pipeline/split/seed do dia10 para comparação direta.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
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
matches_comp['result_ord'] = matches_comp['result'].map({'Away': -1, 'Draw': 0, 'Home': 1})

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

# ============================================================
# 1. Gap de correlação treino(2018, ranking "do futuro") vs teste(2022, ~contemporâneo)
#    -> fingerprint do vazamento por feature, não só rank_diff
# ============================================================
print('=' * 60)
print('1. GAP DE CORRELAÇÃO POR FEATURE (treino 2018 vs teste 2022)')
print('=' * 60)
corr_gaps = {}
for f in feature_cols:
    c_train = train_df[[f, 'result_ord']].fillna(0).corr().iloc[0, 1]
    c_test = test_df[[f, 'result_ord']].fillna(0).corr().iloc[0, 1]
    gap = abs(c_train) - abs(c_test)
    corr_gaps[f] = {'corr_2018_treino': float(c_train), 'corr_2022_teste': float(c_test), 'gap_abs': float(gap)}
    print(f'  {f:<16} corr_2018={c_train:+.3f}  corr_2022={c_test:+.3f}  gap={gap:+.3f}')

mais_afetadas = sorted(corr_gaps, key=lambda f: corr_gaps[f]['gap_abs'], reverse=True)[:2]
print(f'\nFeatures mais afetadas pelo vazamento (maior gap): {mais_afetadas}')

reduced_cols = [f for f in feature_cols if f not in mais_afetadas]
print(f'Feature set reduzido (sem as 2 mais afetadas): {reduced_cols}')

# ============================================================
# 2. Retreinar os 6 combos (dia10) com feature set completo vs reduzido
# ============================================================
le = LabelEncoder()
y_train = le.fit_transform(train_df['result'])
y_test = le.transform(test_df['result'])
rps_order = [list(le.classes_).index(c) for c in ['Home', 'Draw', 'Away']]
y_test_ord = np.array([rps_order.index(y) for y in y_test])

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

def run_combos(cols):
    X_train, X_test = train_df[cols].fillna(0), test_df[cols].fillna(0)
    results = {}
    for name, make in base_models.items():
        for method in ['isotonic', 'sigmoid']:
            cal = CalibratedClassifierCV(make(42), method=method, cv=3)
            cal.fit(X_train, y_train)
            proba = cal.predict_proba(X_test)
            acc = accuracy_score(y_test, cal.predict(X_test))
            score = rps(y_test_ord, proba[:, rps_order])
            results[f'{name} ({method})'] = {'accuracy': float(acc), 'rps': float(score)}
    return results

print('\n' + '=' * 60)
print('2. ACURÁCIA/RPS: FEATURE SET COMPLETO vs REDUZIDO (sem features mais vazadas)')
print('=' * 60)
full_results = run_combos(feature_cols)
reduced_results = run_combos(reduced_cols)

deltas = {}
for name in full_results:
    d_acc = reduced_results[name]['accuracy'] - full_results[name]['accuracy']
    d_rps = reduced_results[name]['rps'] - full_results[name]['rps']
    deltas[name] = {'delta_accuracy': float(d_acc), 'delta_rps': float(d_rps)}
    print(f'  {name:<20} completo: acc={full_results[name]["accuracy"]:.1%} rps={full_results[name]["rps"]:.4f}  '
          f'| reduzido: acc={reduced_results[name]["accuracy"]:.1%} rps={reduced_results[name]["rps"]:.4f}  '
          f'| Δacc={d_acc:+.1%}')

media_delta_acc = np.mean([d['delta_accuracy'] for d in deltas.values()])
print(f'\nΔ médio de acurácia (reduzido - completo) nos 6 combos: {media_delta_acc:+.1%}')
print('Δ negativo grande -> desempenho reportado depende fortemente do sinal vazado.')
print('Δ pequeno/positivo -> modelo é robusto, vazamento não é o principal motor do resultado.')

# ============================================================
# 3. Salvar
# ============================================================
out = {
    'corr_gaps_por_feature': corr_gaps,
    'features_mais_afetadas': mais_afetadas,
    'feature_set_completo': feature_cols,
    'feature_set_reduzido': reduced_cols,
    'resultados_completo': full_results,
    'resultados_reduzido': reduced_results,
    'deltas': deltas,
    'delta_medio_acuracia': float(media_delta_acc),
    'metodologia': (
        'Gap de correlação |corr_2018| - |corr_2022| por feature (fingerprint de vazamento, '
        'generaliza a análise anterior que só olhava rank_diff). Remove as 2 features com maior gap e '
        're-roda o pipeline do dia10 (mesmo split/seed/6 combos) para quantificar quanto da '
        'acurácia/RPS reportados depende do sinal potencialmente inflado pelo ranking único.'
    ),
}
json.dump(out, open(OUTPUTS / 'dia16_sensibilidade_vazamento.json', 'w'), indent=2, ensure_ascii=False)
print(f"\nResultados salvos: {OUTPUTS / 'dia16_sensibilidade_vazamento.json'}")

def _selftest():
    assert len(mais_afetadas) == 2 and len(reduced_cols) == 3
    assert all(0.0 <= r['accuracy'] <= 1.0 for r in full_results.values())
    assert all(0.0 <= r['accuracy'] <= 1.0 for r in reduced_results.values())
    print('selftest OK')

if __name__ == '__main__':
    _selftest()
