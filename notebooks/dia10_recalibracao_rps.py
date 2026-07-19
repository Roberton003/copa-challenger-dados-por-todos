#!/usr/bin/env python3
"""Dia 10 — Recalibração (isotonic vs sigmoid), RPS e estabilidade multi-seed.

Fecha as recomendações da seção 9 do RELATORIO_FINAL:
  9.1 -> multi-seed: média±desvio da acurácia em 10 seeds
  9.2 -> comparar isotonic vs sigmoid por RPS (proper scoring rule) e
         regenerar outputs/previsoes_copa_2026.csv com o vencedor
Features: top-5 estáticas (da_006). Split temporal 2018->2022 mantido.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings, json
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / 'data' / 'raw'
OUTPUTS = BASE / 'outputs'

matches = pd.read_csv(RAW / 'matches_1930_2022.csv')
schedule_2026 = pd.read_csv(RAW / 'schedule_2026.csv')
ranking_2022 = pd.read_csv(RAW / 'fifa_ranking_2022-10-06.csv')
ranking_2026 = pd.read_csv(RAW / 'fifa_ranking_2026-06-08.csv')

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
    # Nomes de seleção divergem entre matches/schedule e o CSV de ranking
    # (ex.: 'United States' vs 'USA') — sem alias, viravam NaN->0.
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
# Ordem para RPS: Home, Draw, Away (ordinal por "resultado do mandante")
rps_order = [list(le.classes_).index(c) for c in ['Home', 'Draw', 'Away']]

def rps(y_true_idx, proba_ordered):
    """Ranked Probability Score (menor = melhor), classes ordenadas Home<Draw<Away."""
    n = len(y_true_idx)
    total = 0.0
    for i in range(n):
        obs = np.zeros(3)
        obs[y_true_idx[i]] = 1
        cum_p = np.cumsum(proba_ordered[i])
        cum_o = np.cumsum(obs)
        total += np.sum((cum_p - cum_o) ** 2) / 2
    return total / n

y_test_ord = np.array([rps_order.index(y) for y in y_test])

# ============================================================
# 1. Isotonic vs Sigmoid por RPS e acurácia (seed 42, comparável ao dia 6/9)
# ============================================================
print('=' * 60)
print('1. CALIBRAÇÃO: ISOTONIC vs SIGMOID (métrica: RPS + acurácia)')
print('=' * 60)

base_models = {
    'LogReg': lambda seed: LogisticRegression(max_iter=1000, random_state=seed, class_weight='balanced'),
    'XGBoost': lambda seed: XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                                          random_state=seed, eval_metric='mlogloss'),
    'LightGBM': lambda seed: LGBMClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                            subsample=0.8, colsample_bytree=0.8,
                                            random_state=seed, verbose=-1),
}

calib_results = {}
for name, make in base_models.items():
    for method in ['isotonic', 'sigmoid']:
        cal = CalibratedClassifierCV(make(42), method=method, cv=3)
        cal.fit(X_train, y_train)
        proba = cal.predict_proba(X_test)
        acc = accuracy_score(y_test, cal.predict(X_test))
        score = rps(y_test_ord, proba[:, rps_order])
        extreme = float(np.mean((proba > 0.99) | (proba < 0.01)))
        calib_results[f'{name} ({method})'] = {
            'accuracy': acc, 'rps': score, 'pct_probs_extremas': extreme}
        print(f'{name:>9} ({method:>8}): acc={acc:.1%}  RPS={score:.4f}  probs extremas={extreme:.1%}')

best_name = min(calib_results, key=lambda k: calib_results[k]['rps'])
print(f'\nMelhor por RPS: {best_name} (RPS={calib_results[best_name]["rps"]:.4f})')

# ============================================================
# 2. Estabilidade multi-seed (9.1) — 10 seeds, modelo calibrado sigmoid
# ============================================================
print('\n' + '=' * 60)
print('2. ESTABILIDADE MULTI-SEED (10 seeds)')
print('=' * 60)

seeds = list(range(10))
seed_stats = {}
for name, make in base_models.items():
    accs = []
    for s in seeds:
        cal = CalibratedClassifierCV(make(s), method='sigmoid', cv=3)
        cal.fit(X_train, y_train)
        accs.append(accuracy_score(y_test, cal.predict(X_test)))
    seed_stats[name] = {'mean': float(np.mean(accs)), 'std': float(np.std(accs)),
                        'min': float(np.min(accs)), 'max': float(np.max(accs))}
    print(f'{name:>9}: {np.mean(accs):.1%} ± {np.std(accs):.1%}  (min {np.min(accs):.1%}, max {np.max(accs):.1%})')

# ============================================================
# 3. Regenerar previsões 2026 com o melhor calibrador por RPS
# ============================================================
print('\n' + '=' * 60)
print('3. PREVISÕES 2026 (recalibradas)')
print('=' * 60)

best_base, best_method = best_name.split(' (')
best_method = best_method.rstrip(')')
final_model = CalibratedClassifierCV(base_models[best_base](42), method=best_method, cv=3)
# Treino final: 2018+2022 completos (previsão real de futuro, não avaliação)
X_all = matches_comp[feature_cols].fillna(0)
y_all = le.transform(matches_comp['result'])
final_model.fit(X_all, y_all)

sched = schedule_2026.copy()
sched = add_rank_features(sched, ranking_2026)
X_2026 = sched[feature_cols].fillna(0)
proba_2026 = final_model.predict_proba(X_2026)
pred_2026 = le.inverse_transform(np.argmax(proba_2026, axis=1))

out = pd.DataFrame({
    'Round': sched['Round'], 'home_team': sched['home_team'], 'away_team': sched['away_team'],
    'prediction': pred_2026,
    'pred_home_prob': proba_2026[:, list(le.classes_).index('Home')],
    'pred_draw_prob': proba_2026[:, list(le.classes_).index('Draw')],
    'pred_away_prob': proba_2026[:, list(le.classes_).index('Away')],
})
out.to_csv(OUTPUTS / 'previsoes_copa_2026.csv', index=False)
print(f'72 previsões salvas — modelo: {best_name}, treino 2018+2022 completo')
print(f'Prob máxima em qualquer previsão: {proba_2026.max():.3f} (antes: 0.99997)')
print(out['prediction'].value_counts().to_string())

json.dump({
    'calibration_comparison': calib_results,
    'best_by_rps': best_name,
    'multi_seed_sigmoid': seed_stats,
    'features': feature_cols,
    'max_prob_2026': float(proba_2026.max()),
}, open(OUTPUTS / 'dia10_recalibracao_rps.json', 'w'), indent=2)
print(f"\nMétricas salvas: {OUTPUTS / 'dia10_recalibracao_rps.json'}")
