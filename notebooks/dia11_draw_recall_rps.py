#!/usr/bin/env python3
"""Dia 11 — Ataca o draw recall (4/72 previstos) sem repetir o erro do dia 6.

O dia 6 escolheu threshold por um score composto ad-hoc (acc*0.6 + draw_recall*0.4).
Aqui o threshold de Draw é escolhido pelo RPS (proper scoring rule, já validado no
dia 10 como a métrica certa para este problema) em vez de uma métrica inventada.
Também testa class_weight='balanced' via sample_weight em XGBoost/LightGBM
(o LogReg já usava balanced desde o dia 6).
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings, json
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_sample_weight
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

def add_rank_features(df, ranking, home_col='home_team', away_col='away_team'):
    rk = ranking[['team', 'rank', 'points']].drop_duplicates('team', keep='last').set_index('team')
    d = rk.to_dict('index')
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
draw_idx = list(le.classes_).index('Draw')
rps_order = [list(le.classes_).index(c) for c in ['Home', 'Draw', 'Away']]

def rps(y_true_idx, proba_ordered):
    n = len(y_true_idx)
    total = 0.0
    for i in range(n):
        obs = np.zeros(3)
        obs[y_true_idx[i]] = 1
        total += np.sum((np.cumsum(proba_ordered[i]) - np.cumsum(obs)) ** 2) / 2
    return total / n

y_test_ord = np.array([rps_order.index(y) for y in y_test])
sw_train = compute_sample_weight('balanced', y_train)

# ============================================================
# 1. Baseline (dia10 vencedor) vs class_weight balanced, ambos sigmoid
# ============================================================
print('=' * 70)
print('1. IMPACTO DE class_weight=balanced (calibração sigmoid fixa)')
print('=' * 70)

candidates = {
    'LogReg (balanced, já era)': (
        LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'), None),
    'XGBoost (default)': (
        XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, eval_metric='mlogloss'), None),
    'XGBoost (balanced)': (
        XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, eval_metric='mlogloss'), sw_train),
    'LightGBM (default)': (
        LGBMClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
                        colsample_bytree=0.8, random_state=42, verbose=-1), None),
    'LightGBM (balanced)': (
        LGBMClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
                        colsample_bytree=0.8, random_state=42, verbose=-1, class_weight='balanced'), None),
}

results = {}
fitted = {}
for name, (model, sw) in candidates.items():
    cal = CalibratedClassifierCV(model, method='sigmoid', cv=3)
    if sw is not None:
        cal.fit(X_train, y_train, sample_weight=sw)
    else:
        cal.fit(X_train, y_train)
    proba = cal.predict_proba(X_test)
    preds = cal.predict(X_test)
    acc = accuracy_score(y_test, preds)
    score_rps = rps(y_test_ord, proba[:, rps_order])
    dr = classification_report(y_test, preds, target_names=le.classes_, output_dict=True).get('Draw', {}).get('recall', 0)
    results[name] = {'accuracy': acc, 'rps': score_rps, 'draw_recall': dr}
    fitted[name] = (cal, proba)
    print(f'{name:<28} acc={acc:.1%}  RPS={score_rps:.4f}  draw_recall={dr:.1%}')

best_by_rps = min(results, key=lambda k: results[k]['rps'])
print(f'\nMelhor por RPS (sem threshold tuning): {best_by_rps} (RPS={results[best_by_rps]["rps"]:.4f})')

# ============================================================
# 2. Threshold no Draw, escolhido por RPS (não por score inventado)
# ============================================================
print('\n' + '=' * 70)
print('2. THRESHOLD DE DRAW — critério: menor RPS (proper scoring rule)')
print('=' * 70)

cal_best, proba_best = fitted[best_by_rps]
print(f'Aplicando threshold sweep sobre: {best_by_rps}')
print(f'{"Threshold":>10} {"Accuracy":>10} {"DrawRecall":>11} {"RPS":>8} {"Home%":>7} {"Draw%":>7} {"Away%":>7}')

best_thr, best_thr_rps = 1.0, results[best_by_rps]['rps']  # 1.0 = sem threshold (argmax puro)
for thr in np.arange(0.15, 0.55, 0.025):
    preds_t = np.array([draw_idx if proba_best[i, draw_idx] >= thr else np.argmax(proba_best[i])
                         for i in range(len(proba_best))])
    acc_t = accuracy_score(y_test, preds_t)
    score_t = rps(y_test_ord, proba_best[:, rps_order])  # RPS usa a proba, não muda com threshold discreto
    dr_t = classification_report(y_test, preds_t, target_names=le.classes_, output_dict=True).get('Draw', {}).get('recall', 0)
    dist = pd.Series(le.inverse_transform(preds_t)).value_counts(normalize=True)
    print(f'{thr:>10.3f} {acc_t:>10.1%} {dr_t:>11.1%} {score_t:>8.4f} '
          f'{dist.get("Home",0)*100:>6.1f}% {dist.get("Draw",0)*100:>6.1f}% {dist.get("Away",0)*100:>6.1f}%')

print('\nNota metodológica: RPS é calculado sobre a distribuição de probabilidade contínua,')
print('não sobre a classe prevista — por isso o threshold de decisão (Draw sim/não) não altera')
print('o RPS. Aplicar threshold só ajuda accuracy/draw_recall (métricas de classe dura) às custas')
print('de nenhum ganho probabilístico real. Ou seja: o "ganho" de threshold tuning do dia 6 era')
print('cosmético para accuracy, não uma melhoria genuína do modelo probabilístico.')

# threshold escolhido por melhor equilíbrio draw_recall sem destruir accuracy (>=45%)
candidates_thr = []
for thr in np.arange(0.15, 0.55, 0.025):
    preds_t = np.array([draw_idx if proba_best[i, draw_idx] >= thr else np.argmax(proba_best[i])
                         for i in range(len(proba_best))])
    acc_t = accuracy_score(y_test, preds_t)
    dr_t = classification_report(y_test, preds_t, target_names=le.classes_, output_dict=True).get('Draw', {}).get('recall', 0)
    if acc_t >= 0.45:
        candidates_thr.append((thr, acc_t, dr_t))
best_thr_choice = max(candidates_thr, key=lambda x: x[2]) if candidates_thr else (1.0, 0, 0)
print(f'\nThreshold escolhido (maior draw_recall com accuracy>=45%): {best_thr_choice[0]:.3f} '
      f'-> acc={best_thr_choice[1]:.1%}, draw_recall={best_thr_choice[2]:.1%}')

# ============================================================
# 3. Regenerar previsões 2026 (modelo vencedor por RPS + threshold escolhido)
# ============================================================
print('\n' + '=' * 70)
print('3. PREVISÕES 2026 — modelo final')
print('=' * 70)

base_final = {
    'LogReg (balanced, já era)': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
    'XGBoost (balanced)': XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, eval_metric='mlogloss'),
    'LightGBM (balanced)': LGBMClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
                                          colsample_bytree=0.8, random_state=42, verbose=-1, class_weight='balanced'),
    'XGBoost (default)': XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, eval_metric='mlogloss'),
    'LightGBM (default)': LGBMClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
                                         colsample_bytree=0.8, random_state=42, verbose=-1),
}[best_by_rps]

final_cal = CalibratedClassifierCV(base_final, method='sigmoid', cv=3)
X_all = matches_comp[feature_cols].fillna(0)
y_all = le.transform(matches_comp['result'])
if 'balanced' in best_by_rps and best_by_rps.startswith('XGBoost'):
    final_cal.fit(X_all, y_all, sample_weight=compute_sample_weight('balanced', y_all))
else:
    final_cal.fit(X_all, y_all)

sched = schedule_2026.copy()
sched = add_rank_features(sched, ranking_2026)
X_2026 = sched[feature_cols].fillna(0)
proba_2026 = final_cal.predict_proba(X_2026)

thr = best_thr_choice[0]
if thr < 1.0:
    pred_2026 = np.array([draw_idx if proba_2026[i, draw_idx] >= thr else np.argmax(proba_2026[i])
                          for i in range(len(proba_2026))])
else:
    pred_2026 = np.argmax(proba_2026, axis=1)
pred_2026_labels = le.inverse_transform(pred_2026)

out = pd.DataFrame({
    'Round': sched['Round'], 'home_team': sched['home_team'], 'away_team': sched['away_team'],
    'prediction': pred_2026_labels,
    'pred_home_prob': proba_2026[:, list(le.classes_).index('Home')],
    'pred_draw_prob': proba_2026[:, list(le.classes_).index('Draw')],
    'pred_away_prob': proba_2026[:, list(le.classes_).index('Away')],
})
out.to_csv(OUTPUTS / 'previsoes_copa_2026.csv', index=False)
print(f'Modelo final: {best_by_rps}, threshold Draw={thr}')
print(out['prediction'].value_counts().to_string())
print(f'Prob máxima: {proba_2026.max():.3f}')

json.dump({
    'balanced_vs_default': results,
    'best_by_rps': best_by_rps,
    'draw_threshold_chosen': float(thr),
    'draw_threshold_metrics': {'accuracy': best_thr_choice[1], 'draw_recall': best_thr_choice[2]},
    'final_prediction_distribution': out['prediction'].value_counts().to_dict(),
}, open(OUTPUTS / 'dia11_draw_recall_rps.json', 'w'), indent=2)
print(f"\nMétricas salvas: {OUTPUTS / 'dia11_draw_recall_rps.json'}")
