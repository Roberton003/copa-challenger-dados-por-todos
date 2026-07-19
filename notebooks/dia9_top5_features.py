#!/usr/bin/env python3
"""Dia 9 — Reteste com top-5 features (da_006: <1000 linhas -> 3-5 features).

Mesma pipeline de notebooks/dia6_corrigido.py (baseline -> RandomizedSearchCV ->
calibração isotônica -> threshold tuning), trocando as 19 features estáticas
pelas 5 de maior importância reportadas em RELATORIO_FINAL_COPA_CHALLENGER.md §8:
rank_diff, home_rank, rank_ratio, rank_pts_diff, away_rank.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings, json
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.calibration import CalibratedClassifierCV
from scipy.stats import uniform, randint
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / 'data' / 'raw'
OUTPUTS = BASE / 'outputs'

print('=' * 60)
print('=' * 60)

matches = pd.read_csv(RAW / 'matches_1930_2022.csv')
ranking_2022 = pd.read_csv(RAW / 'fifa_ranking_2022-10-06.csv')

matches_comp = matches[matches['Year'].isin([2018, 2022])].copy()
matches_comp.columns = matches_comp.columns.str.strip()

def derive_result(row):
    if row['home_score'] > row['away_score']:
        return 'Home'
    elif row['home_score'] == row['away_score']:
        return 'Draw'
    else:
        return 'Away'

matches_comp['result'] = matches_comp.apply(derive_result, axis=1)

team_ranking_2022 = ranking_2022[['team', 'rank', 'points']].drop_duplicates('team', keep='last').copy()
ranking_dict = team_ranking_2022.set_index('team')[['rank', 'points']].to_dict('index')

# Nomes de seleção divergem entre matches_1930_2022.csv e o CSV de ranking
# (ex.: 'United States' vs 'USA') — sem alias, essas linhas viravam NaN->0.
NAME_ALIASES = {'United States': 'USA', 'Cape Verde': 'Cabo Verde',
                 'Bosnia-Herzegovina': 'Bosnia and Herzegovina'}
for alias, canonical in NAME_ALIASES.items():
    if canonical in ranking_dict:
        ranking_dict[alias] = ranking_dict[canonical]

for col_prefix, rank_col, pts_col in [('home', 'home_rank', 'home_rank_pts'), ('away', 'away_rank', 'away_rank_pts')]:
    rank_col_vals, pts_col_vals = [], []
    team_col = f'{col_prefix}_team'
    for _, row in matches_comp.iterrows():
        team = row[team_col]
        if team in ranking_dict:
            rank_col_vals.append(ranking_dict[team]['rank'])
            pts_col_vals.append(ranking_dict[team]['points'])
        else:
            rank_col_vals.append(np.nan)
            pts_col_vals.append(np.nan)
    matches_comp[rank_col] = rank_col_vals
    matches_comp[pts_col] = pts_col_vals

matches_comp['rank_diff'] = matches_comp['away_rank'] - matches_comp['home_rank']
matches_comp['rank_pts_diff'] = matches_comp['home_rank_pts'] - matches_comp['away_rank_pts']
matches_comp['rank_ratio'] = matches_comp['away_rank'] / matches_comp['home_rank'].replace(0, 1)

# Top-5 por importância reportada (RELATORIO_FINAL_COPA_CHALLENGER.md secao 8)
feature_cols = ['rank_diff', 'home_rank', 'rank_ratio', 'rank_pts_diff', 'away_rank']

le = LabelEncoder()
train = matches_comp[matches_comp['Year'] == 2018].copy()
test = matches_comp[matches_comp['Year'] == 2022].copy()

X_train = train[feature_cols].fillna(0)
X_test = test[feature_cols].fillna(0)
y_train = le.fit_transform(train['result'])
y_test = le.transform(test['result'])

print(f'Treino: {len(X_train)}, Teste: {len(X_test)}, Features: {len(feature_cols)} -> {feature_cols}')

# 1. Baselines
baseline_models = {
    'LogReg': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
    'XGBoost': XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                              random_state=42, use_label_encoder=False, eval_metric='mlogloss'),
    'LightGBM': LGBMClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1),
}

baseline_results = {}
for name, model in baseline_models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    draw_recall = report.get('Draw', {}).get('recall', 0)
    baseline_results[name] = {'accuracy': acc, 'draw_recall': draw_recall}
    print(f'{name}: {acc:.1%} accuracy, {draw_recall:.1%} draw recall')

# 2. RandomizedSearchCV (mesmo orçamento do dia6_corrigido.py)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

xgb_params = {
    'n_estimators': randint(50, 300), 'max_depth': randint(2, 8),
    'learning_rate': uniform(0.01, 0.3), 'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4), 'min_child_weight': randint(1, 10),
    'gamma': uniform(0, 0.5), 'reg_alpha': uniform(0, 1), 'reg_lambda': uniform(0, 2),
}
xgb_search = RandomizedSearchCV(
    XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'),
    xgb_params, n_iter=30, cv=cv, scoring='accuracy', random_state=42, n_jobs=-1)
xgb_search.fit(X_train, y_train)
y_pred_xgb = xgb_search.predict(X_test)
xgb_acc = accuracy_score(y_test, y_pred_xgb)
xgb_draw = classification_report(y_test, y_pred_xgb, target_names=le.classes_, output_dict=True).get('Draw', {}).get('recall', 0)
print(f'XGBoost otimizado: {xgb_acc:.1%} accuracy, {xgb_draw:.1%} draw recall (CV={xgb_search.best_score_:.1%})')

lgbm_params = {
    'n_estimators': randint(50, 400), 'max_depth': randint(2, 10),
    'learning_rate': uniform(0.01, 0.3), 'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4), 'min_child_samples': randint(5, 30),
    'reg_alpha': uniform(0, 1), 'reg_lambda': uniform(0, 2), 'num_leaves': randint(10, 60),
}
lgbm_search = RandomizedSearchCV(
    LGBMClassifier(random_state=42, verbose=-1),
    lgbm_params, n_iter=30, cv=cv, scoring='accuracy', random_state=42, n_jobs=-1)
lgbm_search.fit(X_train, y_train)
y_pred_lgbm = lgbm_search.predict(X_test)
lgbm_acc = accuracy_score(y_test, y_pred_lgbm)
lgbm_draw = classification_report(y_test, y_pred_lgbm, target_names=le.classes_, output_dict=True).get('Draw', {}).get('recall', 0)
print(f'LightGBM otimizado: {lgbm_acc:.1%} accuracy, {lgbm_draw:.1%} draw recall (CV={lgbm_search.best_score_:.1%})')

lr_params = {'C': uniform(0.01, 10), 'penalty': ['l1', 'l2'], 'solver': ['liblinear', 'saga']}
lr_search = RandomizedSearchCV(
    LogisticRegression(max_iter=5000, random_state=42, class_weight='balanced'),
    lr_params, n_iter=20, cv=cv, scoring='accuracy', random_state=42, n_jobs=-1)
lr_search.fit(X_train, y_train)
y_pred_lr = lr_search.predict(X_test)
lr_acc = accuracy_score(y_test, y_pred_lr)
lr_draw = classification_report(y_test, y_pred_lr, target_names=le.classes_, output_dict=True).get('Draw', {}).get('recall', 0)
print(f'LogReg otimizado: {lr_acc:.1%} accuracy, {lr_draw:.1%} draw recall (CV={lr_search.best_score_:.1%})')

# 3. Calibração isotônica
cal_models = {}
for name, base_model in [('XGB', xgb_search.best_estimator_), ('LGBM', lgbm_search.best_estimator_), ('LR', lr_search.best_estimator_)]:
    cal = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
    cal.fit(X_train, y_train)
    y_pred_cal = cal.predict(X_test)
    acc_cal = accuracy_score(y_test, y_pred_cal)
    draw_cal = classification_report(y_test, y_pred_cal, target_names=le.classes_, output_dict=True).get('Draw', {}).get('recall', 0)
    cal_models[name] = {'accuracy': acc_cal, 'draw_recall': draw_cal}
    print(f'{name} calibrado: {acc_cal:.1%} accuracy, {draw_cal:.1%} draw recall')

# 4. Comparação final
all_results = {
    'LogReg (baseline)': baseline_results['LogReg'],
    'XGBoost (baseline)': baseline_results['XGBoost'],
    'LightGBM (baseline)': baseline_results['LightGBM'],
    'XGBoost (otimizado)': {'accuracy': xgb_acc, 'draw_recall': xgb_draw},
    'LightGBM (otimizado)': {'accuracy': lgbm_acc, 'draw_recall': lgbm_draw},
    'LogReg (otimizado)': {'accuracy': lr_acc, 'draw_recall': lr_draw},
    'XGBoost (calibrado)': cal_models['XGB'],
    'LightGBM (calibrado)': cal_models['LGBM'],
    'LogReg (calibrado)': cal_models['LR'],
}

print('\n' + '=' * 60)
print('COMPARAÇÃO FINAL — TOP-5 FEATURES')
print('=' * 60)
print(f'{"Modelo":<25} {"Acurácia":>10} {"Draw Recall":>12}')
for name, r in all_results.items():
    print(f'{name:<25} {r["accuracy"]:>10.1%} {r["draw_recall"]:>12.1%}')

best_overall = max(all_results, key=lambda k: all_results[k]['accuracy'])
print(f'\nMelhor acurácia: {best_overall} ({all_results[best_overall]["accuracy"]:.1%})')

output = {
    'feature_cols': feature_cols,
    'n_features': len(feature_cols),
    'all_results': {k: {'accuracy': float(v['accuracy']), 'draw_recall': float(v['draw_recall'])} for k, v in all_results.items()},
    'best_accuracy': {'name': best_overall, 'accuracy': float(all_results[best_overall]['accuracy'])},
}
with open(OUTPUTS / 'dia9_top5_features_results.json', 'w') as f:
    json.dump(output, f, indent=2)
print(f'\nResultados salvos: {OUTPUTS / "dia9_top5_features_results.json"}')
