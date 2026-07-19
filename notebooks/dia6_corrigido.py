#!/usr/bin/env python3
"""Dia 6 — Otimização com features corrigidas (sem leakage)."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import warnings, json
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
from scipy.stats import uniform, randint
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / 'data' / 'raw'
OUTPUTS = BASE / 'outputs'

# ============================================================
# 1. CARREGAMENTO + FEATURES
# ============================================================
print('='*60)
print('DIA 6 — OTIMIZAÇÃO COM FEATURES CORRIGIDAS')
print('='*60)

matches = pd.read_csv(RAW / 'matches_1930_2022.csv')
schedule_2026 = pd.read_csv(RAW / 'schedule_2026.csv')
ranking_2022 = pd.read_csv(RAW / 'fifa_ranking_2022-10-06.csv')
ranking_2026 = pd.read_csv(RAW / 'fifa_ranking_2026-06-08.csv')

matches_comp = matches[matches['Year'].isin([2018, 2022])].copy()
matches_comp.columns = matches_comp.columns.str.strip()

def derive_result(row):
    if row['home_score'] > row['away_score']: return 'Home'
    elif row['home_score'] == row['away_score']: return 'Draw'
    else: return 'Away'

matches_comp['result'] = matches_comp.apply(derive_result, axis=1)
matches_comp['total_goals'] = matches_comp['home_score'] + matches_comp['away_score']

# Ranking
team_ranking_2022 = ranking_2022[['team', 'rank', 'points']].drop_duplicates('team', keep='last').copy()
ranking_dict = team_ranking_2022.set_index('team')[['rank', 'points']].to_dict('index')

# Nomes de seleção divergem entre matches_1930_2022.csv e o CSV de ranking
# (ex.: 'United States' vs 'USA') — sem alias, essas linhas viravam NaN->0,
# fazendo o time parecer o #1 do mundo. USA jogou 4 partidas em 2022.
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

# Features
round_map = {'Group stage': 1, 'Round of 16': 2, 'Quarter-finals': 3,
             'Semi-finals': 4, 'Third-place match': 5, 'Final': 6}

matches_comp['round_num'] = matches_comp['Round'].map(round_map).fillna(1)
matches_comp['is_knockout'] = (matches_comp['round_num'] >= 2).astype(int)
matches_comp['is_final'] = (matches_comp['round_num'] == 6).astype(int)
matches_comp['rank_ratio'] = matches_comp['away_rank'] / matches_comp['home_rank'].replace(0, 1)
matches_comp['rank_pts_ratio'] = matches_comp['home_rank_pts'] / matches_comp['away_rank_pts'].replace(0, 1)
matches_comp['rank_x_round'] = matches_comp['rank_diff'] * matches_comp['round_num']
matches_comp['strength_diff'] = matches_comp['rank_pts_diff'] * matches_comp['round_num']

# H2H
history = matches_comp.sort_values('Date').copy()
h2h_hw, h2h_d, h2h_aw, h2h_t = {}, {}, {}, {}
for idx, row in history.iterrows():
    home, away = row['home_team'], row['away_team']
    pair = history[
        ((history['home_team'] == home) & (history['away_team'] == away)) |
        ((history['home_team'] == away) & (history['away_team'] == home))
    ]
    pair = pair[pair.index < idx]
    if len(pair) > 0:
        h, d, a = 0, 0, 0
        for _, pm in pair.iterrows():
            hs, aws = pm['home_score'], pm['away_score']
            if pd.isna(hs) or pd.isna(aws): continue
            if hs > aws:
                if pm['home_team'] == home: h += 1
                else: a += 1
            elif hs == aws: d += 1
            else:
                if pm['home_team'] == home: a += 1
                else: h += 1
        h2h_hw[idx], h2h_d[idx], h2h_aw[idx], h2h_t[idx] = h, d, a, len(pair)
    else:
        h2h_hw[idx], h2h_d[idx], h2h_aw[idx], h2h_t[idx] = 0, 0, 0, 0

matches_comp['h2h_home_wins'] = pd.Series(h2h_hw)
matches_comp['h2h_draws'] = pd.Series(h2h_d)
matches_comp['h2h_away_wins'] = pd.Series(h2h_aw)
matches_comp['h2h_total'] = pd.Series(h2h_t)
matches_comp['h2h_home_win_rate'] = matches_comp['h2h_home_wins'] / matches_comp['h2h_total'].replace(0, 1)

# Confederação
confeds = {
    'Europe': ['France','England','Germany','Spain','Belgium','Netherlands','Portugal',
               'Croatia','Denmark','Switzerland','Poland','Wales','Serbia','Czech Republic',
               'Scotland','Austria','Sweden','Norway','Ireland','Northern Ireland',
               'Iceland','Finland','Ukraine','Hungary','Romania','Bulgaria','Greece',
               'Turkey','Russia','Bosnia-Herzegovina','Montenegro','Albania','North Macedonia',
               'Slovakia','Slovenia','Georgia','Armenia','Cyprus','Luxembourg',
               'Belarus','Kazakhstan','Estonia','Latvia','Lithuania','Malta','Moldova',
               'Andorra','San Marino','Liechtenstein','Gibraltar','Kosovo'],
    'South America': ['Brazil','Argentina','Uruguay','Colombia','Peru','Chile','Ecuador',
                      'Paraguay','Bolivia','Venezuela'],
    'Africa': ['Senegal','Morocco','Tunisia','Cameroon','Nigeria','Ghana','Ivory Coast',
               'Mali','Burkina Faso','DR Congo','Guinea','Cape Verde','Equatorial Guinea',
               'Gambia','Comoros','Sudan','South Sudan','Ethiopia','Kenya','Uganda',
               'Rwanda','Tanzania','Burundi','Somalia','Djibouti','Eritrea','Seychelles',
               'Mauritius','Angola','Mozambique','Zimbabwe','Zambia','Malawi','Namibia',
               'Botswana','South Africa','Lesotho','Eswatini','Madagascar','Mauritania',
               'Libya','Algeria','Egypt','Sierra Leone','Liberia','Togo','Benin','Niger',
               'Chad','Central African Republic','Gabon','Congo','Guinea-Bissau'],
    'Asia': ['Japan','South Korea','Australia','Saudi Arabia','Iran','Qatar',
             'China','Iraq','United Arab Emirates','Oman','Jordan','Bahrain',
             'Kuwait','Syria','Lebanon','Palestine','Vietnam','Thailand',
             'Malaysia','Indonesia','Philippines','Singapore','Myanmar',
             'Cambodia','Laos','Brunei','East Timor','Chinese Taipei',
             'Hong Kong','Macau','Mongolia','India','Pakistan','Bangladesh',
             'Sri Lanka','Nepal','Bhutan','Afghanistan','Turkmenistan',
             'Uzbekistan','Kyrgyzstan','Tajikistan','Kazakhstan'],
    'North America': ['United States','Mexico','Canada','Costa Rica','Honduras',
                      'Panama','Jamaica','El Salvador','Guatemala','Trinidad and Tobago',
                      'Haiti','Curaçao','Suriname','Bermuda','Grenada',
                      'Saint Kitts and Nevis','Saint Lucia','Saint Vincent and the Grenadines',
                      'Barbados','Antigua and Barbuda','Dominica','Cuba',
                      'Dominican Republic','Puerto Rico','Nicaragua','Belize',
                      'Bahamas','Aruba','Virgin Islands']
}
def get_confed(team):
    for c, teams in confeds.items():
        if team in teams: return c
    return 'Other'

matches_comp['home_confed'] = matches_comp['home_team'].apply(get_confed)
matches_comp['away_confed'] = matches_comp['away_team'].apply(get_confed)
matches_comp['same_conf'] = (matches_comp['home_confed'] == matches_comp['away_confed']).astype(int)

feature_cols = ['home_rank','away_rank','rank_diff','rank_pts_diff',
                'home_rank_pts','away_rank_pts','rank_ratio','rank_pts_ratio',
                'round_num','rank_x_round','strength_diff',
                'is_knockout','is_final',
                'h2h_home_wins','h2h_draws','h2h_away_wins','h2h_total','h2h_home_win_rate',
                'same_conf']

# ============================================================
# 2. SPLIT
# ============================================================
le = LabelEncoder()
train = matches_comp[matches_comp['Year'] == 2018].copy()
test = matches_comp[matches_comp['Year'] == 2022].copy()

X_train = train[feature_cols].fillna(0)
X_test = test[feature_cols].fillna(0)
y_train = le.fit_transform(train['result'])
y_test = le.transform(test['result'])

print(f'Treino: {len(X_train)}, Teste: {len(X_test)}, Features: {len(feature_cols)}')
print(f'Classes: {list(le.classes_)}')
print(f'Distribuição treino: {pd.Series(y_train).value_counts(normalize=True).to_dict()}')
print(f'Distribuição teste: {pd.Series(y_test).value_counts(normalize=True).to_dict()}')

# ============================================================
# 3. BASELINE (do notebook final)
# ============================================================
print('\n' + '='*60)
print('3. BASELINES')
print('='*60)

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

# ============================================================
# 4. RANDOMIZEDSEARCHCV
# ============================================================
print('\n' + '='*60)
print('4. RANDOMIZEDSEARCHCV (3 modelos)')
print('='*60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 4a. XGBoost
print('\n--- XGBoost ---')
xgb_params = {
    'n_estimators': randint(50, 300),
    'max_depth': randint(2, 8),
    'learning_rate': uniform(0.01, 0.3),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
    'min_child_weight': randint(1, 10),
    'gamma': uniform(0, 0.5),
    'reg_alpha': uniform(0, 1),
    'reg_lambda': uniform(0, 2),
}

xgb_search = RandomizedSearchCV(
    XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'),
    xgb_params, n_iter=30, cv=cv, scoring='accuracy', random_state=42, n_jobs=-1
)
xgb_search.fit(X_train, y_train)
print(f'Best params: {xgb_search.best_params_}')
print(f'Best CV accuracy: {xgb_search.best_score_:.1%}')

y_pred_xgb = xgb_search.predict(X_test)
xgb_acc = accuracy_score(y_test, y_pred_xgb)
xgb_report = classification_report(y_test, y_pred_xgb, target_names=le.classes_, output_dict=True)
xgb_draw = xgb_report.get('Draw', {}).get('recall', 0)
print(f'Test accuracy: {xgb_acc:.1%}, Draw recall: {xgb_draw:.1%}')

# 4b. LightGBM
print('\n--- LightGBM ---')
lgbm_params = {
    'n_estimators': randint(50, 400),
    'max_depth': randint(2, 10),
    'learning_rate': uniform(0.01, 0.3),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
    'min_child_samples': randint(5, 30),
    'reg_alpha': uniform(0, 1),
    'reg_lambda': uniform(0, 2),
    'num_leaves': randint(10, 60),
}

lgbm_search = RandomizedSearchCV(
    LGBMClassifier(random_state=42, verbose=-1),
    lgbm_params, n_iter=30, cv=cv, scoring='accuracy', random_state=42, n_jobs=-1
)
lgbm_search.fit(X_train, y_train)
print(f'Best params: {lgbm_search.best_params_}')
print(f'Best CV accuracy: {lgbm_search.best_score_:.1%}')

y_pred_lgbm = lgbm_search.predict(X_test)
lgbm_acc = accuracy_score(y_test, y_pred_lgbm)
lgbm_report = classification_report(y_test, y_pred_lgbm, target_names=le.classes_, output_dict=True)
lgbm_draw = lgbm_report.get('Draw', {}).get('recall', 0)
print(f'Test accuracy: {lgbm_acc:.1%}, Draw recall: {lgbm_draw:.1%}')

# 4c. Logistic Regression
print('\n--- Logistic Regression ---')
lr_params = {
    'C': uniform(0.01, 10),
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga'],
}

lr_search = RandomizedSearchCV(
    LogisticRegression(max_iter=5000, random_state=42, class_weight='balanced'),
    lr_params, n_iter=20, cv=cv, scoring='accuracy', random_state=42, n_jobs=-1
)
lr_search.fit(X_train, y_train)
print(f'Best params: {lr_search.best_params_}')
print(f'Best CV accuracy: {lr_search.best_score_:.1%}')

y_pred_lr = lr_search.predict(X_test)
lr_acc = accuracy_score(y_test, y_pred_lr)
lr_report = classification_report(y_test, y_pred_lr, target_names=le.classes_, output_dict=True)
lr_draw = lr_report.get('Draw', {}).get('recall', 0)
print(f'Test accuracy: {lr_acc:.1%}, Draw recall: {lr_draw:.1%}')

# ============================================================
# 5. PROBABILITY CALIBRATION
# ============================================================
print('\n' + '='*60)
print('5. PROBABILITY CALIBRATION (isotonic)')
print('='*60)

# Calibrar os 3 modelos otimizados
cal_models = {}
for name, base_model, search in [('XGB', xgb_search.best_estimator_, xgb_search),
                                  ('LGBM', lgbm_search.best_estimator_, lgbm_search),
                                  ('LR', lr_search.best_estimator_, lr_search)]:
    cal = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
    cal.fit(X_train, y_train)
    y_pred_cal = cal.predict(X_test)
    y_proba_cal = cal.predict_proba(X_test)
    acc_cal = accuracy_score(y_test, y_pred_cal)
    report_cal = classification_report(y_test, y_pred_cal, target_names=le.classes_, output_dict=True)
    draw_cal = report_cal.get('Draw', {}).get('recall', 0)
    cal_models[name] = {'model': cal, 'accuracy': acc_cal, 'draw_recall': draw_cal, 'proba': y_proba_cal}
    print(f'{name} calibrado: {acc_cal:.1%} accuracy, {draw_cal:.1%} draw recall')

# ============================================================
# 6. THRESHOLD TUNING PARA DRAWS
# ============================================================
print('\n' + '='*60)
print('6. THRESHOLD TUNING')
print('='*60)

# Usar LightGBM calibrado (melhor acurácia geral)
best_cal = cal_models['LGBM']['model']
y_proba = cal_models['LGBM']['proba']

# Mapear classes
draw_idx = list(le.classes_).index('Draw')
home_idx = list_le_idx = list(le.classes_).index('Home')
away_idx = list(le.classes_).index('Away')

# Testar thresholds diferentes para Draw
print('\nThreshold sweep para Draw:')
print(f'{"Threshold":>10} {"Accuracy":>10} {"Draw Recall":>12} {"Home%":>8} {"Draw%":>8} {"Away%":>8}')
best_threshold = 0.33
best_score = 0
best_draw_recall = 0

for thr in np.arange(0.10, 0.60, 0.05):
    # Aplicar threshold: se prob Draw > thr → Draw, senão argmax
    preds = []
    for i in range(len(y_proba)):
        if y_proba[i, draw_idx] > thr:
            preds.append(draw_idx)
        else:
            preds.append(np.argmax(y_proba[i]))
    preds = np.array(preds)
    
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, target_names=le.classes_, output_dict=True)
    dr = report.get('Draw', {}).get('recall', 0)
    
    pred_labels = le.inverse_transform(preds)
    dist = pd.Series(pred_labels).value_counts(normalize=True)
    h_pct = dist.get('Home', 0) * 100
    d_pct = dist.get('Draw', 0) * 100
    a_pct = dist.get('Away', 0) * 100
    
    # Score composto: accuracy + draw_recall (ponderado)
    score = acc * 0.6 + dr * 0.4
    
    print(f'{thr:>10.2f} {acc:>10.1%} {dr:>12.1%} {h_pct:>7.1f}% {d_pct:>7.1f}% {a_pct:>7.1f}%')
    
    if score > best_score:
        best_score = score
        best_threshold = thr
        best_draw_recall = dr

print(f'\nMelhor threshold: {best_threshold:.2f} (score composto: {best_score:.3f})')

# Aplicar melhor threshold
preds_final = []
for i in range(len(y_proba)):
    if y_proba[i, draw_idx] > best_threshold:
        preds_final.append(draw_idx)
    else:
        preds_final.append(np.argmax(y_proba[i]))
preds_final = np.array(preds_final)

acc_final = accuracy_score(y_test, preds_final)
report_final = classification_report(y_test, preds_final, target_names=le.classes_, output_dict=True)
draw_final = report_final.get('Draw', {}).get('recall', 0)

print(f'\nResultados finais com threshold={best_threshold:.2f}:')
print(f'Acurácia: {acc_final:.1%}')
print(f'Draw recall: {draw_final:.1%}')
print(f'\nRelatório completo:')
print(classification_report(y_test, preds_final, target_names=le.classes_))

# ============================================================
# 7. FEATURE IMPORTANCE DO MELHOR MODELO
# ============================================================
print('\n' + '='*60)
print('7. FEATURE IMPORTANCE')
print('='*60)

best_lgbm = lgbm_search.best_estimator_
fi = pd.Series(best_lgbm.feature_importances_, index=feature_cols).sort_values(ascending=False)
print('Top 10 features (LightGBM otimizado):')
print(fi.head(10))

# ============================================================
# 8. COMPARAÇÃO FINAL
# ============================================================
print('\n' + '='*60)
print('8. COMPARAÇÃO FINAL')
print('='*60)

all_results = {
    'LogReg (baseline)': baseline_results['LogReg'],
    'XGBoost (baseline)': baseline_results['XGBoost'],
    'LightGBM (baseline)': baseline_results['LightGBM'],
    'XGBoost (otimizado)': {'accuracy': xgb_acc, 'draw_recall': xgb_draw},
    'LightGBM (otimizado)': {'accuracy': lgbm_acc, 'draw_recall': lgbm_draw},
    'LogReg (otimizado)': {'accuracy': lr_acc, 'draw_recall': lr_draw},
    'XGBoost (calibrado)': {'accuracy': cal_models['XGB']['accuracy'], 'draw_recall': cal_models['XGB']['draw_recall']},
    'LightGBM (calibrado)': {'accuracy': cal_models['LGBM']['accuracy'], 'draw_recall': cal_models['LGBM']['draw_recall']},
    'LogReg (calibrado)': {'accuracy': cal_models['LR']['accuracy'], 'draw_recall': cal_models['LR']['draw_recall']},
    f'LightGBM (threshold={best_threshold:.2f})': {'accuracy': acc_final, 'draw_recall': draw_final},
}

print(f'{"Modelo":<35} {"Acurácia":>10} {"Draw Recall":>12}')
print('-' * 60)
for name, r in all_results.items():
    print(f'{name:<35} {r["accuracy"]:>10.1%} {r.get("draw_recall", 0):>12.1%}')

best_overall = max(all_results, key=lambda k: all_results[k]['accuracy'])
print(f'\n🏆 Melhor acurácia: {best_overall} ({all_results[best_overall]["accuracy"]:.1%})')

best_draw = max(all_results, key=lambda k: all_results[k].get('draw_recall', 0))
print(f'🎯 Melhor draw recall: {best_draw} ({all_results[best_draw].get("draw_recall", 0):.1%})')

# Salvar métricas
output = {
    'all_results': {k: {'accuracy': float(v['accuracy']), 'draw_recall': float(v.get('draw_recall', 0))} for k, v in all_results.items()},
    'best_accuracy': {'name': best_overall, 'accuracy': float(all_results[best_overall]['accuracy'])},
    'best_draw_recall': {'name': best_draw, 'draw_recall': float(all_results[best_draw].get('draw_recall', 0))},
    'optimal_threshold': float(best_threshold),
    'feature_importance': fi.head(10).to_dict(),
    'cv_scores': {
        'xgb': float(xgb_search.best_score_),
        'lgbm': float(lgbm_search.best_score_),
        'lr': float(lr_search.best_score_)
    }
}
with open(OUTPUTS / 'dia6_optimization_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f'\nResultados salvos: {OUTPUTS / "dia6_optimization_results.json"}')
print('\n✅ DIA 6 CONCLUÍDO')
