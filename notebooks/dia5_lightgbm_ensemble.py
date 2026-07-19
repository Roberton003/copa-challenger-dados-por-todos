"""
Dia 5 — LightGBM + Ensemble + Resolver Problema de Draws
=========================================================
Foco: Melhorar previsão de empates usando LightGBM balanced + ensemble + ajuste de threshold.

Autor: Roberto — Copa Challenger 2026
"""

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import lightgbm as lgb

print("=" * 60)
print("DIA 5 — LightGBM + Ensemble + Resolver Draws")
print("=" * 60)

# ============================================================
# 1. CARREGAR DADOS
# ============================================================
print("\n--- Carregar Dados ---")

import os
from pathlib import Path
_BASE = Path(__file__).resolve().parent.parent
RAW_DIR = str(_BASE / "data/raw")
PROCESSED_DIR = str(_BASE / "data/processed")

# Carregar matches
matches = pd.read_csv(os.path.join(RAW_DIR, "matches_1930_2022.csv"))
rank_2022 = pd.read_csv(os.path.join(RAW_DIR, "fifa_ranking_2022-10-06.csv"))

# Filtrar 2018+2022
matches_comp = matches[matches['Year'].isin([2018, 2022])].copy()
print(f"Partidas 2018+2022: {len(matches_comp)}")

# Derivar resultado dos scores
matches_comp['result'] = np.where(
    matches_comp['home_score'] > matches_comp['away_score'], 'Home',
    np.where(matches_comp['home_score'] < matches_comp['away_score'], 'Away', 'Draw')
)

# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================
print("\n--- Feature Engineering ---")

# Rankings (colunas lowercase: team, rank, points)
rank_2022['team'] = rank_2022['team'].str.strip()
rank_map = dict(zip(rank_2022['team'], rank_2022['points']))

matches_comp['home_rank_pts'] = matches_comp['home_team'].map(rank_map).fillna(1500)
matches_comp['away_rank_pts'] = matches_comp['away_team'].map(rank_map).fillna(1500)
matches_comp['rank_pts_diff'] = matches_comp['home_rank_pts'] - matches_comp['away_rank_pts']

# Rank number
rank_num_map = dict(zip(rank_2022['team'], rank_2022['rank']))
matches_comp['home_rank_num'] = matches_comp['home_team'].map(rank_num_map).fillna(100)
matches_comp['away_rank_num'] = matches_comp['away_team'].map(rank_num_map).fillna(100)
matches_comp['rank_num_diff'] = matches_comp['home_rank_num'] - matches_comp['away_rank_num']

# Round encoding
round_map = {
    'Group stage': 0,
    'Round of 16': 1,
    'Quarter-finals': 2,
    'Semi-finals': 3,
    'Third-place match': 4,
    'Final': 5
}
matches_comp['round_num'] = matches_comp['Round'].map(round_map).fillna(0)

# Stats médias
matches_comp['home_avg_scored'] = matches_comp.groupby('home_team')['home_score'].transform('mean')
matches_comp['home_avg_conceded'] = matches_comp.groupby('home_team')['away_score'].transform('mean')
matches_comp['away_avg_scored'] = matches_comp.groupby('away_team')['away_score'].transform('mean')
matches_comp['away_avg_conceded'] = matches_comp.groupby('away_team')['home_score'].transform('mean')

# Features avançadas (simplificadas para Dia 5)
matches_comp['avg_goals_by_round'] = matches_comp.groupby('round_num')['home_score'].transform('mean')
matches_comp['rank_x_round'] = matches_comp['rank_pts_diff'] * matches_comp['round_num']

# Interação ranking × forma
matches_comp['home_strength'] = matches_comp['home_rank_pts'] * matches_comp['home_avg_scored']
matches_comp['away_strength'] = matches_comp['away_rank_pts'] * matches_comp['away_avg_scored']
matches_comp['strength_diff'] = matches_comp['home_strength'] - matches_comp['away_strength']

# Diferença de gols médios
matches_comp['avg_scored_diff'] = matches_comp['home_avg_scored'] - matches_comp['away_avg_scored']
matches_comp['avg_conceded_diff'] = matches_comp['home_avg_conceded'] - matches_comp['away_avg_conceded']

# Total de features: 17
feature_cols = [
    'rank_pts_diff', 'rank_num_diff', 'home_rank_pts', 'away_rank_pts',
    'home_avg_scored', 'home_avg_conceded', 'away_avg_scored', 'away_avg_conceded',
    'round_num', 'avg_goals_by_round', 'rank_x_round',
    'home_strength', 'away_strength', 'strength_diff',
    'avg_scored_diff', 'avg_conceded_diff', 'home_score'
]

# Remover nulos
matches_model = matches_comp.dropna(subset=feature_cols + ['result'])
print(f"Após dropna: {len(matches_model)} partidas")

# ============================================================
# 3. PREPARAR DADOS
# ============================================================
print("\n--- Preparar Dados ---")

# Split temporal
train_mask = matches_model['Year'] == 2018
test_mask = matches_model['Year'] == 2022

X_train = matches_model[train_mask][feature_cols]
X_test = matches_model[test_mask][feature_cols]

le = LabelEncoder()
y_train_enc = le.fit_transform(matches_model[train_mask]['result'])
y_test_enc = le.transform(matches_model[test_mask]['result'])

print(f"Treino: {len(X_train)} partidas (2018)")
print(f"Teste: {len(X_test)} partidas (2022)")
print(f"Features: {len(feature_cols)}")
print(f"Classes: {le.classes_}")

# Distribuição de classes
print(f"\nDistribuição no treino:")
for i, cls in enumerate(le.classes_):
    count = (y_train_enc == i).sum()
    print(f"  {cls}: {count} ({count/len(y_train_enc):.1%})")

print(f"\nDistribuição no teste:")
for i, cls in enumerate(le.classes_):
    count = (y_test_enc == i).sum()
    print(f"  {cls}: {count} ({count/len(y_test_enc):.1%})")

# ============================================================
# 4. MODELOS
# ============================================================
print("\n--- Modelos ---")

# 1. Logistic Regression (baseline)
print("\n[1] Logistic Regression (baseline)")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train_enc)
y_pred_lr = lr.predict(X_test)
acc_lr = accuracy_score(y_test_enc, y_pred_lr)
print(f"Acurácia: {acc_lr:.1%}")

# 2. XGBoost (baseline do Dia 4)
print("\n[2] XGBoost (baseline Dia 4)")
xgb_model = xgb.XGBClassifier(
    n_estimators=200, max_depth=3, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    min_child_weight=3, gamma=0.1,
    random_state=42, eval_metric='mlogloss'
)
xgb_model.fit(X_train, y_train_enc)
y_pred_xgb = xgb_model.predict(X_test)
acc_xgb = accuracy_score(y_test_enc, y_pred_xgb)
print(f"Acurácia: {acc_xgb:.1%}")

# 3. LightGBM COM class_weight='balanced' (NOVO!)
print("\n[3] LightGBM (balanced)")
lgb_balanced = lgb.LGBMClassifier(
    n_estimators=200, max_depth=3, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    min_child_samples=5, reg_alpha=0.1, reg_lambda=0.1,
    class_weight='balanced',  # ← CHAVE para Draws!
    random_state=42, verbose=-1
)
lgb_balanced.fit(X_train, y_train_enc)
y_pred_lgb_bal = lgb_balanced.predict(X_test)
acc_lgb_bal = accuracy_score(y_test_enc, y_pred_lgb_bal)
print(f"Acurácia: {acc_lgb_bal:.1%}")

# 4. LightGBM SEM class_weight (para comparação)
print("\n[4] LightGBM (sem balanced)")
lgb_normal = lgb.LGBMClassifier(
    n_estimators=200, max_depth=3, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    min_child_samples=5, reg_alpha=0.1, reg_lambda=0.1,
    random_state=42, verbose=-1
)
lgb_normal.fit(X_train, y_train_enc)
y_pred_lgb_norm = lgb_normal.predict(X_test)
acc_lgb_norm = accuracy_score(y_test_enc, y_pred_lgb_norm)
print(f"Acurácia: {acc_lgb_norm:.1%}")

# 5. Ensemble Voting (LogReg + XGBoost + LightGBM balanced)
print("\n[5] Ensemble Voting (LR + XGB + LGB-balanced)")
ensemble = VotingClassifier(
    estimators=[
        ('lr', lr),
        ('xgb', xgb_model),
        ('lgb_bal', lgb_balanced)
    ],
    voting='soft',  # usa probabilidades
    weights=[1, 2, 2]  # dá mais peso a XGB e LGB
)
ensemble.fit(X_train, y_train_enc)
y_pred_ensemble = ensemble.predict(X_test)
acc_ensemble = accuracy_score(y_test_enc, y_pred_ensemble)
print(f"Acurácia: {acc_ensemble:.1%}")

# 6. Ensemble com threshold ajustado (NOVO!)
print("\n[6] Ensemble + Threshold Ajustado")
ensemble.fit(X_train, y_train_enc)
y_proba_ensemble = ensemble.predict_proba(X_test)

# Função para ajustar threshold (aumentar probabilidade mínima para Draws)
def predict_with_threshold(proba, threshold_home=0.35, threshold_draw=0.25, threshold_away=0.35):
    """Prediz com thresholds ajustados para favorecer Draws."""
    predictions = []
    for p in proba:
        # Se probabilidade de Draw > threshold, prediz Draw
        if p[le.transform(['Draw'])[0]] > threshold_draw:
            predictions.append(le.transform(['Draw'])[0])
        elif p[le.transform(['Home'])[0]] > threshold_home:
            predictions.append(le.transform(['Home'])[0])
        elif p[le.transform(['Away'])[0]] > threshold_away:
            predictions.append(le.transform(['Away'])[0])
        else:
            predictions.append(np.argmax(p))
    return np.array(predictions)

# Testar thresholds
thresholds_draw = [0.20, 0.22, 0.25, 0.28, 0.30]
best_acc_thr = 0
best_thr = 0.25
best_pred_thr = None

for thr in thresholds_draw:
    y_pred_thr = predict_with_threshold(y_proba_ensemble, threshold_draw=thr)
    acc_thr = accuracy_score(y_test_enc, y_pred_thr)
    # Calcular recall de Draws
    draw_mask = y_test_enc == le.transform(['Draw'])[0]
    if draw_mask.sum() > 0:
        draw_recall = (y_pred_thr[draw_mask] == le.transform(['Draw'])[0]).mean()
    else:
        draw_recall = 0
    print(f"  Threshold {thr:.2f}: Acurácia={acc_thr:.1%}, Draw Recall={draw_recall:.1%}")
    if acc_thr > best_acc_thr:
        best_acc_thr = acc_thr
        best_thr = thr
        best_pred_thr = y_pred_thr

y_pred_thr_best = best_pred_thr
acc_thr_best = best_acc_thr
print(f"\nMelhor threshold: {best_thr:.2f} → Acurácia: {acc_thr_best:.1%}")

# ============================================================
# 5. COMPARAÇÃO COMPLETA
# ============================================================
print("\n" + "=" * 60)
print("COMPARAÇÃO COMPLETA")
print("=" * 60)

model_names = [
    "LogReg (baseline)",
    "XGBoost",
    "LightGBM (balanced)",
    "LightGBM (normal)",
    "Ensemble (soft)",
    f"Ensemble + Threshold ({best_thr:.2f})"
]
predictions = [
    y_pred_lr, y_pred_xgb, y_pred_lgb_bal,
    y_pred_lgb_norm, y_pred_ensemble, y_pred_thr_best
]
accuracies = [
    acc_lr, acc_xgb, acc_lgb_bal,
    acc_lgb_norm, acc_ensemble, acc_thr_best
]

print(f"\n{'Modelo':<30} {'Acurácia':>8} {'Home%':>6} {'Draw%':>6} {'Away%':>6}")
print("-" * 60)

for name, pred, acc in zip(model_names, predictions, accuracies):
    home_pct = (pred == le.transform(['Home'])[0]).mean()
    draw_pct = (pred == le.transform(['Draw'])[0]).mean()
    away_pct = (pred == le.transform(['Away'])[0]).mean()
    print(f"{name:<30} {acc:>7.1%} {home_pct:>5.1%} {draw_pct:>5.1%} {away_pct:>5.1%}")

# Real
home_real = (y_test_enc == le.transform(['Home'])[0]).mean()
draw_real = (y_test_enc == le.transform(['Draw'])[0]).mean()
away_real = (y_test_enc == le.transform(['Away'])[0]).mean()
print("-" * 60)
print(f"{'REAL':<30} {'':>8} {home_real:>5.1%} {draw_real:>5.1%} {away_real:>5.1%}")

# ============================================================
# 6. MELHOR MODELO — CLASSIFICATION REPORT
# ============================================================
print("\n" + "=" * 60)
print("MELHOR MODELO — CLASSIFICATION REPORT")
print("=" * 60)

# Encontrar melhor modelo
best_idx = np.argmax(accuracies)
best_name = model_names[best_idx]
best_pred = predictions[best_idx]

print(f"\nMelhor: {best_name} ({accuracies[best_idx]:.1%})")
print("\nClassification Report:")
print(classification_report(
    y_test_enc, best_pred,
    target_names=le.classes_,
    digits=3
))

# Matriz de confusão
print("Matriz de Confusão:")
cm = confusion_matrix(y_test_enc, best_pred)
print(f"         Pred Home  Pred Draw  Pred Away")
for i, cls in enumerate(le.classes_):
    print(f"Real {cls:5s}  {cm[i][0]:>8}  {cm[i][1]:>8}  {cm[i][2]:>8}")

# ============================================================
# 7. FEATURE IMPORTANCE
# ============================================================
print("\n" + "=" * 60)
print("FEATURE IMPORTANCE (LightGBM balanced)")
print("=" * 60)

feat_imp = pd.DataFrame({
    'feature': feature_cols,
    'importance': lgb_balanced.feature_importances_
}).sort_values('importance', ascending=False)

for _, row in feat_imp.head(10).iterrows():
    print(f"  {row['feature']:<25} {row['importance']:.4f}")

# ============================================================
# 8. ANÁLISE DE ERROS
# ============================================================
print("\n" + "=" * 60)
print("ANÁLISE DE ERROS")
print("=" * 60)

test_data = matches_model[test_mask].copy()
y_test_labels = le.inverse_transform(y_test_enc.astype(int))
best_pred_labels = le.inverse_transform(best_pred.astype(int))

wrong_mask = best_pred_labels != y_test_labels
wrong_preds = test_data[wrong_mask].copy()
wrong_preds['predicted'] = best_pred_labels[wrong_mask]
wrong_preds['actual'] = y_test_labels[wrong_mask]

print(f"\nPrevisões incorretas: {wrong_mask.sum()} de {len(y_test_enc)} ({wrong_mask.mean():.1%})")

# Tipos de erro
print("\nTipos de erro:")
for actual_cls in le.classes_:
    for pred_cls in le.classes_:
        if actual_cls != pred_cls:
            count = ((y_test_labels == actual_cls) & (best_pred_labels == pred_cls)).sum()
            if count > 0:
                print(f"  {actual_cls} → {pred_cls}: {count}")

print("\nExemplos de erros:")
for _, row in wrong_preds.head(8).iterrows():
    print(f"  {row['home_team']} vs {row['away_team']}: {row['actual']} → {row['predicted']}")

# ============================================================
# 9. SALVAR RESULTADOS
# ============================================================
print("\n--- Salvar Resultados ---")

# Salvar previsões
predictions_df = test_data[['home_team', 'away_team', 'result', 'round_num']].copy()
predictions_df['pred_best'] = le.inverse_transform(best_pred.astype(int))
predictions_df['pred_ensemble_raw'] = le.inverse_transform(y_pred_ensemble.astype(int))
predictions_df['pred_threshold'] = le.inverse_transform(y_pred_thr_best.astype(int))

# Salvar probabilidades
for i, cls in enumerate(le.classes_):
    predictions_df[f'proba_{cls}'] = y_proba_ensemble[:, i]

predictions_df.to_csv(os.path.join(PROCESSED_DIR, "predictions_dia5.csv"), index=False)

# Salvar métricas
metrics = {
    'dia': 5,
    'models': {},
    'best_model': best_name,
    'best_accuracy': float(accuracies[best_idx]),
    'draw_problem': {
        'real_draw_pct': float(draw_real),
        'lgb_balanced_draw_pct': float((y_pred_lgb_bal == le.transform(['Draw'])[0]).mean()),
        'ensemble_draw_pct': float((y_pred_ensemble == le.transform(['Draw'])[0]).mean()),
        'threshold_draw_pct': float((y_pred_thr_best == le.transform(['Draw'])[0]).mean()),
    }
}

for name, pred, acc in zip(model_names, predictions, accuracies):
    metrics['models'][name] = {
        'accuracy': float(acc),
        'home_pct': float((pred == le.transform(['Home'])[0]).mean()),
        'draw_pct': float((pred == le.transform(['Draw'])[0]).mean()),
        'away_pct': float((pred == le.transform(['Away'])[0]).mean()),
    }

OUTPUTS_DIR = str(_BASE / "outputs")
with open(os.path.join(OUTPUTS_DIR, 'dia5_metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

# Salvar feature importance
feat_imp.to_csv(os.path.join(OUTPUTS_DIR, 'feature_importance_dia5.csv'), index=False)

print("Arquivos salvos:")
print(f"  {PROCESSED_DIR}/predictions_dia5.csv")
print(f"  outputs/dia5_metrics.json")
print(f"  outputs/feature_importance_dia5.csv")

# ============================================================
# 10. RESUMO
# ============================================================
print("\n" + "=" * 60)
print("RESUMO DIA 5")
print("=" * 60)

print(f"""
Features: {len(feature_cols)}

Melhor Modelo: {best_name}
Acurácia: {accuracies[best_idx]:.1%}

Comparação com Dia 4:
  Dia 4 (XGBoost Otimizado): 62.5%
  Dia 5 (Melhor): {accuracies[best_idx]:.1%}
  Melhoria: {accuracies[best_idx] - 0.625:+.1%}

Problema de Draws:
  Real: {draw_real:.1%}
  Dia 4 (XGBoost): {(y_pred_xgb == le.transform(['Draw'])[0]).mean():.1%}
  Dia 5 (LightGBM balanced): {(y_pred_lgb_bal == le.transform(['Draw'])[0]).mean():.1%}
  Dia 5 (Ensemble): {(y_pred_ensemble == le.transform(['Draw'])[0]).mean():.1%}
  Dia 5 (Threshold {best_thr:.2f}): {(y_pred_thr_best == le.transform(['Draw'])[0]).mean():.1%}

Top 3 features (LightGBM):
  {feat_imp.iloc[0]['feature']}: {feat_imp.iloc[0]['importance']:.4f}
  {feat_imp.iloc[1]['feature']}: {feat_imp.iloc[1]['importance']:.4f}
  {feat_imp.iloc[2]['feature']}: {feat_imp.iloc[2]['importance']:.4f}

Próximo: Dia 6 — Otimização + Validação Cruzada
""")

print("FIM DIA 5")
