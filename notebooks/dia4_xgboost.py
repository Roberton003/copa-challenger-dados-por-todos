"""
Dia 4 — XGBoost + Feature Engineering Avançada
Copa Challenger Dados por Todos

- CLASSIFICAR: data-science-ml (modelos avançados, ensemble)
- EXECUTAR: este script
- REPORTAR: outcomes no final
- VALIDAR: comparação com Dia 3

"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import warnings
import os
import json
from pathlib import Path

warnings.filterwarnings('ignore')

# ============================================================
# 1. CARREGAR DADOS
# ============================================================
print("=" * 60)
print("DIA 4 — XGBoost + Feature Engineering Avançada")
print("=" * 60)

base_path = Path(__file__).resolve().parent.parent
matches = pd.read_csv(base_path / "data/raw/matches_1930_2022.csv")
ranking_2022 = pd.read_csv(base_path / "data/raw/fifa_ranking_2022-10-06.csv")

# Filtrar 2018+2022
matches_comp = matches[matches['Year'].isin([2018, 2022])].copy()
print(f"\nPartidas 2018+2022: {len(matches_comp)}")

# ============================================================
# 2. FEATURE ENGINEERING AVANÇADA
# ============================================================
print("\n--- Feature Engineering Avançada ---")

# Features básicas (do Dia 3)
matches_comp['home_score'] = pd.to_numeric(matches_comp['home_score'], errors='coerce')
matches_comp['away_score'] = pd.to_numeric(matches_comp['away_score'], errors='coerce')
matches_comp['total_goals'] = matches_comp['home_score'] + matches_comp['away_score']
matches_comp['goal_diff'] = matches_comp['home_score'] - matches_comp['away_score']

def get_result(row):
    if row['home_score'] > row['away_score']:
        return 'Home'
    elif row['home_score'] < row['away_score']:
        return 'Away'
    else:
        return 'Draw'

matches_comp['result'] = matches_comp.apply(get_result, axis=1)

# Round encoding
round_map = {
    'Group stage': 0,
    'Round of 16': 1,
    'Quarter-finals': 2,
    'Semi-finals': 3,
    'Third-place match': 4,
    'Final': 5
}
matches_comp['round_num'] = matches_comp['Round'].map(round_map)

# Merge ranking
ranking_2022['rank'] = pd.to_numeric(ranking_2022['rank'], errors='coerce')
ranking_2022['rank_points'] = pd.to_numeric(ranking_2022.get('rank_points', ranking_2022.get('points', 0)), errors='coerce')

# Encontrar coluna de pontos
pts_col = None
for col in ['rank_points', 'points', 'total_points']:
    if col in ranking_2022.columns:
        pts_col = col
        break

if pts_col:
    ranking_dict = ranking_2022.set_index('team')['rank'].to_dict()
    ranking_pts_dict = ranking_2022.set_index('team')[pts_col].to_dict()
else:
    ranking_dict = {}
    ranking_pts_dict = {}

matches_comp['home_rank'] = matches_comp['home_team'].map(ranking_dict)
matches_comp['away_rank'] = matches_comp['away_team'].map(ranking_dict)
matches_comp['home_rank_pts'] = matches_comp['home_team'].map(ranking_pts_dict)
matches_comp['away_rank_pts'] = matches_comp['away_team'].map(ranking_pts_dict)

# Features de ranking
matches_comp['rank_diff'] = matches_comp['home_rank'] - matches_comp['away_rank']
matches_comp['rank_pts_diff'] = matches_comp['home_rank_pts'] - matches_comp['away_rank_pts']

# ============================================================
# 2.1 NOVAS FEATURES AVANÇADAS
# ============================================================
print("Criando features avançadas...")

# 2.1.1 Histórico de confrontos diretos (head-to-head)
def get_h2h_record(df, home_team, away_team, current_idx):
    """Retorna histórico de confrontos diretos antes da partida atual"""
    past = df[(df.index < current_idx) & 
              ((df['home_team'] == home_team) & (df['away_team'] == away_team) |
               (df['home_team'] == away_team) & (df['away_team'] == home_team))]
    
    if len(past) == 0:
        return 0, 0, 0  # home_wins, draws, away_wins
    
    home_wins = 0
    draws = 0
    away_wins = 0
    
    for _, row in past.iterrows():
        if row['home_team'] == home_team:
            if row['result'] == 'Home':
                home_wins += 1
            elif row['result'] == 'Draw':
                draws += 1
            else:
                away_wins += 1
        else:
            if row['result'] == 'Home':
                away_wins += 1
            elif row['result'] == 'Draw':
                draws += 1
            else:
                home_wins += 1
    
    return home_wins, draws, away_wins

# Calcular h2h para cada partida
h2h_home_wins = []
h2h_draws = []
h2h_away_wins = []

for idx in range(len(matches_comp)):
    row = matches_comp.iloc[idx]
    hw, d, aw = get_h2h_record(matches_comp, row['home_team'], row['away_team'], matches_comp.index[idx])
    h2h_home_wins.append(hw)
    h2h_draws.append(d)
    h2h_away_wins.append(aw)

matches_comp['h2h_home_wins'] = h2h_home_wins
matches_comp['h2h_draws'] = h2h_draws
matches_comp['h2h_away_wins'] = h2h_away_wins
matches_comp['h2h_total'] = matches_comp['h2h_home_wins'] + matches_comp['h2h_draws'] + matches_comp['h2h_away_wins']

# 2.1.2 Forma recente (últimos 5 jogos)
def get_team_form(df, team, current_idx, n=5):
    """Retorna forma recente do time (últimos n jogos)"""
    past = df[(df.index < current_idx) & 
              ((df['home_team'] == team) | (df['away_team'] == team))].tail(n)
    
    if len(past) == 0:
        return 0, 0, 0, 0  # wins, draws, losses, goals_scored
    
    wins = 0
    draws = 0
    losses = 0
    goals_scored = 0
    
    for _, row in past.iterrows():
        if row['home_team'] == team:
            goals_scored += row['home_score']
            if row['result'] == 'Home':
                wins += 1
            elif row['result'] == 'Draw':
                draws += 1
            else:
                losses += 1
        else:
            goals_scored += row['away_score']
            if row['result'] == 'Away':
                wins += 1
            elif row['result'] == 'Draw':
                draws += 1
            else:
                losses += 1
    
    return wins, draws, losses, goals_scored

# Calcular forma
home_form_wins = []
home_form_draws = []
home_form_losses = []
home_form_goals = []
away_form_wins = []
away_form_draws = []
away_form_losses = []
away_form_goals = []

for idx in range(len(matches_comp)):
    row = matches_comp.iloc[idx]
    
    hw, hd, hl, hg = get_team_form(matches_comp, row['home_team'], matches_comp.index[idx])
    home_form_wins.append(hw)
    home_form_draws.append(hd)
    home_form_losses.append(hl)
    home_form_goals.append(hg)
    
    aw, ad, al, ag = get_team_form(matches_comp, row['away_team'], matches_comp.index[idx])
    away_form_wins.append(aw)
    away_form_draws.append(ad)
    away_form_losses.append(al)
    away_form_goals.append(ag)

matches_comp['home_form_wins'] = home_form_wins
matches_comp['home_form_draws'] = home_form_draws
matches_comp['home_form_losses'] = home_form_losses
matches_comp['home_form_goals'] = home_form_goals
matches_comp['away_form_wins'] = away_form_wins
matches_comp['away_form_draws'] = away_form_draws
matches_comp['away_form_losses'] = away_form_losses
matches_comp['away_form_goals'] = away_form_goals

# 2.1.3 Diferença de forma
matches_comp['form_wins_diff'] = matches_comp['home_form_wins'] - matches_comp['away_form_wins']
matches_comp['form_goals_diff'] = matches_comp['home_form_goals'] - matches_comp['away_form_goals']

# 2.1.4 Média de gols por fase do torneio
matches_comp['avg_goals_by_round'] = matches_comp.groupby('round_num')['total_goals'].transform('mean')

# 2.1.5 Interação ranking × fase
matches_comp['rank_x_round'] = matches_comp['rank_diff'] * matches_comp['round_num']

# 2.1.6 Dummy de mandante (sempre 1, mas mantido para consistência)
matches_comp['is_home'] = 1

print(f"Features criadas: {len(matches_comp.columns)} colunas totais")

# ============================================================
# 3. PREPARAR DADOS PARA MODELOS
# ============================================================
print("\n--- Preparar Dados ---")

features = [
    'rank_diff', 'rank_pts_diff', 'home_rank', 'away_rank',
    'home_rank_pts', 'away_rank_pts',
    'home_form_wins', 'home_form_draws', 'home_form_losses', 'home_form_goals',
    'away_form_wins', 'away_form_draws', 'away_form_losses', 'away_form_goals',
    'form_wins_diff', 'form_goals_diff',
    'h2h_home_wins', 'h2h_draws', 'h2h_away_wins', 'h2h_total',
    'round_num', 'avg_goals_by_round', 'rank_x_round', 'is_home'
]

# Adicionar médias históricas (do Dia 3)
# Calcular médias por time
home_stats = matches_comp.groupby('home_team').agg({
    'home_score': ['mean', 'std'],
    'away_score': ['mean', 'std']
}).reset_index()
home_stats.columns = ['team', 'home_avg_scored', 'home_avg_conceded', 'home_std_scored', 'home_std_conceded']

away_stats = matches_comp.groupby('away_team').agg({
    'away_score': ['mean', 'std'],
    'home_score': ['mean', 'std']
}).reset_index()
away_stats.columns = ['team', 'away_avg_scored', 'away_avg_conceded', 'away_std_scored', 'away_std_conceded']

matches_comp = matches_comp.merge(home_stats, left_on='home_team', right_on='team', how='left')
matches_comp = matches_comp.merge(away_stats, left_on='away_team', right_on='team', how='left', suffixes=('', '_away'))

# Adicionar features de média
features.extend(['home_avg_scored', 'home_avg_conceded', 'away_avg_scored', 'away_avg_conceded'])

# Preparar X e y
X = matches_comp[features].copy()
y = matches_comp['result'].copy()

# Fill NaN com 0 (para times sem histórico)
X = X.fillna(0)

# Split temporal: treino=2018, teste=2022
train_mask = matches_comp['Year'] == 2018
test_mask = matches_comp['Year'] == 2022

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

print(f"Treino: {len(X_train)} partidas (2018)")
print(f"Teste: {len(X_test)} partidas (2022)")
print(f"Features: {len(features)}")

# Encode target labels
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)
print(f"Classes: {list(le.classes_)}")

# ============================================================
# 4. MODELOS
# ============================================================
print("\n--- Modelos ---")

# 4.1 Logistic Regression (baseline do Dia 3)
print("\n[1] Logistic Regression (baseline)")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train_enc)
y_pred_lr = lr.predict(X_test)
acc_lr = accuracy_score(y_test_enc, y_pred_lr)
print(f"Acurácia: {acc_lr:.1%}")

# 4.2 XGBoost
print("\n[2] XGBoost")
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss'
)
xgb_model.fit(X_train, y_train_enc)
y_pred_xgb = xgb_model.predict(X_test)
acc_xgb = accuracy_score(y_test_enc, y_pred_xgb)
print(f"Acurácia: {acc_xgb:.1%}")

# 4.3 XGBoost com hyperparameters otimizados
print("\n[3] XGBoost Otimizado")
xgb_opt = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss'
)
xgb_opt.fit(X_train, y_train_enc)
y_pred_xgb_opt = xgb_opt.predict(X_test)
acc_xgb_opt = accuracy_score(y_test_enc, y_pred_xgb_opt)
print(f"Acurácia: {acc_xgb_opt:.1%}")

# ============================================================
# 5. COMPARAÇÃO DETALHADA
# ============================================================
print("\n--- Comparação ---")

models = {
    'LogReg (Dia 3)': y_pred_lr,
    'XGBoost': y_pred_xgb,
    'XGBoost Otimizado': y_pred_xgb_opt
}

for name, y_pred in models.items():
    print(f"\n{name}:")
    print(f"  Acurácia: {accuracy_score(y_test_enc, y_pred):.1%}")
    
    # Distribuição de previsões
    pred_labels = le.inverse_transform(y_pred.astype(int))
    pred_dist = pd.Series(pred_labels).value_counts(normalize=True)
    for cls in ['Home', 'Draw', 'Away']:
        pct = pred_dist.get(cls, 0)
        print(f"  Previsões {cls}: {pct:.1%}")

# ============================================================
# 6. FEATURE IMPORTANCE (XGBoost)
# ============================================================
print("\n--- Feature Importance (XGBoost Otimizado) ---")

importance = xgb_opt.feature_importances_
feat_imp = pd.DataFrame({
    'feature': features,
    'importance': importance
}).sort_values('importance', ascending=False)

for _, row in feat_imp.head(10).iterrows():
    print(f"  {row['feature']}: {row['importance']:.3f}")

# ============================================================
# 7. ANÁLISE DE ERROS
# ============================================================
print("\n--- Análise de Erros (XGBoost Otimizado) ---")

# Apenas previsões erradas
y_pred_xgb_opt_labels = le.inverse_transform(y_pred_xgb_opt.astype(int))
y_test_labels = le.inverse_transform(y_test_enc.astype(int))
wrong_mask = y_pred_xgb_opt_labels != y_test_labels
wrong_preds = matches_comp[test_mask][wrong_mask].copy()
wrong_preds['predicted'] = y_pred_xgb_opt_labels[wrong_mask]
wrong_preds['actual'] = y_test_labels[wrong_mask]

print(f"Previsões incorretas: {wrong_mask.sum()} de {len(y_test)} ({wrong_mask.mean():.1%})")
print("\nExemplos de erros:")
for _, row in wrong_preds.head(5).iterrows():
    print(f"  {row['home_team']} vs {row['away_team']}: {row['actual']} → {row['predicted']}")

# ============================================================
# 8. SALVAR RESULTADOS
# ============================================================
print("\n--- Salvar Resultados ---")

# Salvar previsões
predictions = matches_comp[test_mask][['home_team', 'away_team', 'result', 'round_num']].copy()
predictions['pred_lr'] = le.inverse_transform(y_pred_lr.astype(int))
predictions['pred_xgb'] = le.inverse_transform(y_pred_xgb.astype(int))
predictions['pred_xgb_opt'] = le.inverse_transform(y_pred_xgb_opt.astype(int))
predictions.to_csv(f"{base_path}/data/processed/predictions_dia4.csv", index=False)

# Salvar métricas
metrics = {
    'dia': 4,
    'models': {
        'logistic_regression': {'accuracy': float(acc_lr)},
        'xgboost': {'accuracy': float(acc_xgb)},
        'xgboost_optimized': {'accuracy': float(acc_xgb_opt)}
    },
    'features_count': len(features),
    'train_size': int(len(X_train)),
    'test_size': int(len(X_test)),
    'feature_importance_top10': feat_imp.head(10).to_dict('records')
}

with open(f"{base_path}/outputs/dia4_metrics.json", 'w') as f:
    json.dump(metrics, f, indent=2)

# Salvar feature importance
feat_imp.to_csv(f"{base_path}/outputs/feature_importance_dia4.csv", index=False)

print(f"\nArquivos salvos:")
print(f"  data/processed/predictions_dia4.csv")
print(f"  outputs/dia4_metrics.json")
print(f"  outputs/feature_importance_dia4.csv")

# ============================================================
# 9. RESUMO
# ============================================================
print("\n" + "=" * 60)
print("RESUMO DIA 4")
print("=" * 60)
print(f"\nFeatures: {len(features)} (7 básicas + 17 avançadas)")
print(f"\nModelos:")
print(f"  LogReg (baseline): {acc_lr:.1%}")
print(f"  XGBoost: {acc_xgb:.1%}")
print(f"  XGBoost Otimizado: {acc_xgb_opt:.1%}")
print(f"\nMelhoria sobre baseline: +{(acc_xgb_opt - acc_lr)*100:.1f}pp")
print(f"\nTop 3 features:")
for _, row in feat_imp.head(3).iterrows():
    print(f"  {row['feature']}: {row['importance']:.3f}")

