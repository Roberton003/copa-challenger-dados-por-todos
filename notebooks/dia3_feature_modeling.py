"""
Dia 3 — Feature Engineering + Modelagem Inicial
Copa Challenger Dados por Todos

"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. CARREGAR DADOS
# ============================================================
print("=" * 60)
print("DIA 3 — FEATURE ENGINEERING + MODELAGEM INICIAL")
print("=" * 60)

BASE = Path(__file__).resolve().parent.parent
RAW_DIR = BASE / "data/raw"
PROCESSED_DIR = BASE / "data/processed"
OUTPUT_DIR = BASE / "outputs"

print("\n[1] Carregando dados...")
matches = pd.read_csv(PROCESSED_DIR / "matches_2018_2022_dia1.csv")
rankings_2022 = pd.read_csv(RAW_DIR / "fifa_ranking_2022-10-06.csv")
rankings_2026 = pd.read_csv(RAW_DIR / "fifa_ranking_2026-06-08.csv")

print(f"  Matches: {len(matches)} linhas, {len(matches.columns)} colunas")
print(f"  Ranking 2022: {len(rankings_2022)} times")

# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================
print("\n[2] Feature Engineering...")

# 2.1 Normalizar nomes
def normalize_team(name):
    name = str(name).strip()
    mappings = {
        'United States': 'USA', 'South Korea': 'Korea Republic',
        'IR Iran': 'Iran', 'Côte d\'Ivoire': 'Ivory Coast',
        'Bosnia-Herzegovina': 'Bosnia and Herzegovina',
    }
    return mappings.get(name, name)

matches['home_team_norm'] = matches['home_team'].apply(normalize_team)
matches['away_team_norm'] = matches['away_team'].apply(normalize_team)

# 2.2 Build rank maps (coluna = 'points')
rank_map_2022 = {row['team']: {'points': row['points'], 'rank': row['rank']}
                 for _, row in rankings_2022.iterrows()}

# 2.3 Adicionar ranking
matches['home_rank_pts'] = matches['home_team_norm'].map(lambda x: rank_map_2022.get(x, {}).get('points'))
matches['home_rank_num'] = matches['home_team_norm'].map(lambda x: rank_map_2022.get(x, {}).get('rank'))
matches['away_rank_pts'] = matches['away_team_norm'].map(lambda x: rank_map_2022.get(x, {}).get('points'))
matches['away_rank_num'] = matches['away_team_norm'].map(lambda x: rank_map_2022.get(x, {}).get('rank'))

# Ranking differences
matches['rank_pts_diff'] = matches['home_rank_pts'].fillna(1500) - matches['away_rank_pts'].fillna(1500)
matches['rank_num_diff'] = matches['away_rank_num'].fillna(100) - matches['home_rank_num'].fillna(100)

print(f"  Matches com ranking: {matches['home_rank_pts'].notna().sum()}/{len(matches)}")

# 2.4 Média de gols por time (mandante)
team_stats = matches.groupby('home_team').agg(
    home_avg_scored=('home_score', 'mean'),
    home_avg_conceded=('away_score', 'mean'),
).reset_index()
matches = matches.merge(team_stats, on='home_team', how='left')

# 2.5 Encoding de fase
round_mapping = {
    'Group stage': 0, 'Round of 16': 1, 'Quarter-finals': 2,
    'Semi-finals': 3, 'Third-place match': 4, 'Final': 5
}
matches['round_num'] = matches['Round'].map(round_mapping)

# 2.6 Encoding de resultado
result_mapping = {'Home': 0, 'Draw': 1, 'Away': 2}
matches['result_class'] = matches['result'].map(result_mapping)

print(f"  rounds mapeados: {matches['round_num'].notna().sum()}/{len(matches)}")
print(f"  results mapeados: {matches['result_class'].notna().sum()}/{len(matches)}")

# 2.7 Dataset final
feature_cols = [
    'rank_pts_diff', 'rank_num_diff',
    'home_rank_pts', 'away_rank_pts',
    'home_avg_scored', 'home_avg_conceded',
    'round_num',
]

df_model = matches[feature_cols + ['result_class', 'home_team', 'away_team', 'Year', 'Round', 'result']].dropna()
print(f"\n  Dataset final: {len(df_model)} linhas (dropna removeu {len(matches)-len(df_model)})")

train = df_model[df_model['Year'] == 2018].copy()
test = df_model[df_model['Year'] == 2022].copy()
print(f"  Treino: {len(train)} | Teste: {len(test)}")

X_train = train[feature_cols]
y_train = train['result_class']
X_test = test[feature_cols]
y_test = test['result_class']

from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# Poisson para gols mandante e visitante
poisson_h = PoissonRegressor(alpha=1.0, max_iter=1000)
poisson_h.fit(X_train_sc, train['home_score'] if 'home_score' in train.columns else train.get('home_score', pd.Series([1]*len(train))))

poisson_a = PoissonRegressor(alpha=1.0, max_iter=1000)
poisson_a.fit(X_train_sc, train['away_score'] if 'away_score' in train.columns else train.get('away_score', pd.Series([1]*len(train))))

pred_hg = poisson_h.predict(X_test_sc)
pred_ag = poisson_a.predict(X_test_sc)

def goals_to_result(hg, ag):
    return np.array([0 if h > a else (1 if h == a else 2) for h, a in zip(hg, ag)])

y_pred_poisson = goals_to_result(pred_hg, pred_ag)
acc_poisson = accuracy_score(y_test, y_pred_poisson)
print(f"  Poisson Accuracy: {acc_poisson:.3f}")
print(classification_report(y_test, y_pred_poisson, target_names=['Home','Draw','Away'], zero_division=0))

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(multi_class='multinomial', max_iter=1000, random_state=42, class_weight='balanced')
logreg.fit(X_train_sc, y_train)
y_pred_logreg = logreg.predict(X_test_sc)
acc_logreg = accuracy_score(y_test, y_pred_logreg)
print(f"  LogReg Accuracy: {acc_logreg:.3f}")
print(classification_report(y_test, y_pred_logreg, target_names=['Home','Draw','Away'], zero_division=0))

# Feature importance
print("  Coeficientes LogReg:")
for i, col in enumerate(feature_cols):
    coefs = logreg.coef_[:, i]
    print(f"    {col:20s}: H={coefs[0]:+.3f}  D={coefs[1]:+.3f}  A={coefs[2]:+.3f}")

# ============================================================
# 6. COMPARAÇÃO
# ============================================================
print("\n[6] Comparação:")
print(f"  {'Modelo':<25} {'Acurácia':<10}")
print(f"  {'-'*35}")
print(f"  {'Poisson':<25} {acc_poisson:.3f}")
print(f"  {'Logistic Regression':<25} {acc_logreg:.3f}")
print(f"  {'Aleatório (1/3)':<25} {1/3:.3f}")

# ============================================================
# 7. SALVAR
# ============================================================
print("\n[7] Salvando...")
df_model.to_csv(PROCESSED_DIR / "matches_model_features.csv", index=False)

pred_df = test[['home_team', 'away_team', 'Round', 'result']].copy()
pred_df['poisson_pred'] = [(['Home','Draw','Away'])[p] for p in y_pred_poisson]
pred_df['logreg_pred'] = [(['Home','Draw','Away'])[p] for p in y_pred_logreg]
pred_df.to_csv(PROCESSED_DIR / "predictions_2022.csv", index=False)

metrics = {
    'dia': 3,
    'modelos': {
        'Poisson': {'accuracy': float(acc_poisson)},
        'LogisticRegression': {'accuracy': float(acc_logreg)},
    },
    'baseline_random': 1/3,
    'features': feature_cols,
    'split': {'treino': '2018', 'teste': '2022'},
}
with open(OUTPUT_DIR / "dia3_metrics.json", 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"  -> data/processed/matches_model_features.csv")
print(f"  -> data/processed/predictions_2022.csv")
print(f"  -> outputs/dia3_metrics.json")

print("\n" + "=" * 60)
print("DIA 3 CONCLUÍDO")
print("=" * 60)
