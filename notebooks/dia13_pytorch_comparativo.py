#!/usr/bin/env python3
"""Dia 13 — Comparativo empírico PyTorch vs scikit-learn/XGBoost/LightGBM.

A Missão 4 sugere PyTorch/IA Generativa como ferramenta possível. Em vez de só
argumentar "dataset pequeno demais", treina uma MLP real com as mesmas 5
features/split/métricas dos dias 9-10 e compara com o melhor modelo clássico
(LogReg sigmoid, RPS=0.2019) usando dado, não opinião.

Mesma pipeline de dia10_recalibracao_rps.py (alias de nomes, top-5 features,
split temporal 2018→2022). Multi-seed (10 seeds) pela mesma razão do dia10:
com 64 amostras de treino, um único seed não é confiável.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json

import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, StandardScaler

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
train = matches_comp[matches_comp['Year'] == 2018]
test = matches_comp[matches_comp['Year'] == 2022]
X_train_raw, X_test_raw = train[feature_cols].fillna(0).values, test[feature_cols].fillna(0).values
y_train, y_test = le.fit_transform(train['result']), le.transform(test['result'])
rps_order = [list(le.classes_).index(c) for c in ['Home', 'Draw', 'Away']]
y_test_ord = np.array([rps_order.index(y) for y in y_test])

class_counts = np.bincount(y_train)
class_weights = torch.tensor(len(y_train) / (len(class_counts) * class_counts), dtype=torch.float32)

def rps(y_true_idx, proba_ordered):
    n = len(y_true_idx)
    total = 0.0
    for i in range(n):
        obs = np.zeros(3)
        obs[y_true_idx[i]] = 1
        total += np.sum((np.cumsum(proba_ordered[i]) - np.cumsum(obs)) ** 2) / 2
    return total / n

class MLP(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 16), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 3),
        )

    def forward(self, x):
        return self.net(x)

def train_eval(seed):
    torch.manual_seed(seed)
    scaler = StandardScaler().fit(X_train_raw)
    X_train = torch.tensor(scaler.transform(X_train_raw), dtype=torch.float32)
    X_test = torch.tensor(scaler.transform(X_test_raw), dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)

    model = MLP(len(feature_cols))
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-3)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    model.train()
    for _ in range(200):
        opt.zero_grad()
        loss = loss_fn(model(X_train), y_train_t)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        proba = torch.softmax(model(X_test), dim=1).numpy()
    acc = float((proba.argmax(axis=1) == y_test).mean())
    draw_idx = list(le.classes_).index('Draw')
    draw_mask = y_test == draw_idx
    draw_recall = float((proba.argmax(axis=1)[draw_mask] == draw_idx).mean()) if draw_mask.sum() else 0.0
    score = rps(y_test_ord, proba[:, rps_order])
    return acc, draw_recall, score

print('=' * 60)
print('DIA 13 — MLP (PyTorch) vs melhor modelo clássico (LogReg sigmoid)')
print('=' * 60)

seeds = list(range(10))
results = [train_eval(s) for s in seeds]
accs, draws, rpss = zip(*results)

print(f'MLP (PyTorch, 10 seeds): acc={np.mean(accs):.1%} ± {np.std(accs):.1%}  '
      f'draw_recall={np.mean(draws):.1%} ± {np.std(draws):.1%}  '
      f'RPS={np.mean(rpss):.4f} ± {np.std(rpss):.4f}')
print('Referência (dia10, classic ML): LogReg sigmoid acc=57.8%  draw_recall=0.0%  RPS=0.2019')

out = {
    'mlp_pytorch': {
        'accuracy_mean': float(np.mean(accs)), 'accuracy_std': float(np.std(accs)),
        'draw_recall_mean': float(np.mean(draws)), 'draw_recall_std': float(np.std(draws)),
        'rps_mean': float(np.mean(rpss)), 'rps_std': float(np.std(rpss)),
        'seeds': seeds, 'architecture': '5->16->8->3, ReLU, dropout 0.3',
    },
    'classic_ml_reference': {
        'model': 'LogReg (sigmoid)', 'accuracy': 0.578, 'draw_recall': 0.0, 'rps': 0.2019,
        'source': 'outputs/dia10_recalibracao_rps.json',
    },
}
json.dump(out, open(OUTPUTS / 'dia13_pytorch_comparativo.json', 'w'), indent=2)
print(f"\nResultados salvos: {OUTPUTS / 'dia13_pytorch_comparativo.json'}")

def _selftest():
    assert 0.0 <= np.mean(accs) <= 1.0
    assert 0.0 <= np.mean(rpss) <= 1.0
    print('selftest OK')

if __name__ == '__main__':
    _selftest()
