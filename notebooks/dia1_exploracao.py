#!/usr/bin/env python3
"""
Dia 1 — SQL + Entendimento dos Dados
Competição: Copa Challenger Dados por Todos

"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "data" / "raw"
PROCESSED = BASE / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("DIA 1 — SQL + ENTENDIMENTO DOS DADOS")
print("Copa Challenger Dados por Todos")
print("=" * 70)

print("\n📦 1. CARREGANDO DATASETS...")

matches = pd.read_csv(RAW / "matches_1930_2022.csv")
world_cup = pd.read_csv(RAW / "world_cup.csv")
schedule_2026 = pd.read_csv(RAW / "schedule_2026.csv")
ranking_2022 = pd.read_csv(RAW / "fifa_ranking_2022-10-06.csv")
ranking_2026 = pd.read_csv(RAW / "fifa_ranking_2026-06-08.csv")

print(f"  matches_1930_2022: {matches.shape[0]} partidas, {matches.shape[1]} colunas")
print(f"  world_cup: {world_cup.shape[0]} edições")
print(f"  schedule_2026: {schedule_2026.shape[0]} jogos")
print(f"  ranking_2022: {ranking_2022.shape[0]} times")
print(f"  ranking_2026: {ranking_2026.shape[0]} times")

# ============================================================
# 2. EXPLORAR SCHEMA
# ============================================================
print("\n🔍 2. SCHEMA DO DATASET PRINCIPAL...")
print(f"\nColunas ({matches.shape[1]}):")
for i, col in enumerate(matches.columns):
    dtype = matches[col].dtype
    nulls = matches[col].isnull().sum()
    null_pct = nulls / len(matches) * 100
    nunique = matches[col].nunique()
    print(f"  {i+1:2d}. {col:30s} | {str(dtype):10s} | {null_pct:5.1f}% nulos | {nunique:4d} únicos")

print("\n🎯 3. FILTRANDO DADOS DA COMPETIÇÃO (2018 + 2022)...")

# Identificar coluna de ano
year_col = None
for col in ['Year', 'year', 'DATE', 'Date', 'date']:
    if col in matches.columns:
        year_col = col
        break

if year_col:
    matches_comp = matches[matches[year_col].isin([2018, 2022])].copy()
    print(f"  Ano detectado na coluna: {year_col}")
else:
    # Tentar extrair de outras colunas
    print("  ⚠️ Coluna de ano não detectada, verificando...")
    # Mostrar primeiras linhas para debug
    print(matches.head(3).to_string())
    matches_comp = matches.copy()

print(f"  Partidas filtradas: {len(matches_comp)}")
print(f"  Colunas: {len(matches_comp.columns)}")

print("\n📊 4. ANÁLISE BASELINE (Distribuição de Resultados)...")

# Detectar coluna de resultado
result_col = None
for col in ['Result', 'result', 'ResultID', 'result_id', 'FTR']:
    if col in matches_comp.columns:
        result_col = col
        break

if result_col:
    print(f"\n  Coluna de resultado: {result_col}")
    dist = matches_comp[result_col].value_counts(normalize=True) * 100
    for val, pct in dist.items():
        print(f"    {val}: {pct:.1f}%")
else:
    # Derivar resultado de gols
    home_score_col = None
    away_score_col = None
    for col in ['Home Team Goals', 'home_score', 'HomeGoals', 'home_goals']:
        if col in matches_comp.columns:
            home_score_col = col
            break
    for col in ['Away Team Goals', 'away_score', 'AwayGoals', 'away_goals']:
        if col in matches_comp.columns:
            away_score_col = col
            break
    
    if home_score_col and away_score_col:
        print(f"  Derivando resultado de: {home_score_col} vs {away_score_col}")
        matches_comp['result'] = np.where(
            matches_comp[home_score_col] > matches_comp[away_score_col], 'Home',
            np.where(matches_comp[home_score_col] < matches_comp[away_score_col], 'Away', 'Draw')
        )
        result_col = 'result'
        dist = matches_comp['result'].value_counts(normalize=True) * 100
        for val, pct in dist.items():
            label = {'Home': 'Mandante', 'Away': 'Visitante', 'Draw': 'Empate'}.get(val, val)
            print(f"    {label}: {pct:.1f}%")
    else:
        print("  ❌ Não foi possível detectar colunas de placar")

# ============================================================
# 5. ANÁLISE DE GOLS
# ============================================================
print("\n⚽ 5. ANÁLISE DE GOLS...")

if home_score_col and away_score_col:
    matches_comp['total_goals'] = matches_comp[home_score_col] + matches_comp[away_score_col]
    matches_comp['goal_diff'] = matches_comp[home_score_col] - matches_comp[away_score_col]
    
    print(f"  Média de gols por jogo: {matches_comp['total_goals'].mean():.2f}")
    print(f"  Máximo de gols em um jogo: {matches_comp['total_goals'].max()}")
    print(f"  Mediana: {matches_comp['total_goals'].median():.1f}")
    print(f"  Std: {matches_comp['total_goals'].std():.2f}")
    
    print(f"\n  Distribuição total de gols:")
    goals_dist = matches_comp['total_goals'].value_counts().sort_index()
    for g, c in goals_dist.items():
        bar = "█" * int(c / 2)
        print(f"    {int(g)} gols: {c:3d} ({c/len(matches_comp)*100:5.1f}%) {bar}")

# ============================================================
# 6. RANKING FIFA × RESULTADO
# ============================================================
print("\n🏆 6. CRUZAMENTO RANKING FIFA × RESULTADO...")

# Verificar colunas de ranking
print("  Colunas ranking_2022:", list(ranking_2022.columns[:8]))
print("  Colunas ranking_2026:", list(ranking_2026.columns[:8]))

# Detectar colunas relevantes do ranking
rank_team_col = None
rank_val_col = None
for col in ranking_2022.columns:
    if 'team' in col.lower() or 'country' in col.lower() or 'name' in col.lower():
        rank_team_col = col
        break
for col in ranking_2022.columns:
    if 'rank' in col.lower() and 'date' not in col.lower():
        rank_val_col = col
        break

if rank_team_col:
    print(f"  Coluna time no ranking: {rank_team_col}")
    print(f"  Coluna ranking: {rank_val_col}")
    print(f"\n  Top 10 rankings 2022:")
    top10 = ranking_2022.nsmallest(10, rank_val_col) if rank_val_col else ranking_2022.head(10)
    for _, row in top10.iterrows():
        print(f"    #{int(row[rank_val_col]) if rank_val_col else '?'} {row[rank_team_col]}")
else:
    print("  ⚠️ Não foi possível detectar coluna de time no ranking")
    print(f"  Colunas disponíveis: {list(ranking_2022.columns)}")

print("\n🔧 7. FEATURES DERIVADAS...")

# Criar features básicas
features_created = []

if result_col:
    matches_comp['result_encoded'] = matches_comp[result_col].map(
        {'Home': 1, 'Draw': 0, 'Away': -1} if matches_comp[result_col].isin(['Home']).any()
        else {'H': 1, 'D': 0, 'A': -1}
    )
    features_created.append('result_encoded')

if home_score_col and away_score_col:
    matches_comp['had_extra_time'] = 0  # Placeholder — precisa de coluna real
    matches_comp['had_penalties'] = 0   # Placeholder — precisa de coluna real
    features_created.extend(['total_goals', 'goal_diff', 'had_extra_time', 'had_penalties'])

print(f"  Features criadas: {features_created}")

# ============================================================
# 8. SALVAR DADOS PROCESSADOS
# ============================================================
print("\n💾 8. SALVANDO DADOS PROCESSADOS...")

output_path = PROCESSED / "matches_2018_2022_dia1.csv"
matches_comp.to_csv(output_path, index=False)
print(f"  Salvo: {output_path}")
print(f"  Shape: {matches_comp.shape}")

# ============================================================
# 9. RESUMO EXECUTIVO
# ============================================================
print("\n" + "=" * 70)
print("📋 RESUMO EXECUTIVO — DIA 1")
print("=" * 70)
print(f"  Dataset: {len(matches_comp)} partidas (2018 + 2022)")
print(f"  Features: {len(matches_comp.columns)} colunas ({len(features_created)} derivadas)")
if result_col:
    print(f"  Resultado: Mandante {dist.get('Home', 0):.1f}% | Empate {dist.get('Draw', 0):.1f}% | Visitante {dist.get('Away', 0):.1f}%")
print(f"  Arquivo: {output_path.name}")

print("\n🎯 Próximo passo: Dia 2 — EDA + Visualizações")
print("=" * 70)
