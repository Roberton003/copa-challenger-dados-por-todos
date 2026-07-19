#!/usr/bin/env python3
"""
Dia 1 — Parte 2: Cruzamento Ranking × Resultados + Schedule 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "data" / "raw"
PROCESSED = BASE / "data" / "processed"

print("=" * 70)
print("DIA 1 — PARTE 2: RANKING × RESULTADOS + SCHEDULE 2026")
print("=" * 70)

# Carregar dados
matches = pd.read_csv(RAW / "matches_1930_2022.csv")
ranking_2022 = pd.read_csv(RAW / "fifa_ranking_2022-10-06.csv")
ranking_2026 = pd.read_csv(RAW / "fifa_ranking_2026-06-08.csv")
schedule_2026 = pd.read_csv(RAW / "schedule_2026.csv")

# Filtrar 2018+2022
matches_comp = matches[matches['Year'].isin([2018, 2022])].copy()

# Derivar resultado
matches_comp['result'] = np.where(
    matches_comp['home_score'] > matches_comp['away_score'], 'Home',
    np.where(matches_comp['home_score'] < matches_comp['away_score'], 'Away', 'Draw')
)
matches_comp['total_goals'] = matches_comp['home_score'] + matches_comp['away_score']

# ============================================================
# 1. MAPEAR TIMES DO RANKING COM OS TIMES DAS COPAS
# ============================================================
print("\n🔍 1. MAPEAMENTO DE TIMES...")
print("\nTimes únicos nas partidas 2018+2022:")
home_teams = set(matches_comp['home_team'].unique())
away_teams = set(matches_comp['away_team'].unique())
all_teams = home_teams | away_teams
print(f"  Total: {len(all_teams)} times")

print("\nTimes no ranking 2022:")
ranking_teams = set(ranking_2022['team'].unique())
print(f"  Total: {len(ranking_teams)} times")

# Encontrar matches
matched = all_teams & ranking_teams
unmatched = all_teams - ranking_teams
print(f"\n  Matched: {len(matched)} times")
print(f"  Unmatched: {len(unmatched)} times")
if unmatched:
    print(f"  Unmatched teams: {sorted(unmatched)}")

# ============================================================
# 2. RANKING × RESULTADO
# ============================================================
print("\n📊 2. RANKING FIFA × RESULTADO (2018 + 2022)...")

# Criar dicionário de ranking
rank_dict = dict(zip(ranking_2022['team'], ranking_2022['rank']))

# Adicionar ranking ao dataframe
matches_comp['home_rank'] = matches_comp['home_team'].map(rank_dict)
matches_comp['away_rank'] = matches_comp['away_team'].map(rank_dict)
matches_comp['rank_diff'] = matches_comp['home_rank'] - matches_comp['away_rank']

# Analisar por faixa de rank_diff
print("\nFaixa rank_diff vs resultado:")
bins = [-100, -10, -5, 0, 5, 10, 100]
labels = ['< -10', '-10 a -5', '-5 a 0', '0 a 5', '5 a 10', '> 10']
matches_comp['rank_diff_bin'] = pd.cut(matches_comp['rank_diff'], bins=bins, labels=labels)

for label in labels:
    subset = matches_comp[matches_comp['rank_diff_bin'] == label]
    if len(subset) > 0:
        home_pct = (subset['result'] == 'Home').mean() * 100
        draw_pct = (subset['result'] == 'Draw').mean() * 100
        away_pct = (subset['result'] == 'Away').mean() * 100
        print(f"  {label:10s}: {len(subset):3d} jogos | Mand {home_pct:5.1f}% | Emp {draw_pct:5.1f}% | Visit {away_pct:5.1f}%")

# ============================================================
# 3. TOP TIMES POR DESEMPENHO
# ============================================================
print("\n🏆 3. TOP TIMES POR DESEMPENHO (2018 + 2022)...")

# Contar vitórias
home_wins = matches_comp[matches_comp['result'] == 'Home']['home_team'].value_counts()
away_wins = matches_comp[matches_comp['result'] == 'Away']['away_team'].value_counts()

total_wins = home_wins.add(away_wins, fill_value=0).sort_values(ascending=False)
print("\nTop 15 times com mais vitórias:")
for team, wins in total_wins.head(15).items():
    rank = rank_dict.get(team, '?')
    print(f"  {team:25s}: {int(wins):2.0f} vitórias | Ranking FIFA: #{rank}")

# ============================================================
# 4. SCHEDULE 2026 — TIMES PARTICIPANTES
# ============================================================
print("\n📅 4. SCHEDULE 2026 — ANÁLISE DOS GRUPOS...")

# Detectar colunas do schedule
print(f"  Colunas: {list(schedule_2026.columns)}")
print(f"  Shape: {schedule_2026.shape}")
print(f"\nPrimeiras 10 linhas:")
print(schedule_2026.head(10).to_string())

# Extrair times únicos do schedule 2026
schedule_teams = set()
for col in schedule_2026.columns:
    if 'team' in col.lower() or 'home' in col.lower() or 'away' in col.lower():
        vals = schedule_2026[col].dropna().unique()
        schedule_teams.update(vals)

if schedule_teams:
    print(f"\nTimes únicos no schedule 2026: {len(schedule_teams)}")
    # Verificar quais estão no ranking 2026
    rank_2026_dict = dict(zip(ranking_2026['team'], ranking_2026['rank']))
    for team in sorted(schedule_teams):
        rank = rank_2026_dict.get(team, 'Não ranqueado')
        print(f"  {team:30s} | Ranking: #{rank}")

# ============================================================
# 5. SALVAR CRUZAMENTO RANKING
# ============================================================
output_path = PROCESSED / "matches_ranking_2018_2022.csv"
matches_comp.to_csv(output_path, index=False)
print(f"\n💾 Salvo: {output_path}")
print(f"  Shape: {matches_comp.shape}")

# ============================================================
# 6. ESTATÍSTICAS POR FASE
# ============================================================
print("\n📊 6. ESTATÍSTICAS POR FASE DA COPA...")

if 'Round' in matches_comp.columns:
    for phase in matches_comp['Round'].unique():
        subset = matches_comp[matches_comp['Round'] == phase]
        home_pct = (subset['result'] == 'Home').mean() * 100
        draw_pct = (subset['result'] == 'Draw').mean() * 100
        away_pct = (subset['result'] == 'Away').mean() * 100
        avg_goals = subset['total_goals'].mean()
        print(f"  {phase:25s}: {len(subset):2d} jogos | Mand {home_pct:5.1f}% | Emp {draw_pct:5.1f}% | Visit {away_pct:5.1f}% | Gols {avg_goals:.1f}")

print("\n✅ Dia 1 — Parte 2 concluída!")
print("=" * 70)
