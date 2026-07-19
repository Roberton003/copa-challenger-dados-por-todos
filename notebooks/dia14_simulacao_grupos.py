#!/usr/bin/env python3
"""Dia 14 — Simulação Monte Carlo da fase de grupos 2026 + surpresas.

Consome só outputs/previsoes_copa_2026.csv (probabilidades já calculadas) e
data/raw/schedule_2026.csv (estrutura dos grupos, derivada por união de
partidas — o CSV não tem coluna de grupo explícita). Não retreina nada.

Formato real da Copa 2026: 12 grupos de 4, top-2 de cada grupo avança
automaticamente (24 times) + os 8 melhores terceiros colocados (8 times) =
32 times na fase eliminatória.

Limitação assumida: não modelamos placar, só resultado (Home/Draw/Away) —
critérios de desempate reais da FIFA (saldo de gols, gols marcados) não são
simuláveis aqui. Desempate aproximado por nº de vitórias simuladas, depois
aleatório. Documentado explicitamente, não escondido.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / 'data' / 'raw'
OUT = BASE / 'outputs'

N_SIMS = 20000
rng = np.random.default_rng(42)

previsoes = pd.read_csv(OUT / 'previsoes_copa_2026.csv')
schedule = pd.read_csv(RAW / 'schedule_2026.csv')
ranking_2026 = pd.read_csv(RAW / 'fifa_ranking_2026-06-08.csv')

NAME_ALIASES = {'United States': 'USA', 'Cape Verde': 'Cabo Verde',
                 'Bosnia-Herzegovina': 'Bosnia and Herzegovina'}
rank_dict = ranking_2026[['team', 'rank']].drop_duplicates('team', keep='last').set_index('team')['rank'].to_dict()
for alias, canonical in NAME_ALIASES.items():
    if canonical in rank_dict:
        rank_dict[alias] = rank_dict[canonical]

# ============================================================
# 1. Derivar os 12 grupos (união de times conectados por partida)
# ============================================================
parent = {}

def find(x):
    parent.setdefault(x, x)
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[ra] = rb

for _, r in schedule.iterrows():
    union(r['home_team'], r['away_team'])

teams_by_root = defaultdict(list)
for t in sorted(set(schedule['home_team']) | set(schedule['away_team'])):
    teams_by_root[find(t)].append(t)

groups = {f'Grupo {chr(65 + i)}': sorted(teams) for i, teams in enumerate(teams_by_root.values())}
team_group = {t: g for g, teams in groups.items() for t in teams}

print(f'{len(groups)} grupos derivados de {len(schedule)} partidas')

# ============================================================
# 2. Simulação Monte Carlo
# ============================================================
matches = previsoes[['home_team', 'away_team', 'pred_home_prob', 'pred_draw_prob', 'pred_away_prob']].values
all_teams = sorted(team_group.keys())
advance_count = defaultdict(int)
group_winner_count = defaultdict(int)

for _ in range(N_SIMS):
    points = defaultdict(int)
    wins = defaultdict(int)
    tiebreak = {t: rng.random() for t in all_teams}

    for home, away, p_home, p_draw, p_away in matches:
        outcome = rng.choice(['Home', 'Draw', 'Away'], p=[p_home, p_draw, p_away])
        if outcome == 'Home':
            points[home] += 3
            wins[home] += 1
        elif outcome == 'Away':
            points[away] += 3
            wins[away] += 1
        else:
            points[home] += 1
            points[away] += 1

    thirds = []
    for g, teams in groups.items():
        ranked = sorted(teams, key=lambda t: (points[t], wins[t], tiebreak[t]), reverse=True)
        advance_count[ranked[0]] += 1
        advance_count[ranked[1]] += 1
        group_winner_count[ranked[0]] += 1
        thirds.append((ranked[2], points[ranked[2]], wins[ranked[2]], tiebreak[ranked[2]]))

    best_thirds = sorted(thirds, key=lambda x: (x[1], x[2], x[3]), reverse=True)[:8]
    for t, *_ in best_thirds:
        advance_count[t] += 1

advance_prob = {t: advance_count[t] / N_SIMS for t in all_teams}
winner_prob = {t: group_winner_count[t] / N_SIMS for t in all_teams}

print('\nTop 10 favoritos a avançar:')
for t, p in sorted(advance_prob.items(), key=lambda x: -x[1])[:10]:
    print(f'  {t:<20} {p:.1%}  (grupo {team_group[t]})')

# ============================================================
# 3. Possíveis surpresas — grande gap de ranking, mas o modelo não vê favoritismo claro
# (inclui as inversões puras — favorito do ranking com probabilidade menor — como caso extremo)
# ============================================================
RANK_GAP_MIN = 20  # só considera pares com diferença de ranking relevante

surpresas = []
for home, away, p_home, p_draw, p_away in matches:
    rk_home, rk_away = rank_dict.get(home), rank_dict.get(away)
    if rk_home is None or rk_away is None:
        continue
    rank_gap = abs(rk_home - rk_away)
    if rank_gap < RANK_GAP_MIN:
        continue
    favorito_ranking = home if rk_home < rk_away else away
    p_favorito = p_home if favorito_ranking == home else p_away
    azarao = away if favorito_ranking == home else home
    p_azarao = p_away if favorito_ranking == home else p_home
    margem = p_favorito - p_azarao  # negativa = o modelo inverteu o favoritismo do ranking
    surpresas.append({
        'home_team': home, 'away_team': away,
        'favorito_ranking': favorito_ranking, 'rank_favorito': int(rank_dict[favorito_ranking]),
        'azarao': azarao, 'rank_azarao': int(rank_dict[azarao]),
        'rank_gap': int(rank_gap),
        'prob_favorito_vencer': float(p_favorito), 'prob_azarao_vencer': float(p_azarao),
        'margem': float(margem),
    })

surpresas.sort(key=lambda x: x['margem'])
surpresas = surpresas[:10]
print(f'\nTop {len(surpresas)} possíveis surpresas (grande gap de ranking, margem do modelo estreita ou invertida):')
for s in surpresas:
    tag = 'INVERSÃO' if s['margem'] < 0 else 'zebra em aberto'
    print(f"  [{tag}] {s['azarao']} (#{s['rank_azarao']}) vs {s['favorito_ranking']} (#{s['rank_favorito']}, "
          f"gap {s['rank_gap']}) — {s['prob_azarao_vencer']:.1%} vs {s['prob_favorito_vencer']:.1%}")

# ============================================================
# 4. Salvar
# ============================================================
out = {
    'n_sims': N_SIMS,
    'groups': groups,
    'advance_prob': {t: advance_prob[t] for t in all_teams},
    'group_winner_prob': {t: winner_prob[t] for t in all_teams},
    'team_group': team_group,
    'surpresas': surpresas,
    'metodologia': (
        'Monte Carlo (20000 simulações) amostrando resultado (Home/Draw/Away) de cada '
        'partida a partir das probabilidades do modelo final (dia10). Desempate aproximado '
        'por nº de vitórias simuladas + valor aleatório (não modelamos placar/saldo de gols). '
        'Top-2 de cada grupo avança automaticamente + 8 melhores terceiros (formato real 2026).'
    ),
}
json.dump(out, open(OUT / 'dia14_simulacao_grupos.json', 'w'), indent=2, ensure_ascii=False)
print(f"\nResultados salvos: {OUT / 'dia14_simulacao_grupos.json'}")

def _selftest():
    assert len(groups) == 12 and all(len(v) == 4 for v in groups.values())
    assert all(0.0 <= p <= 1.0 for p in advance_prob.values())
    assert abs(sum(winner_prob[t] for t in all_teams) - 12.0) < 0.05
    print('selftest OK')

if __name__ == '__main__':
    _selftest()
