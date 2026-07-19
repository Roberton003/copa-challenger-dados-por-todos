#!/usr/bin/env python3
"""
Dia 2 — EDA + Visualizações
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "data" / "raw"
PROCESSED = BASE / "data" / "processed"
OUTPUT = BASE / "outputs" / "visualizations"
OUTPUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.size': 11,
    'figure.dpi': 150,
})

COLORS = {
    'home': '#2196F3',    # azul — mandante
    'draw': '#9E9E9E',    # cinza — empate (neutro, sem destaque)
    'away': '#F44336',    # vermelho — visitante
    'highlight': '#FF9800', # laranja — destaque principal
    'bg': '#FAFAFA',
}

print("=" * 70)
print("DIA 2 — EDA + VISUALIZAÇÕES")
print("=" * 70)

# ============================================================
# CARREGAR DADOS
# ============================================================
matches = pd.read_csv(RAW / "matches_1930_2022.csv")
ranking_2022 = pd.read_csv(RAW / "fifa_ranking_2022-10-06.csv")
ranking_2026 = pd.read_csv(RAW / "fifa_ranking_2026-06-08.csv")

# Filtrar 2018+2022
df = matches[matches['Year'].isin([2018, 2022])].copy()
df['result'] = np.where(
    df['home_score'] > df['away_score'], 'Home',
    np.where(df['home_score'] < df['away_score'], 'Away', 'Draw')
)
df['total_goals'] = df['home_score'] + df['away_score']

# Ranking
rank_dict = dict(zip(ranking_2022['team'], ranking_2022['rank']))
df['home_rank'] = df['home_team'].map(rank_dict)
df['away_rank'] = df['away_team'].map(rank_dict)
df['rank_diff'] = df['home_rank'] - df['away_rank']

print(f"✅ Dados carregados: {len(df)} partidas (2018+2022)")

# ============================================================
# GRÁFICO 1: Distribuição de Resultados (da_004 — bar plot categórico)
# ============================================================
print("\n📊 1. Distribuição de Resultados...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 1a: Proporção geral
result_counts = df['result'].value_counts()
colors = [COLORS['home'], COLORS['away'], COLORS['draw']]
bars = axes[0].bar(result_counts.index, result_counts.values, color=colors, edgecolor='white', linewidth=1.5)
axes[0].set_title('Distribuição de Resultados\n(2018 + 2022)', fontweight='bold', fontsize=13)
axes[0].set_ylabel('Número de Partidas')
for bar, val in zip(bars, result_counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{val}\n({val/len(df)*100:.1f}%)', ha='center', va='bottom', fontweight='bold')

# 1b: Por edição
for year in [2018, 2022]:
    subset = df[df['Year'] == year]
    result_pct = subset['result'].value_counts(normalize=True) * 100
    x_pos = [0, 1, 2] if year == 2018 else [0.25, 1.25, 2.25]
    width = 0.22
    for i, (res, pct) in enumerate(result_pct.items()):
        color = COLORS[res.lower()]
        axes[1].bar(x_pos[i] - width/2 if year == 2018 else x_pos[i] + width/2,
                    pct, width=width, color=color, alpha=0.7 if year == 2018 else 1.0,
                    label=f'{res} ({year})' if i == 0 else None)

axes[1].set_title('Resultado por Edição', fontweight='bold', fontsize=13)
axes[1].set_ylabel('% dos Jogos')
axes[1].set_xticks([0.125, 1.125, 2.125])
axes[1].set_xticklabels(['Home', 'Away', 'Draw'])
axes[1].legend()

plt.tight_layout()
plt.savefig(OUTPUT / '01_distribuicao_resultados.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ Salvo: outputs/visualizations/01_distribuicao_resultados.png")

# ============================================================
# GRÁFICO 2: Gols por Partida (da_004 — histograma univariado)
# ============================================================
print("\n📊 2. Distribuição de Gols por Partida...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 2a: Histograma de total de gols
axes[0].hist(df['total_goals'], bins=range(0, 10), color=COLORS['highlight'],
             edgecolor='white', linewidth=1.5, alpha=0.85)
axes[0].axvline(df['total_goals'].mean(), color='red', linestyle='--', linewidth=2,
                label=f'Média: {df["total_goals"].mean():.2f}')
axes[0].set_title('Total de Gols por Partida', fontweight='bold', fontsize=13)
axes[0].set_xlabel('Gols')
axes[0].set_ylabel('Frequência')
axes[0].legend()

# 2b: Gols mandante vs visitante
axes[1].hist(df['home_score'], bins=range(0, 7), color=COLORS['home'],
             edgecolor='white', linewidth=1.5, alpha=0.7, label='Mandante')
axes[1].hist(df['away_score'], bins=range(0, 7), color=COLORS['away'],
             edgecolor='white', linewidth=1.5, alpha=0.7, label='Visitante')
axes[1].set_title('Gols: Mandante vs Visitante', fontweight='bold', fontsize=13)
axes[1].set_xlabel('Gols Marcados')
axes[1].set_ylabel('Frequência')
axes[1].legend()

plt.tight_layout()
plt.savefig(OUTPUT / '02_distribuicao_gols.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ Salvo: outputs/visualizations/02_distribuicao_gols.png")

print("\n📊 3. Ranking FIFA × Resultado...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 3a: rank_diff vs resultado
bins = [-100, -10, -5, 0, 5, 10, 100]
labels = ['< -10', '-10 a -5', '-5 a 0', '0 a 5', '5 a 10', '> 10']
df['rank_bin'] = pd.cut(df['rank_diff'], bins=bins, labels=labels)

home_pcts = []
draw_pcts = []
away_pcts = []
counts = []
for label in labels:
    subset = df[df['rank_bin'] == label]
    counts.append(len(subset))
    home_pcts.append((subset['result'] == 'Home').mean() * 100)
    draw_pcts.append((subset['result'] == 'Draw').mean() * 100)
    away_pcts.append((subset['result'] == 'Away').mean() * 100)

x = np.arange(len(labels))
width = 0.25
bars1 = axes[0].bar(x - width, home_pcts, width, color=COLORS['home'], label='Mandante')
bars2 = axes[0].bar(x, draw_pcts, width, color=COLORS['draw'], label='Empate')
bars3 = axes[0].bar(x + width, away_pcts, width, color=COLORS['away'], label='Visitante')

axes[0].set_title('Diferença de Ranking FIFA × Resultado\n(rank_diff = home_rank - away_rank)',
                   fontweight='bold', fontsize=12)
axes[0].set_xlabel('Faixa de rank_diff (negativo = mandante melhor ranqueado)')
axes[0].set_ylabel('% Vitórias')
axes[0].set_xticks(x)
axes[0].set_xticklabels(labels, fontsize=9)
axes[0].legend()

bars1[0].set_edgecolor(COLORS['highlight'])
bars1[0].set_linewidth(3)
axes[0].annotate('Mandante domina\nquando muito\nmelhor ranqueado',
                 xy=(0 - width, home_pcts[0]),
                 xytext=(1.5, home_pcts[0] + 5),
                 arrowprops=dict(arrowstyle='->', color=COLORS['highlight']),
                 fontsize=9, fontweight='bold', color=COLORS['highlight'])

# 3b: Scatter rank_diff vs total_goals
colors_scatter = df['result'].map({'Home': COLORS['home'], 'Draw': COLORS['draw'], 'Away': COLORS['away']})
axes[1].scatter(df['rank_diff'], df['total_goals'], c=colors_scatter, alpha=0.6, s=50, edgecolors='white')
axes[1].axvline(0, color='gray', linestyle='--', alpha=0.5)
axes[1].set_title('Ranking Diff vs Total de Gols', fontweight='bold', fontsize=13)
axes[1].set_xlabel('rank_diff (negativo = mandante melhor)')
axes[1].set_ylabel('Total de Gols')

# Legenda manual
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['home'], markersize=8, label='Mandante vence'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['draw'], markersize=8, label='Empate'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['away'], markersize=8, label='Visitante vence'),
]
axes[1].legend(handles=legend_elements)

plt.tight_layout()
plt.savefig(OUTPUT / '03_ranking_vs_resultado.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ Salvo: outputs/visualizations/03_ranking_vs_resultado.png")

# ============================================================
# GRÁFICO 4: Top Times (da_004 — bar plot horizontal, ordenado)
# ============================================================
print("\n📊 4. Top Times por Desempenho...")

home_wins = df[df['result'] == 'Home']['home_team'].value_counts()
away_wins = df[df['result'] == 'Away']['away_team'].value_counts()
total_wins = home_wins.add(away_wins, fill_value=0).sort_values(ascending=True).tail(15)

fig, ax = plt.subplots(figsize=(10, 7))
colors_bar = [COLORS['highlight'] if v == total_wins.max() else '#607D8B' for v in total_wins.values]
bars = ax.barh(total_wins.index, total_wins.values, color=colors_bar, edgecolor='white', linewidth=1.5)

ax.barh(total_wins.index[-1], total_wins.values[-1], color=COLORS['highlight'], edgecolor='white', linewidth=2)

for bar, val in zip(bars, total_wins.values):
    rank = rank_dict.get(bar.get_y() + bar.get_height()/2, '?')
    ax.text(val + 0.1, bar.get_y() + bar.get_height()/2,
            f'{int(val)} vitórias', va='center', fontsize=10)

ax.set_title('Top 15 Times — Mais Vitórias em Copas (2018 + 2022)\n(cor laranja = mais vitórias)',
             fontweight='bold', fontsize=13)
ax.set_xlabel('Número de Vitórias')
plt.tight_layout()
plt.savefig(OUTPUT / '04_top_times_vitorias.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ Salvo: outputs/visualizations/04_top_times_vitorias.png")

# ============================================================
# GRÁFICO 5: Gols por Fase da Copa (da_004 — box plot)
# ============================================================
print("\n📊 5. Gols por Fase da Copa...")

phase_order = ['Group stage', 'Round of 16', 'Quarter-finals', 'Semi-finals',
               'Third-place match', 'Final']
existing_phases = [p for p in phase_order if p in df['Round'].values]
phase_data = [df[df['Round'] == p]['total_goals'].values for p in existing_phases]

fig, ax = plt.subplots(figsize=(10, 6))
bp = ax.boxplot(phase_data, labels=[p.replace(' stage', '').replace('match', '\n3rd') for p in existing_phases],
                patch_artist=True, notch=True)

colors_phase = ['#E3F2FD', '#BBDEFB', '#90CAF9', '#64B5F6', '#42A5F5', '#1E88E5']
for patch, color in zip(bp['boxes'], colors_phase[:len(bp['boxes'])]):
    patch.set_facecolor(color)
    patch.set_edgecolor('#1565C0')

ax.set_title('Distribuição de Gols por Fase da Copa\n(box plot — mediana e dispersão)',
             fontweight='bold', fontsize=13)
ax.set_ylabel('Total de Gols')
ax.set_xlabel('Fase')
plt.tight_layout()
plt.savefig(OUTPUT / '05_gols_por_fase.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ Salvo: outputs/visualizations/05_gols_por_fase.png")

# ============================================================
# GRÁFICO 6: xG Analysis (se disponível)
# ============================================================
print("\n📊 6. Análise de xG...")

if 'home_xg' in df.columns and df['home_xg'].notna().sum() > 0:
    xg_df = df[df['home_xg'].notna() & df['away_xg'].notna()].copy()
    print(f"  Partidas com xG: {len(xg_df)}")

    if len(xg_df) > 5:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 6a: xG vs Gols Reais
        axes[0].scatter(xg_df['home_xg'], xg_df['home_score'], alpha=0.5, color=COLORS['home'], label='Mandante')
        axes[0].scatter(xg_df['away_xg'], xg_df['away_score'], alpha=0.5, color=COLORS['away'], label='Visitante')
        max_val = max(xg_df['home_xg'].max(), xg_df['away_xg'].max(), xg_df['home_score'].max(), xg_df['away_score'].max())
        axes[0].plot([0, max_val + 0.5], [0, max_val + 0.5], 'k--', alpha=0.3, label='xG = Gols')
        axes[0].set_title('xG vs Gols Reais', fontweight='bold', fontsize=13)
        axes[0].set_xlabel('xG (Expected Goals)')
        axes[0].set_ylabel('Gols Reais')
        axes[0].legend()

        # 6b: Distribuição de xG
        axes[1].hist(xg_df['home_xg'], bins=15, alpha=0.6, color=COLORS['home'], label='Mandante xG')
        axes[1].hist(xg_df['away_xg'], bins=15, alpha=0.6, color=COLORS['away'], label='Visitante xG')
        axes[1].set_title('Distribuição de xG', fontweight='bold', fontsize=13)
        axes[1].set_xlabel('xG')
        axes[1].set_ylabel('Frequência')
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(OUTPUT / '06_xg_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Salvo: outputs/visualizations/06_xg_analysis.png")
    else:
        print(f"  ⚠️ Dados xG insuficientes ({len(xg_df)} partidas)")
else:
    print(f"  ⚠️ xG não disponível ou todos nulos")

# ============================================================
# RESUMO ESTATÍSTICO
# ============================================================
print("\n" + "=" * 70)
print("📊 RESUMO ESTATÍSTICO DO DIA 2")
print("=" * 70)
print(f"\nTotal de partidas: {len(df)}")
print(f"2018: {len(df[df['Year']==2018])} | 2022: {len(df[df['Year']==2022])}")
print(f"\nResultado:")
print(f"  Mandante: {(df['result']=='Home').mean()*100:.1f}%")
print(f"  Empate:   {(df['result']=='Draw').mean()*100:.1f}%")
print(f"  Visitante: {(df['result']=='Away').mean()*100:.1f}%")
print(f"\nGols:")
print(f"  Média por jogo: {df['total_goals'].mean():.2f}")
print(f"  Mediana: {df['total_goals'].median():.0f}")
print(f"  Máximo: {df['total_goals'].max()}")
print(f"\nRanking:")
print(f"  Correlação rank_diff × resultado mandante: {df['rank_diff'].corr((df['result']=='Home').astype(int)):.3f}")
print(f"\nVisualizações geradas: 6 arquivos em outputs/visualizations/")
print("\n✅ Dia 2 — EDA + Visualizações concluído!")
print("=" * 70)
