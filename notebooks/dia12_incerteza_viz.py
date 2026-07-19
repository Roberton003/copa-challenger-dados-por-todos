#!/usr/bin/env python3
"""Dia 12 — Visualização de incerteza (communication-gap-review).

A seção "Limitações Estatísticas" do README/RELATORIO_FINAL já afirma em texto
que as diferenças entre modelos cabem na margem de erro (±12pp, n=64) e que as
previsões 2026 têm baixa confiança (Draw quase nunca vence o argmax). Nenhuma
visualização do projeto mostra isso — só EDA. Este script fecha essa lacuna
com 2 gráficos, reusando números já computados (não recalcula modelos).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / 'outputs' / 'visualizations'

# ---- 1. Comparação de modelos com IC 95% (n=64 no teste) ----
# Valores da tabela "Modelos Avaliados" do README (já reportados, não recalculados aqui).
modelos = {
    'XGBoost\n(calibrado)': 0.625,
    'LightGBM\notimizado': 0.562,
    'LogReg\ncalibrado': 0.531,
    'LogReg\nbaseline': 0.516,
    'LightGBM\nbaseline': 0.500,
    'XGBoost\nbaseline': 0.453,
}
n_teste = 64
z = 1.96
nomes = list(modelos.keys())
acc = np.array(list(modelos.values()))
erro = z * np.sqrt(acc * (1 - acc) / n_teste)

fig, ax = plt.subplots(figsize=(9, 5.5))
cores = ['#2e7d32' if a == acc.max() else '#5b7fa6' for a in acc]
ax.bar(nomes, acc * 100, yerr=erro * 100, capsize=6, color=cores, alpha=0.85)
ax.axhline(33.3, color='crimson', linestyle='--', linewidth=1.2, label='Chance (33.3%)')
ax.set_ylabel('Acurácia (%)')
ax.set_title('Comparação de modelos com IC 95% (n=64) — diferenças cabem no erro')
ax.legend()
ax.set_ylim(0, 85)
plt.tight_layout()
plt.savefig(OUT / '07_incerteza_modelos.png', dpi=120)
plt.close()
print(f"Salvo: {OUT / '07_incerteza_modelos.png'}")

# ---- 2. Distribuição da probabilidade máxima nas previsões 2026 ----
pred = pd.read_csv(BASE / 'outputs' / 'previsoes_copa_2026.csv')
max_prob = pred[['pred_home_prob', 'pred_draw_prob', 'pred_away_prob']].max(axis=1)

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(max_prob * 100, bins=15, color='#5b7fa6', edgecolor='white')
ax.axvline(33.3, color='crimson', linestyle='--', linewidth=1.2, label='Chance (33.3%)')
ax.axvline(max_prob.median() * 100, color='#2e7d32', linestyle='-', linewidth=1.2,
           label=f'Mediana ({max_prob.median()*100:.1f}%)')
ax.set_xlabel('Probabilidade máxima prevista por jogo (%)')
ax.set_ylabel('Nº de jogos (de 72)')
ax.set_title('Confiança das previsões 2026 — quão longe do "chute" de 33%')
ax.legend()
plt.tight_layout()
plt.savefig(OUT / '08_confianca_previsoes_2026.png', dpi=120)
plt.close()
print(f"Salvo: {OUT / '08_confianca_previsoes_2026.png'}")

# self-check (ponytail: menor coisa que quebra se a lógica falhar)
assert (erro > 0).all(), "margem de erro deveria ser positiva para todo modelo"
assert 0 <= max_prob.min() and max_prob.max() <= 1, "probabilidade fora de [0,1]"
print("OK — 2 visualizações de incerteza geradas.")
