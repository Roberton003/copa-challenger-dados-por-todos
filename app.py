"""Copa Challenger — Dashboard (Missão 3: Dashboard e Storytelling).

Consome apenas resultados já calculados (outputs/*.csv, outputs/*.json) —
não recalcula modelos. Rodar: streamlit run app.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BASE = Path(__file__).parent
OUT = BASE / 'outputs'

st.set_page_config(page_title='Copa Challenger — Dashboard', page_icon='⚽', layout='wide')

st.markdown("""
<style>
[data-testid="stMetric"] {
    background: #1B222C;
    border: 1px solid #2A3341;
    border-radius: 10px;
    padding: 14px 16px;
}
[data-testid="stMetricLabel"] { font-size: 0.85rem; opacity: 0.75; }
.hero {
    background: linear-gradient(135deg, #00A859 0%, #0E1117 100%);
    border-radius: 14px;
    padding: 28px 32px;
    margin-bottom: 24px;
}
.hero h1 { margin: 0; font-size: 2rem; }
.hero p { margin: 6px 0 0 0; opacity: 0.9; }
</style>
""", unsafe_allow_html=True)

previsoes = pd.read_csv(OUT / 'previsoes_copa_2026.csv')
rps_info = json.load(open(OUT / 'dia10_recalibracao_rps.json'))
pytorch_info = json.load(open(OUT / 'dia13_pytorch_comparativo.json'))
sim_info = json.load(open(OUT / 'dia14_simulacao_grupos.json'))
pareado_info = json.load(open(OUT / 'dia15_teste_pareado.json'))
sensibilidade_info = json.load(open(OUT / 'dia16_sensibilidade_vazamento.json'))
split_info = json.load(open(OUT / 'dia17_split_invertido.json'))
ensemble_info = json.load(open(OUT / 'dia18_ensemble_probabilidades.json'))
poisson_info = json.load(open(OUT / 'dia19_poisson_gols.json'))

# MODELOS vem do dia10 (pós-correção do bug de nomes de seleção, seleção por RPS) —
# não do dia6_optimization_results.json (XGBoost 60.9%), que é um pipeline anterior
# à correção e não é o modelo que gera outputs/previsoes_copa_2026.csv.
MODELOS = {k: v['accuracy'] for k, v in rps_info['calibration_comparison'].items()}
N_TESTE = pareado_info['n_teste']
Z = 1.96

st.markdown("""
<div class="hero">
  <h1>⚽ Copa Challenger — Predição Copa do Mundo 2026</h1>
  <p>Treinado em Copas 2018+2022 (128 partidas) · dataset oficial da competição, sem dados externos</p>
</div>
""", unsafe_allow_html=True)

max_prob = previsoes[['pred_home_prob', 'pred_draw_prob', 'pred_away_prob']].max(axis=1)
melhor_modelo = max(MODELOS, key=MODELOS.get)

c1, c2, c3, c4 = st.columns(4)
c1.metric('Melhor modelo (teste 2022)', melhor_modelo, f'{MODELOS[melhor_modelo]:.1%}')
c2.metric('Jogos previstos (2026)', len(previsoes))
c3.metric('Confiança média das previsões', f'{max_prob.mean():.1%}', 'vs. 33.3% chance')
c4.metric('Empates previstos', int((previsoes['prediction'] == 'Draw').sum()), f"de {len(previsoes)} jogos")

tab_modelos, tab_previsoes, tab_simulacao, tab_corrida, tab_limitacoes = st.tabs(
    ['📊 Modelos', '🔮 Previsões 2026', '🎲 Simulação de Grupos', '🏁 Corrida ao Vivo', '⚠️ Limitações'])

with tab_modelos:
    st.subheader('Comparação de modelos (n=64 no teste — IC 95%)')
    st.caption('As diferenças entre modelos cabem majoritariamente na margem de erro; só a comparação contra o acaso (33.3%) é estatisticamente defensável.')

    nomes = list(MODELOS.keys())
    acc = np.array(list(MODELOS.values()))
    erro = Z * np.sqrt(acc * (1 - acc) / N_TESTE)

    fig = go.Figure(go.Bar(
        x=acc, y=nomes, orientation='h',
        error_x=dict(type='data', array=erro, color='#FAFAFA', thickness=1.5, width=4),
        marker_color=['#00A859' if a == acc.max() else '#3D6FB4' for a in acc],
    ))
    fig.add_trace(go.Scatter(
        x=acc + erro + 0.015, y=nomes, mode='text',
        text=[f'{a:.1%}' for a in acc], textposition='middle right', textfont_color='#FAFAFA',
        showlegend=False, hoverinfo='skip',
    ))
    fig.add_vline(x=1 / 3, line_dash='dash', line_color='crimson',
                  annotation_text='Chance (33.3%)', annotation_font_color='#FAFAFA')
    fig.update_layout(
        xaxis=dict(tickformat='.0%', range=[0, float((acc + erro).max()) + 0.15], gridcolor='#2A3341'),
        yaxis=dict(autorange='reversed'),
        height=380, margin=dict(l=10, r=60, t=10, b=10), showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#FAFAFA',
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander('As diferenças entre modelos são estatisticamente reais? (teste pareado)'):
        st.markdown(f"""
O IC 95% acima é univariado (cada modelo isolado) — não prova nem descarta diferença entre modelos,
porque os erros são correlacionados (testados nos mesmos {pareado_info['n_teste']} jogos). Rodamos
McNemar exato e bootstrap pareado sobre as mesmas previsões para responder isso de verdade
(`notebooks/dia15_teste_pareado.py`).

**Melhor por acurácia neste teste: {pareado_info['best_model']}** ({pareado_info['accuracies'][pareado_info['best_model']]:.1%})
""")
        pareado_df = pd.DataFrame([
            {
                'Comparação': f"{pareado_info['best_model']} vs {nome}",
                'Discordantes': mc['discordantes'],
                'p (McNemar)': mc['p_value'],
                'Δ acurácia (bootstrap)': bs['delta_mean'],
                'IC 95% do Δ': f"[{bs['ci_95_low']:+.1%}, {bs['ci_95_high']:+.1%}]",
                'Significativo?': 'Não' if bs['ci_contains_zero'] else 'Sim',
            }
            for nome, mc in pareado_info['mcnemar'].items()
            for bs in [pareado_info['bootstrap_delta_acc'][nome]]
        ])
        st.dataframe(
            pareado_df.style.format({'p (McNemar)': '{:.3f}', 'Δ acurácia (bootstrap)': '{:+.1%}'}),
            hide_index=True, use_container_width=True,
        )
        st.caption('Nenhuma comparação é significativa a 5% (IC do delta sempre cobre 0) — com n=64, '
                   'o teste pareado confirma que as diferenças entre modelos não são estatisticamente '
                   'distinguíveis, mesmo olhando erro a erro em vez de IC isolado.')

    with st.expander('O quanto o resultado depende do vazamento do ranking? (teste de sensibilidade)'):
        sens = sensibilidade_info
        afetadas = sens['features_mais_afetadas']
        st.markdown(f"""
Todas as 5 features usadas vêm de **um único snapshot** de ranking FIFA (out/2022), aplicado tanto
ao treino (2018) quanto ao teste (2022) — vazamento temporal documentado, mas nunca antes testado
quanto ao impacto real no desempenho. `notebooks/dia16_sensibilidade_vazamento.py` mede o gap de
correlação treino-vs-teste por feature (quanto maior o gap, mais "contaminada" pelo vazamento) e
retreina os 6 combos removendo as 2 features com maior gap: **{afetadas[0]}** e **{afetadas[1]}**.
""")
        colS1, colS2 = st.columns(2)
        colS1.metric(f"{pareado_info['best_model']} — completo (5 features)",
                     f"{sens['resultados_completo'][pareado_info['best_model']]['accuracy']:.1%}")
        colS2.metric(f"{pareado_info['best_model']} — sem {afetadas[0]}/{afetadas[1]}",
                     f"{sens['resultados_reduzido'][pareado_info['best_model']]['accuracy']:.1%}",
                     f"{sens['deltas'][pareado_info['best_model']]['delta_accuracy']:+.1%}")
        st.markdown(f"""
**Δ médio de acurácia nos 6 combos ao remover as features mais vazadas: {sens['delta_medio_acuracia']:+.1%}**
— próximo de zero (e positivo para o melhor modelo). O desempenho reportado **não depende** do sinal
inflado pelo vazamento: tirar as duas features mais contaminadas não piora o modelo, às vezes até
melhora. Isso não corrige o vazamento (segue documentado em §9.4c), mas mostra que ele não é o
motor do resultado.
""")

    with st.expander('O split treino/teste é robusto? (split invertido + 5-fold)'):
        sp = split_info
        st.markdown("""
Todo o pipeline usa um único split fixo (treino=2018, teste=2022). Com só 128 partidas, isso é
uma amostra de validação, não uma verdade estável — testamos invertendo o split (treino=2022,
teste=2018) e rodando 5-fold estratificado sobre as 128 partidas combinadas
(`notebooks/dia17_split_invertido.py`).
""")
        split_df = pd.DataFrame([
            {
                'Modelo': nome,
                'Original (treino 2018→teste 2022)': sp['split_original_treino2018_teste2022'][nome]['accuracy'],
                'Invertido (treino 2022→teste 2018)': sp['split_invertido_treino2022_teste2018'][nome]['accuracy'],
                '5-fold (média±desvio)': f"{sp['kfold_5_estratificado'][nome]['accuracy_mean']:.1%} ± {sp['kfold_5_estratificado'][nome]['accuracy_std']:.1%}",
            }
            for nome in sp['split_original_treino2018_teste2022']
        ])
        st.dataframe(
            split_df.style.format({'Original (treino 2018→teste 2022)': '{:.1%}',
                                    'Invertido (treino 2022→teste 2018)': '{:.1%}'}),
            hide_index=True, use_container_width=True,
        )
        st.markdown("""
**Achado sem maquiagem: o LogReg (modelo de destaque, 57.8%) não é robusto à direção do split** —
cai para 39.1% quando treinado em 2022 e testado em 2018 (queda de ~19pp). XGBoost é estável nas
duas direções (50.0%) e LightGBM até melhora invertido (56.2%→59.4%). O 5-fold é o retrato mais
honesto: os três modelos ficam sobrepostos em ~51-56% com desvio padrão de 3-8pp — consistente com
o teste pareado acima (nenhuma diferença estatisticamente significativa). O 57.8% de destaque no
topo do dashboard é real, mas é o resultado de **um único split entre vários possíveis**, não uma
medida estável do modelo.
""")

    with st.expander('Ensemble por média de probabilidades resolve a indecisão entre modelos?'):
        en = ensemble_info
        st.markdown("""
Como nenhum modelo vence os outros de forma estatisticamente significativa (teste pareado acima) e
o "melhor" muda de acordo com a direção do split, a recomendação padrão (Kaggle Book, cap. 9 —
"Averaging models into an ensemble") é não escolher: tirar a média das probabilidades calibradas dos
três modelos (`notebooks/dia18_ensemble_probabilidades.py`).
""")
        ens_df = pd.DataFrame([
            {
                'Modelo': nome,
                'Original (treino 2018→teste 2022)': en['split_original'][nome]['accuracy'],
                'Invertido (treino 2022→teste 2018)': en['split_invertido'][nome]['accuracy'],
                '5-fold (média±desvio)': f"{en['kfold_5_estratificado'][nome]['accuracy_mean']:.1%} ± {en['kfold_5_estratificado'][nome]['accuracy_std']:.1%}",
            }
            for nome in en['split_original']
        ])
        st.dataframe(
            ens_df.style.format({'Original (treino 2018→teste 2022)': '{:.1%}',
                                  'Invertido (treino 2022→teste 2018)': '{:.1%}'}),
            hide_index=True, use_container_width=True,
        )
        st.markdown(f"""
O ensemble empata com o melhor individual no split original ({en['split_original']['Ensemble (média)']['accuracy']:.1%})
e é bem mais estável que o LogReg sozinho no invertido ({en['split_invertido']['Ensemble (média)']['accuracy']:.1%} vs
39.1% do LogReg puro), sem cair tanto quanto o LightGBM sobe. No 5-fold, tem o menor desvio padrão
entre as opções ({en['kfold_5_estratificado']['Ensemble (média)']['accuracy_std']:.1%}) — não é o modelo
com maior pico de acurácia, mas é a opção que menos depende de qual split calhou de ser usado.
""")

    with st.expander('Regressão de Poisson nos gols — dá pra prever empates de verdade?'):
        po = poisson_info
        st.markdown("""
O pipeline de classificação nunca prevê "Draw" nas 72 partidas de 2026 — sinal de que classificar
H/D/A direto é ruim pra essa classe minoritária. O dataset já traz `home_score`/`away_score`; em vez
de classificar o resultado, modelamos os gols de cada lado via regressão de Poisson (mesmas 5
features de ranking) e derivamos P(Home)/P(Draw)/P(Away) somando a matriz de probabilidade de
placares — abordagem clássica de modelagem de futebol (`notebooks/dia19_poisson_gols.py`).
""")
        colq1, colq2, colq3 = st.columns(3)
        colq1.metric('Acurácia (teste 2022)', f"{po['acuracia_teste_2022']:.1%}")
        colq2.metric('P(Draw) médio', f"{po['prob_draw_media_teste_2022']:.1%}", 'vs ~0% do classificador')
        colq3.metric('Empates previstos (2026)', f"{po['empates_previstos_2026']}/{po['n_jogos_2026']}")
        st.markdown(f"""
**Achado sem maquiagem**: o Poisson dá um sinal de empate real — P(Draw) médio de
{po['prob_draw_media_teste_2022']:.1%} contra confiança ~0% do classificador — e o "Draw" foi a
**segunda** opção mais provável em {po['quase_empates_teste_2022']}/{po['n_teste_2022']} jogos do
teste 2022. Mas a previsão "dura" (a opção mais provável) ainda não escolhe Draw em nenhum jogo, nem
em 2022 nem em 2026. Isso é um fenômeno conhecido de modelos de Poisson independentes (correção
Dixon-Coles de 1997 existe justamente pra isso, fora do escopo aqui): o modelo melhora a
**probabilidade** atribuída ao empate, mas não muda a previsão pontual. Diagnóstico mais honesto que
solução definitiva.
""")

    with st.expander('Testamos PyTorch — o que aconteceu'):
        mlp = pytorch_info['mlp_pytorch']
        ref = pytorch_info['classic_ml_reference']
        colp1, colp2, colp3 = st.columns(3)
        colp1.metric('Acurácia', f"{mlp['accuracy_mean']:.1%}", f"vs {ref['accuracy']:.1%} (clássico)")
        colp2.metric('RPS (menor é melhor)', f"{mlp['rps_mean']:.3f}", f"vs {ref['rps']:.3f} (clássico)", delta_color='inverse')
        colp3.metric('Draw recall', f"{mlp['draw_recall_mean']:.1%}", f"vs {ref['draw_recall']:.1%} (clássico)")
        st.markdown(f"""
Em vez de só argumentar que o dataset é pequeno demais para redes neurais, treinamos uma MLP real (mesmas 5 features, mesmo split 2018→2022, 10 seeds) e comparamos com o melhor modelo clássico ({ref['model']}).

Resultado: a MLP tem acurácia e RPS piores, mas prevê empates de verdade (draw recall {mlp['draw_recall_mean']:.0%} contra 0% do modelo clássico) — um trade-off real, não hipotético. Ficamos com o modelo clássico porque acurácia geral pesa mais na avaliação da competição, mas o resultado da MLP é interessante o suficiente para não descartar de cara numa próxima iteração.
""")

with tab_previsoes:
    st.subheader('Previsões — Copa do Mundo 2026 (fase de grupos)')
    times = sorted(set(previsoes['home_team']) | set(previsoes['away_team']))
    filtro = st.multiselect('Filtrar por seleção', times)
    tabela = previsoes.copy()
    if filtro:
        tabela = tabela[tabela['home_team'].isin(filtro) | tabela['away_team'].isin(filtro)]

    tabela_fmt = tabela.rename(columns={
        'home_team': 'Mandante', 'away_team': 'Visitante', 'prediction': 'Previsão',
        'pred_home_prob': 'P(Casa)', 'pred_draw_prob': 'P(Empate)', 'pred_away_prob': 'P(Fora)',
    })[['Mandante', 'Visitante', 'Previsão', 'P(Casa)', 'P(Empate)', 'P(Fora)']]
    st.dataframe(
        tabela_fmt.style.format({'P(Casa)': '{:.1%}', 'P(Empate)': '{:.1%}', 'P(Fora)': '{:.1%}'})
            .background_gradient(subset=['P(Casa)', 'P(Empate)', 'P(Fora)'], cmap='Greens'),
        hide_index=True, use_container_width=True, height=420,
    )

    st.subheader('Confiança das previsões')
    st.caption(f"Modelo final: {rps_info['best_by_rps']} · treino completo 2018+2022 · prob. máx. observada: {rps_info['max_prob_2026']:.1%}")
    fig2 = go.Figure(go.Histogram(x=max_prob * 100, marker_color='#3D6FB4', nbinsx=15))
    fig2.add_vline(x=33.3, line_dash='dash', line_color='crimson', annotation_text='Chance (33.3%)')
    fig2.add_vline(x=max_prob.median() * 100, line_color='#00A859', annotation_text=f'Mediana ({max_prob.median():.1%})')
    fig2.update_layout(
        xaxis_title='Probabilidade máxima prevista por jogo (%)', yaxis_title='Nº de jogos',
        height=340, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#FAFAFA',
    )
    st.plotly_chart(fig2, use_container_width=True)

with tab_simulacao:
    st.subheader(f"Simulação Monte Carlo da fase de grupos ({sim_info['n_sims']:,} rodadas)".replace(',', '.'))
    st.caption('Cada simulação sorteia o resultado de cada partida a partir das probabilidades do modelo. '
               'Formato real 2026: top-2 de cada grupo avança + os 8 melhores terceiros colocados.')

    advance = pd.Series(sim_info['advance_prob']).sort_values(ascending=False)
    grupo_de = sim_info['team_group']

    grupos_disponiveis = ['Todos'] + sorted(sim_info['groups'].keys())
    grupo_sel = st.selectbox('Filtrar por grupo', grupos_disponiveis)
    if grupo_sel != 'Todos':
        advance_view = advance[[t for t in advance.index if grupo_de[t] == grupo_sel]]
    else:
        advance_view = advance

    fig3 = go.Figure(go.Bar(
        x=advance_view.values, y=advance_view.index, orientation='h',
        marker_color=['#00A859' if v >= 0.5 else '#3D6FB4' for v in advance_view.values],
        text=[f'{v:.0%}' for v in advance_view.values], textposition='outside', textfont_color='#FAFAFA',
    ))
    fig3.add_vline(x=0.5, line_dash='dash', line_color='crimson')
    fig3.update_layout(
        xaxis=dict(tickformat='.0%', range=[0, 1.08], gridcolor='#2A3341', title='Probabilidade de avançar'),
        yaxis=dict(autorange='reversed'),
        height=max(320, 26 * len(advance_view)), margin=dict(l=10, r=60, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#FAFAFA',
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader('Possíveis surpresas')
    st.caption('Jogos com grande diferença de ranking FIFA, mas onde o modelo não vê favoritismo claro '
               '(ou até inverte o favorito do ranking). Desempate de grupo aproximado por vitórias simuladas — '
               'não modelamos placar, então critérios reais de saldo de gols não entram na simulação.')
    surp = pd.DataFrame(sim_info['surpresas'])
    surp_fmt = surp.rename(columns={
        'azarao': 'Zebra', 'rank_azarao': 'Rank zebra', 'favorito_ranking': 'Favorito (ranking)',
        'rank_favorito': 'Rank favorito', 'rank_gap': 'Gap de ranking',
        'prob_azarao_vencer': 'P(zebra vence)', 'prob_favorito_vencer': 'P(favorito vence)',
    })[['Zebra', 'Rank zebra', 'Favorito (ranking)', 'Rank favorito', 'Gap de ranking',
        'P(zebra vence)', 'P(favorito vence)']]
    st.dataframe(
        surp_fmt.style.format({'P(zebra vence)': '{:.1%}', 'P(favorito vence)': '{:.1%}'})
            .background_gradient(subset=['P(zebra vence)'], cmap='Oranges'),
        hide_index=True, use_container_width=True,
    )

with tab_corrida:
    st.subheader('Jornada analítica em 5 pontos')
    st.caption('Os momentos de pensamento crítico que mais mudaram o resultado final — não só o que funcionou, também o que foi testado e rejeitado.')
    jornada = [
        ('🐛', 'Bug de nomes', 'USA / Cabo Verde / Bósnia ficavam sem ranking por divergência de nome entre datasets — corrigido com alias antes de qualquer modelo rodar.'),
        ('🎯', 'Threshold rejeitado', 'Ajustar o limiar de decisão para acertar mais empates previu 45.8% de empates em 2026 (vs. ~22% histórico) — descartado, ficou o argmax puro.'),
        ('⏳', 'Vazamento documentado', 'Ranking FIFA usa 1 snapshot para treino 2018 e teste 2022 — sinal inflado detectado, documentado no relatório em vez de escondido.'),
        ('🔥', 'PyTorch testado, não só argumentado', 'Treinamos uma MLP real (10 seeds) contra o modelo clássico — perdeu em acurácia e RPS, mas venceu em draw recall: trade-off medido, não hipotético.'),
        ('🎲', 'Formato real do torneio', 'Simulação Monte Carlo implementa top-2 + 8 melhores terceiros (formato oficial 2026), não a simplificação comum de "top-2 por grupo".'),
    ]
    cols_jornada = st.columns(5)
    for col, (emoji, titulo, texto) in zip(cols_jornada, jornada):
        col.markdown(f"**{emoji} {titulo}**")
        col.caption(texto)

    st.divider()
    st.subheader('Corrida da fase de grupos — rodada a rodada')
    st.caption('Cada grupo joga 3 rodadas (2 jogos simultâneos por rodada). A tabela evolui conforme o resultado previsto (argmax) de cada jogo — clique em ▶ Play.')

    schedule_dates = pd.read_csv(BASE / 'data' / 'raw' / 'schedule_2026.csv')[['Date', 'home_team', 'away_team']]
    merged = previsoes.merge(schedule_dates, on=['home_team', 'away_team'], how='left')

    grupo_corrida = st.selectbox('Grupo', sorted(sim_info['groups'].keys()), key='grupo_corrida')
    times_grupo = sim_info['groups'][grupo_corrida]
    jogos_grupo = merged[merged['home_team'].isin(times_grupo) & merged['away_team'].isin(times_grupo)].copy()
    jogos_grupo['rodada'] = jogos_grupo['Date'].rank(method='dense').astype(int)
    jogos_grupo = jogos_grupo.sort_values('rodada')

    pontos = {t: 0 for t in times_grupo}
    frames_pontos = [dict(pontos)]  # rodada 0
    for rodada in sorted(jogos_grupo['rodada'].unique()):
        for _, j in jogos_grupo[jogos_grupo['rodada'] == rodada].iterrows():
            if j['prediction'] == 'Home':
                pontos[j['home_team']] += 3
            elif j['prediction'] == 'Away':
                pontos[j['away_team']] += 3
            else:
                pontos[j['home_team']] += 1
                pontos[j['away_team']] += 1
        frames_pontos.append(dict(pontos))

    ordem_final = sorted(times_grupo, key=lambda t: frames_pontos[-1][t], reverse=True)
    cores = {t: ('#00A859' if i < 2 else '#3D6FB4') for i, t in enumerate(ordem_final)}

    frames = [
        go.Frame(
            name=str(i),
            data=[go.Bar(x=[snap[t] for t in ordem_final], y=ordem_final, orientation='h',
                          marker_color=[cores[t] for t in ordem_final],
                          text=[str(snap[t]) for t in ordem_final], textposition='outside', textfont_color='#FAFAFA')],
        )
        for i, snap in enumerate(frames_pontos)
    ]
    fig4 = go.Figure(data=frames[0].data, frames=frames[1:])
    fig4.update_layout(
        xaxis=dict(range=[0, 10], gridcolor='#2A3341', title='Pontos'),
        yaxis=dict(autorange='reversed'),
        height=280, margin=dict(l=10, r=60, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#FAFAFA', showlegend=False,
        updatemenus=[dict(
            type='buttons', showactive=False, x=0.0, y=1.15, xanchor='left',
            buttons=[dict(label='▶ Play', method='animate',
                           args=[None, dict(frame=dict(duration=900, redraw=True), fromcurrent=True, transition=dict(duration=300))])],
        )],
        sliders=[dict(
            steps=[dict(method='animate', args=[[str(i)], dict(frame=dict(duration=0, redraw=True), mode='immediate')],
                         label=f'Rodada {i}' if i else 'Início')
                   for i in range(len(frames_pontos))],
            x=0.0, y=-0.05, len=1.0,
        )],
    )
    st.plotly_chart(fig4, use_container_width=True)
    st.caption('🟢 zona de classificação direta (top-2) · 🔵 fora da zona nesta rodada · desempate por saldo de gols não é simulado aqui (só resultado).')

with tab_limitacoes:
    st.markdown("""
### Limitações estatísticas (ler antes de interpretar os números acima)

- **n=64 no teste** → margem de erro ±12pp: diferenças entre modelos não são estatisticamente significativas entre si.
- **Threshold tuning para Draw foi testado e rejeitado** — um threshold "otimizado" no teste de 2022 previu 45.8% de empates em 2026 (vs. ~22% histórico). Modelo final usa argmax puro.
- **Vazamento temporal no ranking FIFA** — features de ranking usam snapshot único (out/2022) para treino 2018 e teste 2022; correlação rank_diff×resultado é 0.536 em 2018 vs. 0.346 em 2022, evidência de sinal de treino inflado. Não corrigível dentro do escopo oficial de dados (PRD §1.2).
- **Modelo final não prevê nenhum empate em 2026** (0 de 72 jogos) — consistente com o draw recall estruturalmente baixo do modelo (ponto acima), mas vale ter em mente ao ler as previsões.
- Detalhes completos: `RELATORIO_FINAL_COPA_CHALLENGER.md` §9.
""")
