import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import StructuredTool
from langchain_experimental.tools import PythonAstREPLTool
import re 

# Obtenção da chave de API
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Configurações do LLM
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="openai/gpt-oss-120b", # Mantido modelo rápido e estável do código anterior
    temperature=0
)

# =====================================================================
# 1. FUNÇÃO DE PRÉ-CÁLCULO 
# =====================================================================
def calcular_metricas_consolidadas(df: pd.DataFrame) -> pd.DataFrame:
    """Pré-calcula todas as estatísticas e eficiências do campeonato para evitar
    sobrecarregar o prompt do LLM com cálculos em tempo real ou strings massivas."""
    
    base = df[df['set_no'] == 'MATCH'].copy()
    colunas_soma = [
        'attack_points', 'attack_attempts', 'attack_faults',
        'serve_aces', 'serve_attempts', 'serve_faults',
        'block_points', 'block_attempts', 'block_faults',
        'reception_perfect', 'reception_attempts', 'reception_faults',
        'dig_success', 'dig_attempts',
        'total_points', 'minutes_played', 'total_faults'
    ]
    
    col_existentes = [c for c in colunas_soma if c in base.columns]
    agrupado = base.groupby('player_name')[col_existentes].sum().reset_index()

    div_segura = lambda num, den: (num / den * 100).replace([np.inf, -np.inf], 0).fillna(0)

    if 'attack_points' in agrupado and 'attack_attempts' in agrupado:
        agrupado['aproveitamento_ataque_pct'] = div_segura(agrupado['attack_points'], agrupado['attack_attempts'])
        if 'attack_faults' in agrupado:
            agrupado['eficiencia_ataque_pct'] = div_segura((agrupado['attack_points'] - agrupado['attack_faults']), agrupado['attack_attempts'])

    if 'reception_perfect' in agrupado and 'reception_faults' in agrupado:
        agrupado['aproveitamento_recepcao_pct'] = div_segura(agrupado['reception_perfect'], agrupado['reception_attempts'])
        agrupado['eficiencia_recepcao_pct'] = div_segura((agrupado['reception_perfect'] - agrupado['reception_faults']), agrupado['reception_attempts'])

    if 'serve_aces' in agrupado and 'serve_faults' in agrupado:
        agrupado['eficiencia_saque_pct'] = div_segura((agrupado['serve_aces'] - agrupado['serve_faults']), agrupado['serve_attempts'])

    if 'dig_success' in agrupado and 'dig_attempts' in agrupado:
        agrupado['aproveitamento_defesa_pct'] = div_segura(agrupado['dig_success'], agrupado['dig_attempts'])

    if 'block_points' in agrupado and 'block_faults' in agrupado and 'block_attempts' in agrupado:
        agrupado['aproveitamento_bloqueio_pct'] = div_segura(agrupado['block_points'], agrupado['block_attempts'])
        agrupado['eficiencia_bloqueio_pct'] = div_segura((agrupado['block_points'] - agrupado['block_faults']), agrupado['block_attempts'])

    if 'total_points' in agrupado and 'total_faults' in agrupado:
        agrupado['saldo_pontuacao'] = agrupado['total_points'] - agrupado['total_faults']

    return agrupado.round(2)

# =====================================================================
# 2. RADAR COMPARATIVO (COM LEGENDA RESTAURADA)
# =====================================================================
def radar_comparativo_atletas(entrada: str, df: pd.DataFrame) -> str:
    """Compara dois jogadores gerando um radar técnico."""
    partes = [p.strip() for p in entrada.split(",")]
    if len(partes) < 2:
        return "Erro: Informe ao menos os dois nomes separados por vírgula, ex: 'Maicon, Paulo'."
    
    atleta1, atleta2 = partes[0], partes[1]
    modo = partes[2].lower() if len(partes) > 2 else 'relativo'

    validos = df['player_name'].unique()
    faltantes = [a for a in (atleta1, atleta2) if a not in validos]
    if faltantes:
        return f"Jogador(es) não encontrado(s): {', '.join(faltantes)}."

    base = df[df['set_no'] == 'MATCH'].copy()
    colunas = [c for c in ['attack_attempts', 'attack_points', 'attack_faults', 'block_attempts', 'block_points', 'block_faults', 'serve_attempts', 'serve_aces', 'serve_faults', 'reception_attempts', 'reception_perfect', 'reception_faults', 'dig_attempts', 'dig_success', 'total_points'] if c in base.columns]
    
    stats = base.groupby('player_name')[colunas].sum().reset_index()

    if modo == 'absoluto':
        categorias = ['Attack Pts', 'Block Pts', 'Aces', 'Digs', 'Rec Perf', 'Total Pts']
        max_ranges = [100, 20, 10, 40, 60, 120]
        def extrair_valores(nome):
            row = stats[stats['player_name'] == nome].iloc[0]
            return [row.get('attack_points', 0), row.get('block_points', 0), row.get('serve_aces', 0), row.get('dig_success', 0), row.get('reception_perfect', 0), row.get('total_points', 0)]
    else:
        categorias = ['Kill %', 'Att Eff %', 'Rec Eff %', 'Srv Eff %', 'Blk Eff %', 'Dig %']
        max_ranges = [80, 60, 50, 20, 30, 90]
        def extrair_valores(nome):
            row = stats[stats['player_name'] == nome].iloc[0]
            div_segura = lambda num, den: (num / den * 100) if den > 0 else 0.0
            return [
                div_segura(row.get('attack_points', 0), row.get('attack_attempts', 1)),
                div_segura((row.get('attack_points', 0) - row.get('attack_faults', 0)), row.get('attack_attempts', 1)),
                div_segura((row.get('reception_perfect', 0) - row.get('reception_faults', 0)), row.get('reception_attempts', 1)),
                div_segura((row.get('serve_aces', 0) - row.get('serve_faults', 0)), row.get('serve_attempts', 1)),
                div_segura((row.get('block_points', 0) - row.get('block_faults', 0)), row.get('block_attempts', 1)),
                div_segura(row.get('dig_success', 0), row.get('dig_attempts', 1))
            ]

    valores1 = extrair_valores(atleta1)
    valores2 = extrair_valores(atleta2)
    val1_norm = [min(1.0, max(0.0, r / m)) if pd.notna(r) else 0.0 for r, m in zip(valores1, max_ranges)]
    val2_norm = [min(1.0, max(0.0, r / m)) if pd.notna(r) else 0.0 for r, m in zip(valores2, max_ranges)]

    N = len(categorias)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    val1_norm += val1_norm[:1]
    val2_norm += val2_norm[:1]

    fig, ax = plt.subplots(figsize=(8, 9), subplot_kw=dict(polar=True), facecolor="#0D1117")
    ax.set_facecolor("#0D1117")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1.0)
    ax.set_yticklabels([])
    ax.spines['polar'].set_visible(False)
    ax.grid(color="#21262D", linestyle='-', linewidth=1)
    
    ax.plot(angles, val1_norm, color="#58A6FF", linewidth=2.2, label=atleta1)
    ax.fill(angles, val1_norm, color="#58A6FF", alpha=0.25)
    ax.plot(angles, val2_norm, color="#FF7B72", linewidth=2.2, label=atleta2)
    ax.fill(angles, val2_norm, color="#FF7B72", alpha=0.25)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categorias, fontsize=10, weight='bold', color="#E6EDF3")
    fig.text(0.5, 0.94, f"RADAR DE PERFORMANCE - {'ABSOLUTAS' if modo == 'absoluto' else 'RELATIVAS (%)'}", fontsize=14, weight='bold', color="#E6EDF3", ha='center')
    
    # RESTAURANDO A LEGENDA COLORIDA
    fig.text(0.35, 0.86, f"● {atleta1}", color="#58A6FF", fontsize=11, weight='bold', ha='center')
    fig.text(0.65, 0.86, f"● {atleta2}", color="#FF7B72", fontsize=11, weight='bold', ha='center')
    
    st.pyplot(fig)
    plt.close(fig)

    return f"Gráfico de radar renderizado. Valores de {atleta1}: {[round(v, 1) for v in valores1]}. Valores de {atleta2}: {[round(v, 1) for v in valores2]}."

# =====================================================================
# 3. GERADOR DE GRÁFICOS (COM FILTRO REGEX)
# =====================================================================
def gerador_graficos_estatisticos(instrucao: str, df: pd.DataFrame) -> str:
    """Gera gráficos Matplotlib/Seaborn."""
    colunas_info = "\n".join([f"- {col} ({dtype})" for col, dtype in df.dtypes.items()])
    amostra_dados = df.head(3).to_dict(orient='records')

    template_grafico = PromptTemplate(
        template="""
        Você é um cientista de dados. Escreva EXCLUSIVAMENTE o código Python para gerar o gráfico.
        ## Instrução: {instrucao}
        ## Metadados DataFrame `df`: {colunas}
        ## Amostra: {amostra}
        
        Regras:
        1. Para métricas gerais da partida, filtre com `df_plot = df[df['set_no'] == 'MATCH'].copy()`.
        2. Estética: Fundo escuro '#0e1117' em `fig` e `ax`, textos em branco.
        3. Apenas código Python.
        """,
        input_variables=["instrucao", "colunas", "amostra"]
    )

    cadeia_grafico = template_grafico | llm | StrOutputParser()
    resposta_llm = cadeia_grafico.invoke({"instrucao": instrucao, "colunas": colunas_info, "amostra": amostra_dados})
    
    # FILTRO INTELIGENTE: Pega apenas o que estiver dentro de blocos de código python, 
    # ignorando conversas extras que o LLM possa tentar adicionar.
    match = re.search(r"```(?:python)?(.*?)```", resposta_llm, re.DOTALL)
    if match:
        codigo_python = match.group(1).strip()
    else:
        codigo_python = resposta_llm.replace("```python", "").replace("```", "").strip()

    ambiente = {"df": df, "plt": plt, "sns": sns, "np": np, "pd": pd}

    try:
        plt.close('all')
        exec(codigo_python, ambiente)
        fig = plt.gcf()
        if not fig.axes: return "Erro: Nenhum gráfico gerado."
        st.pyplot(fig)
        plt.close(fig)
        return "Gráfico gerado e exibido com sucesso na tela."
    except Exception as e:
        return f"Falha ao gerar o gráfico. Erro técnico: {str(e)}"

# =====================================================================
# 4. INSTANCIADOR DE FERRAMENTAS (SEM LAMBDAS)
# =====================================================================
def obter_todas_ferramentas(df: pd.DataFrame):
    df_metricas = calcular_metricas_consolidadas(df)

    # 1. Ferramenta Python (já arrumada no passo anterior)
    def executor_pandas_seguro(query: str) -> str:
        query_limpa = query.replace("```python", "").replace("```", "").strip()
        repl = PythonAstREPLTool(locals={"df": df, "df_metricas": df_metricas, "pd": pd, "np": np})
        try:
            return str(repl.invoke({"query": query_limpa}))
        except Exception as e:
            return f"Erro na execução Python: {str(e)}"

    ferramenta_python = StructuredTool.from_function(
        func=executor_pandas_seguro,
        name="motor_calculo_python",
        description="""Utilize para consultas no Pandas. Envie a expressão obrigatoriamente pelo parâmetro 'query'."""
    )

    # 2. Wrapper para o Radar (Resolve erro de validação do LLM)
    def wrapper_radar(entrada: str) -> str:
        """Gera um radar comparativo entre dois atletas. """
        return radar_comparativo_atletas(entrada, df)

    ferramenta_radar = StructuredTool.from_function(
        func=wrapper_radar,
        name="radar_comparativo_atletas",
        description="Gera um radar técnico. Envie a string obrigatoriamente no parâmetro 'entrada' com os dois nomes separados por vírgula (Ex: 'Yan, Paulo')."
    )

    # 3. Wrapper para o Gráfico (Resolve erro de validação do LLM)
    def wrapper_graficos(instrucao: str) -> str:
        """Gera gráficos estatísticos."""
        return gerador_graficos_estatisticos(instrucao, df)

    ferramenta_graficos = StructuredTool.from_function(
        func=wrapper_graficos,
        name="gerador_graficos_estatisticos",
        description="Gera gráficos livres. Envie a requisição obrigatoriamente através do parâmetro em texto chamado 'instrucao'."
    )
    
    return [ferramenta_python, ferramenta_radar, ferramenta_graficos]