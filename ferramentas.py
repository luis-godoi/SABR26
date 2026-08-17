import os
import numpy as np
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from langchain_core.tools import tool, Tool
from langchain_experimental.tools import PythonAstREPLTool
from langchain_core.tools import StructuredTool

# Obtenção da chave de api
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Configurações do LLM
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="openai/gpt-oss-120b",
    temperature=0
)

# Ferramentas 

def criar_motor_calculo(df: pd.DataFrame) -> Tool:
    """
    Inicializa a ferramenta de execução Python com o DataFrame injetado no ambiente local.
    """
    
    def executor_pandas(expressao: str) -> str:
        """Executa a string contendo a expressão Pandas e retorna o resultado."""
        try:
            # O eval() executa a expressão injetando o df, pd e np no contexto
            resultado = eval(expressao, {"df": df, "pd": pd, "np": np})
            return str(resultado)
        except Exception as e:
            # Retorna o erro para o LLM tentar corrigir
            return f"Erro de sintaxe/execução: {str(e)}"

    ferramenta_motor_calculo = Tool(
        name="motor_calculo_python",
        func=executor_pandas,
        description="""Utilize esta ferramenta para cálculos pontuais ou filtragens compostas
        diretamente sobre o DataFrame pandas carregado na variável `df`.
        
        Exemplos de uso:
        - "Quantos pontos o jogador X fez no 2º set contra Y?"
        - "Qual a média de idade do time?"
        - "Quantos bloqueios aconteceram no set 3 do jogo Z?"
        
        Regras Inegociáveis:
        1. Escreva APENAS a expressão Python/Pandas que retorna o valor desejado (ex: `df[(df['player_name'] == 'X')]['total_points'].sum()`).
        2. NÃO use `print()`. Apenas coloque a expressão.
        3. NÃO use para gerar relatórios gerais, rankings consolidados ou gráficos — temos ferramentas específicas para isso.
        """
    )
    
    return ferramenta_motor_calculo




# Métricas calculadas

def metricas_calculadas_jogadores(query: str, df: pd.DataFrame) -> str:
    """Utilize esta ferramenta para rankings globais, comparações de eficiência, 
    maiores pontuadores ou líderes de fundamentos específicos (ataque, saque, 
    bloqueio, recepção, defesa) agregando todo o campeonato."""

    # 1. Filtra apenas o consolidado de cada partida
    base = df[df['set_no'] == 'MATCH'].copy()

    # 2. Define as colunas brutas que precisam ser somadas antes do cálculo
    # Adicionado 'block_points' que faltava para os líderes de fundamento
    colunas_soma = [
        'attack_points', 'attack_attempts', 'attack_faults',
        'serve_aces', 'serve_attempts', 'serve_faults',
        'block_points', 'reception_perfect', 'reception_attempts',
        'dig_success', 'dig_attempts',
        'total_points', 'minutes_played', 'total_faults'
    ]

    # 3. Agrupa por jogador e soma os volumes totais do campeonato
    agrupado = base.groupby('player_name')[colunas_soma].sum().reset_index()

    # 4. Recalcula as eficiências e métricas derivadas em cima do montante agregado
    agrupado['kill_pct'] = (agrupado['attack_points'] / agrupado['attack_attempts'] * 100).replace([np.inf, -np.inf], 0).fillna(0)
    agrupado['attack_eff_pct'] = ((agrupado['attack_points'] - agrupado['attack_faults']) / agrupado['attack_attempts'] * 100).replace([np.inf, -np.inf], 0).fillna(0)
    agrupado['ace_pct'] = (agrupado['serve_aces'] / agrupado['serve_attempts'] * 100).replace([np.inf, -np.inf], 0).fillna(0)
    agrupado['serve_eff_pct'] = ((agrupado['serve_aces'] - agrupado['serve_faults']) / agrupado['serve_attempts'] * 100).replace([np.inf, -np.inf], 0).fillna(0)
    agrupado['reception_pct'] = (agrupado['reception_perfect'] / agrupado['reception_attempts'] * 100).replace([np.inf, -np.inf], 0).fillna(0)
    agrupado['dig_pct'] = (agrupado['dig_success'] / agrupado['dig_attempts'] * 100).replace([np.inf, -np.inf], 0).fillna(0)
    agrupado['points_per_min'] = (agrupado['total_points'] / agrupado['minutes_played']).replace([np.inf, -np.inf], 0).fillna(0)
    
    agrupado['scorer_efficiency'] = agrupado['total_points'] - agrupado['total_faults']

    # 5. INCLUSÃO DAS COLUNAS ABSOLUTAS: total_points, attack_points, block_points, serve_aces
    tabela = agrupado[[
        'player_name', 'total_points', 'attack_points', 'block_points', 'serve_aces',
        'kill_pct', 'attack_eff_pct', 'ace_pct',
        'serve_eff_pct', 'reception_pct', 'dig_pct', 'points_per_min', 'scorer_efficiency'
    ]].round(1).to_string(index=False)

    # 6. Prompt Refinado com Dicionário de Dados
    template_resposta = PromptTemplate(
        template="""
        Você é um analista técnico de voleibol. O usuário pediu: "{query}"
        
        Dicionário de colunas para sua referência:
        - total_points: Pontos totais feitos pelo jogador (use para "maior pontuador")
        - attack_points: Pontos de ataque
        - block_points: Pontos de bloqueio
        - serve_aces: Pontos de saque (aces)
        - attack_eff_pct: Eficiência de ataque (%)
        - serve_eff_pct: Eficiência de saque (%)
        - scorer_efficiency: Eficiência de pontuação (saldo total_points - total_faults)

        Tabela consolidada e agregada de métricas por jogador (total do campeonato):
        {tabela}

        Responda de forma direta e técnica, aplicando o critério de ordenação/filtro
        pedido. Destaque o(s) jogador(es) relevante(s) com os valores exatos da tabela.
        Se a pergunta for sobre "maior pontuador", olhe EXCLUSIVAMENTE para a coluna `total_points`.
        Nunca invente ou recalcule valores — use somente os números apresentados na tabela acima.
        """,
        input_variables=["query", "tabela"]
    )

    cadeia = template_resposta | llm | StrOutputParser()
    return cadeia.invoke({"query": query, "tabela": tabela})






# Radar de comparações

def radar_comparativo_atletas(entrada: str, df: pd.DataFrame) -> str:
    """Utilize esta ferramenta quando o usuário pedir para comparar dois jogadores
    ou gerar um radar/perfil técnico entre eles. 
    A entrada deve conter os dois nomes separados por vírgula, e opcionalmente 
    o tipo de métrica ('relativo' ou 'absoluto'). 
    Exemplos: 'Maicon, Paulo', 'Maicon, Paulo, absoluto', 'Léo, Sabino, relativo'."""

    partes = [p.strip() for p in entrada.split(",")]
    if len(partes) < 2:
        return "Erro: Informe ao menos os dois nomes separados por vírgula, ex: 'Maicon, Paulo'."
    
    atleta1 = partes[0]
    atleta2 = partes[1]
    modo = partes[2].lower() if len(partes) > 2 else 'relativo'

    # Validar se os atletas existem na base
    validos = df['player_name'].unique()
    faltantes = [a for a in (atleta1, atleta2) if a not in validos]
    if faltantes:
        return (f"Jogador(es) não encontrado(s): {', '.join(faltantes)}. "
                f"Atletas disponíveis: {', '.join(sorted(validos))}.")

    # 1. Filtrar e Agregar o volume total do campeonato
    base = df[df['set_no'] == 'MATCH'].copy()
    colunas_soma = [
        'attack_attempts', 'attack_points', 'attack_faults',
        'block_attempts', 'block_points', 'block_faults',
        'serve_attempts', 'serve_aces', 'serve_faults',
        'reception_attempts', 'reception_perfect', 'reception_faults',
        'dig_attempts', 'dig_success', 'total_points'
    ]
    stats = base.groupby('player_name')[colunas_soma].sum().reset_index()

    # 2. Definição das métricas conforme o modo escolhido
    if modo == 'absoluto':
        categorias = ['Attack Pts', 'Block Pts', 'Aces', 'Digs', 'Rec Perf', 'Total Pts']
        max_ranges = [100, 20, 10, 40, 60, 120]  # Tetos de referência para normalização
        
        def extrair_valores(nome):
            row = stats[stats['player_name'] == nome].iloc[0]
            return [
                row['attack_points'], row['block_points'], row['serve_aces'],
                row['dig_success'], row['reception_perfect'], row['total_points']
            ]
    else:  # Modo 'relativo' (padrão)
        categorias = ['Kill %', 'Att Eff %', 'Rec Eff %', 'Srv Eff %', 'Blk Eff %', 'Dig %']
        max_ranges = [80, 60, 50, 20, 30, 90]  # Tetos percentuais de referência
        
        def extrair_valores(nome):
            row = stats[stats['player_name'] == nome].iloc[0]
            
            # Cálculo com divisão segura (evitando NaN e Infinity)
            div_segura = lambda num, den: (num / den * 100) if den > 0 else 0.0
            
            return [
                div_segura(row['attack_points'], row['attack_attempts']),
                div_segura((row['attack_points'] - row['attack_faults']), row['attack_attempts']),
                div_segura((row['reception_perfect'] - row['reception_faults']), row['reception_attempts']),
                div_segura((row['serve_aces'] - row['serve_faults']), row['serve_attempts']),
                div_segura((row['block_points'] - row['block_faults']), row['block_attempts']),
                div_segura(row['dig_success'], row['dig_attempts'])
            ]

    valores1 = extrair_valores(atleta1)
    valores2 = extrair_valores(atleta2)

    # 3. Escalas de Normalização: Limitando entre 0.0 e 1.0 (trata percentuais negativos)
    val1_norm = [min(1.0, max(0.0, r / m)) if pd.notna(r) else 0.0 for r, m in zip(valores1, max_ranges)]
    val2_norm = [min(1.0, max(0.0, r / m)) if pd.notna(r) else 0.0 for r, m in zip(valores2, max_ranges)]

    # 4. Fechamento do polígono para o Matplotlib polar
    N = len(categorias)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    val1_norm += val1_norm[:1]
    val2_norm += val2_norm[:1]

    # Paleta Dark Tech fornecida
    bg_color, grid_color, text_color = "#0D1117", "#21262D", "#E6EDF3"
    accent_p1, accent_p2 = "#58A6FF", "#FF7B72"
    
    fig = plt.figure(figsize=(8, 9), facecolor=bg_color)
    ax = fig.add_axes([0.15, 0.12, 0.70, 0.65], polar=True)
    ax.set_facecolor(bg_color)
    
    # Orientação e estética
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.25, 0.50, 0.75, 1.0])
    ax.set_yticklabels([])
    ax.spines['polar'].set_visible(False)
    ax.grid(color=grid_color, linestyle='-', linewidth=1, alpha=0.9)
    
    # Plotagem do Jogador 1
    ax.plot(angles, val1_norm, color=accent_p1, linewidth=2.2, label=atleta1, zorder=4)
    ax.fill(angles, val1_norm, color=accent_p1, alpha=0.25, zorder=3)
    ax.scatter(angles[:-1], val1_norm[:-1], color=accent_p1, s=35, zorder=5)
    
    # Plotagem do Jogador 2
    ax.plot(angles, val2_norm, color=accent_p2, linewidth=2.2, label=atleta2, zorder=4)
    ax.fill(angles, val2_norm, color=accent_p2, alpha=0.25, zorder=3)
    ax.scatter(angles[:-1], val2_norm[:-1], color=accent_p2, s=35, zorder=5)
    
    # Rótulos, títulos e legendas
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categorias, fontsize=10, weight='bold', color=text_color)
    ax.tick_params(pad=16)
    
    tipo_titulo = "MÉTRICAS ABSOLUTAS" if modo == 'absoluto' else "MÉTRICAS RELATIVAS (%)"
    fig.text(0.5, 0.94, f"RADAR DE PERFORMANCE - {tipo_titulo}", fontsize=14, weight='bold', color=text_color, ha='center')
    fig.text(0.5, 0.91, "Comparativo de Desempenho Individual Agregado", fontsize=9.5, color="#8B949E", ha='center')
    fig.text(0.35, 0.86, f"● {atleta1}", color=accent_p1, fontsize=11, weight='bold', ha='center')
    fig.text(0.65, 0.86, f"● {atleta2}", color=accent_p2, fontsize=11, weight='bold', ha='center')
    
    # Exibe no Streamlit e limpa a memória
    st.pyplot(fig)
    plt.close(fig)

    # Retorno estruturado para o agente IA ler e interpretar no chat
    return (f"Gráfico de radar '{modo}' renderizado na tela. "
            f"Eixos utilizados: {', '.join(categorias)}. "
            f"Valores ({modo}) de {atleta1}: {[round(v, 1) for v in valores1]}. "
            f"Valores ({modo}) de {atleta2}: {[round(v, 1) for v in valores2]}. "
            f"Forneça um breve resumo analítico apontando a principal vantagem de cada atleta baseando-se nestes números exatos.")




# Gerador de gráficos estatísticos

def gerador_graficos_estatisticos(instrucao: str, df: pd.DataFrame) -> str:
    """Utilize esta ferramenta para gráficos livres de distribuição ou correlação
    (barras, dispersão, séries temporais por set/partida). Exemplos: 'Mostre a
    distribuição de erros de saque por partida', 'Plote os pontos de bloqueio por adversário'."""

    # 1. Template rigoroso para extrair código Python e guiar o pré-processamento
    template_grafico = PromptTemplate(
        template="""
        Você é um cientista de dados especialista em Python, Pandas e Matplotlib focando em Voleibol.
        Escreva EXCLUSIVAMENTE o código Python para gerar o gráfico solicitado.
        NÃO inclua marcações markdown como ```python, texto explicativo ou introduções. Apenas o código puro.

        O DataFrame já está carregado na variável `df`.
        Colunas disponíveis no DataFrame: {colunas}

        Instrução do usuário para o gráfico: {instrucao}

        Regras ESTRITAS para os Dados e Gráfico:
        1. TRATAMENTO DE DADOS: Se a métrica for "por partida" ou "por adversário", PRIMEIRO filtre os dados consolidados: `df_plot = df[df['set_no'] == 'MATCH'].copy()`.
        2. AGREGAÇÃO: Agrupe corretamente. Exemplo: para bloqueios da equipe por partida, faça `agrupado = df_plot.groupby(['match_id', 'opponent'])['block_points'].sum().reset_index()`.
        3. EIXO X LEGÍVEL: NUNCA coloque o `match_id` numérico no eixo X. Se for uma linha do tempo ou progressão por partida, use a coluna `opponent` (o nome do país) no eixo X.
        4. ESTÉTICA: 
           - Crie a figura com `fig, ax = plt.subplots(figsize=(10, 6))`
           - Fundo escuro: `fig.patch.set_facecolor('#0e1117')` e `ax.set_facecolor('#0e1117')`
           - Textos, bordas e ticks em branco ou cinza: `ax.tick_params(colors='white')`, `ax.xaxis.label.set_color('white')`, `ax.yaxis.label.set_color('white')`, `ax.title.set_color('white')`
           - Para gráficos de progressão (linha), adicione marcadores: `ax.plot(..., marker='o', color='#00e5ff')`
        5. ROTAÇÃO: Se o eixo X contiver texto (como os nomes dos adversários), aplique `plt.xticks(rotation=45, ha='right')`.
        6. NÃO chame `plt.show()`. O gráfico será capturado posteriormente pela ferramenta pai.
        """,
        input_variables=["instrucao", "colunas"]
    )

    colunas_str = ", ".join(df.columns.astype(str).tolist())

    # 2. Gera o código usando o LLM
    cadeia_grafico = template_grafico | llm | StrOutputParser()
    codigo_python = cadeia_grafico.invoke({"instrucao": instrucao, "colunas": colunas_str})

    # Limpeza de segurança caso o LLM insista em colocar blocos markdown
    codigo_python = codigo_python.replace("```python", "").replace("```", "").strip()

    # 3. Sandbox de Execução
    ambiente = {
        "df": df,
        "plt": plt,
        "sns": sns,
        "np": np,
        "pd": pd
    }

    try:
        # Fecha qualquer figura residual antes de começar
        plt.close('all')
        
        # Executa o código gerado no escopo restrito do dicionário "ambiente"
        exec(codigo_python, ambiente)
        
        # 4. Captura e exibe no Streamlit
        fig = plt.gcf()
        
        if not fig.axes:
            return "Erro: O código foi executado, mas nenhum gráfico foi gerado."
            
        st.pyplot(fig)
        plt.close(fig)
        
        return "Gráfico gerado e exibido com sucesso na interface. Diga ao usuário que o gráfico está na tela."

    except Exception as e:
        return f"Falha ao gerar o gráfico. Erro técnico retornado: {str(e)}"






    


def obter_todas_ferramentas(df: pd.DataFrame):
    
    # Ferramenta de métricas
    ferramenta_metricas = StructuredTool.from_function(
        func=lambda query: metricas_calculadas_jogadores(query, df),
        name="metricas_calculadas_jogadores",
        description="Utilize esta ferramenta para rankings globais, comparações de eficiência, maiores pontuadores ou líderes de fundamentos específicos (ataque, saque, bloqueio, recepção, defesa) agregando todo o campeonato."
    )

    # Ferramenta de radar
    ferramenta_radar = StructuredTool.from_function(
        func=lambda entrada: radar_comparativo_atletas(entrada, df),
        name="radar_comparativo_atletas",
        description="Utilize esta ferramenta quando o usuário pedir para comparar dois jogadores ou gerar um radar/perfil técnico entre eles. A entrada deve conter os dois nomes separados por vírgula, e opcionalmente o tipo de métrica ('relativo' ou 'absoluto')."
    )

    # CORREÇÃO DA FERRAMENTA DE GRÁFICOS
    ferramenta_graficos = StructuredTool.from_function(
        func=lambda instrucao: gerador_graficos_estatisticos(instrucao, df),
        name="gerador_graficos_estatisticos",
        description="Utilize esta ferramenta para gráficos livres de distribuição ou correlação (barras, dispersão, séries temporais por set/partida)."
    )
    
    ferramenta_motor = criar_motor_calculo(df)

    return [ferramenta_metricas, ferramenta_radar, ferramenta_graficos, ferramenta_motor]