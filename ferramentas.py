import os
import numpy as np
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from langchain.agents import Tool
from langchain_experimental.tools import PythonAstREPLTool

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

# Métricas calculadas

@tool
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