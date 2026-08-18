from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from ferramentas import llm, obter_todas_ferramentas


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "brazil_matches_performance2.csv"

load_dotenv()

st.set_page_config(
    page_title="Agente de Análise de Voleibol",
    page_icon="🏐",
    layout="wide",
)


def _validate_env() -> None:
    missing = []
    if not os.getenv("GROQ_API_KEY"):
        missing.append("GROQ_API_KEY")
    if missing:
        st.error(
            "Faltam variáveis de ambiente: "
            + ", ".join(missing)
            + ". Crie um arquivo .env baseado em .env.example."
        )
        st.stop()


def _resolve_data_path() -> Path | None:
    """Decide de onde carregar os dados: variável de ambiente,
    caminho padrão ou qualquer CSV já presente na pasta data."""
    configured = os.getenv("DATA_CSV_PATH")
    if configured:
        data_path = Path(configured)
        if not data_path.is_absolute():
            data_path = (PROJECT_ROOT / data_path).resolve()
        if data_path.exists():
            return data_path

    if DEFAULT_DATA_PATH.exists():
        return DEFAULT_DATA_PATH

    # Procura automaticamente qualquer CSV na pasta data/ ou raiz
    for pasta in (PROJECT_ROOT / "data", PROJECT_ROOT):
        if pasta.exists():
            csvs = sorted(pasta.glob("*.csv"))
            if csvs:
                return csvs[0]

    return None


@st.cache_data(show_spinner=False)
def _carregar_dataframe(path_str: str) -> pd.DataFrame:
    return pd.read_csv(path_str)


def _construir_agente(df: pd.DataFrame) -> AgentExecutor:
    ferramentas = obter_todas_ferramentas(df)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Você é um assistente analítico especializado em estatísticas de voleibol.
Use as ferramentas para buscar a informação solicitada.

Regras de Resposta:
1. SEJA DIRETO: Se o usuário perguntar "quem é o maior X", responda APENAS o nome do jogador e o valor da estatística.
2. NÃO REPITA A TABELA: Não liste todos os fundamentos (attack_points, serve_aces, etc.) a menos que o usuário peça uma análise completa.
3. Foque na pergunta: Se perguntaram sobre bloqueio, responda apenas sobre bloqueio.
4. Use as ferramentas apenas para buscar o dado; a formatação do texto final deve ser limpa e minimalista.
5. Nunca invente números — use exclusivamente o que as ferramentas retornarem.
6. Responda em português, de forma técnica e objetiva.""",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm, ferramentas, prompt)
    return AgentExecutor(agent=agent, tools=ferramentas, verbose=True, handle_parsing_errors=True)


def _historico_para_mensagens(historico: list[dict]) -> list:
    mensagens = []
    for item in historico:
        if item["role"] == "user":
            mensagens.append(HumanMessage(content=item["content"]))
        else:
            mensagens.append(AIMessage(content=item["content"]))
    return mensagens


# --------------------------------------------------------------------------- #
# Interface Streamlit
# --------------------------------------------------------------------------- #

def main() -> None:
    _validate_env()

    st.title("🏐 Agente de Análise de Voleibol")
    st.caption("Converse com o agente sobre rankings, comparações e gráficos do campeonato.")

    with st.sidebar:
        st.header("Dados")
        data_path = _resolve_data_path()

        if data_path is None:
            st.error(
                "Nenhum CSV encontrado. Certifique-se de que o arquivo 'brazil_matches_performance2.csv' "
                "está localizado na pasta 'data/'."
            )
            st.stop()

        st.success(f"Dados carregados: {data_path.name}")

        if st.button("🔄 Recarregar dados"):
            _carregar_dataframe.clear()
            st.session_state.pop("agente", None)
            st.rerun()

        if st.button("🧹 Limpar conversa"):
            st.session_state["historico"] = []
            st.rerun()

    df = _carregar_dataframe(str(data_path))

    with st.sidebar:
        st.metric("Linhas", len(df))
        st.metric("Colunas", len(df.columns))
        with st.expander("Visualizar amostra dos dados"):
            st.dataframe(df.head(20))

    if "agente" not in st.session_state:
        with st.spinner("Preparando o agente..."):
            st.session_state["agente"] = _construir_agente(df)

    if "historico" not in st.session_state:
        st.session_state["historico"] = []

    for item in st.session_state["historico"]:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])

    pergunta = st.chat_input("Pergunte algo sobre os jogadores ou o campeonato...")
    if pergunta:
        st.session_state["historico"].append({"role": "user", "content": pergunta})
        with st.chat_message("user"):
            st.markdown(pergunta)

        with st.chat_message("assistant"):
            with st.spinner("Analisando..."):
                try:
                    resultado = st.session_state["agente"].invoke(
                        {
                            "input": pergunta,
                            "chat_history": _historico_para_mensagens(st.session_state["historico"][:-1]),
                        }
                    )
                    resposta = resultado["output"]
                except Exception as exc:
                    resposta = f"Ocorreu um erro ao processar sua pergunta: {exc}"

                st.markdown(resposta)

        st.session_state["historico"].append({"role": "assistant", "content": resposta})


if __name__ == "__main__":
    main()