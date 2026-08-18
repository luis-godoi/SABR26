import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Importa o LLM e a função de ferramentas do nosso novo ferramentas.py
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
            + ". Crie um arquivo .env para rodar a aplicação."
        )
        st.stop()

def _resolve_data_path() -> Path | None:
    """Decide de onde carregar os dados."""
    configured = os.getenv("DATA_CSV_PATH")
    if configured:
        data_path = Path(configured)
        if not data_path.is_absolute():
            data_path = (PROJECT_ROOT / data_path).resolve()
        if data_path.exists():
            return data_path

    if DEFAULT_DATA_PATH.exists():
        return DEFAULT_DATA_PATH

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
    # Obtém as ferramentas com o df_metricas já pré-calculado em background
    ferramentas = obter_todas_ferramentas(df)

    # NOVO PROMPT: Totalmente focado em orientar o LLM a usar o motor Python corretamente
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Você é um assistente analítico especializado em estatísticas de voleibol.
Use as ferramentas para buscar a informação solicitada.

ACESSO AOS DADOS (VIA motor_calculo_python):
Você tem acesso a dois DataFrames no `motor_calculo_python`:
1. `df`: O dataset bruto, com cada linha sendo um set de uma partida.
2. `df_metricas`: Um dataset consolidado (já agrupado por jogador) com os totais do campeonato e métricas avançadas. 
   -> SEMPRE use o `df_metricas` para perguntas sobre totais do campeonato, quem fez mais pontos, líder de fundamento ou eficiência.
   -> Exemplo de uso na ferramenta: df_metricas.sort_values('attack_points', ascending=False).head(3)
   -> Exemplo de busca: df_metricas[df_metricas['player_name'] == 'Yan']['eficiencia_ataque_pct'].values[0]

COLUNAS DISPONÍVEIS NO `df_metricas` (Use EXATAMENTE estes nomes nas suas consultas):
- ATAQUE: attack_points, attack_attempts, attack_faults, aproveitamento_ataque_pct, eficiencia_ataque_pct
- SAQUE: serve_aces, serve_attempts, serve_faults, eficiencia_saque_pct
- BLOQUEIO: block_points, block_attempts, block_faults, aproveitamento_bloqueio_pct, eficiencia_bloqueio_pct
- RECEPÇÃO: reception_perfect, reception_attempts, reception_faults, aproveitamento_recepcao_pct, eficiencia_recepcao_pct
- DEFESA/DIG: dig_success, dig_attempts, aproveitamento_defesa_pct
- GERAL: total_points, total_faults, saldo_pontuacao

GLOSSÁRIO DE FUNDAMENTOS (NUNCA confunda um com o outro):
- ATAQUE: finalizar a jogada atacando a bola por cima da rede.
- SAQUE: o saque inicial do rally.
- BLOQUEIO: interceptar o ataque adversário junto à rede.
- RECEPÇÃO: receber o SAQUE do adversário.
- DEFESA / DIG ("manchete"): defender o ATAQUE adversário na quadra. "Maior defensor", "quem fez mais defesas" = olhar a coluna `dig_success` ou `aproveitamento_defesa_pct`. NUNCA use recepção ou bloqueio para responder sobre defesa.

AVISOS MATEMÁTICOS CRÍTICOS:
1. APROVEITAMENTO vs EFICIÊNCIA: "Aproveitamento" ignora erros (acertos/tentativas). "Eficiência" desconta os erros ((acertos-erros)/tentativas).
2. CONTAGEM vs TAXA: Se a pergunta for "quem fez mais X" (volume), consulte a coluna bruta (ex: dig_success). Se for "quem foi mais eficiente" (taxa), consulte a coluna percentual (ex: eficiencia_ataque_pct).
3. Nunca invente valores. Execute a consulta no `motor_calculo_python` e responda com o resultado exato.

Regras de Resposta:
1. Seja direto, cite o fundamento correto.
2. Responda em português de forma técnica e exata. Formate os números com 2 casas decimais quando aplicável.""",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm, ferramentas, prompt)
    # handle_parsing_errors evita que a aplicação quebre se o LLM errar a formatação da ferramenta
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
    st.caption("Converse com o agente sobre rankings, comparações, métricas e gráficos do campeonato.")

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
        st.metric("Linhas (sets jogados)", len(df))
        st.metric("Colunas brutas", len(df.columns))
        with st.expander("Visualizar amostra dos dados"):
            st.dataframe(df.head(20))

    # Inicialização do agente
    if "agente" not in st.session_state:
        with st.spinner("Preparando o agente e calculando métricas em background..."):
            st.session_state["agente"] = _construir_agente(df)

    # Renderiza o histórico do chat
    if "historico" not in st.session_state:
        st.session_state["historico"] = []

    for item in st.session_state["historico"]:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])

    # Input do usuário
    pergunta = st.chat_input("Pergunte algo sobre os jogadores, gráficos ou o campeonato...")
    if pergunta:
        st.session_state["historico"].append({"role": "user", "content": pergunta})
        with st.chat_message("user"):
            st.markdown(pergunta)

        with st.chat_message("assistant"):
            with st.spinner("Analisando os dados 🦜..."):
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