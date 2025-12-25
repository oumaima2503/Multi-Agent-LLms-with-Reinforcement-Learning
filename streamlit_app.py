"""
Interface Streamlit interactive pour tester / comparer / évaluer
SFT vs MAGRPO + Workflow Multi-Agent réel

ROBUSTE :
- JSON valide
- JSON invalide
- Texte libre
- Debug complet
"""

import os
import json
import time

import streamlit as st
import pandas as pd
import plotly.express as px

from Compared_sft_magrpo.compare_sft_magrpo import test_agent_with_checkpoint
from evaluate_response_quality import evaluate_agent_response

try:
    from interact_magrpo import MAGRPOMultiAgentSystem
except Exception:
    MAGRPOMultiAgentSystem = None


# ============================================================
# CONFIG
# ============================================================

AGENTS = ["orchestrator", "researcher", "code_writer", "critic"]
DEFAULT_EPOCHS = [10, 15, 20]

st.set_page_config(
    page_title="MAGRPO vs SFT – Multi-Agent Interface",
    page_icon="🤖",
    layout="wide"
)

st.title("Multi-Agent Evaluation Interface")


# ============================================================
# UTILS ROBUSTES
# ============================================================

def safe_parse_output(raw):
    """
    Ne plante jamais.
    Retourne toujours un dict exploitable.
    """
    if raw is None:
        return {
            "raw_output": "",
            "is_valid_json": False
        }

    if isinstance(raw, dict):
        return {
            **raw,
            "raw_output": json.dumps(raw, indent=2, ensure_ascii=False),
            "is_valid_json": True
        }

    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return {
                **parsed,
                "raw_output": raw,
                "is_valid_json": True
            }
        except Exception:
            return {
                "raw_output": raw,
                "is_valid_json": False
            }

    return {
        "raw_output": str(raw),
        "is_valid_json": False
    }


def safe_evaluate(agent, parsed_output, query):
    """
    Évaluation robuste : ne casse jamais l'UI
    """
    try:
        return evaluate_agent_response(agent, parsed_output, query)
    except Exception:
        return {
            "overall_quality": 0.0,
            "format_quality": 0.0,
            "content_quality": 0.0
        }


def extract_agent_outputs_from_history(history):
    """
    Version ULTRA-ROBUSTE
    - history = list[str] (logs)
    - capture TOUT texte / JSON / erreurs
    - ne plante jamais
    """

    outputs = []
    current_agent = "unknown"
    buffer = []

    def flush():
        nonlocal buffer, current_agent
        if buffer:
            raw = "\n".join(buffer).strip()
            outputs.append({
                "agent": current_agent,
                "parsed_output": safe_parse_output(raw),
                "raw_output": raw
            })
        buffer.clear()

    for line in history:
        if not isinstance(line, str):
            continue

        # Détection agent actif
        if "Agent actif:" in line:
            flush()
            current_agent = (
                line.split("Agent actif:")[-1]
                    .strip()
                    .lower()
            )
            continue

        # Détection réponse agent
        if "Réponse de" in line:
            flush()
            current_agent = (
                line.split("Réponse de")[-1]
                    .replace(":", "")
                    .strip()
                    .lower()
            )
            continue

        # Séparateurs
        if line.startswith("===") or line.startswith("➡️"):
            flush()
            continue

        buffer.append(line)

    flush()
    return outputs

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("⚙️ Paramètres")

    agent = st.selectbox("Agent principal testé", AGENTS)
    run_sft = st.checkbox("Tester SFT", True)
    run_magrpo = st.checkbox("Tester MAGRPO", True)
    epochs = st.multiselect("Epochs MAGRPO", DEFAULT_EPOCHS, DEFAULT_EPOCHS)

    run_multi_agent = st.checkbox("Workflow Multi-Agent MAGRPO", True)
    offline = st.checkbox("Mode Offline", False)

    st.markdown("---")
    st.caption("Même requête utilisée pour tous les tests.")


# ============================================================
# INPUT USER
# ============================================================

query = st.text_area(
    "Entrez votre requête",
    height=100,
    placeholder="Ex: Écrire une fonction Python qui calcule le max d'une liste."
)

run_btn = st.button(" Lancer l'évaluation")


# ============================================================
# EXECUTION
# ============================================================

if run_btn and query.strip():

    with st.spinner(" Évaluation en cours…"):

        if offline:
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_HUB_OFFLINE"] = "1"

        results = []
        multi_agent_outputs = []

        start_all = time.time()

        # ============================
        # SFT
        # ============================
        if run_sft:
            res = test_agent_with_checkpoint(agent, query, "sft")
            parsed = safe_parse_output(res.get("result"))
            q = safe_evaluate(agent, parsed, query)

            results.append({
                "agent": agent,
                "checkpoint": "SFT",
                "epoch": 0,
                "reward": res.get("reward", 0.0),
                "time_s": res.get("time", 0.0),
                "success": res.get("success", False),
                "valid_json": parsed["is_valid_json"],
                **q
            })

        # ============================
        # MAGRPO (single-agent)
        # ============================
        if run_magrpo:
            for ep in epochs:
                res = test_agent_with_checkpoint(agent, query, "magrpo", epoch=ep)
                parsed = safe_parse_output(res.get("result"))
                q = safe_evaluate(agent, parsed, query)

                results.append({
                    "agent": agent,
                    "checkpoint": f"MAGRPO_E{ep}",
                    "epoch": ep,
                    "reward": res.get("reward", 0.0),
                    "time_s": res.get("time", 0.0),
                    "success": res.get("success", False),
                    "valid_json": parsed["is_valid_json"],
                    **q
                })

        # ============================
        # MULTI-AGENT WORKFLOW
        # ============================
        if run_multi_agent and MAGRPOMultiAgentSystem:
            system = MAGRPOMultiAgentSystem(
                epoch=max(epochs) if epochs else 20,
                offline=offline
            )
            system.run(query)

            agent_outputs = extract_agent_outputs_from_history(system.history)

            for item in agent_outputs:
                agent_name = item["agent"]
                parsed = item["parsed_output"]
                q = safe_evaluate(agent_name, parsed, query)

                quality = q.get("overall_quality", 0.0)
                if quality == 0.0 and parsed.get("raw_output"):
                    quality = 0.25  # fallback texte libre

                multi_agent_outputs.append({
                    "agent": agent_name,
                    "overall_quality": quality,
                    "valid_json": parsed["is_valid_json"],
                    "raw_output": parsed["raw_output"][:1000]
                })

        elapsed = time.time() - start_all

    st.success(f" Évaluation terminée en {elapsed:.1f}s")


    # ========================================================
    # DATAFRAMES
    # ========================================================

    df_results = pd.DataFrame(results)
    df_agents = pd.DataFrame(multi_agent_outputs) if multi_agent_outputs else None


    # ========================================================
    # TABS
    # ========================================================

    tab_tables, tab_graphs, tab_agents = st.tabs([
        " Tables",
        " Graphiques",
        " Agents"
    ])

    # ========================================================
    # TAB TABLES
    # ========================================================

    with tab_tables:
        st.subheader("Résultats SFT vs MAGRPO")
        st.dataframe(df_results, width="stretch")

        st.download_button(
            "⬇ Télécharger CSV",
            df_results.to_csv(index=False),
            file_name="comparison_results.csv"
        )

        if df_agents is not None:
            st.subheader("Résultats Multi-Agent")
            st.dataframe(df_agents, width="stretch")

    # ========================================================
    # TAB GRAPHES
    # ========================================================

    with tab_graphs:
        st.subheader("Reward par checkpoint")
        st.plotly_chart(
            px.bar(df_results, x="checkpoint", y="reward", color="checkpoint"),
            width="stretch"
        )

        st.subheader("Qualité globale")
        metrics = [
    "overall_score",
    "instruction_clear",
    "delegation_relevant",
    "keys_present"
            ]

        df_m = df_results.melt(
                id_vars=["checkpoint"],
                value_vars=[m for m in metrics if m in df_results.columns],
                var_name="metric",
                value_name="score"
            )

        st.plotly_chart(
                px.bar(df_m, x="checkpoint", y="score", color="metric", barmode="group"),
                width="stretch"
            )


        df_ep = df_results[df_results["epoch"] > 0]
        if not df_ep.empty:
            st.subheader("Apprentissage MAGRPO")
            st.plotly_chart(
                px.line(df_ep, x="epoch", y="reward", markers=True),
                width="stretch"
            )

    # ========================================================
    # TAB AGENTS
    # ========================================================

    with tab_agents:
        if df_agents is not None and not df_agents.empty:
            st.subheader("Qualité par agent")
            st.plotly_chart(
                px.bar(df_agents, x="agent", y="overall_quality", color="agent"),
                width="stretch"
            )

            st.subheader("Réponses complètes des agents")
            for _, row in df_agents.iterrows():
                with st.expander(
                    f"{row['agent']} | valid_json={row['valid_json']} | score={row['overall_quality']:.2f}"
                ):
                    st.code(row["raw_output"], language="text")
        else:
            st.warning("Workflow exécuté mais aucune sortie capturée.")

else:
    st.info(" Entrez une requête puis cliquez sur **Lancer l'évaluation**.")
