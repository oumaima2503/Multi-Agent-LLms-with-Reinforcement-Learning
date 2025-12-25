"""
Interface Streamlit interactive pour tester / comparer / évaluer agents SFT vs MAGRPO.

Usage:
    pip install streamlit pandas plotly
    streamlit run interactive_ui.py
"""
import os
import json
import time
import argparse

try:
    import streamlit as st
    import pandas as pd
    import plotly.express as px
except Exception as e:
    st = None
    pd = None
    px = None

from Compared_sft_magrpo.compare_sft_magrpo import test_agent_with_checkpoint
from evaluate_response_quality import evaluate_agent_response

try:
    from interact_magrpo import MAGRPOMultiAgentSystem
except Exception:
    MAGRPOMultiAgentSystem = None

AGENTS = ["orchestrator", "researcher", "code_writer", "critic"]
DEFAULT_EPOCHS = [10, 15, 20]

st.set_page_config(page_title="MAGRPO vs SFT - Test Interface", layout="wide")

st.title("🤖 Test des Agents SFT vs MAGRPO")
st.markdown("**Interface interactive** : saisissez votre requête, choisissez un agent et comparez les performances.")

# ============================================================================
# SECTION CONTRÔLE (Sidebar)
# ============================================================================
with st.sidebar:
    st.header("⚙️ Paramètres")
    agent = st.selectbox("Sélectionnez un agent", AGENTS, index=0)
    run_sft = st.checkbox("Tester SFT", value=True)
    run_magrpo = st.checkbox("Tester MAGRPO (epochs)", value=True)
    epochs = st.multiselect("Époques MAGRPO à tester", DEFAULT_EPOCHS, default=DEFAULT_EPOCHS)
    offline = st.checkbox("Mode hors-ligne", value=False)
    st.markdown("---")
    st.write("**Instructions:**\n1. Saisissez votre requête\n2. Cliquez 'Lancer le test'\n3. Observez réponses + métriques")

# ============================================================================
# SECTION PRINCIPALE : Saisie et Résultats
# ============================================================================

st.header("📝 Votre Requête")
query = st.text_area(
    "Saisissez votre question (tous les agents répondront à cette même requête)",
    value="",
    height=80,
    placeholder="Ex: Planifie une analyse comparative entre le Pixel 8 et l'iPhone 15."
)

if st.button("🚀 Lancer le test", key="submit_button"):
    if not query.strip():
        st.error("❌ Veuillez saisir une requête valide.")
    else:
        if offline:
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HUGGINGFACE_HUB_OFFLINE"] = "1"

        st.info(f"⏳ Tests en cours pour l'agent **{agent.upper()}**... (ceci peut prendre quelques minutes)")
        
        results = []
        start_all = time.time()

        # Run SFT
        if run_sft:
            with st.spinner(f"🔵 Test SFT..."):
                try:
                    res = test_agent_with_checkpoint(agent, query, "sft")
                except Exception as e:
                    res = {"success": False, "error": str(e)}
                
                res_row = {
                    "checkpoint": "SFT",
                    "success": bool(res.get("success", False)),
                    "is_json": bool(res.get("is_json", False)),
                    "has_expected_keys": bool(res.get("has_expected_keys", False)),
                    "time_s": float(res.get("time", 0.0) or 0.0),
                    "reward": float(res.get("reward", 0.0) or 0.0),
                    "num_keys": int(res.get("num_keys", 0) or 0),
                    "delegated_agent": res.get("delegated_agent"),
                    "raw_result": res.get("result")
                }
                try:
                    quality = evaluate_agent_response(agent, res.get("result", {}) if isinstance(res.get("result"), dict) else {}, query)
                except Exception:
                    quality = {}
                res_row.update(quality)
                results.append(res_row)

        # Run MAGRPO epochs
        if run_magrpo:
            for ep in epochs:
                with st.spinner(f"🟢 Test MAGRPO Epoch {ep}..."):
                    try:
                        res = test_agent_with_checkpoint(agent, query, "magrpo", epoch=ep)
                    except Exception as e:
                        res = {"success": False, "error": str(e)}
                    
                    res_row = {
                        "checkpoint": f"MAGRPO_E{ep}",
                        "success": bool(res.get("success", False)),
                        "is_json": bool(res.get("is_json", False)),
                        "has_expected_keys": bool(res.get("has_expected_keys", False)),
                        "time_s": float(res.get("time", 0.0) or 0.0),
                        "reward": float(res.get("reward", 0.0) or 0.0),
                        "num_keys": int(res.get("num_keys", 0) or 0),
                        "delegated_agent": res.get("delegated_agent"),
                        "raw_result": res.get("result")
                    }
                    try:
                        quality = evaluate_agent_response(agent, res.get("result", {}) if isinstance(res.get("result"), dict) else {}, query)
                    except Exception:
                        quality = {}
                    res_row.update(quality)
                    results.append(res_row)

        elapsed_all = time.time() - start_all
        st.success(f"✅ Tests terminés en {elapsed_all:.1f}s")

        # ================================================================
        # AFFICHAGE : Deux colonnes (Réponses | Métriques)
        # ================================================================
        col_left, col_right = st.columns([2, 1.5])

        # COLONNE GAUCHE : Réponses brutes
        with col_left:
            st.subheader("📄 Réponses Brutes des Agents")
            st.write(f"*Requête: {query[:100]}...*" if len(query) > 100 else f"*Requête: {query}*")
            
            for r in results:
                cp = r.get("checkpoint", "?")
                raw = r.get("raw_result")
                
                exp_label = f"{cp}  •  ✅ {r.get('success')}  •  🎯 {r.get('reward', 0):.0f}"
                with st.expander(exp_label, expanded=(cp == "SFT")):
                    if isinstance(raw, dict):
                        # Afficher champs importants en priorité
                        code = raw.get("python_code") or raw.get("code")
                        final_answer = raw.get("final_answer")
                        instruction = raw.get("instruction")
                        research_query = raw.get("research_query")
                        critique_ok = raw.get("critique_ok")
                        suggestions = raw.get("suggestions")
                        
                        if code:
                            st.markdown("**Code Python :**")
                            st.code(code, language="python")
                        if final_answer:
                            st.markdown("**Réponse Finale :**")
                            st.write(final_answer)
                        if instruction:
                            st.markdown("**Instruction :**")
                            st.write(instruction)
                        if research_query:
                            st.markdown("**Requête de Recherche :**")
                            st.write(research_query)
                        if critique_ok is not None:
                            st.markdown(f"**Critique OK :** {'✅ Oui' if critique_ok else '❌ Non'}")
                        if suggestions:
                            st.markdown("**Suggestions :**")
                            st.write(suggestions)
                        
                        # Afficher les autres champs en JSON
                        other_keys = {k: v for k, v in raw.items() 
                                     if k not in ("python_code", "code", "final_answer", "instruction", 
                                                 "research_query", "critique_ok", "suggestions")}
                        if other_keys:
                            st.markdown("**Autres champs :**")
                            st.json(other_keys)
                    else:
                        st.write(str(raw))

        # COLONNE DROITE : Métriques et Évaluation
        with col_right:
            st.subheader("📊 Métriques & Évaluation")
            
            try:
                df = pd.DataFrame(results)
                
                # Afficher les KPIs en haut
                st.markdown("**Résumé Global**")
                avg_reward = df["reward"].mean()
                avg_time = df["time_s"].mean()
                success_rate = (df["success"].sum() / len(df)) * 100
                
                st.metric("Reward Moyen", f"{avg_reward:.0f}", delta=None)
                st.metric("Temps Moyen (s)", f"{avg_time:.2f}", delta=None)
                st.metric("Taux de Succès", f"{success_rate:.0f}%", delta=None)
                
                # Tableau récapitulatif
                st.markdown("**Tableau Comparatif**")
                display_df = df[["checkpoint", "success", "reward", "time_s", "is_json", "has_expected_keys"]].copy()
                st.dataframe(display_df, use_container_width=True)
                
                # Graphiques
                st.markdown("**Visualisations**")
                try:
                    fig_reward = px.bar(
                        df, x="checkpoint", y="reward",
                        title="Reward", text="reward",
                        color="reward", color_continuous_scale="Viridis"
                    )
                    st.plotly_chart(fig_reward, use_container_width=True)
                    
                    fig_time = px.bar(
                        df, x="checkpoint", y="time_s",
                        title="Temps (s)", text="time_s",
                        color="time_s", color_continuous_scale="Reds"
                    )
                    st.plotly_chart(fig_time, use_container_width=True)
                except Exception as e:
                    st.warning(f"⚠️ Erreur graphiques: {e}")
                
                # Évaluation détaillée
                st.markdown("**Évaluation Détaillée**")
                eval_cols = [col for col in df.columns if col in 
                           ("format_valid", "content_relevance", "completeness", "overall_quality", "overall_score")]
                if eval_cols:
                    eval_df = df[["checkpoint"] + eval_cols].set_index("checkpoint")
                    st.dataframe(eval_df, use_container_width=True)
                
                # Export
                st.markdown("**Export**")
                csv = df.to_csv(index=False)
                st.download_button(
                    "📥 Télécharger CSV",
                    csv,
                    file_name=f"{agent}_results_{int(time.time())}.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"Erreur traitement résultats: {e}")

else:
    st.info("👉 Saisissez une requête et cliquez sur **'Lancer le test'** pour démarrer.")

# ============================================================================
# SERVEUR FLASK (optionnel)
# ============================================================================
def create_flask_app():
    try:
        from flask import Flask, request, jsonify, render_template_string
    except Exception:
        raise RuntimeError("Installez flask: pip install flask")

    app = Flask(__name__)

    INDEX_HTML = """
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8"/>
      <title>MAGRPO vs SFT - Web Interface</title>
      <style>
        body{font-family:Arial,Helvetica,sans-serif;margin:20px;background:#f5f5f5}
        .container{max-width:1200px;margin:0 auto;background:white;padding:20px;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.1)}
        h2{color:#333}
        textarea{width:100%;height:120px;padding:8px;border:1px solid #ddd;border-radius:4px}
        .row{display:flex;gap:12px;margin:12px 0}
        .col{flex:1}
        button{background:#4CAF50;color:white;padding:10px 20px;border:none;border-radius:4px;cursor:pointer;font-weight:bold}
        button:hover{background:#45a049}
        pre{background:#f4f4f4;padding:12px;border-radius:4px;overflow:auto;max-height:400px;border-left:4px solid #4CAF50}
        .metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:20px 0}
        .metric-box{background:#f9f9f9;padding:15px;border-radius:4px;border:1px solid #ddd}
        .metric-value{font-size:24px;font-weight:bold;color:#4CAF50}
        .metric-label{font-size:12px;color:#666;margin-top:5px}
      </style>
    </head>
    <body>
      <div class="container">
        <h2>🤖 MAGRPO vs SFT - Web Interface</h2>
        <div>
          <label>Agent:</label>
          <select id="agent">
            <option value="orchestrator">orchestrator</option>
            <option value="researcher">researcher</option>
            <option value="code_writer">code_writer</option>
            <option value="critic">critic</option>
          </select>
        </div>
        <div style="margin:12px 0">
          <label>Requête:</label>
          <textarea id="query" placeholder="Saisissez votre requête ici..."></textarea>
        </div>
        <div class="row">
          <div class="col"><label><input type="checkbox" id="run_sft" checked> SFT</label></div>
          <div class="col"><label><input type="checkbox" id="run_magrpo" checked> MAGRPO</label></div>
          <div class="col"><label>Epochs:<input type="text" id="epochs" value="10,15,20" style="width:80px"/></label></div>
        </div>
        <button onclick="runTest()">🚀 Lancer le test</button>
        <div id="status" style="margin:15px 0;color:#666"></div>
        <h3>📄 Réponses Brutes</h3>
        <pre id="rawout">—</pre>
        <h3>📊 Métriques</h3>
        <pre id="tableout">—</pre>
      </div>
      <script>
      async function runTest(){
        document.getElementById('status').innerText = "⏳ Chargement...";
        const agent = document.getElementById('agent').value;
        const query = document.getElementById('query').value;
        const run_sft = document.getElementById('run_sft').checked;
        const run_magrpo = document.getElementById('run_magrpo').checked;
        const epochs = document.getElementById('epochs').value.split(',').map(s=>parseInt(s.trim())).filter(Boolean);
        const payload = { agent, query, run_sft, run_magrpo, epochs };
        const res = await fetch('/api/test', {
          method:'POST',
          headers:{'Content-Type':'application/json'},
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        document.getElementById('status').innerText = data.message || "✅ Terminé";
        const rawBlocks = data.results.map(r => {
          const raw = r.raw_result !== undefined ? r.raw_result : r.result || {};
          const header = `${r.checkpoint} | Success:${r.success} | Reward:${r.reward} | Time:${(r.time_s||0).toFixed(2)}s`;
          const body = typeof raw === "object" ? JSON.stringify(raw, null, 2) : String(raw);
          return header + "\\n" + body;
        });
        document.getElementById('rawout').innerText = rawBlocks.join("\\n\\n---\\n\\n");
        let rows = data.results.map(r => `${r.checkpoint} | success:${r.success} | reward:${r.reward} | time:${(r.time_s||0).toFixed(2)}s`);
        document.getElementById('tableout').innerText = rows.join('\\n');
      }
      </script>
    </body>
    </html>
    """

    @app.route("/", methods=["GET"])
    def index():
        return render_template_string(INDEX_HTML)

    @app.route("/api/test", methods=["POST"])
    def api_test():
        payload = request.get_json() or {}
        agent = payload.get("agent")
        query = payload.get("query", "")
        run_sft = bool(payload.get("run_sft", True))
        run_magrpo = bool(payload.get("run_magrpo", True))
        epochs = payload.get("epochs", [10,15,20])
        
        start_all = time.time()
        results = []

        if run_sft:
            res = test_agent_with_checkpoint(agent, query, "sft")
            q = evaluate_agent_response(agent, res.get("result", {}) if isinstance(res.get("result"), dict) else {}, query)
            row = {"checkpoint":"SFT", "success": bool(res.get("success", False)), "reward": res.get("reward", 0.0), 
                   "time_s": res.get("time", 0.0), "raw_result": res.get("result"), "quality": q}
            results.append(row)

        if run_magrpo:
            for ep in epochs:
                res = test_agent_with_checkpoint(agent, query, "magrpo", epoch=ep)
                q = evaluate_agent_response(agent, res.get("result", {}) if isinstance(res.get("result"), dict) else {}, query)
                row = {"checkpoint":f"MAGRPO_E{ep}", "success": bool(res.get("success", False)), "reward": res.get("reward", 0.0),
                       "time_s": res.get("time", 0.0), "raw_result": res.get("result"), "quality": q}
                results.append(row)

        elapsed = time.time() - start_all
        return jsonify({"message": f"Terminé en {elapsed:.2f}s", "results": results})

    return app

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive UI (Streamlit or Flask)")
    parser.add_argument("--server", action="store_true", help="Lance serveur Flask")
    parser.add_argument("--port", type=int, default=8050, help="Port Flask")
    args = parser.parse_args()

    if args.server:
        app = create_flask_app()
        print(f"▶️ Serveur web sur http://localhost:{args.port}")
        app.run(host="0.0.0.0", port=args.port, debug=False)