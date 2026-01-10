#Évaluation Globale - Combine tous les tests et évaluations.
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import time
from datetime import datetime
from collections import defaultdict

# Imports agents
from agents.base_agents import OrchestratorAgent, ResearcherAgent, CodeWriterAgent, CriticAgent

# Imports optionnels
try:
    import pandas as pd
    HAS_PANDAS = True
except Exception:
    HAS_PANDAS = False
    pd = None

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False
    plt = None
    sns = None

try:
    import nbformat
    from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
    HAS_NBFORMAT = True
except Exception:
    HAS_NBFORMAT = False


# CONFIGURATION GLOBALE

AGENTS = ["orchestrator", "researcher", "code_writer", "critic"]
EPOCHS = [10, 15, 20]
TEST_QUERIES = {
    "orchestrator": "Planifie une analyse comparative entre le Pixel 8 et l'iPhone 15.",
    "researcher": "Cherche la date de sortie exacte du Google Pixel 8 Pro.",
    "code_writer": "Fais un script Python pour calculer une remise de 15% sur un prix de 899€.",
    "critic": "Évalue ceci : 'Le smartphone est cher mais puissant'."
}

# FONCTIONS D'ÉVALUATION


def test_agent_checkpoint(agent_name: str, query: str, checkpoint_type: str, epoch: int = None):

    agent_classes = {
        "orchestrator": OrchestratorAgent,
        "researcher": ResearcherAgent,
        "code_writer": CodeWriterAgent,
        "critic": CriticAgent
    }
    
    if agent_name not in agent_classes:
        return {"success": False, "error": f"Agent inconnu: {agent_name}"}
    
    agent = agent_classes[agent_name]()

   
    if checkpoint_type == "sft":
        agent.lora_path = f"checkpoints/{agent_name}_lora"
    elif checkpoint_type == "magrpo":
        cp_map = {k: k for k in agent_classes.keys()}
        agent.lora_path = f"checkpoints/magrpo_rl/epoch{epoch}_{cp_map[agent_name]}_rl"
    else:
        return {"success": False, "error": f"Type inconnu: {checkpoint_type}"}

    if not os.path.exists(agent.lora_path):
        fallback = f"checkpoints/{agent_name}_lora"
        if os.path.exists(fallback):
            agent.lora_path = fallback

    try:
        agent._load_model()
    except Exception:
        pass

    try:
        start_time = time.time()
        result = agent.act(query, fast_mode=False)
        elapsed = time.time() - start_time

        is_json = isinstance(result, dict)
        has_expected_keys = False
        num_keys = 0
        delegated_agent = None

        if is_json:
            num_keys = len(result.keys())
            expected_keys = {
                "orchestrator": ["delegated_agent", "instruction"],
                "researcher": ["research_query", "final_answer"],
                "code_writer": ["python_code", "result_explanation"],
                "critic": ["critique_ok", "suggestions"]
            }
            keys = expected_keys.get(agent_name, [])
            has_expected_keys = all(k in result for k in keys) if keys else True
            delegated_agent = result.get("delegated_agent") if isinstance(result.get("delegated_agent"), str) else None

        # Reward proxy
        reward_value = 0.0
        reward_value += 40.0 if is_json else 0.0
        reward_value += 60.0 if has_expected_keys else 0.0

        return {
            "success": True,
            "result": result,
            "is_json": is_json,
            "has_expected_keys": has_expected_keys,
            "time": elapsed,
            "reward": reward_value,
            "num_keys": num_keys,
            "delegated_agent": delegated_agent
        }
    except Exception as e:
        return {"success": False, "error": str(e), "time": time.time() - start_time}

def evaluate_response_quality(agent_name: str, response: dict, query: str) -> dict:
    """Évalue la qualité d'une réponse."""
    scores = {
        "format_valid": isinstance(response, dict),
        "content_relevance": 0.0,
        "completeness": 0.0,
        "overall_quality": 0.0
    }

    if not isinstance(response, dict):
        return scores

    # Pertinence du contenu
    query_lower = query.lower()
    response_str = json.dumps(response, ensure_ascii=False).lower()
    query_words = set(query_lower.split())
    response_words = set(response_str.split())
    overlap = len(query_words.intersection(response_words)) / max(len(query_words), 1)
    scores["content_relevance"] = min(overlap, 1.0)

    # Complétude
    expected_keys = {
        "orchestrator": ["delegated_agent", "instruction"],
        "researcher": ["research_query", "final_answer"],
        "code_writer": ["python_code", "result_explanation"],
        "critic": ["critique_ok", "suggestions"]
    }
    keys = expected_keys.get(agent_name, [])
    if keys:
        present_keys = sum(1 for k in keys if k in response)
        scores["completeness"] = present_keys / len(keys)
    else:
        scores["completeness"] = 1.0

    # Score global
    scores["overall_quality"] = (
        0.33 * scores["format_valid"] +
        0.33 * scores["content_relevance"] +
        0.34 * scores["completeness"]
    )

    return scores

def run_comprehensive_evaluation(agent_name: str, output_dir: str = "reports"):
   
    os.makedirs(output_dir, exist_ok=True)
    agent_dir = os.path.join(output_dir, agent_name)
    os.makedirs(agent_dir, exist_ok=True)

    query = TEST_QUERIES[agent_name]
    print(f"\n{'='*80}")
    print(f"     ÉVALUATION COMPLÈTE : {agent_name.upper()}")
    print(f"{'='*80}")
    print(f"Requête: {query}\n")

    all_data = []

    
    print("     Test SFT...")
    sft_result = test_agent_checkpoint(agent_name, query, "sft")
    sft_quality = evaluate_response_quality(agent_name, sft_result.get("result", {}), query)
    
    row = {
        "checkpoint": "SFT",
        "success": sft_result.get("success", False),
        "is_json": sft_result.get("is_json", False),
        "has_expected_keys": sft_result.get("has_expected_keys", False),
        "time_s": sft_result.get("time", 0.0),
        "reward": sft_result.get("reward", 0.0),
        "num_keys": sft_result.get("num_keys", 0),
        "format_valid": sft_quality["format_valid"],
        "content_relevance": sft_quality["content_relevance"],
        "completeness": sft_quality["completeness"],
        "quality_score": sft_quality["overall_quality"],
        "delegated_agent": sft_result.get("delegated_agent")
    }
    all_data.append(row)

    # 2. Test MAGRPO (toutes les époques)
    for epoch in EPOCHS:
        print(f" Test MAGRPO Epoch {epoch}...")
        magrpo_result = test_agent_checkpoint(agent_name, query, "magrpo", epoch)
        magrpo_quality = evaluate_response_quality(agent_name, magrpo_result.get("result", {}), query)
        
        row = {
            "checkpoint": f"MAGRPO-E{epoch}",
            "success": magrpo_result.get("success", False),
            "is_json": magrpo_result.get("is_json", False),
            "has_expected_keys": magrpo_result.get("has_expected_keys", False),
            "time_s": magrpo_result.get("time", 0.0),
            "reward": magrpo_result.get("reward", 0.0),
            "num_keys": magrpo_result.get("num_keys", 0),
            "format_valid": magrpo_quality["format_valid"],
            "content_relevance": magrpo_quality["content_relevance"],
            "completeness": magrpo_quality["completeness"],
            "quality_score": magrpo_quality["overall_quality"],
            "delegated_agent": magrpo_result.get("delegated_agent")
        }
        all_data.append(row)

    # Créer DataFrame
    if not HAS_PANDAS:
        print("     pandas non disponible, skip visualizations")
        return None

    df = pd.DataFrame(all_data)
    
    # Sauvegarder CSV
    csv_path = os.path.join(agent_dir, f"{agent_name}_comprehensive_evaluation.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  CSV sauvegardé: {csv_path}")

    # Afficher tableau
    print(f"\n{'='*80}")
    print("     TABLEAU D'ÉVALUATION")
    print(f"{'='*80}\n")
    print(df.to_string(index=False))

    return df, agent_dir

def generate_visualizations(df, agent_name: str, agent_dir: str):
    
    if not HAS_MATPLOTLIB or df is None:
        return

    print(f"\n{'='*80}")
    print("     GÉNÉRATION DES VISUALISATIONS")
    print(f"{'='*80}\n")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="checkpoint", y="reward", data=df, palette="viridis", ax=ax)
    plt.title(f"{agent_name} - Comparaison Reward (SFT vs MAGRPO)")
    plt.ylabel("Reward (0-100)")
    plt.xlabel("Checkpoint")
    plt.xticks(rotation=45)
    plt.tight_layout()
    png_path = os.path.join(agent_dir, f"{agent_name}_01_reward.png")
    plt.savefig(png_path, dpi=100)
    plt.close()
    print(f"     Graph reward: {png_path}")

    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="checkpoint", y="time_s", data=df, palette="magma", ax=ax)
    plt.title(f"{agent_name} - Temps de Réponse")
    plt.ylabel("Temps (secondes)")
    plt.xlabel("Checkpoint")
    plt.xticks(rotation=45)
    plt.tight_layout()
    png_path = os.path.join(agent_dir, f"{agent_name}_02_time.png")
    plt.savefig(png_path, dpi=100)
    plt.close()
    print(f"     Graph temps: {png_path}")

    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="checkpoint", y="quality_score", data=df, palette="RdYlGn", ax=ax)
    plt.title(f"{agent_name} - Score de Qualité Global")
    plt.ylabel("Score de Qualité (0-1)")
    plt.xlabel("Checkpoint")
    plt.ylim(0, 1.1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    png_path = os.path.join(agent_dir, f"{agent_name}_03_quality.png")
    plt.savefig(png_path, dpi=100)
    plt.close()
    print(f"     Graph qualité: {png_path}")

    
    metrics_cols = ["is_json", "has_expected_keys", "format_valid", "completeness"]
    metrics_data = df[["checkpoint"] + metrics_cols].set_index("checkpoint").astype(int)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(metrics_data.T, annot=True, fmt="d", cmap="YlGn", ax=ax, cbar_kws={'label': 'Count'})
    plt.title(f"{agent_name} - Matrice de Métriques")
    plt.xlabel("Checkpoint")
    plt.tight_layout()
    png_path = os.path.join(agent_dir, f"{agent_name}_04_metrics.png")
    plt.savefig(png_path, dpi=100)
    plt.close()
    print(f"     Heatmap métriques: {png_path}")

    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    for idx, col in enumerate(["content_relevance", "completeness", "quality_score"]):
        sns.barplot(x="checkpoint", y=col, data=df, palette="coolwarm", ax=axes[idx])
        axes[idx].set_title(col.replace("_", " ").title())
        axes[idx].set_ylabel("Score")
        axes[idx].set_xlabel("Checkpoint")
        axes[idx].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    png_path = os.path.join(agent_dir, f"{agent_name}_05_quality_components.png")
    plt.savefig(png_path, dpi=100)
    plt.close()
    print(f"     Graph composantes qualité: {png_path}")

def generate_global_report(all_dfs: dict, output_dir: str):
    """Génère un rapport global comparatif."""
    if not HAS_PANDAS:
        return

    print(f"\n{'='*80}")
    print("     RAPPORT GLOBAL COMPARATIF")
    print(f"{'='*80}\n")

    
    combined_data = []
    for agent_name, df in all_dfs.items():
        df_copy = df.copy()
        df_copy.insert(0, "agent", agent_name)
        combined_data.append(df_copy)
    
    global_df = pd.concat(combined_data, ignore_index=True)

    
    csv_path = os.path.join(output_dir, "global_comprehensive_evaluation.csv")
    global_df.to_csv(csv_path, index=False)
    print(f"     CSV global: {csv_path}")

    
    print("\n     STATISTIQUES GLOBALES PAR AGENT:")
    print(f"{'Agent':<20} {'Reward Moyen':<15} {'Temps Moyen':<15} {'Qualité Moy.':<15}")
    print("-" * 65)
    
    for agent_name in AGENTS:
        agent_df = global_df[global_df["agent"] == agent_name]
        if len(agent_df) > 0:
            avg_reward = agent_df["reward"].mean()
            avg_time = agent_df["time_s"].mean()
            avg_quality = agent_df["quality_score"].mean()
            print(f"{agent_name:<20} {avg_reward:<15.2f} {avg_time:<15.2f} {avg_quality:<15.2f}")

    # Comparison SFT vs MAGRPO globale
    print("\n     COMPARAISON SFT vs MAGRPO (Global):")
    sft_data = global_df[global_df["checkpoint"] == "SFT"]
    magrpo_data = global_df[global_df["checkpoint"].str.contains("MAGRPO")]
    
    if len(sft_data) > 0 and len(magrpo_data) > 0:
        print(f"{'Métrique':<25} {'SFT':<15} {'MAGRPO':<15} {'Amélior.':<15}")
        print("-" * 70)
        
        metrics = ["reward", "time_s", "quality_score"]
        for metric in metrics:
            sft_val = sft_data[metric].mean()
            magrpo_val = magrpo_data[metric].mean()
            if metric == "time_s":
                improvement = ((sft_val - magrpo_val) / sft_val * 100) if sft_val > 0 else 0
            else:
                improvement = ((magrpo_val - sft_val) / sft_val * 100) if sft_val > 0 else 0
            print(f"{metric:<25} {sft_val:<15.2f} {magrpo_val:<15.2f} {improvement:+.1f}%")

    # Générer visualisations globales
    if HAS_MATPLOTLIB:
        print("\n     Génération des graphiques globaux...")

        # 1. Reward par agent et checkpoint
        fig, ax = plt.subplots(figsize=(12, 6))
        pivot_reward = global_df.pivot_table(values="reward", index="agent", columns="checkpoint", aggfunc="first")
        pivot_reward.plot(kind="bar", ax=ax, width=0.8)
        plt.title("Reward Global - Tous les Agents et Checkpoints")
        plt.ylabel("Reward (0-100)")
        plt.xlabel("Agent")
        plt.legend(title="Checkpoint", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.tight_layout()
        png_path = os.path.join(output_dir, "global_01_reward_comparison.png")
        plt.savefig(png_path, dpi=100)
        plt.close()
        print(f"     Graph comparaison reward: {png_path}")

        # 2. Qualité par agent
        fig, ax = plt.subplots(figsize=(12, 6))
        pivot_quality = global_df.pivot_table(values="quality_score", index="agent", columns="checkpoint", aggfunc="first")
        pivot_quality.plot(kind="bar", ax=ax, width=0.8)
        plt.title("Score de Qualité Global - Tous les Agents")
        plt.ylabel("Score de Qualité (0-1)")
        plt.xlabel("Agent")
        plt.ylim(0, 1.1)
        plt.legend(title="Checkpoint", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.tight_layout()
        png_path = os.path.join(output_dir, "global_02_quality_comparison.png")
        plt.savefig(png_path, dpi=100)
        plt.close()
        print(f"     Graph comparaison qualité: {png_path}")

        # 3. Taux de succès
        fig, ax = plt.subplots(figsize=(12, 6))
        success_data = global_df.groupby(["agent", "checkpoint"])["success"].apply(lambda x: (x.sum() / len(x)) * 100)
        success_pivot = success_data.unstack()
        success_pivot.plot(kind="bar", ax=ax, width=0.8)
        plt.title("Taux de Succès - Tous les Agents")
        plt.ylabel("Taux (%)")
        plt.xlabel("Agent")
        plt.ylim(0, 110)
        plt.legend(title="Checkpoint", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.tight_layout()
        png_path = os.path.join(output_dir, "global_03_success_rate.png")
        plt.savefig(png_path, dpi=100)
        plt.close()
        print(f"     Graph taux de succès: {png_path}")

def generate_html_report(all_dfs: dict, output_dir: str):
   
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Évaluation Globale - SFT vs MAGRPO</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background: #f5f5f5; }}
        h1, h2 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
        .summary {{ background: white; padding: 20px; border-radius: 8px; margin: 20px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        table {{ width: 100%; border-collapse: collapse; background: white; margin: 20px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        th {{ background: #4CAF50; color: white; padding: 12px; text-align: left; }}
        td {{ padding: 10px; border-bottom: 1px solid #ddd; }}
        tr:hover {{ background: #f5f5f5; }}
        .success {{ color: green; font-weight: bold; }}
        .fail {{ color: red; font-weight: bold; }}
        .warning {{ color: orange; font-weight: bold; }}
        .metric-box {{ display: inline-block; margin: 10px; padding: 15px; background: white; border-radius: 8px; min-width: 200px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #4CAF50; }}
        .metric-label {{ font-size: 12px; color: #666; margin-top: 5px; }}
        img {{ max-width: 100%; height: auto; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        .agent-section {{ background: white; padding: 20px; margin: 20px 0; border-radius: 8px; border-left: 4px solid #4CAF50; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
    </style>
</head>
<body>
    <h1>     Rapport Complet d'Évaluation - SFT vs MAGRPO</h1>
    <p>Généré: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <div class="summary">
        <h2>     Vue d'Ensemble Globale</h2>
        <div class="metric-box">
            <div class="metric-value">{len(all_dfs)}</div>
            <div class="metric-label">Agents Évalués</div>
        </div>
        <div class="metric-box">
            <div class="metric-value">{len(EPOCHS) + 1}</div>
            <div class="metric-label">Checkpoints (SFT + {len(EPOCHS)} MAGRPO)</div>
        </div>
        <div class="metric-box">
            <div class="metric-value">{len(all_dfs) * (len(EPOCHS) + 1)}</div>
            <div class="metric-label">Tests Totaux</div>
        </div>
    </div>
"""

    # Ajouter sections par agent
    for agent_name in AGENTS:
        if agent_name in all_dfs:
            df = all_dfs[agent_name]
            avg_reward = df["reward"].mean()
            avg_quality = df["quality_score"].mean()
            success_rate = (df["success"].sum() / len(df)) * 100
            
            html_content += f"""
    <div class="agent-section">
        <h2> Agent: {agent_name.upper()}</h2>
        <div class="metric-box">
            <div class="metric-value">{avg_reward:.1f}</div>
            <div class="metric-label">Reward Moyen</div>
        </div>
        <div class="metric-box">
            <div class="metric-value">{avg_quality:.2%}</div>
            <div class="metric-label">Qualité Moyenne</div>
        </div>
        <div class="metric-box">
            <div class="metric-value">{success_rate:.0f}%</div>
            <div class="metric-label">Taux de Succès</div>
        </div>
        <h3>Tableau Détaillé</h3>
        <table>
            <tr>
                <th>Checkpoint</th>
                <th>Succès</th>
                <th>JSON</th>
                <th>Clés OK</th>
                <th>Reward</th>
                <th>Qualité</th>
                <th>Temps (s)</th>
            </tr>
"""
            for _, row in df.iterrows():
                success_class = "success" if row["success"] else "fail"
                html_content += f"""
            <tr>
                <td>{row['checkpoint']}</td>
                <td class="{success_class}">{'    ' if row['success'] else '    '}</td>
                <td>{'    ' if row['is_json'] else '    '}</td>
                <td>{'    ' if row['has_expected_keys'] else '    '}</td>
                <td>{row['reward']:.1f}</td>
                <td>{row['quality_score']:.2%}</td>
                <td>{row['time_s']:.2f}</td>
            </tr>
"""
            html_content += """
        </table>
        <h3>Visualisations</h3>
"""
            # Ajouter images si elles existent
            agent_dir = os.path.join(output_dir, agent_name)
            for i in range(1, 6):
                img_path = os.path.join(agent_dir, f"{agent_name}_{i:02d}_*.png")
                import glob
                imgs = glob.glob(img_path)
                for img in imgs:
                    rel_path = os.path.relpath(img, output_dir)
                    html_content += f'        <img src="{rel_path}" alt="Graph {i}">\n'
            
            html_content += """
    </div>
"""

    html_content += """
    <div class="summary">
        <h2>     Graphiques Globaux</h2>
"""
    # Ajouter images globales
    import glob
    for img_path in glob.glob(os.path.join(output_dir, "global_*.png")):
        rel_path = os.path.relpath(img_path, output_dir)
        html_content += f'        <img src="{rel_path}" alt="Global Graph">\n'

    html_content += """
    </div>
    <footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #666;">
        <p>Rapport généré automatiquement par comprehensive_evaluation.py</p>
    </footer>
</body>
</html>
"""

    html_path = os.path.join(output_dir, "evaluation_report.html")
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"\n     Rapport HTML: {html_path}")

# FONCTION PRINCIPALE


def main(agents: list = None, output_dir: str = "reports", offline: bool = False):
    """Exécute l'évaluation complète."""
    if agents is None:
        agents = AGENTS

    if offline:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HUGGINGFACE_HUB_OFFLINE"] = "1"

    print("="*80)
    print("     ÉVALUATION GLOBALE COMPLÈTE")
    print("="*80)
    print(f"Agents: {', '.join(agents)}")
    print(f"Checkpoints: SFT + MAGRPO (epochs {EPOCHS})")
    print(f"Output: {output_dir}")
    print("="*80)

    os.makedirs(output_dir, exist_ok=True)

    all_dfs = {}
    for agent_name in agents:
        result = run_comprehensive_evaluation(agent_name, output_dir)
        if result:
            df, agent_dir = result
            all_dfs[agent_name] = df
            generate_visualizations(df, agent_name, agent_dir)

    # Rapport global
    if all_dfs:
        generate_global_report(all_dfs, output_dir)
        generate_html_report(all_dfs, output_dir)

    print(f"\n{'='*80}")
    print("     ÉVALUATION COMPLÈTE TERMINÉE")
    print(f"{'='*80}")
    print(f"\n     Rapports sauvegardés dans: {output_dir}/")
    print("\nFichiers générés:")
    print("  - CSV par agent: {agent}/")
    print("  - Graphiques par agent: {agent}/*_*.png")
    print("  - Rapport global: global_comprehensive_evaluation.csv")
    print("  - Graphiques globaux: global_*.png")
    print("  - Rapport HTML interactif: evaluation_report.html")
    print("\nOuverture du rapport HTML:")
    import webbrowser
    html_path = os.path.join(output_dir, "evaluation_report.html")
    if os.path.exists(html_path):
        print(f"  → {html_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évaluation Globale Complète")
    parser.add_argument("--agent", type=str, help="Agent à évaluer")
    parser.add_argument("--all", action="store_true", help="Tous les agents")
    parser.add_argument("--output", type=str, default="reports", help="Répertoire de sortie")
    parser.add_argument("--offline", action="store_true", help="Mode hors-ligne")

    args = parser.parse_args()

    agents = AGENTS if args.all else ([args.agent] if args.agent else AGENTS)
    main(agents=agents, output_dir=args.output, offline=args.offline)
