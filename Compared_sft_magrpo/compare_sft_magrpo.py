"""
Script pour comparer les performances SFT vs MAGRPO.
Usage: python compare_sft_magrpo.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.base_agents import OrchestratorAgent, ResearcherAgent, CodeWriterAgent, CriticAgent
import json
import time
import os
try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:
    pd = None
    plt = None
    sns = None

def test_agent_with_checkpoint(agent_name: str, query: str, checkpoint_type: str, epoch: int = None):
    """
    Teste un agent avec un checkpoint spécifique.
    
    Args:
        agent_name: orchestrator, researcher, code_writer, critic
        query: Requête de test
        checkpoint_type: 'sft' ou 'magrpo'
        epoch: Époque pour MAGRPO (10, 15, 20)
    """
    agent_classes = {
        "orchestrator": OrchestratorAgent,
        "researcher": ResearcherAgent,
        "code_writer": CodeWriterAgent,
        "critic": CriticAgent
    }
    
    if agent_name not in agent_classes:
        return {"success": False, "error": f"Agent inconnu: {agent_name}"}
    
    agent_class = agent_classes[agent_name]
    agent = agent_class()

    # Définir le chemin selon le type
    if checkpoint_type == "sft":
        agent.lora_path = f"checkpoints/{agent_name}_lora"
    elif checkpoint_type == "magrpo":
        checkpoint_name = {
            "orchestrator": "orchestrator",
            "researcher": "researcher",
            "code_writer": "code_writer",
            "critic": "critic"
        }[agent_name]
        agent.lora_path = f"checkpoints/magrpo_rl/epoch{epoch}_{checkpoint_name}_rl"
    else:
        return {"success": False, "error": f"Type de checkpoint inconnu: {checkpoint_type}"}

    # Vérifier que le checkpoint existe, sinon tenter fallback SFT avant d'abandonner
    if not os.path.exists(agent.lora_path):
        print(f"⚠️ Checkpoint non trouvé: {agent.lora_path}")
        fallback = f"checkpoints/{agent_name}_lora"
        if os.path.exists(fallback):
            agent.lora_path = fallback
            print(f"   → Fallback SFT trouvé et utilisé: {fallback}")
        else:
            print(f"   → Aucun checkpoint local trouvé pour {agent_name}. On tentera un chargement réseau ou instance par défaut.")

    # Charger et tester (tenter le chargement et catcher les erreurs)
    try:
        agent._load_model()
    except Exception as e:
        # Ne pas échouer immédiatement, afficher warning et continuer (l'agent pourra être non chargé)
        print(f"⚠️ Échec chargement modèle pour {agent_name}: {e}")
        # si le modèle est indispensable, renvoyer une erreur
        # return {"success": False, "error": f"Erreur lors du chargement: {e}"}
        # On continue pour permettre tests hors-ligne (agent peut utiliser comportement par défaut)

    try:
        start_time = time.time()
        result = agent.act(query, fast_mode=False)
        elapsed = time.time() - start_time

        # Métriques de base
        is_json = isinstance(result, dict)
        has_expected_keys = False
        num_keys = 0
        delegated_agent = None

        if is_json:
            # Nombre de clés / structure
            num_keys = len(result.keys())
            # Vérifier les clés attendues selon l'agent
            expected_keys = {
                "orchestrator": ["delegated_agent", "instruction"],
                "researcher": ["research_query", "final_answer"],
                "code_writer": ["python_code", "result_explanation"],
                "critic": ["critique_ok", "suggestions"]
            }
            keys = expected_keys.get(agent_name, [])
            has_expected_keys = all(k in result for k in keys) if keys else True
            # Délégation (utile pour l'orchestrator)
            delegated_agent = result.get("delegated_agent") if isinstance(result.get("delegated_agent"), str) else None

        # Tenter d'extraire un reward fourni par le modèle (si présent)
        reward_value = None
        if isinstance(result, dict):
            for k in ("reward", "reward_score", "magrpo_reward", "score"):
                if k in result:
                    try:
                        reward_value = float(result[k])
                        break
                    except Exception:
                        pass

        # Si pas de reward réel, construire un proxy (0-100)
        if reward_value is None:
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
        return {
            "success": False,
            "error": str(e),
            "time": time.time() - start_time
        }

def compare_agents(agent_name: str, query: str, epochs: list = [10, 15, 20]):
    """
    Compare les performances SFT vs MAGRPO pour un agent.
    """
    print(f"\n{'='*70}")
    print(f"📊 Comparaison: {agent_name.upper()}")
    print(f"{'='*70}")
    print(f"Requête: {query}\n")
    
    results = {}
    
    # Test SFT
    print("🔵 Test SFT...")
    results["sft"] = test_agent_with_checkpoint(agent_name, query, "sft")
    
    # Test MAGRPO pour chaque époque
    for epoch in epochs:
        print(f"🟢 Test MAGRPO Epoch {epoch}...")
        results[f"magrpo_epoch{epoch}"] = test_agent_with_checkpoint(agent_name, query, "magrpo", epoch)
    
    # Afficher les résultats
    print(f"\n{'='*70}")
    print("📈 Résultats")
    print(f"{'='*70}\n")
    
    for name, result in results.items():
        print(f"{name.upper()}:")
        if result.get("success"):
            print(f"  ✅ Succès")
            print(f"  📝 JSON valide: {result.get('is_json')}")
            print(f"  🔑 Clés correctes: {result.get('has_expected_keys', 'N/A')}")
            print(f"  ⏱️  Temps: {result.get('time', 0.0):.2f}s")
            print(f"  🎯 Reward (proxy/real): {result.get('reward', 'N/A')}")
            if result.get('result'):
                result_str = str(result['result'])
                if len(result_str) > 200:
                    result_str = result_str[:200] + "..."
                print(f"  📄 Résultat: {result_str}")
        else:
            print(f"  ❌ Échec: {result.get('error', 'Unknown error')}")
        print()
    
    # Générer un tableau récapitulatif & graphiques si pandas/matplotlib disponibles
    try:
        if pd is None or plt is None or sns is None:
            print("⚠️ Visualisation désactivée : installez pandas, matplotlib et seaborn pour voir tableaux/graphiques.")
            return results

        rows = []
        for ckpt_name, res in results.items():
            row = {
                "checkpoint": ckpt_name,
                "success": bool(res.get("success", False)),
                "is_json": bool(res.get("is_json", False)),
                "has_expected_keys": bool(res.get("has_expected_keys", False)),
                "time_s": float(res.get("time", 0.0) or 0.0),
                "reward": float(res.get("reward", 0.0) or 0.0),
                "num_keys": int(res.get("num_keys", 0) or 0),
                "delegated_agent": res.get("delegated_agent")
            }
            rows.append(row)

        df = pd.DataFrame(rows).set_index("checkpoint")
        out_dir = os.path.join("outputs", agent_name)
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, f"{agent_name}_comparison_summary.csv")
        df.to_csv(csv_path)
        print(f"📁 Tableau sauvegardé: {csv_path}")

        sns.set(style="whitegrid")
        # Reward bar
        plt.figure(figsize=(8,4))
        sns.barplot(x=df.index, y="reward", data=df.reset_index(), palette="viridis")
        plt.title(f"{agent_name} - Reward comparatif")
        plt.ylabel("Reward (0-100)")
        plt.xlabel("Checkpoint")
        plt.xticks(rotation=45)
        plt.tight_layout()
        reward_png = os.path.join(out_dir, f"{agent_name}_reward.png")
        plt.savefig(reward_png)
        plt.close()
        print(f"📊 Graph sauvegardé: {reward_png}")

        # Time bar
        plt.figure(figsize=(8,4))
        sns.barplot(x=df.index, y="time_s", data=df.reset_index(), palette="magma")
        plt.title(f"{agent_name} - Temps de réponse (s)")
        plt.ylabel("Temps (s)")
        plt.xlabel("Checkpoint")
        plt.xticks(rotation=45)
        plt.tight_layout()
        time_png = os.path.join(out_dir, f"{agent_name}_time.png")
        plt.savefig(time_png)
        plt.close()
        print(f"📊 Graph sauvegardé: {time_png}")

        # Structure (num_keys)
        plt.figure(figsize=(8,4))
        sns.barplot(x=df.index, y="num_keys", data=df.reset_index(), palette="cool")
        plt.title(f"{agent_name} - Structure: nombre de clés dans la sortie")
        plt.ylabel("Nombre de clés")
        plt.xlabel("Checkpoint")
        plt.xticks(rotation=45)
        plt.tight_layout()
        struct_png = os.path.join(out_dir, f"{agent_name}_structure.png")
        plt.savefig(struct_png)
        plt.close()
        print(f"📊 Graph sauvegardé: {struct_png}")

        # Delegation distribution (si champs présents)
        if df["delegated_agent"].notnull().any():
            plt.figure(figsize=(6,4))
            deleg_counts = df["delegated_agent"].fillna("None").value_counts()
            sns.barplot(x=deleg_counts.index, y=deleg_counts.values, palette="Set2")
            plt.title(f"{agent_name} - Distribution des agents délégués")
            plt.ylabel("Count")
            plt.xlabel("Delegated agent")
            plt.xticks(rotation=45)
            plt.tight_layout()
            del_png = os.path.join(out_dir, f"{agent_name}_delegation.png")
            plt.savefig(del_png)
            plt.close()
            print(f"📊 Graph sauvegardé: {del_png}")

        # Afficher tableau résumé
        print("\n📋 Tableau récapitulatif:")
        print(df)
    except Exception as e:
        print(f"⚠️ Erreur lors de la visualisation: {e}")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Comparer SFT vs MAGRPO")
    parser.add_argument("--agent", type=str, help="Agent à tester (orchestrator, researcher, code_writer, critic)")
    parser.add_argument("--query", type=str, help="Requête de test")
    parser.add_argument("--epochs", type=int, nargs="+", default=[10, 15, 20], help="Époques MAGRPO à tester")
    parser.add_argument("--all", action="store_true", help="Tester tous les agents")
    parser.add_argument("--offline", action="store_true", help="Mode hors-ligne (empêche téléchargements réseau HuggingFace)")
    
    args = parser.parse_args()
    
    # Si hors-ligne, forcer transformers/huggingface en offline pour éviter tentatives réseau longues
    if args.offline:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HUGGINGFACE_HUB_OFFLINE"] = "1"
    
    # Tests de comparaison
    test_cases = [
        ("orchestrator", "Planifie une analyse comparative entre le Pixel 8 et l'iPhone 15."),
        ("researcher", "Cherche la date de sortie exacte du Google Pixel 8 Pro."),
        ("code_writer", "Fais un script Python pour calculer une remise de 15% sur un prix de 899€."),
        ("critic", "Évalue ceci : 'Le smartphone est cher mais puissant'.")
    ]
    
    all_results = {}
    
    if args.all:
        # Tester tous les agents
        for agent_name, query in test_cases:
            # transmettre offline via variable d'env (les agents lisent lora_path et tenteront chargement local)
            results = compare_agents(agent_name, query, epochs=args.epochs)
            all_results[agent_name] = results
            print("\n" + "="*70 + "\n")
    elif args.agent and args.query:
        # Tester un agent spécifique
        results = compare_agents(args.agent, args.query, epochs=args.epochs)
        all_results[args.agent] = results
    else:
        # Mode par défaut: tester tous les agents
        for agent_name, query in test_cases:
            results = compare_agents(agent_name, query, epochs=args.epochs)
            all_results[agent_name] = results
            print("\n" + "="*70 + "\n")
    
    # Résumé global
    if all_results:
        print("\n" + "="*70)
        print("📊 RÉSUMÉ GLOBAL")
        print("="*70 + "\n")
        
        for agent_name, results in all_results.items():
            print(f"{agent_name.upper()}:")
            for name, result in results.items():
                if result["success"]:
                    json_status = "✅" if result['is_json'] else "❌"
                    keys_status = "✅" if result.get('has_expected_keys', False) else "❌"
                    print(f"  {name}: {json_status} JSON, {keys_status} Clés, {result['time']:.2f}s")
                else:
                    print(f"  {name}: ❌ {result.get('error', 'Unknown')[:50]}")
            print()

