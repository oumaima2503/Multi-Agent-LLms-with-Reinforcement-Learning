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
        
        # Métriques
        is_json = isinstance(result, dict)
        has_expected_keys = False
        
        if is_json:
            # Vérifier les clés attendues selon l'agent
            expected_keys = {
                "orchestrator": ["delegated_agent", "instruction"],
                "researcher": ["research_query", "final_answer"],
                "code_writer": ["python_code", "result_explanation"],
                "critic": ["critique_ok", "suggestions"]
            }
            keys = expected_keys.get(agent_name, [])
            has_expected_keys = all(k in result for k in keys) if keys else True
        
        return {
            "success": True,
            "result": result,
            "is_json": is_json,
            "has_expected_keys": has_expected_keys,
            "time": elapsed
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
        if result["success"]:
            print(f"  ✅ Succès")
            print(f"  📝 JSON valide: {result['is_json']}")
            print(f"  🔑 Clés correctes: {result.get('has_expected_keys', 'N/A')}")
            print(f"  ⏱️  Temps: {result['time']:.2f}s")
            if result.get('result'):
                result_str = str(result['result'])
                if len(result_str) > 100:
                    result_str = result_str[:100] + "..."
                print(f"  📄 Résultat: {result_str}")
        else:
            print(f"  ❌ Échec: {result.get('error', 'Unknown error')}")
        print()
    
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

