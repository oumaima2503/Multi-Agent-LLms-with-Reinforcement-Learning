import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.base_agents import OrchestratorAgent, ResearcherAgent, CodeWriterAgent, CriticAgent
import json

def test_magrpo_agent(agent_name: str, query: str, epoch: int = 20):
   
    print(f"\n{'='*60}")
    print(f" Test Agent: {agent_name.upper()} (MAGRPO Epoch {epoch})")
    print(f"{'='*60}")
    print(f"Requête: {query}\n")
    
    # Mapping des agents
    agent_classes = {
        "orchestrator": OrchestratorAgent,
        "researcher": ResearcherAgent,
        "code_writer": CodeWriterAgent,
        "critic": CriticAgent
    }
    
    if agent_name not in agent_classes:
        print(f" Agent inconnu: {agent_name}")
        return None
    
    # Créer l'agent
    agent_class = agent_classes[agent_name]
    
    # Modifier le chemin LoRA pour pointer vers MAGRPO
    agent = agent_class()
    
    # Remplacer le chemin LoRA
    checkpoint_name = {
        "orchestrator": "orchestrator",  # Nom correct dans les checkpoints MAGRPO
        "researcher": "researcher",
        "code_writer": "code_writer",  # Nom correct dans les checkpoints MAGRPO
        "critic": "critic"
    }[agent_name]
    
    agent.lora_path = f"checkpoints/magrpo_rl/epoch{epoch}_{checkpoint_name}_rl"
    
    # Vérifier que le checkpoint existe
    if not os.path.exists(agent.lora_path):
        print(f" Checkpoint non trouvé: {agent.lora_path}")
        print(f"   Vérifiez que le checkpoint MAGRPO existe à cet emplacement.")
        return None
    
    # Recharger le modèle
    print(f" Chargement du checkpoint: {agent.lora_path}")
    try:
        agent._load_model()
    except Exception as e:
        print(f" Erreur lors du chargement: {e}")
        return None
    
    # Tester
    print(f"\n Génération de la réponse...")
    try:
        result = agent.act(query, fast_mode=False)
        print(f"\n Résultat:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        # Vérifier le format
        if isinstance(result, dict):
            print(f"\n Format JSON valide")
            
            # Vérifier les clés attendues
            expected_keys = {
                "orchestrator": ["delegated_agent", "instruction"],
                "researcher": ["research_query", "final_answer"],
                "code_writer": ["python_code", "result_explanation"],
                "critic": ["critique_ok", "suggestions"]
            }
            
            keys = expected_keys.get(agent_name, [])
            if keys:
                missing_keys = [k for k in keys if k not in result]
                if missing_keys:
                    print(f"  Clés manquantes: {missing_keys}")
                else:
                    print(f" Toutes les clés attendues sont présentes")
        else:
            print(f"\n Format non-JSON")
        
        return result
            
    except Exception as e:
        print(f"\n Erreur: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Tests de base
    test_queries = {
        "orchestrator": "Planifie une analyse comparative entre le Pixel 8 et l'iPhone 15.",
        "researcher": "Cherche la date de sortie exacte du Google Pixel 8 Pro.",
        "code_writer": "Fais un script Python pour calculer une remise de 15% sur un prix de 899€.",
        "critic": "Évalue ceci : 'Le smartphone est cher mais puissant'."
    }
    
    # Demander l'époque si non spécifiée
    import argparse
    parser = argparse.ArgumentParser(description="Tester un agent MAGRPO")
    parser.add_argument("--agent", type=str, help="Nom de l'agent (orchestrator, researcher, code_writer, critic)")
    parser.add_argument("--query", type=str, help="Requête de test")
    parser.add_argument("--epoch", type=int, default=20, help="Époque du checkpoint (10, 15, 20)")
    parser.add_argument("--all", action="store_true", help="Tester tous les agents")
    
    args = parser.parse_args()
    
    if args.all:
        # Tester tous les agents
        for agent_name, query in test_queries.items():
            test_magrpo_agent(agent_name, query, epoch=args.epoch)
            print("\n" + "-"*60 + "\n")
    elif args.agent and args.query:
        # Tester un agent spécifique
        test_magrpo_agent(args.agent, args.query, epoch=args.epoch)
    else:
        # Mode interactif
        print("="*60)
        print(" Test des Agents MAGRPO")
        print("="*60)
        print("\nOptions:")
        print("1. Tester tous les agents: python test_magrpo_agent.py --all")
        print("2. Tester un agent spécifique: python test_magrpo_agent.py --agent orchestrator --query 'Votre requête'")
        print("\nOu testez tous les agents avec les requêtes par défaut:\n")
        
        for agent_name, query in test_queries.items():
            test_magrpo_agent(agent_name, query, epoch=args.epoch)
            print("\n" + "-"*60 + "\n")

