import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
import time
from test_magrpo_agent import test_magrpo_agent
from Compared_sft_magrpo.compare_sft_magrpo import test_agent_with_checkpoint

def evaluate_on_dataset(agent_name: str, dataset_path: str, checkpoint_type: str = "magrpo", epoch: int = 20, max_samples: int = None):
 
    if not os.path.exists(dataset_path):
        print(f"     Dataset non trouvé: {dataset_path}")
        return None
    
    # Charger le dataset
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = [json.loads(line) for line in f]
    
    # Limiter si nécessaire
    if max_samples and len(dataset) > max_samples:
        dataset = dataset[:max_samples]
        print(f"      Dataset limité à {max_samples} échantillons pour l'évaluation")
    
    metrics = {
        "total": len(dataset),
        "success": 0,
        "json_valid": 0,
        "keys_correct": 0,
        "avg_time": 0.0,
        "errors": []
    }
    
    print(f"\n    Évaluation {agent_name.upper()} ({checkpoint_type})")
    print(f"Dataset: {len(dataset)} échantillons\n")
    
    # Clés attendues par agent
    expected_keys = {
        "orchestrator": ["delegated_agent", "instruction"],
        "researcher": ["research_query", "final_answer"],
        "code_writer": ["python_code", "result_explanation"],
        "critic": ["critique_ok", "suggestions"]
    }
    
    for i, sample in enumerate(dataset):
        query = sample.get("instruction", sample.get("query", ""))
        if not query:
            continue
        
        start_time = time.time()
        
        try:
            if checkpoint_type == "magrpo":
                result = test_magrpo_agent(agent_name, query, epoch)
                elapsed = time.time() - start_time
                
                if result:
                    metrics["success"] += 1
                    if isinstance(result, dict):
                        metrics["json_valid"] += 1
                        # Vérifier les clés
                        keys = expected_keys.get(agent_name, [])
                        if keys and all(k in result for k in keys):
                            metrics["keys_correct"] += 1
                    metrics["avg_time"] += elapsed
                else:
                    metrics["errors"].append(f"Sample {i+1}: Résultat None")
            else:
                result = test_agent_with_checkpoint(agent_name, query, checkpoint_type, epoch)
                elapsed = time.time() - start_time
                
                if result["success"]:
                    metrics["success"] += 1
                    if result["is_json"]:
                        metrics["json_valid"] += 1
                    if result.get("has_expected_keys", False):
                        metrics["keys_correct"] += 1
                    metrics["avg_time"] += elapsed
                else:
                    metrics["errors"].append(f"Sample {i+1}: {result.get('error', 'Unknown')}")
        except Exception as e:
            metrics["errors"].append(f"Sample {i+1}: {str(e)}")
        
        if (i + 1) % 10 == 0:
            print(f"  Progression: {i+1}/{len(dataset)}")
    
    # Calculer les moyennes
    if metrics["total"] > 0:
        metrics["success_rate"] = metrics["success"] / metrics["total"]
        metrics["json_rate"] = metrics["json_valid"] / metrics["total"]
        metrics["keys_rate"] = metrics["keys_correct"] / metrics["total"]
        if metrics["success"] > 0:
            metrics["avg_time"] = metrics["avg_time"] / metrics["success"]
    
    return metrics

def compare_sft_vs_magrpo(agent_name: str, dataset_path: str, epochs: list = [10, 15, 20], max_samples: int = 100):
    """
    Compare SFT vs MAGRPO pour un agent.
    """
    print(f"\n{'='*70}")
    print(f"    Comparaison SFT vs MAGRPO : {agent_name.upper()}")
    print(f"{'='*70}\n")
    
    # Évaluer SFT
    print("     Évaluation SFT...")
    sft_metrics = evaluate_on_dataset(agent_name, dataset_path, "sft", max_samples=max_samples)
    
    # Évaluer MAGRPO pour chaque époque
    magrpo_results = {}
    for epoch in epochs:
        print(f"\n     Évaluation MAGRPO Epoch {epoch}...")
        magrpo_metrics = evaluate_on_dataset(agent_name, dataset_path, "magrpo", epoch=epoch, max_samples=max_samples)
        if magrpo_metrics:
            magrpo_results[epoch] = magrpo_metrics
    
    # Afficher les résultats
    print(f"\n{'='*70}")
    print("    Résultats Comparatifs")
    print(f"{'='*70}\n")
    
    if sft_metrics:
        
        header = f"{'Métrique':<25} {'SFT':<15} "
        for epoch in epochs:
            header += f"{ 'MAGRPO E{epoch}':<15}     "
        print(header.rstrip())
        print(f"{'-'*70}")
        
        # Taux de succès
        row = f"{'Taux de succès':<25} {sft_metrics['success_rate']:.2%} "
        for epoch in epochs:
            if epoch in magrpo_results:
                row += f"{magrpo_results[epoch]['success_rate']:.2%} "
            else:
                row += f"{'N/A':<15} "
        print(row)
        
        # JSON valide
        row = f"{'JSON valide':<25} {sft_metrics['json_rate']:.2%} "
        for epoch in epochs:
            if epoch in magrpo_results:
                row += f"{magrpo_results[epoch]['json_rate']:.2%} "
            else:
                row += f"{'N/A':<15} "
        print(row)
        
        # Clés correctes
        row = f"{'Clés correctes':<25} {sft_metrics['keys_rate']:.2%} "
        for epoch in epochs:
            if epoch in magrpo_results:
                row += f"{magrpo_results[epoch]['keys_rate']:.2%} "
            else:
                row += f"{'N/A':<15} "
        print(row)
        
        # Temps moyen
        row = f"{'Temps moyen (s)':<25} {sft_metrics['avg_time']:.2f} "
        for epoch in epochs:
            if epoch in magrpo_results:
                row += f"{magrpo_results[epoch]['avg_time']:.2f} "
            else:
                row += f"{'N/A':<15} "
        print(row)
    
    return {
        "sft": sft_metrics,
        "magrpo": magrpo_results
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Évaluer les agents sur un dataset")
    parser.add_argument("--agent", type=str, help="Nom de l'agent")
    parser.add_argument("--dataset", type=str, help="Chemin vers le dataset")
    parser.add_argument("--checkpoint", type=str, default="magrpo", choices=["sft", "magrpo"], help="Type de checkpoint")
    parser.add_argument("--epoch", type=int, default=20, help="Époque pour MAGRPO")
    parser.add_argument("--max-samples", type=int, default=100, help="Nombre max d'échantillons à évaluer")
    parser.add_argument("--compare", action="store_true", help="Comparer SFT vs MAGRPO")
    parser.add_argument("--all", action="store_true", help="Évaluer tous les agents")
    
    args = parser.parse_args()
    
    # Datasets par défaut
    datasets = {
        "orchestrator": "data/processed_sft/orchestrator_sft.jsonl",
        "researcher": "data/processed_sft/researcher_sft.jsonl",
        "code_writer": "data/processed_sft/code_writer_sft.jsonl",
        "critic": "data/processed_sft/critic_sft.jsonl"
    }
    
    if args.compare:
        # Comparer SFT vs MAGRPO
        if args.agent and args.agent in datasets:
            compare_sft_vs_magrpo(args.agent, datasets[args.agent], epochs=[10, 15, 20], max_samples=args.max_samples)
        elif args.all:
            for agent_name, dataset_path in datasets.items():
                if os.path.exists(dataset_path):
                    compare_sft_vs_magrpo(agent_name, dataset_path, epochs=[10, 15, 20], max_samples=args.max_samples)
                    print("\n" + "="*70 + "\n")
        else:
            print("Usage: python evaluate_on_dataset.py --compare --agent orchestrator")
            print("   ou: python evaluate_on_dataset.py --compare --all")
    elif args.agent and args.dataset:
        # Évaluer un agent spécifique
        metrics = evaluate_on_dataset(args.agent, args.dataset, args.checkpoint, args.epoch, args.max_samples)
        if metrics:
            print(f"\n{'='*70}")
            print("    Résultats")
            print(f"{'='*70}\n")
            print(f"Taux de succès: {metrics['success_rate']:.2%}")
            print(f"JSON valide: {metrics['json_rate']:.2%}")
            print(f"Clés correctes: {metrics['keys_rate']:.2%}")
            print(f"Temps moyen: {metrics['avg_time']:.2f}s")
    else:
        print("Usage:")
        print("  python evaluate_on_dataset.py --compare --all")
        print("  python evaluate_on_dataset.py --compare --agent orchestrator")
        print("  python evaluate_on_dataset.py --agent orchestrator --dataset data/processed_sft/orchestrator_sft.jsonl")

