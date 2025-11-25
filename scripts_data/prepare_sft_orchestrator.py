import json
from datasets import load_dataset
from tqdm import tqdm
import os

# --- Configuration (à adapter dans config.py si vous utilisez un fichier de configuration) ---
ORCHESTRATOR_SFT_OUTPUT = "data/processed_sft/orchestrator_sft.jsonl"
ORCHESTRATOR_SYSTEM_PROMPT = "Tu es l'Orchestrateur central. Ton objectif est de planifier la prochaine étape et de déléguer la tâche à l'un des agents Exécuteurs (CODE_WRITER, CRITIC, RESEARCHER) ou de terminer la tâche (FIN). Ta réponse DOIT être un objet JSON valide avec les clés 'AGENT_CIBLE' et 'COMMANDE'."
# -----------------------------------------------------------------------------------------

def format_orchestrator_example(instruction: str, expected_next_action: dict) -> dict:
    """Formate un exemple pour l'Orchestrateur, où la réponse est une action JSON."""
    return {
        "system_prompt": ORCHESTRATOR_SYSTEM_PROMPT,
        "instruction": f"[ÉTAT ACTUEL] : {instruction}",
        "response": json.dumps(expected_next_action, ensure_ascii=False)
    }

def process_metamath_for_delegation(example: dict) -> list:
    """
    Simule la délégation à partir de MetaMathQA en utilisant la première étape
    pour demander de la recherche/vérification.
    """
    problem = example['query']
    # En SFT, nous nous concentrons sur l'étape de décision la plus probable.
    # Dans un problème complexe (MetaMath), la première action est souvent de vérifier
    # les conditions ou d'établir la méthode, ce qui peut nécessiter une recherche/un code.
    
    # Hypothèse : Le problème est nouveau, première action = Recherche ou Code.
    if "code" in problem.lower() or "implement" in problem.lower():
        target_agent = "CODE_WRITER"
        command = f"Écris le code Python pour résoudre le problème mathématique : {problem}"
    else:
        target_agent = "RESEARCHER"
        command = f"Vérifie la formule ou le concept initial nécessaire pour résoudre : {problem}"

    initial_action = {
        "AGENT_CIBLE": target_agent, 
        "COMMANDE": command
    }
    # Pour MetaMath, on génère une seule étape de délégation par question pour simplifier
    return [format_orchestrator_example(problem, initial_action)]

def process_hotpotqa_for_delegation(example: dict) -> list:
    """
    Simule la délégation à partir de HotpotQA. La question multi-sauts est une tâche typique de l'Orchestrateur.
    """
    question = example['question']
    
    # Pour HotpotQA (distractor), l'Orchestrateur doit d'abord déléguer à la recherche.
    initial_action = {
        "AGENT_CIBLE": "RESEARCHER", 
        "COMMANDE": f"Recherche les informations nécessaires pour répondre à la question : {question}"
    }
    # Pour simuler une seconde étape (synthèse), nous ajoutons un exemple où la recherche a déjà été faite
    synthesis_action = {
        "AGENT_CIBLE": "FIN",
        "COMMANDE": f"Synthétise les informations et fournis la réponse finale au problème initial : {question}"
    }
    
    # Retourne deux exemples : le premier pour la délégation, le second pour la synthèse (FIN)
    return [
        format_orchestrator_example(question, initial_action),
        format_orchestrator_example(f"Résultats de la recherche reçus pour : {question}. Les faits pertinents sont les suivants : {example['supporting_facts']}", synthesis_action)
    ]

def prepare_orchestrator_data(limit=10000):
    os.makedirs(os.path.dirname(ORCHESTRATOR_SFT_OUTPUT), exist_ok=True)
    all_examples = []

    # 1. MetaMathQA
    print("Chargement et traitement de MetaMathQA...")
    try:
        # Limiter à 10% ou un nombre spécifique pour l'exemple
        ds_math = load_dataset("meta-math/MetaMathQA", split=f'train[:{limit}]') 
        for example in tqdm(ds_math, desc="Processing MetaMath"):
            all_examples.extend(process_metamath_for_delegation(example))
    except Exception as e:
        print(f"Erreur lors du chargement de MetaMathQA: {e}")

    # 2. HotpotQA
    print("Chargement et traitement de HotpotQA...")
    try:
        ds_hotpot = load_dataset("hotpotqa", "distractor", split=f'train[:{limit//2}]') 
        for example in tqdm(ds_hotpot, desc="Processing HotpotQA"):
            all_examples.extend(process_hotpotqa_for_delegation(example))
    except Exception as e:
        print(f"Erreur lors du chargement de HotpotQA: {e}")
    
    # Écriture dans le fichier de sortie
    with open(ORCHESTRATOR_SFT_OUTPUT, 'w', encoding='utf-8') as f:
        for item in all_examples:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Sauvegarde de {len(all_examples)} exemples pour l'Orchestrateur dans {ORCHESTRATOR_SFT_OUTPUT}")

if __name__ == "__main__":
    prepare_orchestrator_data(limit=5000) # Utiliser une limite raisonnable pour un test