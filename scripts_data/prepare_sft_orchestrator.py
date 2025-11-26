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
    Simule la délégation à partir de MetaMathQA.
    """
    problem = example['query']
    
    if "code" in problem.lower() or "implement" in problem.lower() or "function" in problem.lower():
        target_agent = "CODE_WRITER"
        command = f"Écris le code Python pour résoudre le problème mathématique : {problem}"
    else:
        target_agent = "RESEARCHER"
        command = f"Vérifie la formule ou le concept initial nécessaire pour résoudre : {problem}"

    initial_action = {
        "AGENT_CIBLE": target_agent, 
        "COMMANDE": command
    }
    return [format_orchestrator_example(problem, initial_action)]

def process_hotpotqa_for_delegation(example: dict) -> list:
    """
    Simule la délégation à partir de HotpotQA. La question multi-sauts est une tâche typique de l'Orchestrateur.
    """
    question = example['question']
    
    # 1. Action de délégation initiale à la recherche
    initial_action = {
        "AGENT_CIBLE": "RESEARCHER", 
        "COMMANDE": f"Recherche les informations nécessaires pour répondre à la question multi-sauts : {question}"
    }
    
    # 2. Action de synthèse finale (simulant un état après réception des faits)
    if 'answer' in example and example['answer']:
        synthesis_action = {
            "AGENT_CIBLE": "FIN",
            "COMMANDE": f"Synthétise les informations factuelles pour répondre à la question : {question}"
        }
        return [
            format_orchestrator_example(question, initial_action),
            format_orchestrator_example(f"Résultats de la recherche reçus. Infos factuelles trouvées : {example['answer']}", synthesis_action)
        ]
    
    return [format_orchestrator_example(question, initial_action)]


def prepare_orchestrator_data(limit=5000):
    os.makedirs(os.path.dirname(ORCHESTRATOR_SFT_OUTPUT), exist_ok=True)
    all_examples = []

    # 1. MetaMathQA
    print("Chargement et traitement de MetaMathQA...")
    try:
        ds_math = load_dataset("meta-math/MetaMathQA", split=f'train[:{limit}]') 
        for example in tqdm(ds_math, desc="Processing MetaMath"):
            all_examples.extend(process_metamath_for_delegation(example))
    except Exception as e:
        print(f"Erreur lors du chargement de MetaMathQA: {e}")

    # 2. HotpotQA (Correction: accès direct au split 'train')
    print("Chargement et traitement de HotpotQA...")
    try:
        # CORRECTION MAJEURE: On charge explicitement le split 'train' avec une limite.
        # Si vous utilisez ds_hotpot = load_dataset("hotpotqa/hotpot_qa", "distractor"),
        # l'accès doit se faire via ds_hotpot['train'].
        # La méthode ci-dessous est plus directe et évite l'erreur de clé.
        ds_hotpot = load_dataset("hotpotqa/hotpot_qa", "distractor", split=f'train[:{limit//2}]') 
        
        for example in tqdm(ds_hotpot, desc="Processing HotpotQA"):
            all_examples.extend(process_hotpotqa_for_delegation(example))
    except Exception as e:
        # Si l'erreur persiste, c'est probablement un problème de nom de colonne ou de format dans le dataset
        print(f"Erreur lors du chargement ou du traitement de HotpotQA. Détail de l'erreur : {e}")
        
    # Écriture dans le fichier de sortie
    with open(ORCHESTRATOR_SFT_OUTPUT, 'w', encoding='utf-8') as f:
        for item in all_examples:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Sauvegarde de {len(all_examples)} exemples pour l'Orchestrateur dans {ORCHESTRATOR_SFT_OUTPUT}")

if __name__ == "__main__":
    prepare_orchestrator_data(limit=10000)