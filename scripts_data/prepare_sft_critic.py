import json
from datasets import load_dataset
from tqdm import tqdm
import os

# --- Configuration ---
CRITIC_SFT_OUTPUT = "data/processed_sft/critic_sft.jsonl"
CRITIC_SYSTEM_PROMPT = "Tu es CRITIC. Tu es responsable d'analyser les soumissions de code ou de texte. Réponds par un rapport structuré décrivant l'erreur, la cause, et la suggestion de correction."
# ---------------------

def format_critic_example(input_to_critique: str, critique_report: str) -> dict:
    """Formate une paire input/rapport de critique."""
    return {
        "system_prompt": CRITIC_SYSTEM_PROMPT,
        "instruction": f"Analyse la soumission suivante et génère un rapport de critique : {input_to_critique}",
        "response": critique_report.strip()
    }

def synthesize_human_eval_critiques(ds_humaneval):
    """Génère des exemples en simulant des erreurs courantes dans les solutions HumanEval."""
    synthesized_examples = []
    
    # Exemples d'erreurs simulées (nécessite une connaissance du domaine)
    for i, example in enumerate(ds_humaneval):
        original_code = example['canonical_solution']
        
        # Cas 1: Erreur de cas limite (Empty Input Check)
        problem_description = f"Code à critiquer pour la tâche : {example['prompt']}"
        # Code volontairement sans gestion de cas limite:
        input_code_error_1 = original_code.replace("if not items:", "") # Exemple de suppression de check
        
        critique_1 = """
        **ERREUR** : Manque la vérification des cas limites.
        **CAUSE** : La fonction ne gère pas le cas où la liste d'entrée est vide ou nulle, ce qui pourrait provoquer une erreur d'indexation ou un comportement inattendu.
        **SUGGESTION** : Ajouter une vérification explicite au début : `if not liste: return valeur_par_defaut`.
        """
        synthesized_examples.append(format_critic_example(f"{problem_description}\n\n{input_code_error_1}", critique_1))

        # Cas 2: Erreur de performance/logique simple (simulé)
        if "for" in original_code: # Si c'est une boucle, simuler un problème d'efficacité
             critique_2 = """
            **ERREUR** : Inefficacité algorithmique.
            **CAUSE** : L'approche utilise une complexité temporelle de O(n^2) alors qu'une solution en O(n log n) ou O(n) est possible.
            **SUGGESTION** : Réévaluer l'algorithme pour réduire la complexité en utilisant des structures de données (ex: dictionnaire, ensemble) plus appropriées.
            """
             synthesized_examples.append(format_critic_example(f"{problem_description}\n\n{original_code}", critique_2))

    return synthesized_examples


def prepare_critic_data(limit=1000):
    os.makedirs(os.path.dirname(CRITIC_SFT_OUTPUT), exist_ok=True)
    all_examples = []

    print("Chargement et synthèse de critiques de code à partir de HumanEval...")
    try:
        # Nous utilisons le split 'test' pour le "seed" de nos critiques synthétisées
        ds_humaneval = load_dataset("bigcode/humanevalpack", "python", split=f'test[:{limit}]') 
        all_examples.extend(synthesize_human_eval_critiques(ds_humaneval))
    except Exception as e:
        print(f"Erreur lors du chargement de HumanEval: {e}")

    # Pour les critiques de raisonnement (logique), on pourrait utiliser MetaMath
    # ... (Ajouter la logique de synthèse pour MetaMath ici) ...

    with open(CRITIC_SFT_OUTPUT, 'w', encoding='utf-8') as f:
        for item in all_examples:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Sauvegarde de {len(all_examples)} exemples pour le Critic dans {CRITIC_SFT_OUTPUT}")

if __name__ == "__main__":
    prepare_critic_data(limit=500)