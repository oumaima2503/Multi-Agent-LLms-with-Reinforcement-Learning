import json
from datasets import load_dataset
from tqdm import tqdm
import os

# --- Configuration ---
CODE_WRITER_SFT_OUTPUT = "data/processed_sft/code_writer_sft.jsonl"
CODE_WRITER_SYSTEM_PROMPT = "Tu es CODE_WRITER. Réponds uniquement avec le code Python demandé, sans explication ni bloc de code Markdown. Le code doit être complet et fonctionnel."
# ---------------------

def format_code_example(instruction: str, code: str) -> dict:
    """Formate une paire instruction/code."""
    return {
        "system_prompt": CODE_WRITER_SYSTEM_PROMPT,
        "instruction": instruction,
        "response": code.strip()
    }

def prepare_code_writer_data(limit=5000):
    os.makedirs(os.path.dirname(CODE_WRITER_SFT_OUTPUT), exist_ok=True)
    all_examples = []

    # 1. MBPP
    print("Chargement et traitement de MBPP...")
    try:
        ds_mbpp = load_dataset("google-research-datasets/mbpp", "full", split=f'train[:{limit}]')
        for example in tqdm(ds_mbpp, desc="Processing MBPP"):
            instruction = f"Implémente la fonction Python suivante : {example['text']}"
            all_examples.append(format_code_example(instruction, example['code']))
    except Exception as e:
        print(f"Erreur lors du chargement de MBPP: {e}")

    # 2. HumanEvalPack (Python)
    print("Chargement et traitement de HumanEvalPack (Python)...")
    try:
        ds_humaneval = load_dataset("bigcode/humanevalpack", "python", split='train')
        # HumanEval est petit, pas besoin de limite sur le split train
        for example in tqdm(ds_humaneval, desc="Processing HumanEvalPack"):
            instruction = f"Implémente la fonction demandée par la commande de l'Orchestrateur : {example['prompt']}"
            all_examples.append(format_code_example(instruction, example['canonical_solution']))
    except Exception as e:
        print(f"Erreur lors du chargement de HumanEvalPack: {e}")

    # Écriture dans le fichier de sortie
    with open(CODE_WRITER_SFT_OUTPUT, 'w', encoding='utf-8') as f:
        for item in all_examples:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Sauvegarde de {len(all_examples)} exemples pour le Code Writer dans {CODE_WRITER_SFT_OUTPUT}")

if __name__ == "__main__":
    prepare_code_writer_data()