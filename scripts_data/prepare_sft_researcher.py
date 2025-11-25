import json
from datasets import load_dataset
from tqdm import tqdm
import os

# --- Configuration ---
RESEARCHER_SFT_OUTPUT = "data/processed_sft/researcher_sft.jsonl"
RESEARCHER_SYSTEM_PROMPT = "Tu es RESEARCHER. Fournis une réponse factuelle, concise et directe à la commande de recherche. Évite les préambules et les fioritures."
# ---------------------

def format_researcher_example(question: str, answer: str) -> dict:
    """Formate une paire question/réponse factuelle."""
    return {
        "system_prompt": RESEARCHER_SYSTEM_PROMPT,
        "instruction": f"Recherche et synthétise l'information pour la question : {question}",
        "response": answer.strip()
    }

def prepare_researcher_data(limit=5000):
    os.makedirs(os.path.dirname(RESEARCHER_SFT_OUTPUT), exist_ok=True)
    all_examples = []

    # 1. HotpotQA
    print("Chargement et traitement de HotpotQA...")
    try:
        ds_hotpot = load_dataset("hotpotqa", "distractor", split=f'train[:{limit//2}]') 
        for example in tqdm(ds_hotpot, desc="Processing HotpotQA"):
            if example['answer']:
                all_examples.append(format_researcher_example(example['question'], example['answer']))
    except Exception as e:
        print(f"Erreur lors du chargement de HotpotQA: {e}")

    # 2. Natural Questions (NQ)
    print("Chargement et traitement de Natural Questions...")
    try:
        ds_nq = load_dataset("sentence-transformers/natural-questions", split=f'train[:{limit}]')
        for example in tqdm(ds_nq, desc="Processing Natural Questions"):
            if example['answer_text'] and example['question_text']:
                # Utiliser la première réponse courte comme réponse factuelle
                all_examples.append(format_researcher_example(example['question_text'], example['answer_text'][0]))
    except Exception as e:
        print(f"Erreur lors du chargement de NQ: {e}")

    # 3. ELI5 (pour les explications concises)
    print("Chargement et traitement de ELI5...")
    try:
        ds_eli5 = load_dataset("sentence-transformers/eli5", split=f'train[:{limit}]')
        for example in tqdm(ds_eli5, desc="Processing ELI5"):
            # Utiliser le titre comme question et la première réponse longue/synthétisée
            if example['answers']['text'] and example['title']:
                all_examples.append(format_researcher_example(example['title'], example['answers']['text'][0]))
    except Exception as e:
        print(f"Erreur lors du chargement de ELI5: {e}")

    # Écriture dans le fichier de sortie
    with open(RESEARCHER_SFT_OUTPUT, 'w', encoding='utf-8') as f:
        for item in all_examples:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Sauvegarde de {len(all_examples)} exemples pour le Researcher dans {RESEARCHER_SFT_OUTPUT}")

if __name__ == "__main__":
    prepare_researcher_data()