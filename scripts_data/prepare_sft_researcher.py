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
        # HotpotQA fonctionne, les clés 'question' et 'answer' sont correctes.
        ds_hotpot = load_dataset("hotpotqa/hotpot_qa", "distractor", split=f'train[:{limit//2}]') 
        for example in tqdm(ds_hotpot, desc="Processing HotpotQA"):
            if example['answer']:
                all_examples.append(format_researcher_example(example['question'], example['answer']))
    except Exception as e:
        print(f"Erreur lors du chargement de HotpotQA: {e}")

    # 2. Natural Questions (NQ) - CORRECTION FINALE (NQ)
    print("Chargement et traitement de Natural Questions...")
    try:
        ds_nq = load_dataset("sentence-transformers/natural-questions", split=f'train[:{limit}]')
        for example in tqdm(ds_nq, desc="Processing Natural Questions"):
            # Correction: Changement de 'question_text' (qui causait l'erreur) à 'question'.
            # Le champ 'answer' est un dictionnaire contenant 'answer_text' qui est une liste.
            question = example.get('question', example.get('question_text')) # Tente de récupérer 'question' puis 'question_text'
            
            if question and 'answer' in example and example['answer']['answer_text']:
                 # On prend la première réponse courte disponible dans la liste
                answer_texts = example['answer']['answer_text']
                if answer_texts:
                    # Ici, on utilise question au lieu de example['question_text']
                    all_examples.append(format_researcher_example(question, answer_texts[0]))
    except Exception as e:
        print(f"Erreur lors du chargement de NQ: {e}")

    # 3. ELI5 (pour les explications concises) - CORRECTION FINALE (ELI5)
    print("Chargement et traitement de ELI5...")
    try:
        ds_eli5 = load_dataset("sentence-transformers/eli5", split=f'train[:{limit}]')
        for example in tqdm(ds_eli5, desc="Processing ELI5"):
            # Correction: Changement de 'title' (qui causait l'erreur) à 'q_title' ou 'title' avec fallback.
            question = example.get('q_title', example.get('title')) # Tente de récupérer 'q_title' puis 'title'
            
            # On s'assure que 'answers' est présent et que la liste 'text' n'est pas vide.
            if question and 'answers' in example and 'text' in example['answers'] and example['answers']['text']:
                 # Utiliser la première réponse longue/synthétisée
                all_examples.append(format_researcher_example(question, example['answers']['text'][0]))
    except Exception as e:
        print(f"Erreur lors du chargement de ELI5: {e}")

    # Écriture dans le fichier de sortie
    with open(RESEARCHER_SFT_OUTPUT, 'w', encoding='utf-8') as f:
        for item in all_examples:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Sauvegarde de {len(all_examples)} exemples pour le Researcher dans {RESEARCHER_SFT_OUTPUT}")

if __name__ == "__main__":
    prepare_researcher_data()