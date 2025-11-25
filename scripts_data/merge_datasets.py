import json
import os
import random

# Définition des chemins de fichiers
SFT_FILES = [
    "data/processed_sft/orchestrator_sft.jsonl",
    "data/processed_sft/code_writer_sft.jsonl",
    "data/processed_sft/critic_sft.jsonl",
    "data/processed_sft/researcher_sft.jsonl",
]
MERGED_SFT_OUTPUT = "data/processed_sft/merged_sft_train.jsonl"

def merge_and_sample_datasets(file_paths: list, target_total_size=50000):
    """
    Charge les fichiers JSONL, les fusionne et les échantillonne pour créer un
    dataset de SFT potentiellement plus équilibré et mélangé.
    """
    all_data = []
    
    # 1. Chargement de tous les exemples
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Attention: Le fichier {file_path} est manquant et sera ignoré.")
            continue
            
        print(f"Chargement de {file_path}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    all_data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Erreur de décodage JSON dans {file_path}: {e}")

    print(f"Total des exemples chargés : {len(all_data)}")
    
    # 2. Échantillonnage (si le total dépasse la taille cible)
    if len(all_data) > target_total_size:
        print(f"Échantillonnage aléatoire pour atteindre la taille cible de {target_total_size}...")
        random.shuffle(all_data)
        sampled_data = all_data[:target_total_size]
    else:
        print("Taille totale inférieure à la cible. Mélange des données...")
        random.shuffle(all_data)
        sampled_data = all_data

    # 3. Sauvegarde
    os.makedirs(os.path.dirname(MERGED_SFT_OUTPUT), exist_ok=True)
    with open(MERGED_SFT_OUTPUT, 'w', encoding='utf-8') as f:
        for item in sampled_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f"Sauvegarde du dataset fusionné et mélangé dans {MERGED_SFT_OUTPUT}. Taille finale: {len(sampled_data)}")
    return sampled_data

if __name__ == "__main__":
    # Vous pouvez ajuster la taille cible ici
    merge_and_sample_datasets(SFT_FILES, target_total_size=20000)