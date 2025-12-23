# Structure de Sortie du Fine-Tuning

## 📁 Structure du Répertoire de Sortie

Après le fine-tuning, le répertoire `checkpoints/{agent_name}_lora/` devrait contenir :

### Fichiers Essentiels

1. **Modèle LoRA**
   - `adapter_model.bin` ou `adapter_model.safetensors` : Poids du modèle LoRA fine-tuné
   - `adapter_config.json` : Configuration LoRA (r, alpha, dropout, target_modules)

2. **Tokenizer**
   - `tokenizer.json` : Tokenizer principal
   - `tokenizer_config.json` : Configuration du tokenizer
   - `special_tokens_map.json` : Mapping des tokens spéciaux
   - `vocab.json` ou `vocab.txt` : Vocabulaire (selon le type de tokenizer)

3. **Métriques et Documentation**
   - `training_metrics.json` : **Toutes les métriques d'entraînement et d'évaluation**
   - `README.md` : Documentation complète du modèle
   - `training_args.bin` : Arguments d'entraînement (optionnel)

### Checkpoints Intermédiaires (si activés)

- `checkpoint-{step}/` : Dossiers contenant les checkpoints à différentes étapes
  - Chaque checkpoint contient les mêmes fichiers que le modèle final

## 📊 Contenu de `training_metrics.json`

```json
{
  "agent_name": "orchestrator",
  "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "training_config": {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 2,
    "per_device_eval_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "learning_rate": 0.0002,
    "optimizer": "paged_adamw_32bit",
    "lr_scheduler": "cosine",
    "warmup_steps": 100,
    "fp16": true,
    "bf16": false
  },
  "peft_config": {
    "r": 64,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
  },
  "dataset_info": {
    "dataset_path": "data/processed_sft/orchestrator_sft.jsonl",
    "training_samples": 18000,
    "validation_samples": 2000,
    "eval_split": 0.1
  },
  "training_results": {
    "training_loss": 1.2345,
    "training_steps": 27000,
    "training_epochs": 3.0
  },
  "evaluation_results": {
    "eval_loss": 1.1234,
    "eval_token_accuracy": 0.8567,
    "eval_exact_match_rate": 0.2345,
    "eval_f1_score": 0.7890,
    "eval_precision": 0.8123,
    "eval_recall": 0.7654,
    "eval_perplexity": 3.1234,
    "eval_json_valid_rate": 0.9876,
    "eval_json_parseable_rate": 0.9956,
    "eval_key_match_rate": 0.9234
  }
}
```

## 📝 Contenu de `README.md`

Le README devrait contenir :
- Informations générales (agent, modèle de base)
- Configuration d'entraînement complète
- Configuration LoRA
- Informations sur les données utilisées
- Résultats d'entraînement et d'évaluation
- Instructions d'utilisation du modèle

## ✅ Checklist de Vérification

Après le fine-tuning, vérifiez que vous avez :

- [ ] Modèle LoRA sauvegardé (`adapter_model.*`)
- [ ] Configuration LoRA (`adapter_config.json`)
- [ ] Tokenizer complet (tous les fichiers)
- [ ] Métriques complètes (`training_metrics.json`)
- [ ] Documentation (`README.md`)
- [ ] Checkpoints intermédiaires (si nécessaire)

## 🔄 Utilisation du Modèle Fine-Tuné

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Charger le modèle de base
base_model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Charger les poids LoRA
model = PeftModel.from_pretrained(base_model, "checkpoints/orchestrator_lora")

# Charger le tokenizer
tokenizer = AutoTokenizer.from_pretrained("checkpoints/orchestrator_lora")

# Utiliser le modèle
inputs = tokenizer("Your prompt here", return_tensors="pt")
outputs = model.generate(**inputs, max_length=512)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## 📈 Métriques Importantes à Surveiller

1. **Accuracy (tokens)** : Pourcentage de tokens correctement prédits
2. **F1 Score** : Score F1 basé sur le chevauchement des mots
3. **Exact Match Rate** : Pourcentage de séquences exactement correctes
4. **Loss** : Perte d'entraînement et d'évaluation
5. **Perplexity** : Perplexité du modèle
6. **Métriques JSON** (pour orchestrator) : Validité et parseabilité du JSON généré

