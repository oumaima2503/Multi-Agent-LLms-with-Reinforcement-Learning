# Modèle Fine-Tuné: researcher

## Informations Générales

- **Agent**: researcher
- **Modèle de base**: TinyLlama/TinyLlama-1.1B-Chat-v1.0
- **Date d'entraînement**: N/A

## Configuration d'Entraînement

- **Époques**: 1
- **Batch size (train)**: 1
- **Batch size (eval)**: 1
- **Gradient accumulation**: 8
- **Learning rate**: 0.0002
- **Optimizer**: OptimizerNames.PAGED_ADAMW

## Configuration LoRA

- **r**: 64
- **alpha**: 16
- **dropout**: 0.1
- **Target modules**: k_proj, o_proj, v_proj, q_proj

## Données

- **Échantillons d'entraînement**: 2250
- **Échantillons de validation**: 250
- **Source**: /kaggle/input/processed-sft/processed_sft/researcher_sft.jsonl

## Résultats

### Entraînement
- **Loss finale**: 1.0217

### Évaluation

## Fichiers Sauvegardés

- `adapter_model.bin` ou `adapter_model.safetensors`: Poids LoRA
- `adapter_config.json`: Configuration LoRA
- `tokenizer.json`, `tokenizer_config.json`: Tokenizer
- `training_metrics.json`: Métriques complètes (JSON)
- `README.md`: Ce fichier

## Utilisation

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = AutoModelForCausalLM.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0')
model = PeftModel.from_pretrained(base_model, '/kaggle/working/checkpoints/researcher_lora')
tokenizer = AutoTokenizer.from_pretrained('/kaggle/working/checkpoints/researcher_lora')
```
