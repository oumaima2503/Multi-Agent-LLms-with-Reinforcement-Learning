# 📘 Guide de Configuration Kaggle pour MAGRPO

## 🚀 Étapes pour Utiliser le Notebook sur Kaggle

### 1. Préparer vos Fichiers

#### A. Créer un Dataset Kaggle avec vos Checkpoints SFT

1. **Créer un nouveau Dataset sur Kaggle** :
   - Allez sur https://www.kaggle.com/datasets
   - Cliquez sur "New Dataset"
   - Nommez-le (ex: `multi-agent-sft-checkpoints`)

2. **Structure du Dataset** :
   ```
   checkpoints/
   ├── orchestrator_lora/
   │   ├── adapter_model.safetensors
   │   ├── adapter_config.json
   │   └── tokenizer.json
   ├── researcher_lora/
   │   ├── adapter_model.safetensors
   │   ├── adapter_config.json
   │   └── tokenizer.json
   ├── code_writer_lora/
   │   ├── adapter_model.safetensors
   │   ├── adapter_config.json
   │   └── tokenizer.json
   └── critic_lora/
       ├── adapter_model.safetensors
       ├── adapter_config.json
       └── tokenizer.json
   ```

3. **Uploader le dataset** :
   - Compressez le dossier `checkpoints/` en ZIP
   - Uploadez le ZIP sur Kaggle
   - Publiez le dataset

#### B. Créer un Dataset avec votre Dataset d'Entraînement

1. **Créer un nouveau Dataset** :
   - Nommez-le (ex: `orchestrator-sft-dataset`)

2. **Uploader votre fichier JSONL** :
   - Uploadez `orchestrator_sft.jsonl` (ou votre fichier de dataset)
   - Publiez le dataset

---

### 2. Créer le Notebook sur Kaggle

1. **Créer un nouveau Notebook** :
   - Allez sur https://www.kaggle.com/code
   - Cliquez sur "New Notebook"
   - Nommez-le (ex: `magrpo-training`)

2. **Uploader le notebook** :
   - Uploadez `train_magrpo_kaggle.ipynb`
   - Ou copiez-collez le contenu

3. **Ajouter les Datasets** :
   - Cliquez sur "Add Data" (en haut à droite)
   - Recherchez votre dataset de checkpoints
   - Recherchez votre dataset de données d'entraînement
   - Ajoutez-les

---

### 3. Configurer le Notebook

#### A. Ajuster les Chemins

Dans la cellule de configuration, ajustez les chemins selon vos datasets :

```python
# Si votre dataset s'appelle "multi-agent-sft-checkpoints"
CHECKPOINTS_DIR = "/kaggle/input/multi-agent-sft-checkpoints/checkpoints"

# Si votre dataset s'appelle "orchestrator-sft-dataset"
DATASET_PATH = "/kaggle/input/orchestrator-sft-dataset/orchestrator_sft.jsonl"
```

**Comment trouver le bon chemin :**
1. Dans le notebook, allez dans l'onglet "Data"
2. Regardez le chemin exact de vos fichiers
3. Utilisez ce chemin dans la configuration

#### B. Activer le GPU

1. **Dans les paramètres du notebook** :
   - Cliquez sur "Settings" (⚙️)
   - Activez "GPU" dans "Accelerator"
   - Choisissez "T4 x2" ou "P100" selon disponibilité

---

### 4. Exécuter le Notebook

1. **Exécuter toutes les cellules** :
   - Cliquez sur "Run All" (ou exécutez cellule par cellule)

2. **Surveiller l'entraînement** :
   - Les logs s'affichent dans la sortie
   - Surveillez les métriques (loss, KL, value_mean)

3. **Sauvegarder les résultats** :
   - Les checkpoints sont automatiquement sauvegardés dans `/kaggle/working/`
   - Ils sont disponibles dans l'onglet "Output"

---

### 5. Télécharger les Résultats

1. **Après l'entraînement** :
   - Allez dans l'onglet "Output"
   - Téléchargez le dossier `checkpoints/magrpo_rl/`

2. **Structure des fichiers téléchargés** :
   ```
   checkpoints/magrpo_rl/
   ├── epoch5_orchestrator_rl/
   ├── epoch5_researcher_rl/
   ├── epoch5_code_writer_rl/
   ├── epoch5_critic_rl/
   ├── epoch10_orchestrator_rl/
   └── ...
   ```

---

## ⚠️ Problèmes Courants

### 1. "Missing adapter for {agent}"

**Solution** :
- Vérifiez que le chemin `CHECKPOINTS_DIR` est correct
- Vérifiez que tous les dossiers `{agent}_lora` sont présents
- Vérifiez la structure de votre dataset Kaggle

### 2. "Dataset not found"

**Solution** :
- Vérifiez que vous avez ajouté le dataset au notebook
- Vérifiez le chemin `DATASET_PATH`
- Vérifiez le nom exact du fichier dans le dataset

### 3. "Out of memory"

**Solution** :
- Réduisez `max_episodes` dans `collect_trajectories` (de 2 à 1)
- Réduisez `TOTAL_EPOCHS` (de 10 à 5)
- Utilisez un GPU plus puissant si disponible

### 4. "No transitions collected"

**Solution** :
- Vérifiez que le dataset contient bien le champ "instruction"
- Vérifiez que le dataset n'est pas vide
- Vérifiez les logs pour voir les erreurs

---

## 📊 Optimisation pour Kaggle

### Limites Kaggle

- **Temps d'exécution** : 9 heures maximum (GPU) / 12 heures (CPU)
- **GPU** : T4 x2 ou P100 (selon disponibilité)
- **RAM** : 30 GB
- **Espace disque** : 20 GB

### Recommandations

1. **Commencez petit** :
   - `TOTAL_EPOCHS = 5` pour tester
   - `max_episodes = 1` dans `collect_trajectories`

2. **Monitorer l'utilisation** :
   - Surveillez l'utilisation GPU/RAM
   - Ajustez selon les ressources disponibles

3. **Sauvegarder régulièrement** :
   - `SAVE_FREQ = 3` pour sauvegarder plus souvent
   - Téléchargez les checkpoints intermédiaires

---

## ✅ Checklist Finale

Avant de lancer :

- [ ] Datasets créés et publiés sur Kaggle
- [ ] Notebook créé avec le code MAGRPO
- [ ] Datasets ajoutés au notebook
- [ ] Chemins configurés correctement
- [ ] GPU activé dans les paramètres
- [ ] Hyperparamètres ajustés selon vos ressources

---

## 🎯 Prochaines Étapes

Après l'entraînement sur Kaggle :

1. **Télécharger les checkpoints RL**
2. **Tester les agents** avec `test_agents.ipynb`
3. **Évaluer les résultats**
4. **Itérer** si nécessaire

---

*Guide créé pour faciliter l'utilisation de MAGRPO sur Kaggle*

