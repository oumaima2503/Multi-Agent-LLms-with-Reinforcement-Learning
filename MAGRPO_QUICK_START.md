# 🚀 MAGRPO - Guide de Démarrage Rapide

## 📋 Prérequis

Avant de lancer MAGRPO, assurez-vous d'avoir :

- [x] ✅ Fine-tuning SFT terminé pour tous les agents
- [x] ✅ Checkpoints SFT présents dans `checkpoints/{agent}_lora/`
- [x] ✅ Dataset d'entraînement RL prêt
- [x] ✅ GPU disponible (recommandé) ou CPU

---

## 🎯 Étapes Rapides

### 1. Vérifier les Checkpoints SFT

```bash
# Vérifier que tous les checkpoints existent
ls checkpoints/orchestrator_lora/
ls checkpoints/researcher_lora/
ls checkpoints/code_writer_lora/
ls checkpoints/critic_lora/
```

**Fichiers requis :**
- `adapter_model.safetensors` ou `adapter_model.bin`
- `adapter_config.json`
- `tokenizer.json`

### 2. Configurer les Chemins

Éditez `main_train.py` et vérifiez :

```python
CHECKPOINTS_DIR = "checkpoints"  # Chemin vers vos checkpoints SFT
DATASET_PATH = "data/processed_sft/orchestrator_sft.jsonl"  # Dataset RL
SAVE_FOLDER = "checkpoints/magrpo_rl"  # Où sauvegarder les checkpoints RL
```

### 3. Lancer l'Entraînement

```bash
python main_train.py
```

### 4. Monitorer les Progrès

Surveillez les logs pour :
- **Loss** : Devrait diminuer progressivement
- **KL Divergence** : Devrait rester faible (< 0.1)
- **Value Mean** : Devrait augmenter
- **Récompenses** : Devraient augmenter

---

## 📊 Métriques Attendues

### Époques 1-5
- Loss : 0.5 - 2.0
- KL : 0.01 - 0.1
- Récompense moyenne : Variable

### Époques 6-10
- Loss : Devrait diminuer
- KL : Devrait rester stable
- Récompense moyenne : Devrait augmenter

---

## ⚠️ Problèmes Courants

### 1. "Missing adapter for {agent}"
**Solution** : Vérifiez que les checkpoints SFT sont présents

### 2. "Out of memory"
**Solution** : Réduisez `max_episodes` dans `collect_trajectories`

### 3. "No transitions collected"
**Solution** : Vérifiez que le dataset est accessible

---

## 🔧 Ajustements Recommandés

### Si les récompenses ne s'améliorent pas :

1. **Augmenter les époques** : `TOTAL_EPOCHS = 20`
2. **Ajuster le learning rate** : `lr=5e-6` (plus bas)
3. **Améliorer le système de récompenses** : Voir `MAGRPO_IMPLEMENTATION_GUIDE.md`

### Si la loss explose :

1. **Réduire le learning rate** : `lr=1e-6`
2. **Augmenter clip_epsilon** : `clip_epsilon=0.3`
3. **Vérifier les checkpoints SFT** : Peut-être nécessiter un ré-entraînement

---

## 📝 Checklist Finale

- [ ] Tous les checkpoints SFT présents
- [ ] Chemins configurés correctement
- [ ] Dataset accessible
- [ ] GPU/CPU disponible
- [ ] Monitoring en place

---

## 🎯 Prochaines Étapes

Après 10 époques :
1. Évaluer les résultats
2. Tester les agents avec `test_agents.ipynb`
3. Si résultats positifs : Continuer jusqu'à 20 époques
4. Si résultats négatifs : Ajuster les hyperparamètres ou améliorer SFT

---

*Voir `MAGRPO_IMPLEMENTATION_GUIDE.md` pour plus de détails*

