# ⚡ Démarrage Rapide : Utiliser les Checkpoints MAGRPO

## 🎯 Étapes Rapides

### 1. Télécharger les Checkpoints

Depuis Kaggle → Onglet "Output" → Télécharger `checkpoints/magrpo_rl/`

### 2. Tester un Agent Individuel

```bash
# Tester tous les agents avec les requêtes par défaut
python test_magrpo_agent.py --all

# Tester un agent spécifique
python test_magrpo_agent.py --agent orchestrator --query "Votre requête ici" --epoch 20
```

### 3. Interagir avec le Système Multi-Agent

```bash
# Mode interactif (CLI)
python interact_magrpo.py --interactive --epoch 20

# Avec une requête spécifique
python interact_magrpo.py --query "Compare le Pixel 8 et l'iPhone 15" --epoch 20
```

### 4. Comparer SFT vs MAGRPO

```bash
# Comparer tous les agents
python compare_sft_magrpo.py --all

# Comparer un agent spécifique
python compare_sft_magrpo.py --agent orchestrator --query "Votre requête" --epochs 10 15 20
```

---

## 📋 Structure des Checkpoints

```
checkpoints/
├── magrpo_rl/
│   ├── epoch10_orchestratc_rl/  (ou orchestrator_rl)
│   ├── epoch10_researcher_rl/
│   ├── epoch10_code_write_rl/
│   ├── epoch10_critic_rl/
│   ├── epoch15_.../
│   └── epoch20_.../
```

**Note** : Vérifiez si c'est `orchestratc` ou `orchestrator` dans vos fichiers.

---

## 🔧 Utilisation en Python

```python
from agents.base_agents import OrchestratorAgent

# Charger un agent avec checkpoint MAGRPO
agent = OrchestratorAgent()
agent.lora_path = "checkpoints/magrpo_rl/epoch20_orchestratc_rl"
agent._load_model()

# Utiliser l'agent
result = agent.act("Votre requête")
print(result)
```

---

## 📊 Workflow Recommandé

1. ✅ **Tester individuellement** chaque agent avec `test_magrpo_agent.py`
2. ✅ **Comparer** SFT vs MAGRPO avec `compare_sft_magrpo.py`
3. ✅ **Identifier** le meilleur checkpoint (epoch 10, 15, ou 20)
4. ✅ **Tester le système complet** avec `interact_magrpo.py`
5. ✅ **Déployer** le meilleur checkpoint en production

---

## 🎯 Prochaines Étapes

- Voir `GUIDE_POST_MAGRPO.md` pour le guide complet
- Voir `MAGRPO_THEORETICAL_EXPLANATION.md` pour la théorie
- Voir `MAGRPO_TRAINING_SUCCESS.md` pour les résultats d'entraînement

