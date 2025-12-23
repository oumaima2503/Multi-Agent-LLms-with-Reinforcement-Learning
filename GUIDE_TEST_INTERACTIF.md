# 🎯 Guide de Test Interactif - Tester avec Votre Phrase

## 🚀 Utilisation Simple

### Fonction Principale : `test_interactif()`

La fonction la plus simple pour tester un agent avec votre propre phrase :

```python
test_interactif("Votre phrase ici")
```

---

## 📝 Exemples d'Utilisation

### 1. Test avec Orchestrator (par défaut)

```python
test_interactif("Compare le Pixel 8 et l'iPhone 15")
```

**Output attendu :**
```json
{
  "delegated_agent": "Researcher",
  "instruction": "Trouve des informations comparatives...",
  "final_answer": ""
}
```

### 2. Test avec Researcher

```python
test_interactif("Cherche la date de sortie du Google Pixel 8 Pro", agent="researcher")
```

**Output attendu :**
```json
{
  "research_query": "Google Pixel 8 Pro release date",
  "final_answer": "Le Pixel 8 Pro a été lancé le 4 octobre 2023"
}
```

### 3. Test avec CodeWriter

```python
test_interactif("Calcule une remise de 15% sur un prix de 899€", agent="code_writer")
```

**Output attendu :**
```json
{
  "python_code": "price = 899\ndiscount = price * 0.15",
  "result_explanation": "La remise est de 134.85€"
}
```

### 4. Test avec Critic

```python
test_interactif("Évalue ce code: def test(): return 1", agent="critic")
```

**Output attendu :**
```json
{
  "critique_ok": true,
  "suggestions": "Le code est correct mais pourrait être amélioré..."
}
```

---

## 🔄 Tester avec Tous les Agents

Pour tester votre phrase avec tous les agents en une fois :

```python
test_tous_agents("Compare le Pixel 8 et l'iPhone 15")
```

Cela testera votre phrase avec :
1. Orchestrator
2. Researcher
3. CodeWriter
4. Critic

---

## ⚡ Mode Rapide

Pour accélérer les tests (2-3x plus rapide) :

```python
test_interactif("Votre phrase", fast_mode=True)
```

---

## 📋 Format des Réponses Attendues

### ✅ Test Réussi

Vous verrez :
```
✅ SUCCÈS : Format JSON valide pour [Agent].
   → [Détails spécifiques]
```

### ❌ Test Échoué

Vous verrez :
```
❌ ÉCHEC DU PARSING : [Message d'erreur]
📄 Sortie brute du modèle: [JSON généré]
🔍 Analyse de la sortie:
   → Clés trouvées: [...]
   ⚠️ Clés manquantes: {...}
```

---

## 🎯 Exemples de Phrases par Agent

### Pour Orchestrator
- "Planifie une analyse comparative entre le Pixel 8 et l'iPhone 15"
- "Organise une recherche sur les smartphones Android"
- "Gère un workflow pour comparer deux produits"

### Pour Researcher
- "Cherche la date de sortie du Google Pixel 8 Pro"
- "Trouve les spécifications techniques de l'iPhone 15"
- "Recherche le prix du Pixel 8"

### Pour CodeWriter
- "Écris un script Python pour calculer une remise de 15% sur 899€"
- "Crée une fonction pour additionner deux nombres"
- "Génère du code pour trier une liste"

### Pour Critic
- "Évalue ce code: def test(): return 1"
- "Critique cette solution: Le smartphone est cher"
- "Analyse cette réponse: [votre texte]"

---

## 💡 Astuces

1. **Mode rapide** : Utilisez `fast_mode=True` pour des tests plus rapides
2. **Phrases claires** : Plus votre phrase est claire, meilleure sera la réponse
3. **Un agent à la fois** : Testez un agent spécifique pour des résultats plus prévisibles
4. **Tous les agents** : Utilisez `test_tous_agents()` pour voir comment chaque agent interprète votre phrase

---

## 📊 Interprétation des Résultats

### Critère de Succès Principal

**Le test passe si** :
- ✅ Aucune clé `"action_type": "ERROR"` dans la réponse
- ✅ Toutes les clés requises sont présentes
- ✅ Message `✅ SUCCÈS` dans la console

**Le test échoue si** :
- ❌ Clé `"action_type": "ERROR"` présente
- ❌ Clés requises manquantes
- ❌ Message `❌ ÉCHEC DU PARSING` dans la console

---

*Voir `TEST_SUCCESS_CRITERIA.md` pour plus de détails sur les formats attendus*

