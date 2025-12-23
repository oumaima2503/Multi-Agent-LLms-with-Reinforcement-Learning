# ✅ Critères de Succès des Tests - Outputs Attendus

## 📋 Vue d'Ensemble

Ce document définit les **outputs attendus** et les **critères de validation** pour déterminer si un test d'agent passe avec succès.

---

## 🎯 Critère Principal de Succès

Un test **passe** si :
- ✅ Le JSON est **valide** (parsable)
- ✅ Toutes les **clés requises** sont présentes
- ✅ Les **types de données** sont corrects
- ✅ **Aucune clé `"action_type": "ERROR"`** dans la réponse

Un test **échoue** si :
- ❌ `"action_type": "ERROR"` est présent
- ❌ Des clés requises sont manquantes
- ❌ Les types de données sont incorrects
- ❌ Le JSON n'est pas parsable

---

## 1. 🎭 ORCHESTRATOR

### ✅ Format JSON Attendu (Succès)

```json
{
  "delegated_agent": "Researcher" | "CodeWriter" | "Critic" | "FINISHED",
  "instruction": "description de la tâche pour l'agent délégué",
  "final_answer": "réponse finale si la tâche est terminée, sinon chaîne vide"
}
```

### 📝 Exemples de Succès

**Exemple 1 : Délégation à Researcher**
```json
{
  "delegated_agent": "Researcher",
  "instruction": "Trouve des informations sur les caractéristiques du Pixel 8",
  "final_answer": ""
}
```
✅ **Test passe** : Toutes les clés requises présentes, valeurs valides

**Exemple 2 : Tâche terminée**
```json
{
  "delegated_agent": "FINISHED",
  "instruction": "",
  "final_answer": "L'analyse comparative est terminée. Le Pixel 8 a un meilleur rapport qualité-prix."
}
```
✅ **Test passe** : Tâche complétée avec réponse finale

**Exemple 3 : Délégation à CodeWriter**
```json
{
  "delegated_agent": "CodeWriter",
  "instruction": "Calcule le prix après remise de 15% sur 899€",
  "final_answer": ""
}
```
✅ **Test passe** : Délégation correcte à CodeWriter

### ❌ Exemples d'Échec

**Exemple 1 : Clés manquantes**
```json
{
  "data": {"delegation_info": null},
  "errors": [{"message": "..."}],
  "action_type": "ERROR",
  "error_message": "Orchestrator action missing required keys (delegated_agent, instruction, final_answer)."
}
```
❌ **Test échoue** : Clés requises absentes, `action_type: ERROR` présent

**Exemple 2 : Format incorrect**
```json
{
  "query": "How long did it take...",
  "$$$context": false,
  "action_type": "ERROR",
  "error_message": "Orchestrator action missing required keys..."
}
```
❌ **Test échoue** : Structure incorrecte, pas de clés requises

### 🔍 Validation

Le test passe si :
- ✅ `delegated_agent` est présent et dans `["Researcher", "CodeWriter", "Critic", "FINISHED"]`
- ✅ `instruction` est présent (peut être vide si `delegated_agent == "FINISHED"`)
- ✅ `final_answer` est présent (peut être vide)
- ✅ **Aucune** clé `action_type: "ERROR"`

---

## 2. 🔍 RESEARCHER

### ✅ Format JSON Attendu (Succès)

```json
{
  "research_query": "requête de recherche spécifique",
  "final_answer": "résumé des résultats de recherche ou chaîne vide"
}
```

### 📝 Exemples de Succès

**Exemple 1 : Requête de recherche**
```json
{
  "research_query": "Google Pixel 8 Pro release date",
  "final_answer": ""
}
```
✅ **Test passe** : Clés requises présentes, format correct

**Exemple 2 : Avec réponse finale**
```json
{
  "research_query": "Pixel 8 specifications",
  "final_answer": "Le Google Pixel 8 Pro a été lancé le 4 octobre 2023 avec un écran de 6.7 pouces."
}
```
✅ **Test passe** : Recherche complétée avec réponse

**Exemple 3 : Variante camelCase (normalisée automatiquement)**
```json
{
  "researchQuery": "iPhone 15 price",
  "finalAnswer": "L'iPhone 15 coûte 799€"
}
```
✅ **Test passe** : Normalisation automatique camelCase → snake_case

### ❌ Exemples d'Échec

**Exemple 1 : Clés manquantes**
```json
{
  "@context": "https://schema.org",
  "@type": ["Thing"],
  "action_type": "ERROR",
  "error_message": "Researcher action missing required keys (research_query, final_answer)."
}
```
❌ **Test échoue** : Structure incorrecte, clés requises absentes

**Exemple 2 : Format incorrect**
```json
{
  "search_term": "Pixel 8",
  "response": {"facts": []},
  "action_type": "ERROR",
  "error_message": "Researcher action missing required keys..."
}
```
❌ **Test échoue** : Clés incorrectes (sera normalisé mais peut échouer si normalisation échoue)

### 🔍 Validation

Le test passe si :
- ✅ `research_query` est présent (string non vide recommandé)
- ✅ `final_answer` est présent (peut être vide)
- ✅ **Aucune** clé `action_type: "ERROR"`

---

## 3. 💻 CODE_WRITER

### ✅ Format JSON Attendu (Succès)

```json
{
  "python_code": "code Python à exécuter",
  "result_explanation": "explication du résultat ou chaîne vide"
}
```

### 📝 Exemples de Succès

**Exemple 1 : Code simple**
```json
{
  "python_code": "result = 899 * 0.15",
  "result_explanation": "La remise de 15% sur 899€ est de 134.85€"
}
```
✅ **Test passe** : Code Python valide, explication présente

**Exemple 2 : Code avec fonction**
```json
{
  "python_code": "def calculate_discount(price, rate):\n    return price * rate / 100\n\nresult = calculate_discount(899, 15)",
  "result_explanation": ""
}
```
✅ **Test passe** : Code fonctionnel, explication optionnelle

**Exemple 3 : Variante camelCase (normalisée)**
```json
{
  "pythonCode": "price = 899\ndiscount = price * 0.15",
  "ResultExplain": "Calcul de la remise"
}
```
✅ **Test passe** : Normalisation automatique des clés

### ❌ Exemples d'Échec

**Exemple 1 : Clés manquantes**
```json
{
  "python_Code": "result = 899 * 0.15",
  "action_type": "ERROR",
  "error_message": "CodeWriter action missing required keys (python_code, result_explanation)."
}
```
❌ **Test échoue** : `result_explanation` manquant (sera ajouté automatiquement si normalisation fonctionne)

**Exemple 2 : JSON malformé**
```json
{
  "action_type": "ERROR",
  "error_message": "Erreur de format JSON/Validation: Aucun bloc JSON ({...}) trouvé dans la réponse.",
  "raw_output": "{'python_Code': '...', 'ResultExplain': '...'}"
}
```
❌ **Test échoue** : JSON non parsable (guillemets simples au lieu de doubles)

### 🔍 Validation

Le test passe si :
- ✅ `python_code` est présent (string non vide)
- ✅ `result_explanation` est présent (peut être vide, ajouté automatiquement si manquant)
- ✅ **Aucune** clé `action_type: "ERROR"`

---

## 4. 🎯 CRITIC

### ✅ Format JSON Attendu (Succès)

```json
{
  "critique_ok": true | false,
  "suggestions": "suggestions concrètes ou message de confirmation"
}
```

### 📝 Exemples de Succès

**Exemple 1 : Critique positive**
```json
{
  "critique_ok": true,
  "suggestions": "La solution est complète et correcte. Le code est bien structuré."
}
```
✅ **Test passe** : Boolean correct, suggestions présentes

**Exemple 2 : Critique négative**
```json
{
  "critique_ok": false,
  "suggestions": "Le code a une erreur de syntaxe à la ligne 3. Il manque un deux-points après la définition de fonction."
}
```
✅ **Test passe** : Boolean correct, suggestions constructives

**Exemple 3 : Variante camelCase (normalisée)**
```json
{
  "critiqueOk": "true",
  "suggestions": "La solution est correcte"
}
```
✅ **Test passe** : Normalisation automatique (`"true"` → `true`)

### ❌ Exemples d'Échec

**Exemple 1 : Clés manquantes**
```json
{
  "message": "La recommandation suivante...",
  "critiqueOk": true,
  "action_type": "ERROR",
  "error_message": "Critic action missing required keys (critique_ok, suggestions)."
}
```
❌ **Test échoue** : `suggestions` manquant (sera ajouté automatiquement si normalisation fonctionne)

**Exemple 2 : Type incorrect**
```json
{
  "critique_ok": "true/false",
  "suggestions": "string",
  "action_type": "ERROR",
  "error_message": "Critic 'critique_ok' must be a boolean."
}
```
❌ **Test échoue** : `critique_ok` n'est pas un boolean (sera normalisé automatiquement si possible)

### 🔍 Validation

Le test passe si :
- ✅ `critique_ok` est présent et est un **boolean** (`true` ou `false`, pas une string)
- ✅ `suggestions` est présent (peut être vide, ajouté automatiquement si manquant)
- ✅ **Aucune** clé `action_type: "ERROR"`

---

## 📊 Tableau Récapitulatif

| Agent | Clés Requises | Types | Test Passe Si |
|-------|--------------|-------|---------------|
| **Orchestrator** | `delegated_agent`, `instruction`, `final_answer` | String | Toutes présentes, `delegated_agent` valide |
| **Researcher** | `research_query`, `final_answer` | String | Toutes présentes |
| **CodeWriter** | `python_code`, `result_explanation` | String | Toutes présentes, `python_code` non vide |
| **Critic** | `critique_ok`, `suggestions` | Boolean, String | Toutes présentes, `critique_ok` est boolean |

---

## 🔍 Comment Interpréter les Résultats

### ✅ Indicateurs de Succès

Dans la sortie du test, vous verrez :
```
✅ SUCCÈS : Format JSON valide pour [Agent].
   → [Détails spécifiques selon l'agent]
```

**Exemple pour Orchestrator :**
```
✅ SUCCÈS : Format JSON valide pour Orchestrator.
   → Agent délégué: Researcher
   → Instruction: Trouve des informations sur le Pixel 8...
```

**Exemple pour Researcher :**
```
✅ SUCCÈS : Format JSON valide pour Researcher.
   → Requête de recherche: Google Pixel 8 release date
   → Réponse finale: Le Pixel 8 a été lancé le 4 octobre 2023
```

**Exemple pour CodeWriter :**
```
✅ SUCCÈS : Format JSON valide pour CodeWriter.
   → Code Python généré (25 caractères)
   → Code: result = 899 * 0.15
```

**Exemple pour Critic :**
```
✅ SUCCÈS : Format JSON valide pour Critic.
   → Critique OK: True
   → Suggestions: La solution est complète et correcte
```

### ❌ Indicateurs d'Échec

Dans la sortie du test, vous verrez :
```
❌ ÉCHEC DU PARSING : [Message d'erreur]
📄 Sortie brute du modèle: [Sortie générée]
🔍 Analyse de la sortie:
   → [Diagnostics détaillés]
```

**Exemple d'échec :**
```
❌ ÉCHEC DU PARSING : Researcher action missing required keys (research_query, final_answer).

📄 Sortie brute du modèle:
{"researchQuery": "Pixel 8", "finalAnswer": "..."}

🔍 Analyse de la sortie:
   ✅ La sortie commence par '{' (bon signe)
   → Clés JSON trouvées dans la sortie: ['researchQuery', 'finalAnswer']
   ⚠️ Clés manquantes: {'research_query', 'final_answer'}
```

**Note** : Si les clés sont trouvées mais avec un format différent (camelCase), la normalisation devrait les convertir automatiquement. Si l'échec persiste, c'est que la normalisation n'a pas fonctionné.

---

## 🎯 Critères de Validation Détaillés

### 1. Validation du JSON

✅ **JSON Valide** :
- Peut être parsé par `json.loads()`
- Structure correcte (accolades, guillemets, etc.)

❌ **JSON Invalide** :
- Erreur de parsing
- Guillemets simples au lieu de doubles (non réparé)
- Accolades non fermées (non réparé)

### 2. Validation des Clés

✅ **Clés Présentes** :
- Toutes les clés requises sont dans le dictionnaire
- Les variantes (camelCase) sont normalisées automatiquement

❌ **Clés Manquantes** :
- Une ou plusieurs clés requises absentes
- Normalisation échouée

### 3. Validation des Types

✅ **Types Corrects** :
- `critique_ok` est un boolean (pas une string)
- Toutes les autres valeurs sont des strings

❌ **Types Incorrects** :
- `critique_ok` est une string `"true"` (sera normalisé automatiquement)
- Types inattendus

### 4. Validation des Valeurs

✅ **Valeurs Valides** :
- `delegated_agent` dans `["Researcher", "CodeWriter", "Critic", "FINISHED"]`
- `python_code` non vide
- `critique_ok` est `true` ou `false`

❌ **Valeurs Invalides** :
- `delegated_agent` avec valeur inconnue
- `python_code` vide
- `critique_ok` avec valeur non booléenne (sera normalisé si possible)

---

## 📝 Exemples Complets de Tests

### Test Réussi - Orchestrator

**Input :**
```python
test_single_agent('orchestrator', 'Planifie une analyse comparative entre le Pixel 8 et l\'iPhone 15.')
```

**Output Attendu (Succès) :**
```json
{
  "delegated_agent": "Researcher",
  "instruction": "Trouve des informations comparatives sur le Pixel 8 et l'iPhone 15",
  "final_answer": ""
}
```

**Sortie Console :**
```
✅ SUCCÈS : Format JSON valide pour Orchestrator.
   → Agent délégué: Researcher
   → Instruction: Trouve des informations comparatives sur le Pixel 8 et l'iPhone 15...
```

### Test Réussi - Researcher

**Input :**
```python
test_single_agent('researcher', 'Cherche la date de sortie exacte du Google Pixel 8 Pro.')
```

**Output Attendu (Succès) :**
```json
{
  "research_query": "Google Pixel 8 Pro release date",
  "final_answer": "Le Google Pixel 8 Pro a été lancé le 4 octobre 2023"
}
```

**Sortie Console :**
```
✅ SUCCÈS : Format JSON valide pour Researcher.
   → Requête de recherche: Google Pixel 8 Pro release date
   → Réponse finale: Le Google Pixel 8 Pro a été lancé le 4 octobre 2023
```

### Test Réussi - CodeWriter

**Input :**
```python
test_single_agent('code_writer', 'Fais un script Python pour calculer une remise de 15% sur un prix de 899€.')
```

**Output Attendu (Succès) :**
```json
{
  "python_code": "price = 899\ndiscount = price * 0.15\nfinal_price = price - discount",
  "result_explanation": "La remise de 15% sur 899€ est de 134.85€, prix final: 764.15€"
}
```

**Sortie Console :**
```
✅ SUCCÈS : Format JSON valide pour CodeWriter.
   → Code Python généré (85 caractères)
   → Code: price = 899
discount = price * 0.15
final_price = price - discount
```

### Test Réussi - Critic

**Input :**
```python
test_single_agent('critic', 'Évalue ceci : "Le smartphone est cher mais puissant".')
```

**Output Attendu (Succès) :**
```json
{
  "critique_ok": false,
  "suggestions": "L'évaluation est trop vague. Précisez le modèle de smartphone, le prix exact, et les caractéristiques de performance."
}
```

**Sortie Console :**
```
✅ SUCCÈS : Format JSON valide pour Critic.
   → Critique OK: False
   → Suggestions: L'évaluation est trop vague. Précisez le modèle de smartphone...
```

---

## ⚠️ Cas Limites et Normalisation

### Normalisation Automatique

Le système normalise automatiquement :
- ✅ **camelCase → snake_case** : `researchQuery` → `research_query`
- ✅ **String → Boolean** : `"true"` → `true` pour `critique_ok`
- ✅ **Variantes de clés** : `pythonCode`, `python_Code` → `python_code`
- ✅ **Valeurs par défaut** : `result_explanation` et `suggestions` ajoutés si manquants

### Quand la Normalisation Échoue

Si la normalisation échoue, le test échouera avec :
```
❌ ÉCHEC DU PARSING : [Agent] action missing required keys ([clés requises]).
```

Dans ce cas, vérifiez :
1. La sortie brute du modèle
2. Les clés trouvées vs manquantes
3. Si les clés sont présentes mais avec un format non reconnu

---

## 🎯 Checklist de Validation Rapide

Pour chaque agent, vérifiez :

- [ ] Le JSON est valide (pas d'erreur de parsing)
- [ ] Toutes les clés requises sont présentes
- [ ] Aucune clé `"action_type": "ERROR"` dans la réponse
- [ ] Les types de données sont corrects (boolean pour `critique_ok`)
- [ ] Les valeurs sont cohérentes (ex: `delegated_agent` dans les valeurs autorisées)

---

*Document créé pour clarifier les critères de succès des tests*
*Dernière mise à jour : Après implémentation de la normalisation des clés*

