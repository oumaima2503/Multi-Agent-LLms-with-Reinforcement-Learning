# ⚡ Guide d'Optimisation des Tests

## 🐌 Problèmes de Performance Identifiés

Les tests prenaient beaucoup de temps à cause de :

1. **CPU au lieu de GPU** : Le modèle tourne sur CPU (très lent)
2. **Trop de tokens générés** : `max_new_tokens=512` est élevé
3. **Mécanisme de retry** : Jusqu'à 3 tentatives par test
4. **Paramètres de génération** : Sampling avec température (plus lent)
5. **Chargement répété** : Chaque agent charge son modèle séparément

---

## ✅ Optimisations Implémentées

### 1. **Mode Rapide (Fast Mode)**

Un mode rapide a été ajouté qui :
- ✅ Réduit `max_new_tokens` de 512 → 256 (2x plus rapide)
- ✅ Désactive le retry (1 seule tentative)
- ✅ Utilise un mode déterministe (`do_sample=False`)
- ✅ Réduit `top_k` de 50 → 20
- ✅ Réduit `repetition_penalty` de 1.3 → 1.2
- ✅ Réduit `no_repeat_ngram_size` de 3 → 2

**Gain estimé** : **2-3x plus rapide** sur CPU

### 2. **Paramètres Optimisés pour CPU**

En mode rapide :
- Mode déterministe (pas de sampling)
- Moins de tokens candidats à évaluer
- Paramètres simplifiés

---

## 🚀 Comment Utiliser le Mode Rapide

### Option 1 : Variable d'Environnement (Recommandé)

**Windows PowerShell :**
```powershell
$env:FAST_MODE="true"
# Puis exécuter le notebook
```

**Windows CMD :**
```cmd
set FAST_MODE=true
```

**Linux/macOS :**
```bash
export FAST_MODE=true
```

### Option 2 : Dans le Notebook

```python
# Activer le mode rapide globalement
import os
os.environ["FAST_MODE"] = "true"

# Puis exécuter les tests
test_single_agent('orchestrator', 'Votre requête')
```

### Option 3 : Par Fonction

```python
# Activer le mode rapide pour un test spécifique
test_single_agent('orchestrator', 'Votre requête', fast_mode=True)
```

---

## 📊 Comparaison des Performances

| Mode | max_new_tokens | Retry | Sampling | Temps Estimé (CPU) |
|------|----------------|-------|----------|-------------------|
| **Normal** | 512 | 2 tentatives | Oui | ~30-60s par agent |
| **Rapide** | 256 | 0 tentative | Non | ~10-20s par agent |

**Gain** : **2-3x plus rapide** en mode rapide

---

## 💡 Recommandations

### Pour les Tests Rapides
```python
# Utiliser le mode rapide
test_single_agent('orchestrator', 'Test rapide', fast_mode=True)
```

### Pour les Tests Complets
```python
# Utiliser le mode normal (meilleure qualité)
test_single_agent('orchestrator', 'Test complet', fast_mode=False)
```

### Pour les Tests par Défaut
```python
# Définir FAST_MODE dans l'environnement avant d'exécuter
# Le notebook utilisera automatiquement le mode rapide
```

---

## 🔧 Optimisations Supplémentaires

### 1. **Utiliser un GPU** (Si Disponible)

Le code détecte automatiquement un GPU. Si vous avez un GPU :
- Les tests seront **10-20x plus rapides**
- Le mode rapide n'est pas nécessaire avec GPU

### 2. **Tester un Agent à la Fois**

Au lieu de tester tous les agents :
```python
# Au lieu de :
for agent_name, query in default_tests:
    test_single_agent(agent_name, query)

# Faire :
test_single_agent('orchestrator', 'Votre requête', fast_mode=True)
```

### 3. **Réduire le Nombre de Tests**

Commenter les tests non nécessaires dans le notebook :
```python
default_tests = [
    # ("orchestrator", "Planifie une analyse..."),  # Commenté
    ("researcher", "Cherche la date..."),
    # ("code_writer", "Fais un script..."),  # Commenté
    # ("critic", "Évalue ceci..."),  # Commenté
]
```

---

## 📝 Exemple d'Utilisation

### Test Rapide d'un Agent
```python
# Mode rapide activé
test_single_agent('researcher', 'Cherche la date de sortie du Pixel 8', fast_mode=True)
```

### Test Complet avec Qualité
```python
# Mode normal (meilleure qualité, plus lent)
test_single_agent('researcher', 'Cherche la date de sortie du Pixel 8', fast_mode=False)
```

### Configuration Globale
```python
# Au début du notebook
import os
os.environ["FAST_MODE"] = "true"  # Tous les tests seront rapides

# Puis tous les tests utiliseront le mode rapide automatiquement
test_single_agent('orchestrator', 'Test')
```

---

## ⚠️ Limitations du Mode Rapide

1. **Moins de tokens** : Réponses potentiellement plus courtes
2. **Pas de retry** : Si la première tentative échoue, pas de seconde chance
3. **Mode déterministe** : Moins de variété dans les réponses

**Recommandation** : Utiliser le mode rapide pour les tests de développement, et le mode normal pour les tests finaux.

---

## 🎯 Résumé

| Action | Commande |
|--------|----------|
| **Activer mode rapide global** | `os.environ["FAST_MODE"] = "true"` |
| **Test rapide spécifique** | `test_single_agent(..., fast_mode=True)` |
| **Test normal** | `test_single_agent(..., fast_mode=False)` |
| **Vérifier le mode** | Le notebook affiche `⚡ Mode rapide actuel: True/False` |

---

*Optimisations appliquées pour réduire le temps d'exécution des tests*
*Gain estimé : 2-3x plus rapide en mode rapide*

