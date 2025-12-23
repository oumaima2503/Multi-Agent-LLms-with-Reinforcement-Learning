# ⚡ Optimisations pour Accélérer l'Entraînement avec Llama-3

## 🎯 Problème Initial

L'entraînement avec Llama-3 8B prenait **6 heures par époque**, ce qui est trop lent.

## ✅ Optimisations Appliquées

### 1. **Réduction de la Longueur Maximale** (Gain: ~2-3x)
- **Avant** : 4096 tokens
- **Après** : 2048 tokens
- **Impact** : Réduit significativement le temps de calcul (complexité quadratique de l'attention)

### 2. **Optimisation du Batch Size** (Gain: ~1.5-2x)
- **Avant** : `batch_size=1`, `gradient_accumulation=8`
- **Après** : `batch_size=2`, `gradient_accumulation=16` (GPU)
- **Impact** : Meilleure utilisation du GPU, moins d'overhead

### 3. **Optimisation LoRA** (Gain: ~1.2-1.5x)
- **Avant** : `r=64`, `dropout=0.1`
- **Après** : `r=32`, `dropout=0.05`
- **Impact** : Moins de paramètres à entraîner = plus rapide

### 4. **Optimisation Optimizer** (Gain: ~1.2x)
- **Avant** : `paged_adamw_32bit`
- **Après** : `paged_adamw_8bit`
- **Impact** : Optimizer 8-bit plus rapide que 32-bit

### 5. **Désactivation du Gradient Checkpointing** (Gain: ~1.3-1.5x)
- **Avant** : `use_gradient_checkpointing=True`
- **Après** : `use_gradient_checkpointing=False`
- **Impact** : Plus rapide mais utilise plus de mémoire

### 6. **Compilation du Modèle** (Gain: ~1.2-1.5x si disponible)
- **Ajout** : `torch.compile(model)` si PyTorch 2.0+
- **Impact** : Accélération significative via compilation JIT

### 7. **Optimisation du Dataloader** (Gain: ~1.1-1.2x)
- **Avant** : `num_workers=0`
- **Après** : `num_workers=2`, `prefetch_factor=2`
- **Impact** : Parallélisation du chargement des données

### 8. **Optimisation du Préprocessing** (Gain: ~1.2x)
- **Avant** : `batch_size` par défaut, cache activé
- **Après** : `batch_size=1000`, `load_from_cache_file=False`
- **Impact** : Préprocessing plus rapide, moins d'I/O

### 9. **Réduction des Logs et Sauvegardes** (Gain: ~1.1x)
- **Avant** : Logs fréquents, sauvegardes fréquentes
- **Après** : Logs et sauvegardes moins fréquents
- **Impact** : Moins d'I/O, plus de temps pour l'entraînement

### 10. **Learning Rate et Scheduler** (Gain: convergence plus rapide)
- **Avant** : `lr=2e-4`, `scheduler=cosine`
- **Après** : `lr=3e-4`, `scheduler=linear`
- **Impact** : Convergence potentiellement plus rapide

---

## 📊 Gains Attendus

### Estimation Totale
- **Gain combiné** : ~**3-5x plus rapide**
- **Temps estimé** : **1-2 heures par époque** (au lieu de 6h)

### Gains par Optimisation
| Optimisation | Gain Estimé |
|-------------|-------------|
| Réduction longueur (4096→2048) | 2-3x |
| Batch size optimisé | 1.5-2x |
| LoRA r=32 vs r=64 | 1.2-1.5x |
| Optimizer 8-bit | 1.2x |
| Pas de gradient checkpointing | 1.3-1.5x |
| Compilation (si disponible) | 1.2-1.5x |
| Dataloader optimisé | 1.1-1.2x |
| Préprocessing optimisé | 1.2x |
| **TOTAL (multiplicatif)** | **~3-5x** |

---

## ⚙️ Paramètres Optimisés

### Training Arguments
```python
per_device_train_batch_size=2  # Au lieu de 1
gradient_accumulation_steps=16  # Au lieu de 8
learning_rate=3e-4  # Au lieu de 2e-4
optim="paged_adamw_8bit"  # Au lieu de 32-bit
dataloader_num_workers=2  # Au lieu de 0
dataloader_prefetch_factor=2  # Nouveau
prediction_loss_only=True  # Au lieu de False
lr_scheduler_type="linear"  # Au lieu de cosine
max_grad_norm=1.0  # Nouveau (stabilité)
```

### LoRA Config
```python
r=32  # Au lieu de 64
lora_dropout=0.05  # Au lieu de 0.1
```

### Model Config
```python
model_max_len=2048  # Au lieu de 4096
use_gradient_checkpointing=False  # Au lieu de True
torch.compile(model)  # Nouveau (si disponible)
```

---

## ⚠️ Trade-offs

### Avantages
- ✅ **Beaucoup plus rapide** (3-5x)
- ✅ **Meilleure utilisation GPU**
- ✅ **Moins d'I/O**

### Inconvénients
- ⚠️ **Utilise plus de mémoire** (pas de gradient checkpointing)
- ⚠️ **Longueur réduite** (2048 au lieu de 4096 tokens)
- ⚠️ **Moins de paramètres LoRA** (r=32 au lieu de r=64)

---

## 🔧 Ajustements Possibles

### Si vous avez plus de mémoire GPU
```python
per_device_train_batch_size=4  # Augmenter
gradient_accumulation_steps=8  # Réduire proportionnellement
model_max_len=3072  # Augmenter si nécessaire
```

### Si vous avez moins de mémoire
```python
per_device_train_batch_size=1  # Réduire
gradient_accumulation_steps=32  # Augmenter
use_gradient_checkpointing=True  # Réactiver
model_max_len=1024  # Réduire encore
```

### Pour tests rapides
```python
# Décommentez dans le notebook :
MAX_SAMPLES = 1000
if len(dataset) > MAX_SAMPLES:
    dataset = dataset.select(range(MAX_SAMPLES))
```

---

## 📈 Monitoring

Surveillez ces métriques pour vérifier que les optimisations fonctionnent :

1. **Temps par step** : Devrait être réduit de 3-5x
2. **Utilisation GPU** : Devrait être plus élevée (>80%)
3. **Mémoire GPU** : Surveiller pour éviter OOM
4. **Loss** : Devrait converger normalement malgré les changements

---

## 🎯 Résultat Attendu

Avec ces optimisations, vous devriez passer de :
- **6 heures/époque** → **1-2 heures/époque**

Soit une **amélioration de 3-5x** en vitesse d'entraînement.

---

## 💡 Conseils Supplémentaires

1. **Testez d'abord avec un petit dataset** (décommentez MAX_SAMPLES=1000)
2. **Surveillez la mémoire GPU** - ajustez batch_size si nécessaire
3. **Vérifiez que torch.compile fonctionne** - peut donner un boost supplémentaire
4. **Si toujours trop lent**, considérez :
   - Utiliser un modèle plus petit (Llama-3 1B si disponible)
   - Réduire encore plus la longueur (1024 tokens)
   - Utiliser moins d'époques avec un learning rate plus élevé

---

*Optimisations appliquées le 22/12/2025*

