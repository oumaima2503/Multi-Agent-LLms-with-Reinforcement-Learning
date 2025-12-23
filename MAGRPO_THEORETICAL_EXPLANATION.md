# 📚 Explication Théorique de MAGRPO et Alignement avec l'Implémentation

## 🎯 Vue d'Ensemble

**MAGRPO (Multi-Agent Group Relative Policy Optimization)** est un algorithme de Reinforcement Learning adapté pour entraîner plusieurs agents LLM à collaborer. Il combine les concepts de **PPO (Proximal Policy Optimization)**, **GAE (Generalized Advantage Estimation)**, et un **Critic Centralisé** pour l'apprentissage multi-agent.

---

## 📖 1. Fondements Théoriques

### 1.1 Formalisation : Dec-POMDP

Le problème est modélisé comme un **Decentralized Partially Observable Markov Decision Process (Dec-POMDP)** :

```
Dec-POMDP = (N, S, A, O, T, R, γ)
```

Où :
- **N** : Nombre d'agents (4 dans notre cas : orchestrator, researcher, code_writer, critic)
- **S** : Espace d'états (historique de dialogue global)
- **A** : Espace d'actions (génération de texte par chaque agent)
- **O** : Observations (chaque agent voit son propre prompt)
- **T** : Fonction de transition (déterminée par les actions des agents)
- **R** : Fonction de récompense (évalue la qualité de la collaboration)
- **γ** : Facteur de discount (0.99 dans notre implémentation)

### 1.2 Objectif de MAGRPO

L'objectif est de maximiser la **récompense attendue jointe** :

```
J(θ) = E_{τ ~ π_θ} [Σ_{t=0}^T γ^t R(s_t, a_t)]
```

Où :
- `π_θ` : Politique jointe des agents (paramétrée par θ)
- `τ` : Trajectoire (séquence d'états-actions)
- `R(s_t, a_t)` : Récompense à l'étape t

**Dans le code** : Cet objectif est optimisé via PPO avec un critic centralisé.

---

## 🏗️ 2. Architecture MAGRPO

### 2.1 Principe : Centralized Training, Decentralized Execution (CTDE)

```
┌─────────────────────────────────────────────────────────┐
│              MAGRPO Architecture (CTDE)                 │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ENTRAÎNEMENT (Centralized)                             │
│  ┌──────────────────────────────────────────────┐     │
│  │  Centralized Critic V(s)                      │     │
│  │  - Évalue l'état global                        │     │
│  │  - Utilisé pour calculer les avantages       │     │
│  └──────────────────────────────────────────────┘     │
│           │                                             │
│           ▼                                             │
│  ┌──────────────────────────────────────────────┐     │
│  │  Agents (Actors)                             │     │
│  │  - Orchestrator π_θ₁                         │     │
│  │  - Researcher π_θ₂                           │     │
│  │  - CodeWriter π_θ₃                           │     │
│  │  - Critic π_θ₄                               │     │
│  └──────────────────────────────────────────────┘     │
│                                                          │
│  EXÉCUTION (Decentralized)                             │
│  - Chaque agent génère indépendamment                  │
│  - Pas besoin du critic à l'inférence                  │
└─────────────────────────────────────────────────────────┘
```

**Dans le code** :
```python
# Centralized Critic (ligne 102-124)
class CentralizedCritic(nn.Module):
    """Évalue l'état global S_t"""
    def forward(self, state_emb: torch.Tensor):
        v = self.net(state_emb)  # V(s)
        return v.squeeze(-1)
```

---

## 🔧 3. Composants Clés et Implémentation

### 3.1 Centralized Critic

**Théorie** : Le critic centralisé `V(s)` estime la **valeur de l'état global** `s`, qui inclut l'historique complet de la conversation et le tour actuel.

**Formule** :
```
V(s) = E_{a ~ π} [R(s, a) + γ V(s')]
```

**Dans le code** (lignes 102-124) :
```python
class CentralizedCritic(nn.Module):
    def __init__(self, input_dim: int = 384, hidden: int = 512):
        # MLP simple : embedding d'état → valeur
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),    # 384 → 512
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),  # 512 → 256
            nn.ReLU(),
            nn.Linear(hidden // 2, 1)         # 256 → 1 (valeur)
        )
```

**Utilisation** (lignes 315-319) :
```python
# Calculer les valeurs pour chaque état
values = []
for emb in batch_state_embs:
    v = self.critic(emb.to("cpu")).item()  # V(s)
    values.append(v)
```

**Pourquoi centralisé ?** : Permet d'évaluer la **qualité globale** de la collaboration, pas seulement les actions individuelles.

---

### 3.2 Encodage de l'État Global

**Théorie** : L'état global `s_t` doit capturer :
- L'historique complet de la conversation
- Le tour actuel
- L'agent actif

**Dans le code** (lignes 145-155) :
```python
def encode_global_state(history_text: str, turn: int, current_agent: str, device: str = "cpu"):
    """
    Encode S_t → embedding tensor
    Format: "[TURN=t][AGENT=a] {history}"
    """
    text = f"[TURN={turn}][AGENT={current_agent}] {history_text}"
    inputs = _state_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = _state_encoder(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1).squeeze(0)  # [emb_dim]
    return emb.to(device)
```

**Utilisation** (ligne 737) :
```python
state_emb = encode_global_state(env.current_state, env.turn_count, agent_name, device="cpu")
```

---

### 3.3 Generalized Advantage Estimation (GAE)

**Théorie** : GAE combine les avantages Monte-Carlo et TD pour réduire la variance tout en gardant un biais faible.

**Formule** :
```
δ_t = r_t + γ V(s_{t+1}) - V(s_t)          # TD error
A_t^GAE = Σ_{l=0}^∞ (γλ)^l δ_{t+l}        # GAE
```

Où :
- `λ` : Paramètre de trade-off variance/biais (0.95 dans notre code)
- `γ` : Facteur de discount (0.99)

**Dans le code** (lignes 296-304) :
```python
def compute_gae(self, rewards: List[float], values: List[float], gamma=0.99, lam=0.95):
    advs = []
    gae = 0.0
    values_ext = values + [0.0]  # V(s_{T+1}) = 0 (état terminal)
    
    # Calcul rétrograde (backward)
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values_ext[t+1] - values_ext[t]  # δ_t
        gae = delta + gamma * lam * gae  # A_t^GAE
        advs.insert(0, gae)
    return advs
```

**Utilisation** (lignes 321-324) :
```python
# Calculer les avantages avec GAE
advantages = self.compute_gae(batch_rewards, values)
advantages = torch.tensor(advantages, dtype=torch.float32)
# Normalisation (réduit la variance)
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

**Pourquoi GAE ?** : Réduit la variance des gradients tout en gardant un biais faible, crucial pour la stabilité de l'entraînement.

---

### 3.4 Proximal Policy Optimization (PPO)

**Théorie** : PPO limite les changements de politique pour éviter les mises à jour trop agressives.

**Formule de la loss PPO** :
```
L^CLIP(θ) = E[min(
    r(θ) * A,                    # Unclipped
    clip(r(θ), 1-ε, 1+ε) * A     # Clipped
)]
```

Où :
- `r(θ) = π_θ(a|s) / π_θ_old(a|s)` : Ratio de probabilités
- `A` : Avantage (calculé via GAE)
- `ε` : Paramètre de clipping (0.2 dans notre code)

**Dans le code** (lignes 449-453) :
```python
# Calculer le ratio de probabilités
ratio = torch.exp(new_logp - old_logp)  # r(θ) = exp(log π_new - log π_old)

# PPO clipped loss
unclipped = ratio * adv_tensor
clipped = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * adv_tensor
policy_loss = -torch.min(unclipped, clipped)  # Min pour maximiser (négation)
```

**Pourquoi PPO ?** : Évite les mises à jour trop grandes qui pourraient dégrader la politique, crucial pour l'entraînement stable de LLMs.

---

### 3.5 Calcul des Log-Probabilités

**Théorie** : Pour calculer `r(θ)`, on a besoin de :
- `log π_θ_new(a|s)` : Log-probabilité sous la nouvelle politique (actor)
- `log π_θ_old(a|s)` : Log-probabilité sous l'ancienne politique (ref model)

**Formule** :
```
log π(a|s) = Σ_{i=1}^L log π(a_i | s, a_{<i})
```

Où `L` est la longueur de la séquence générée.

**Dans le code** (lignes 157-224) :
```python
def compute_logprobs(model, input_ids, gen_ids, tokenizer, requires_grad=False):
    """
    Calcule la somme des log-probabilités des tokens générés.
    """
    # Concaténer input + génération
    full = torch.cat([input_ids, gen_ids], dim=0).unsqueeze(0)
    
    # Obtenir les logits
    if requires_grad:
        outputs = model(full)  # Avec gradients pour l'actor
    else:
        with torch.no_grad():
            outputs = model(full)  # Sans gradients pour le ref model
    
    logits = outputs.logits  # [1, L_full, vocab_size]
    lps = F.log_softmax(logits, dim=-1)  # Log-probabilités
    
    # Extraire les log-probs pour les tokens générés
    selected_log_probs = []
    for i in range(L_gen):
        pos = L_in + i
        token_id = gen_ids[i].item()
        selected_log_probs.append(lps[0, pos - 1, token_id])
    
    # Somme (préserve les gradients si requires_grad=True)
    total = torch.stack(selected_log_probs).sum()
    return total
```

**Utilisation** (lignes 375-383) :
```python
# Ref model (ancienne politique) - pas de gradients
with torch.no_grad():
    old_logp = compute_logprobs(self.ref, q_ids, gen_ids, self.tokenizer, requires_grad=False)

# Actor (nouvelle politique) - avec gradients
new_logp = compute_logprobs(self.actor, q_ids, gen_ids, self.tokenizer, requires_grad=True)
```

---

### 3.6 KL Divergence (Métrique de Contrôle)

**Théorie** : La KL divergence mesure l'écart entre l'ancienne et la nouvelle politique :

```
KL(π_old || π_new) = E_{a ~ π_old} [log π_old(a|s) - log π_new(a|s)]
```

**Pourquoi ?** : Si KL est trop grande, la politique change trop rapidement → risque d'instabilité.

**Dans le code** (ligne 471) :
```python
# KL divergence (détaché car métrique seulement)
kls.append((old_logp.detach() - new_logp.detach()).cpu())
```

**Utilisation** : Surveillée pendant l'entraînement pour détecter les mises à jour trop agressives.

---

## 🔄 4. Flux d'Entraînement MAGRPO

### 4.1 Boucle Principale

```
Pour chaque époque :
    1. Collection de trajectoires
       └─> Pour chaque épisode :
           - Reset avec instruction
           - Pour chaque tour :
               a. Encoder état global s_t
               b. Agent actif génère action a_t
               c. Obtenir récompense r_t
               d. Transition vers s_{t+1}
           - Stocker (s_t, a_t, r_t) pour chaque tour
    
    2. Calcul des valeurs
       └─> Pour chaque état s_t :
           - V(s_t) = critic(s_t)
    
    3. Calcul des avantages (GAE)
       └─> A_t = compute_gae(rewards, values)
    
    4. Mise à jour des politiques (PPO)
       └─> Pour chaque agent :
           - Calculer old_logp (ref model)
           - Calculer new_logp (actor)
           - Calculer ratio = exp(new_logp - old_logp)
           - Calculer loss = -min(r*A, clip(r,1-ε,1+ε)*A)
           - Backward + optimizer.step()
    
    5. Sauvegarde des checkpoints (si époque % SAVE_FREQ == 0)
```

**Dans le code** (lignes 845-890) :
```python
for epoch in range(TOTAL_EPOCHS):
    # 1. Collection
    transitions = collect_trajectories(env, dataset, max_episodes=2)
    
    # 2. Grouper par agent
    batches = {n: {"query":[], "response":[], "reward":[], "state":[]} for n in agents_list}
    for t in transitions:
        batches[t["agent"]]["query"].append(t["query"])
        batches[t["agent"]]["response"].append(t["response"])
        batches[t["agent"]]["reward"].append(t["reward"])
        batches[t["agent"]]["state"].append(t["state_emb"])
    
    # 3. Mise à jour
    for name in agents_list:
        stats = trainers[name].step(
            b["query"], b["response"], b["reward"], b["state"]
        )
    
    # 4. Sauvegarde
    if (epoch + 1) % SAVE_FREQ == 0:
        env.agents[name].model.save_pretrained(save_path)
```

---

### 4.2 Collection de Trajectoires

**Théorie** : Une trajectoire `τ = (s_0, a_0, r_0, s_1, a_1, r_1, ..., s_T)` capture un épisode complet.

**Dans le code** (lignes 724-764) :
```python
def collect_trajectories(env: MARL_Env, dataset, max_episodes: int):
    trajs = []
    for _ in range(max_episodes):
        # Reset avec instruction aléatoire
        instr = dataset[random.randint(0, len(dataset)-1)]["instruction"]
        env.reset(instr)
        
        episode_steps = []
        done = False
        final_reward = 0.0
        
        while not done:
            # Encoder état global
            state_emb = encode_global_state(
                env.current_state, 
                env.turn_count, 
                agent_name, 
                device="cpu"
            )
            
            # Agent génère action
            new_state, r, done, info = env.step()
            
            # Stocker transition
            episode_steps.append({
                "agent": agent_name,
                "query": q_ids,      # Prompt tokenisé
                "response": resp_ids, # Réponse générée tokenisée
                "state_emb": state_emb
            })
            
            if done:
                final_reward = r
                # Attribuer la récompense finale à toutes les transitions
                for s in episode_steps:
                    trajs.append({
                        "agent": s["agent"],
                        "query": s["query"],
                        "response": s["response"],
                        "reward": float(final_reward),  # Récompense épisodique
                        "state_emb": s["state_emb"]
                    })
    
    return trajs
```

**Point important** : La récompense est **épisodique** (attribuée à la fin) et partagée par toutes les transitions de l'épisode. Cela encourage la collaboration.

---

### 4.3 Environnement Multi-Agent

**Théorie** : L'environnement gère les transitions entre agents selon un workflow prédéfini.

**Dans le code** (lignes 634-706) :
```python
class MARL_Env:
    def step(self):
        agent = self.agents[self.current_agent]
        
        # Agent génère action
        text, gen = agent.generate_action(self.current_state)
        
        # Logique de transition
        if agent_name == "orchestrator":
            try:
                j = json.loads(text)
                tgt = j.get("AGENT_CIBLE", "").lower()
                if tgt == "end":
                    done = True
                    reward = 5.0  # Succès
                elif tgt in self.agents:
                    self.current_agent = tgt  # Transition vers autre agent
                    self.current_state += f"\n[ORCH->{tgt}]: {cmd}"
                else:
                    done = True
                    reward = -3.0  # Agent invalide
            except:
                done = True
                reward = -5.0  # JSON invalide
        else:
            # Autres agents retournent à l'orchestrator
            self.current_state += f"\n[{agent_name.upper()}]: {text}"
            self.current_agent = "orchestrator"
        
        return new_state, reward, done, info
```

**Workflow** :
```
Orchestrator → Researcher → Orchestrator → CodeWriter → Orchestrator → Critic → Orchestrator → END
```

---

## 🎯 5. Alignement Théorie ↔ Code

### 5.1 Tableau de Correspondance

| Concept Théorique | Implémentation | Lignes de Code |
|-------------------|----------------|----------------|
| **Dec-POMDP** | `MARL_Env` avec agents multiples | 634-706 |
| **État global s_t** | `encode_global_state()` | 145-155 |
| **Critic centralisé V(s)** | `CentralizedCritic` | 102-124 |
| **GAE** | `compute_gae()` | 296-304 |
| **PPO Clipped Loss** | `MAGRPOTrainer.step()` | 449-453 |
| **Log-probabilités** | `compute_logprobs()` | 157-224 |
| **Ratio r(θ)** | `exp(new_logp - old_logp)` | 450 |
| **KL Divergence** | `old_logp - new_logp` | 471 |
| **Collection trajectoires** | `collect_trajectories()` | 724-764 |
| **Optimisation LoRA** | `Adam(peft_params)` | 294 |

---

### 5.2 Points Clés d'Alignement

#### ✅ **Centralized Training**
- **Théorie** : Un critic centralisé évalue l'état global
- **Code** : `CentralizedCritic` partagé par tous les agents (ligne 810)
- **Avantage** : Permet d'évaluer la qualité globale de la collaboration

#### ✅ **Decentralized Execution**
- **Théorie** : Chaque agent génère indépendamment à l'inférence
- **Code** : `agent.generate_action()` (ligne 625-632)
- **Avantage** : Pas besoin du critic à l'inférence → plus rapide

#### ✅ **Group-Relative Advantages**
- **Théorie** : Les avantages sont calculés relativement au groupe
- **Code** : Normalisation des avantages (ligne 324)
- **Avantage** : Réduit la variance et améliore la stabilité

#### ✅ **PPO Clipping**
- **Théorie** : Limite les changements de politique
- **Code** : `torch.clamp(ratio, 1-ε, 1+ε)` (ligne 452)
- **Avantage** : Évite les mises à jour trop agressives

#### ✅ **Reference Model**
- **Théorie** : Ancre la politique pour éviter la dérive
- **Code** : `ref_model` chargé depuis SFT (ligne 829)
- **Avantage** : Maintient la qualité de base du modèle

---

## 📊 6. Optimisations Techniques

### 6.1 Gestion Mémoire (CPU/GPU)

**Problème** : Les modèles LLM sont trop grands pour tenir en mémoire GPU.

**Solution** : Modèles sur CPU, déplacés vers GPU uniquement pendant les calculs.

**Dans le code** :
```python
# Déplacer vers GPU pour calcul
move_model_to_device(self.actor, device)  # Ligne 336

# Calculer
new_logp = compute_logprobs(self.actor, ...)  # Ligne 383

# Retourner sur CPU immédiatement
offload_model_to_cpu(self.actor)  # Ligne 474
```

### 6.2 Quantization 4-bit

**Théorie** : Réduit la mémoire de 4x en utilisant seulement 4 bits par paramètre.

**Dans le code** (lignes 536-541) :
```python
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, quantization_config=bnb)
```

### 6.3 LoRA (Low-Rank Adaptation)

**Théorie** : Entraîne seulement une petite matrice de rang faible au lieu de tous les paramètres.

**Dans le code** :
- Seulement les paramètres LoRA sont entraînables (lignes 272-282)
- Optimizer ne contient que ces paramètres (ligne 294)

**Avantage** : Réduit drastiquement le nombre de paramètres entraînables (2.84% du modèle total).

---

## 🎓 7. Résumé : Pourquoi MAGRPO Fonctionne

### 7.1 Avantages par Rapport à l'Entraînement Individuel

1. **Collaboration** : Les agents apprennent à travailler ensemble
2. **Récompenses Globales** : La récompense épisodique encourage la coordination
3. **Critic Centralisé** : Évalue la qualité globale, pas seulement individuelle
4. **Stabilité** : PPO + Reference Model évitent la dérive

### 7.2 Différences avec PPO Standard

| Aspect | PPO Standard | MAGRPO |
|--------|--------------|--------|
| **Critic** | Par agent | Centralisé (partagé) |
| **Récompense** | Par action | Épisodique (partagée) |
| **État** | Local à l'agent | Global (historique complet) |
| **Avantages** | Individuels | Group-relative (normalisés) |

---

## 📝 8. Conclusion

Le notebook `train_magrpo_kaggle.ipynb` implémente fidèlement l'algorithme MAGRPO :

1. ✅ **Centralized Critic** : Évalue l'état global
2. ✅ **GAE** : Calcule les avantages avec réduction de variance
3. ✅ **PPO** : Optimise les politiques avec clipping
4. ✅ **Decentralized Execution** : Chaque agent génère indépendamment
5. ✅ **Group-Relative Advantages** : Normalisation pour stabilité
6. ✅ **Reference Model** : Ancre la politique pour éviter la dérive

L'implémentation est **optimisée pour la mémoire** (4-bit quantization, LoRA, CPU/GPU management) et **stable** (PPO clipping, KL monitoring).

---

## 🔗 Références

- **Article** : "LLM Collaboration with Multi-Agent Reinforcement Learning"
- **PPO** : Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
- **GAE** : Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation" (2016)
- **LoRA** : Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021)

