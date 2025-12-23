# 🚀 Guide d'Implémentation MAGRPO - Basé sur l'Article

## 📚 Référence

**Article** : "LLM Collaboration with Multi-Agent Reinforcement Learning"  
**Auteurs** : Shuo Liu, Tianle Chen, Zeyu Liang, Xueguang Lyu, Christopher Amato  
**Code** : https://github.com/OpenMLRL/CoMLRL

---

## 🎯 Concepts Clés de MAGRPO

### 1. **Multi-Agent Group Relative Policy Optimization (MAGRPO)**

MAGRPO est basé sur **GRPO (Group Relative Policy Optimization)** et adapté pour l'entraînement multi-agent coopératif.

**Principes :**
- ✅ **Entraînement centralisé** : Utilise un critic centralisé pour l'optimisation jointe
- ✅ **Exécution décentralisée** : Chaque agent génère ses réponses indépendamment
- ✅ **Avantages group-relative** : Les avantages sont calculés relativement au groupe
- ✅ **Coopération** : Les agents apprennent à collaborer pour des réponses jointes

### 2. **Modélisation : Dec-POMDP**

Le problème est formalisé comme un **Decentralized Partially Observable Markov Decision Process (Dec-POMDP)** :

- **Agents** : Multiple LLMs avec rôles spécialisés
- **État** : État global de l'environnement (historique de dialogue)
- **Actions** : Génération de réponses par chaque agent
- **Observations** : Chaque agent voit son propre prompt/instruction
- **Récompenses** : Récompenses alignées avec les préférences humaines

---

## 🏗️ Architecture MAGRPO

### Composants Principaux

```
┌─────────────────────────────────────────────────────────┐
│              MAGRPO Architecture                       │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐      ┌──────────────┐               │
│  │   Agent 1    │      │   Agent 2    │               │
│  │  (Actor)     │      │  (Actor)     │               │
│  └──────┬───────┘      └──────┬───────┘               │
│         │                      │                        │
│         └──────────┬───────────┘                        │
│                    │                                    │
│         ┌──────────▼──────────┐                        │
│         │ Centralized Critic  │                        │
│         │   (V(s))            │                        │
│         └──────────┬───────────┘                        │
│                    │                                    │
│         ┌──────────▼──────────┐                        │
│         │  Group-Relative     │                        │
│         │  Advantages         │                        │
│         └─────────────────────┘                        │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### 1. **Actors (Agents)**

Chaque agent est un modèle LLM fine-tuné avec LoRA :
- **Actor Model** : Modèle actuel en entraînement
- **Reference Model** : Modèle de référence (frozen) pour calculer KL divergence
- **Tokenizer** : Tokenizer partagé

### 2. **Centralized Critic**

Un critic centralisé qui évalue l'état global :
- **Input** : État global encodé (embedding)
- **Output** : Valeur V(s) de l'état
- **Rôle** : Fournir des valeurs pour calculer les avantages

### 3. **Group-Relative Advantages**

Les avantages sont calculés relativement au groupe :
- Utilise **GAE (Generalized Advantage Estimation)**
- Normalisation par rapport au groupe
- Permet l'optimisation jointe tout en gardant l'exécution décentralisée

---

## 📋 Implémentation Détaillée

### Étape 1 : Configuration

```python
# Configuration basée sur l'article
BASE_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
AGENTS_LIST = ["orchestrator", "researcher", "code_writer", "critic"]
TOTAL_EPOCHS = 10  # Commencez avec 10, ajustez selon résultats
SAVE_FREQ = 5
CHECKPOINTS_DIR = "checkpoints"  # Chemin vers vos checkpoints SFT
SAVE_FOLDER = "checkpoints/magrpo_rl"
```

### Étape 2 : Chargement des Modèles

```python
def load_agent_policy(agent_name: str, is_training=True):
    """
    Charge un agent fine-tuné avec SFT.
    
    Args:
        agent_name: Nom de l'agent
        is_training: Si True, prépare pour l'entraînement
    
    Returns:
        model, tokenizer
    """
    lora_path = os.path.join(CHECKPOINTS_DIR, f"{agent_name}_lora")
    
    # Vérifier que le checkpoint existe
    if not os.path.exists(os.path.join(lora_path, "adapter_config.json")):
        raise FileNotFoundError(f"Missing adapter for {agent_name} at {lora_path}")
    
    # Configuration 4-bit quantization
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Charger tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Charger modèle de base
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, 
        quantization_config=bnb, 
        device_map="cpu"
    )
    
    # Préparer pour l'entraînement si nécessaire
    if is_training:
        prepare_model_for_kbit_training(base)
        base.config.use_cache = False
    else:
        base.config.use_cache = True
    
    # Charger LoRA
    lora_cfg = LoraConfig.from_pretrained(lora_path)
    model = get_peft_model(base, lora_cfg)
    
    return model, tokenizer
```

### Étape 3 : Système de Récompenses

**Selon l'article, les récompenses doivent être :**
- ✅ **Alignées avec les préférences humaines**
- ✅ **Vérifiables** (par exemple, tests unitaires pour le code)
- ✅ **Multi-aspects** (qualité, format, coopération)

**Exemple de fonction de récompense :**

```python
def compute_reward(response: str, expected_format: dict, agent_name: str) -> float:
    """
    Calcule la récompense pour une réponse d'agent.
    
    Args:
        response: Réponse générée par l'agent
        expected_format: Format JSON attendu
        agent_name: Nom de l'agent
    
    Returns:
        Récompense (float)
    """
    reward = 0.0
    
    # 1. Récompense pour JSON valide
    try:
        parsed = json.loads(response)
        reward += 0.3  # Base reward pour JSON valide
    except:
        return -1.0  # Pénalité forte pour JSON invalide
    
    # 2. Récompense pour clés présentes
    expected_keys = expected_format.get("required_keys", [])
    keys_present = sum(1 for k in expected_keys if k in parsed)
    key_reward = (keys_present / len(expected_keys)) * 0.4 if expected_keys else 0.0
    reward += key_reward
    
    # 3. Récompense pour format exact
    if all(k in parsed for k in expected_keys):
        reward += 0.2  # Bonus pour format parfait
    
    # 4. Récompense pour qualité du contenu (selon l'agent)
    if agent_name == "orchestrator":
        # Vérifier que delegated_agent est valide
        if parsed.get("delegated_agent") in ["Researcher", "CodeWriter", "Critic", "FINISHED"]:
            reward += 0.1
    elif agent_name == "code_writer":
        # Vérifier que le code est exécutable
        if "python_code" in parsed:
            try:
                compile(parsed["python_code"], "<string>", "exec")
                reward += 0.1
            except:
                reward -= 0.1
    
    return reward
```

### Étape 4 : Environnement Multi-Agent

```python
class MARL_Env:
    """
    Environnement multi-agent pour la collaboration LLM.
    Modélise un Dec-POMDP où les agents coopèrent.
    """
    
    def __init__(self, agents_list):
        self.agents = {}
        for name in agents_list:
            agent = LLMAgent(name)
            agent.load_policy()
            self.agents[name] = agent
        
        self.current_state = ""
        self.current_agent = "orchestrator"
        self.turn_count = 0
        self.max_turns = 10
    
    def reset(self, instruction: str):
        """Réinitialise l'environnement avec une nouvelle instruction."""
        self.current_state = f"Instruction: {instruction}"
        self.current_agent = "orchestrator"
        self.turn_count = 0
        return self.current_state
    
    def step(self):
        """
        Exécute une étape dans l'environnement.
        
        Returns:
            new_state, reward, done, info
        """
        agent_name = self.current_agent
        agent = self.agents[agent_name]
        
        # Offload autres agents pour économiser la mémoire
        for n, a in self.agents.items():
            if n != agent_name:
                offload_model_to_cpu(a.model)
        
        # Déplacer l'agent actif sur GPU
        move_model_to_device(agent.model, "cuda")
        
        # Générer la réponse
        prompt = format_prompt(agent.system_prompt, self.current_state)
        text, gen_tokens = agent.generate_action(self.current_state)
        
        # Offload l'agent
        offload_model_to_cpu(agent.model)
        
        # Calculer la récompense
        reward = self._compute_reward(text, agent_name)
        
        # Mettre à jour l'état
        done = False
        if agent_name == "orchestrator":
            try:
                j = json.loads(text)
                delegated = j.get("delegated_agent", "").strip()
                if delegated == "FINISHED":
                    done = True
                    reward += 5.0  # Bonus pour terminaison réussie
                elif delegated in self.agents:
                    self.current_agent = delegated
                    instruction = j.get("instruction", "")
                    self.current_state += f"\n[ORCH->{delegated}]: {instruction}"
                else:
                    done = True
                    reward -= 3.0
            except:
                done = True
                reward -= 5.0
        else:
            self.current_state += f"\n[{agent_name.upper()}]: {text}"
            self.current_agent = "orchestrator"
        
        self.turn_count += 1
        if self.turn_count >= self.max_turns:
            done = True
            reward -= 5.0  # Pénalité pour dépassement de limite
        
        return self.current_state, reward, done, {
            "response": text,
            "agent": agent_name,
            "tokens": gen_tokens
        }
```

### Étape 5 : Collection de Trajectoires

```python
def collect_trajectories(env: MARL_Env, dataset, max_episodes: int):
    """
    Collecte des trajectoires pour l'entraînement MAGRPO.
    
    Selon l'article, les trajectoires sont collectées de manière épisodique.
    """
    trajectories = []
    
    num_episodes = min(max_episodes, len(dataset))
    
    for _ in range(num_episodes):
        # Sélectionner un épisode aléatoire
        idx = random.randint(0, len(dataset) - 1)
        instruction = dataset[idx]["instruction"]
        
        # Réinitialiser l'environnement
        env.reset(instruction)
        
        episode_steps = []
        done = False
        final_reward = 0.0
        
        while not done:
            agent_name = env.current_agent
            
            # Encoder l'état global
            state_emb = encode_global_state(
                env.current_state,
                env.turn_count,
                agent_name,
                device="cpu"
            )
            
            # Exécuter une étape
            new_state, reward, done, info = env.step()
            
            if info.get("response") is not None:
                agent = env.agents[agent_name]
                
                # Encoder query et response
                query_ids = agent.tokenizer(
                    env.last_query,
                    return_tensors="pt",
                    truncation=True
                ).input_ids.squeeze(0).cpu()
                
                response_ids = info.get("tokens", torch.tensor([], dtype=torch.long)).cpu()
                
                episode_steps.append({
                    "agent": agent_name,
                    "query": query_ids,
                    "response": response_ids,
                    "state_emb": state_emb.detach().cpu(),
                    "reward": reward
                })
            
            if done:
                final_reward = reward
                # Assigner la récompense finale à toutes les étapes
                for step in episode_steps:
                    step["reward"] = final_reward
                    trajectories.append(step)
                break
    
    logging.info(f"Collected {len(trajectories)} transitions.")
    return trajectories
```

### Étape 6 : Entraînement MAGRPO

```python
def train_marl_magrpo(agents_list):
    """
    Entraînement principal MAGRPO.
    
    Basé sur l'article :
    - Centralized critic pour l'optimisation jointe
    - Decentralized execution pour chaque agent
    - Group-relative advantages
    """
    # Initialiser l'environnement
    env = MARL_Env(agents_list)
    
    # Charger le dataset
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    
    # Initialiser le critic centralisé
    state_dim = 384  # Dimension de l'embedding d'état
    critic = CentralizedCritic(input_dim=state_dim, hidden=512)
    critic.to("cpu")
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-4)
    
    # Préparer les trainers pour chaque agent
    trainers = {}
    for name in agents_list:
        actor = env.agents[name].model
        ref_model, _ = load_agent_policy(name, is_training=False)
        offload_model_to_cpu(ref_model)
        
        trainer = MAGRPOTrainer(
            actor_model=actor,
            ref_model=ref_model,
            tokenizer=env.agents[name].tokenizer,
            critic=critic,
            lr=1.41e-5,  # Learning rate selon l'article
            clip_epsilon=0.2,
            device="cuda"
        )
        trainers[name] = trainer
        logging.info(f"Trainer ready for {name}")
    
    # Boucle d'entraînement
    for epoch in range(TOTAL_EPOCHS):
        logging.info(f"Epoch {epoch + 1}/{TOTAL_EPOCHS}")
        
        # Collecter des trajectoires
        transitions = collect_trajectories(env, dataset, max_episodes=2)
        
        if not transitions:
            logging.warning("No transitions collected.")
            break
        
        # Grouper par agent
        batches = {
            n: {"query": [], "response": [], "reward": [], "state": []}
            for n in agents_list
        }
        
        for t in transitions:
            batches[t["agent"]]["query"].append(t["query"])
            batches[t["agent"]]["response"].append(t["response"])
            batches[t["agent"]]["reward"].append(t["reward"])
            batches[t["agent"]]["state"].append(t["state_emb"])
        
        # Mettre à jour chaque agent
        for name in agents_list:
            b = batches[name]
            if not b["query"]:
                continue
            
            # Step du trainer MAGRPO
            stats = trainers[name].step(
                b["query"],
                b["response"],
                b["reward"],
                b["state"]
            )
            
            logging.info(
                f"{name} update: "
                f"loss {stats['loss']:.4f} "
                f"kl {stats['kl']:.6f} "
                f"val_mean {stats['value_mean']:.3f}"
            )
        
        # Sauvegarder les checkpoints
        if (epoch + 1) % SAVE_FREQ == 0:
            os.makedirs(SAVE_FOLDER, exist_ok=True)
            for name in agents_list:
                save_path = os.path.join(
                    SAVE_FOLDER,
                    f"epoch{epoch+1}_{name}_rl"
                )
                try:
                    env.agents[name].model.save_pretrained(save_path)
                    logging.info(f"Saved RL LoRA for {name} -> {save_path}")
                except Exception as e:
                    logging.warning(f"Failed to save {name}: {e}")
    
    logging.info("Training finished.")
```

---

## 🔧 Améliorations Basées sur l'Article

### 1. **Group-Relative Advantages**

L'article mentionne que MAGRPO utilise des avantages group-relative. L'implémentation actuelle utilise GAE standard. Pour être plus fidèle à l'article :

```python
def compute_group_relative_advantages(rewards, values, group_rewards, group_values):
    """
    Calcule les avantages group-relative selon l'article.
    
    Les avantages sont calculés relativement au groupe plutôt qu'individuellement.
    """
    # Calculer la moyenne du groupe
    group_mean_reward = np.mean(group_rewards)
    group_mean_value = np.mean(group_values)
    
    # Ajuster les avantages par rapport au groupe
    relative_advantages = []
    for r, v in zip(rewards, values):
        # Avantage relatif au groupe
        relative_adv = (r - group_mean_reward) + (v - group_mean_value)
        relative_advantages.append(relative_adv)
    
    return relative_advantages
```

### 2. **Système de Récompenses Multi-Aspects**

Selon l'article, les récompenses doivent être :
- **Multi-aspects** : Qualité, format, coopération
- **Process-supervised** : Récompenses pour le processus, pas seulement le résultat
- **Human-aligned** : Alignées avec les préférences humaines

### 3. **Critic Centralisé Amélioré**

Le critic peut être amélioré pour mieux capturer l'état global :

```python
class ImprovedCentralizedCritic(nn.Module):
    """
    Critic centralisé amélioré basé sur l'article.
    Prend en compte l'état global et les actions jointes.
    """
    def __init__(self, input_dim: int = 384, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden // 2, 1)
        )
    
    def forward(self, state_emb: torch.Tensor):
        if state_emb.dim() == 1:
            state_emb = state_emb.unsqueeze(0)
        v = self.net(state_emb)
        return v.squeeze(-1)
```

---

## 📊 Métriques à Surveiller

Selon l'article, surveillez :

1. **Loss RL** : Perte de la politique
2. **KL Divergence** : Distance entre actor et reference
3. **Value Mean** : Valeur moyenne du critic
4. **Récompense moyenne** : Récompense moyenne par épisode
5. **Taux de succès** : Pourcentage d'épisodes réussis
6. **Efficacité** : Nombre de tours nécessaires pour compléter une tâche

---

## 🚀 Prochaines Étapes

1. ✅ Vérifier que tous les checkpoints SFT sont présents
2. ✅ Configurer le système de récompenses
3. ✅ Lancer l'entraînement MAGRPO
4. ✅ Monitorer les métriques
5. ✅ Ajuster les hyperparamètres si nécessaire

---

## 📝 Notes Importantes

- **Commencez petit** : 10 époques, puis évaluez
- **Monitorer activement** : Surveillez les métriques toutes les 2-3 époques
- **Sauvegarder régulièrement** : Sauvegardez toutes les 5 époques
- **Itérer** : Ajustez les hyperparamètres selon les résultats

---

*Guide basé sur l'article "LLM Collaboration with Multi-Agent Reinforcement Learning" (2508.04652v7)*

