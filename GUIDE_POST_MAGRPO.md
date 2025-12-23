# 🚀 Guide Post-MAGRPO : Utiliser les Agents Entraînés

## 📋 Vue d'Ensemble

Après l'entraînement MAGRPO, vous avez des checkpoints optimisés pour la collaboration multi-agent. Ce guide vous montre comment :
1. **Charger** les checkpoints MAGRPO
2. **Tester** les agents individuellement
3. **Interagir** avec le système multi-agent complet
4. **Comparer** les performances SFT vs MAGRPO
5. **Évaluer** l'amélioration apportée par le RL

---

## 📦 Étape 1 : Préparer les Checkpoints

### 1.1 Télécharger depuis Kaggle

Si vous avez entraîné sur Kaggle :

1. Allez dans l'onglet **"Output"** du notebook
2. Téléchargez le dossier `checkpoints/magrpo_rl/`
3. Placez-le dans votre projet local : `checkpoints/magrpo_rl/`

### 1.2 Structure Attendue

```
checkpoints/
├── magrpo_rl/
│   ├── epoch10_orchestratc_rl/  (ou orchestrator_rl)
│   ├── epoch10_researcher_rl/
│   ├── epoch10_code_write_rl/
│   ├── epoch10_critic_rl/
│   ├── epoch15_.../
│   └── epoch20_.../
└── [checkpoints SFT originaux]
    ├── orchestrator_lora/
    ├── researcher_lora/
    ├── code_writer_lora/
    └── critic_lora/
```

---

## 🔧 Étape 2 : Modifier les Agents pour Charger MAGRPO

### 2.1 Option A : Modifier `BaseAgent` pour Accepter un Chemin Personnalisé

Créez une fonction helper pour charger les checkpoints MAGRPO :

```python
# Dans agents/base_agents.py ou un nouveau fichier

def load_magrpo_checkpoint(agent_name: str, epoch: int = 20, base_path: str = "checkpoints/magrpo_rl"):
    """
    Charge un checkpoint MAGRPO pour un agent.
    
    Args:
        agent_name: Nom de l'agent (orchestrator, researcher, code_writer, critic)
        epoch: Époque du checkpoint (10, 15, 20)
        base_path: Chemin de base vers les checkpoints MAGRPO
    
    Returns:
        Chemin complet vers le checkpoint
    """
    # Mapping des noms d'agents
    agent_mapping = {
        "orchestrator": "orchestratc",  # Note: vérifiez le nom exact dans vos checkpoints
        "researcher": "researcher",
        "code_writer": "code_write",
        "critic": "critic_rl"
    }
    
    checkpoint_name = agent_mapping.get(agent_name, agent_name)
    checkpoint_path = os.path.join(base_path, f"epoch{epoch}_{checkpoint_name}_rl")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint MAGRPO non trouvé: {checkpoint_path}")
    
    return checkpoint_path
```

### 2.2 Option B : Créer une Classe Agent MAGRPO

Créez `agents/magrpo_agents.py` :

```python
from agents.base_agents import BaseAgent, OrchestratorAgent, ResearcherAgent, CodeWriterAgent, CriticAgent
import os

class MAGRPOrchestratorAgent(OrchestratorAgent):
    """Orchestrator avec checkpoint MAGRPO"""
    def __init__(self, epoch: int = 20, magrpo_base_path: str = "checkpoints/magrpo_rl"):
        # Chemin vers le checkpoint MAGRPO
        checkpoint_name = "orchestratc"  # Vérifiez le nom exact
        lora_folder = os.path.join(magrpo_base_path, f"epoch{epoch}_{checkpoint_name}_rl")
        
        # Appeler le constructeur parent avec le nouveau chemin
        super().__init__()
        # Remplacer le chemin LoRA
        self.lora_path = lora_folder
        # Recharger le modèle
        self._load_model()

# Répéter pour les autres agents...
```

---

## 🧪 Étape 3 : Tester un Agent Individuel

### 3.1 Script de Test Simple

Créez `test_magrpo_agent.py` :

```python
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agents import OrchestratorAgent, ResearcherAgent, CodeWriterAgent, CriticAgent
import json

def test_magrpo_agent(agent_name: str, query: str, epoch: int = 20):
    """
    Teste un agent avec un checkpoint MAGRPO.
    
    Args:
        agent_name: orchestrator, researcher, code_writer, critic
        query: Requête de test
        epoch: Époque du checkpoint (10, 15, 20)
    """
    print(f"\n{'='*60}")
    print(f"🧪 Test Agent: {agent_name.upper()} (MAGRPO Epoch {epoch})")
    print(f"{'='*60}")
    print(f"Requête: {query}\n")
    
    # Mapping des agents
    agent_classes = {
        "orchestrator": OrchestratorAgent,
        "researcher": ResearcherAgent,
        "code_writer": CodeWriterAgent,
        "critic": CriticAgent
    }
    
    if agent_name not in agent_classes:
        print(f"❌ Agent inconnu: {agent_name}")
        return
    
    # Créer l'agent
    agent_class = agent_classes[agent_name]
    
    # Modifier le chemin LoRA pour pointer vers MAGRPO
    agent = agent_class()
    
    # Remplacer le chemin LoRA
    checkpoint_name = {
        "orchestrator": "orchestratc",
        "researcher": "researcher",
        "code_writer": "code_write",
        "critic": "critic_rl"
    }[agent_name]
    
    agent.lora_path = f"checkpoints/magrpo_rl/epoch{epoch}_{checkpoint_name}_rl"
    
    # Recharger le modèle
    print(f"📦 Chargement du checkpoint: {agent.lora_path}")
    agent._load_model()
    
    # Tester
    print(f"\n🔄 Génération de la réponse...")
    try:
        result = agent.act(query, fast_mode=False)
        print(f"\n✅ Résultat:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        # Vérifier le format
        if isinstance(result, dict):
            print(f"\n✅ Format JSON valide")
        else:
            print(f"\n⚠️  Format non-JSON")
            
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Tests de base
    test_queries = {
        "orchestrator": "Planifie une analyse comparative entre le Pixel 8 et l'iPhone 15.",
        "researcher": "Cherche la date de sortie exacte du Google Pixel 8 Pro.",
        "code_writer": "Fais un script Python pour calculer une remise de 15% sur un prix de 899€.",
        "critic": "Évalue ceci : 'Le smartphone est cher mais puissant'."
    }
    
    # Tester tous les agents
    for agent_name, query in test_queries.items():
        test_magrpo_agent(agent_name, query, epoch=20)
        print("\n" + "-"*60 + "\n")
```

### 3.2 Exécution

```bash
python test_magrpo_agent.py
```

---

## 🤝 Étape 4 : Interagir avec le Système Multi-Agent Complet

### 4.1 Script d'Interaction Interactive

Créez `interact_magrpo.py` :

```python
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agents import OrchestratorAgent, ResearcherAgent, CodeWriterAgent, CriticAgent
import json

class MAGRPOMultiAgentSystem:
    """
    Système multi-agent utilisant les checkpoints MAGRPO.
    Simule le workflow de collaboration entre agents.
    """
    def __init__(self, epoch: int = 20):
        self.epoch = epoch
        self.agents = {}
        self.history = []
        self.current_agent = "orchestrator"
        
        # Charger tous les agents
        self._load_agents()
    
    def _load_agents(self):
        """Charge tous les agents avec les checkpoints MAGRPO"""
        print("📦 Chargement des agents MAGRPO...")
        
        agent_configs = {
            "orchestrator": (OrchestratorAgent, "orchestratc"),
            "researcher": (ResearcherAgent, "researcher"),
            "code_writer": (CodeWriterAgent, "code_write"),
            "critic": (CriticAgent, "critic_rl")
        }
        
        for name, (agent_class, checkpoint_name) in agent_configs.items():
            print(f"   Chargement de {name}...")
            agent = agent_class()
            agent.lora_path = f"checkpoints/magrpo_rl/epoch{self.epoch}_{checkpoint_name}_rl"
            agent._load_model()
            self.agents[name] = agent
            print(f"   ✅ {name} chargé")
        
        print("✅ Tous les agents chargés\n")
    
    def reset(self, initial_query: str):
        """Réinitialise le système avec une nouvelle requête"""
        self.history = [f"User: {initial_query}"]
        self.current_agent = "orchestrator"
        print(f"\n🔄 Nouvelle session: {initial_query}\n")
    
    def step(self):
        """Exécute une étape du workflow multi-agent"""
        if self.current_agent not in self.agents:
            print(f"❌ Agent inconnu: {self.current_agent}")
            return None, True
        
        agent = self.agents[self.current_agent]
        current_state = "\n".join(self.history)
        
        print(f"🤖 Agent actif: {self.current_agent.upper()}")
        print(f"📝 État actuel: {current_state[:100]}...\n")
        
        # Agent génère une action
        try:
            result = agent.act(current_state, fast_mode=False)
            print(f"✅ Réponse de {self.current_agent}:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            
            # Ajouter à l'historique
            response_text = json.dumps(result, ensure_ascii=False)
            self.history.append(f"[{self.current_agent.upper()}]: {response_text}")
            
            # Déterminer le prochain agent (logique simplifiée)
            if self.current_agent == "orchestrator":
                # L'orchestrator décide du prochain agent
                if isinstance(result, dict):
                    next_agent = result.get("delegated_agent", "").lower()
                    if next_agent in self.agents:
                        self.current_agent = next_agent
                    elif next_agent == "end":
                        return result, True  # Terminé
                    else:
                        self.current_agent = "orchestrator"  # Retour à l'orchestrator
                else:
                    self.current_agent = "orchestrator"
            else:
                # Les autres agents retournent à l'orchestrator
                self.current_agent = "orchestrator"
            
            return result, False
            
        except Exception as e:
            print(f"❌ Erreur: {e}")
            import traceback
            traceback.print_exc()
            return None, True
    
    def run(self, initial_query: str, max_turns: int = 10):
        """Exécute un workflow complet"""
        self.reset(initial_query)
        
        for turn in range(max_turns):
            print(f"\n{'='*60}")
            print(f"Tour {turn + 1}/{max_turns}")
            print(f"{'='*60}\n")
            
            result, done = self.step()
            
            if done:
                print(f"\n✅ Workflow terminé après {turn + 1} tours")
                return result
            
            if turn >= max_turns - 1:
                print(f"\n⚠️  Nombre maximum de tours atteint")
                return result
        
        return None

if __name__ == "__main__":
    # Créer le système
    system = MAGRPOMultiAgentSystem(epoch=20)
    
    # Test interactif
    print("="*60)
    print("🚀 Système Multi-Agent MAGRPO")
    print("="*60)
    
    # Exemple 1
    print("\n📋 Exemple 1: Analyse comparative")
    result1 = system.run("Compare le Pixel 8 et l'iPhone 15", max_turns=5)
    
    # Exemple 2
    print("\n\n📋 Exemple 2: Recherche et code")
    result2 = system.run("Trouve la date de sortie du Pixel 8 et crée un script pour calculer son prix avec remise", max_turns=6)
```

### 4.2 Mode Interactif (CLI)

Créez `interact_cli.py` :

```python
from interact_magrpo import MAGRPOMultiAgentSystem

def main():
    system = MAGRPOMultiAgentSystem(epoch=20)
    
    print("="*60)
    print("🚀 Système Multi-Agent MAGRPO - Mode Interactif")
    print("="*60)
    print("\nTapez 'quit' pour quitter\n")
    
    while True:
        query = input("Vous: ")
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("👋 Au revoir!")
            break
        
        if not query.strip():
            continue
        
        result = system.run(query, max_turns=10)
        
        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()
```

---

## 📊 Étape 5 : Comparer SFT vs MAGRPO

### 5.1 Script de Comparaison

Créez `compare_sft_magrpo.py` :

```python
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agents import OrchestratorAgent, ResearcherAgent, CodeWriterAgent, CriticAgent
import json
import time

def test_agent_with_checkpoint(agent_name: str, query: str, checkpoint_type: str, epoch: int = None):
    """
    Teste un agent avec un checkpoint spécifique.
    
    Args:
        agent_name: orchestrator, researcher, code_writer, critic
        query: Requête de test
        checkpoint_type: 'sft' ou 'magrpo'
        epoch: Époque pour MAGRPO (10, 15, 20)
    """
    agent_classes = {
        "orchestrator": OrchestratorAgent,
        "researcher": ResearcherAgent,
        "code_writer": CodeWriterAgent,
        "critic": CriticAgent
    }
    
    agent_class = agent_classes[agent_name]
    agent = agent_class()
    
    # Définir le chemin selon le type
    if checkpoint_type == "sft":
        agent.lora_path = f"checkpoints/{agent_name}_lora"
    elif checkpoint_type == "magrpo":
        checkpoint_name = {
            "orchestrator": "orchestratc",
            "researcher": "researcher",
            "code_writer": "code_write",
            "critic": "critic_rl"
        }[agent_name]
        agent.lora_path = f"checkpoints/magrpo_rl/epoch{epoch}_{checkpoint_name}_rl"
    else:
        raise ValueError(f"Type de checkpoint inconnu: {checkpoint_type}")
    
    # Charger et tester
    agent._load_model()
    
    start_time = time.time()
    try:
        result = agent.act(query, fast_mode=False)
        elapsed = time.time() - start_time
        
        # Métriques
        is_json = isinstance(result, dict)
        has_expected_keys = False
        
        if is_json:
            # Vérifier les clés attendues selon l'agent
            expected_keys = {
                "orchestrator": ["delegated_agent", "instruction"],
                "researcher": ["research_query", "final_answer"],
                "code_writer": ["python_code", "result_explanation"],
                "critic": ["critique_ok", "suggestions"]
            }
            keys = expected_keys.get(agent_name, [])
            has_expected_keys = all(k in result for k in keys) if keys else True
        
        return {
            "success": True,
            "result": result,
            "is_json": is_json,
            "has_expected_keys": has_expected_keys,
            "time": elapsed
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "time": time.time() - start_time
        }

def compare_agents(agent_name: str, query: str, epochs: list = [10, 15, 20]):
    """
    Compare les performances SFT vs MAGRPO pour un agent.
    """
    print(f"\n{'='*70}")
    print(f"📊 Comparaison: {agent_name.upper()}")
    print(f"{'='*70}")
    print(f"Requête: {query}\n")
    
    results = {}
    
    # Test SFT
    print("🔵 Test SFT...")
    results["sft"] = test_agent_with_checkpoint(agent_name, query, "sft")
    
    # Test MAGRPO pour chaque époque
    for epoch in epochs:
        print(f"🟢 Test MAGRPO Epoch {epoch}...")
        results[f"magrpo_epoch{epoch}"] = test_agent_with_checkpoint(agent_name, query, "magrpo", epoch)
    
    # Afficher les résultats
    print(f"\n{'='*70}")
    print("📈 Résultats")
    print(f"{'='*70}\n")
    
    for name, result in results.items():
        print(f"{name.upper()}:")
        if result["success"]:
            print(f"  ✅ Succès")
            print(f"  📝 JSON valide: {result['is_json']}")
            print(f"  🔑 Clés correctes: {result.get('has_expected_keys', 'N/A')}")
            print(f"  ⏱️  Temps: {result['time']:.2f}s")
            if result.get('result'):
                print(f"  📄 Résultat: {str(result['result'])[:100]}...")
        else:
            print(f"  ❌ Échec: {result.get('error', 'Unknown error')}")
        print()
    
    return results

if __name__ == "__main__":
    # Tests de comparaison
    test_cases = [
        ("orchestrator", "Planifie une analyse comparative entre le Pixel 8 et l'iPhone 15."),
        ("researcher", "Cherche la date de sortie exacte du Google Pixel 8 Pro."),
        ("code_writer", "Fais un script Python pour calculer une remise de 15% sur un prix de 899€."),
        ("critic", "Évalue ceci : 'Le smartphone est cher mais puissant'.")
    ]
    
    all_results = {}
    
    for agent_name, query in test_cases:
        results = compare_agents(agent_name, query, epochs=[10, 15, 20])
        all_results[agent_name] = results
        print("\n" + "="*70 + "\n")
    
    # Résumé global
    print("\n" + "="*70)
    print("📊 RÉSUMÉ GLOBAL")
    print("="*70 + "\n")
    
    for agent_name, results in all_results.items():
        print(f"{agent_name.upper()}:")
        for name, result in results.items():
            if result["success"]:
                print(f"  {name}: ✅ JSON={result['is_json']}, Clés={result.get('has_expected_keys', 'N/A')}")
            else:
                print(f"  {name}: ❌")
        print()
```

---

## 📈 Étape 6 : Évaluation Quantitative

### 6.1 Script d'Évaluation

Créez `evaluate_magrpo.py` :

```python
import json
import os
from compare_sft_magrpo import test_agent_with_checkpoint

def evaluate_on_dataset(agent_name: str, dataset_path: str, checkpoint_type: str, epoch: int = None):
    """
    Évalue un agent sur un dataset de test.
    """
    # Charger le dataset
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = [json.loads(line) for line in f]
    
    metrics = {
        "total": len(dataset),
        "success": 0,
        "json_valid": 0,
        "keys_correct": 0,
        "avg_time": 0.0
    }
    
    print(f"\n📊 Évaluation {agent_name} ({checkpoint_type})")
    print(f"Dataset: {len(dataset)} échantillons\n")
    
    for i, sample in enumerate(dataset):
        query = sample.get("instruction", sample.get("query", ""))
        
        result = test_agent_with_checkpoint(agent_name, query, checkpoint_type, epoch)
        
        if result["success"]:
            metrics["success"] += 1
            if result["is_json"]:
                metrics["json_valid"] += 1
            if result.get("has_expected_keys", False):
                metrics["keys_correct"] += 1
            metrics["avg_time"] += result["time"]
        
        if (i + 1) % 10 == 0:
            print(f"  Progression: {i+1}/{len(dataset)}")
    
    # Calculer les moyennes
    metrics["success_rate"] = metrics["success"] / metrics["total"]
    metrics["json_rate"] = metrics["json_valid"] / metrics["total"]
    metrics["keys_rate"] = metrics["keys_correct"] / metrics["total"]
    metrics["avg_time"] = metrics["avg_time"] / metrics["total"]
    
    return metrics

if __name__ == "__main__":
    # Évaluer sur les datasets de test
    datasets = {
        "orchestrator": "data/test/orchestrator_test.jsonl",
        "researcher": "data/test/researcher_test.jsonl",
        "code_writer": "data/test/code_writer_test.jsonl",
        "critic": "data/test/critic_test.jsonl"
    }
    
    for agent_name, dataset_path in datasets.items():
        if not os.path.exists(dataset_path):
            print(f"⚠️  Dataset non trouvé: {dataset_path}")
            continue
        
        # Évaluer SFT
        sft_metrics = evaluate_on_dataset(agent_name, dataset_path, "sft")
        
        # Évaluer MAGRPO
        magrpo_metrics = evaluate_on_dataset(agent_name, dataset_path, "magrpo", epoch=20)
        
        # Comparer
        print(f"\n{'='*70}")
        print(f"📊 Comparaison {agent_name.upper()}")
        print(f"{'='*70}\n")
        print(f"{'Métrique':<20} {'SFT':<15} {'MAGRPO':<15} {'Amélioration':<15}")
        print(f"{'-'*70}")
        print(f"{'Taux de succès':<20} {sft_metrics['success_rate']:.2%} {'':<10} {magrpo_metrics['success_rate']:.2%} {'':<10} {(magrpo_metrics['success_rate'] - sft_metrics['success_rate']):.2%}")
        print(f"{'JSON valide':<20} {sft_metrics['json_rate']:.2%} {'':<10} {magrpo_metrics['json_rate']:.2%} {'':<10} {(magrpo_metrics['json_rate'] - sft_metrics['json_rate']):.2%}")
        print(f"{'Clés correctes':<20} {sft_metrics['keys_rate']:.2%} {'':<10} {magrpo_metrics['keys_rate']:.2%} {'':<10} {(magrpo_metrics['keys_rate'] - sft_metrics['keys_rate']):.2%}")
        print(f"{'Temps moyen':<20} {sft_metrics['avg_time']:.2f}s {'':<10} {magrpo_metrics['avg_time']:.2f}s {'':<10}")
        print()
```

---

## 🎯 Checklist Complète

- [ ] ✅ Checkpoints MAGRPO téléchargés depuis Kaggle
- [ ] ✅ Structure des dossiers vérifiée
- [ ] ✅ Scripts de test créés
- [ ] ✅ Agents individuels testés
- [ ] ✅ Système multi-agent testé
- [ ] ✅ Comparaison SFT vs MAGRPO effectuée
- [ ] ✅ Évaluation quantitative réalisée
- [ ] ✅ Meilleur checkpoint identifié (epoch 10, 15, ou 20)

---

## 🚀 Prochaines Étapes Recommandées

1. **Identifier le meilleur checkpoint** : Comparez epoch 10, 15, 20
2. **Déployer en production** : Utilisez le meilleur checkpoint pour l'inférence
3. **Continuer l'entraînement** : Si nécessaire, entraînez plus d'époques
4. **Fine-tuning** : Ajustez les hyperparamètres si les performances ne sont pas optimales

---

## 📝 Notes Importantes

- **Nom des checkpoints** : Vérifiez si c'est `orchestratc` ou `orchestrator` dans vos fichiers
- **Compatibilité** : Les checkpoints MAGRPO sont compatibles avec la même structure que SFT
- **Performance** : Les agents MAGRPO devraient avoir de meilleures performances de collaboration
- **Mémoire** : Assurez-vous d'avoir assez de mémoire GPU/CPU pour charger tous les agents

