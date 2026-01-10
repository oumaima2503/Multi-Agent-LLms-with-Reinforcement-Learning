import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.base_agents import OrchestratorAgent, ResearcherAgent, CodeWriterAgent, CriticAgent
import json

def validate_python_code(code: str) -> bool:
    """Valide que le code Python est syntaxiquement correct"""
    if not code or len(code.strip()) < 5:
        return False
    try:
        compile(code, '<string>', 'exec')
        return True
    except SyntaxError:
        return False
    except Exception:
        return False

def is_code_satisfactory(code: str) -> bool:
    """Vérifie si le code semble satisfaisant"""
    if not code:
        return False
    
    # Vérifier syntaxe
    if not validate_python_code(code):
        return False
    
    # Vérifier qu'il contient des éléments de base Python
    code_lower = code.lower()
    python_keywords = ['def ', 'return ', 'max(', 'min(', 'import ', 'print(']
    if not any(kw in code_lower for kw in python_keywords):
        return False
    
    # Vérifier qu'il n'y a pas de code manifestement incorrect
    bad_patterns = ['get_min', 'getmax', 'stupid_array', '{%-', '<!--']
    if any(pattern in code_lower for pattern in bad_patterns):
        return False
    
    return True

class MAGRPOMultiAgentSystem:
 
    def __init__(self, epoch: int = 20, fast_mode: bool = False, offline: bool = False):
        self.epoch = epoch
        self.agents = {}
        self.history = []
        self.current_agent = "orchestrator"
        self.original_query = ""  
        self.fast_mode = fast_mode 
        self.offline = offline
        
        if self.offline:
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HUGGINGFACE_HUB_OFFLINE"] = "1"
        
        self._load_agents()
    
    def _load_agents(self):
       
        print("    Chargement des agents MAGRPO...")

        agent_configs = {
            "orchestrator": (OrchestratorAgent, "orchestrator"),
            "researcher": (ResearcherAgent, "researcher"),
            "code_writer": (CodeWriterAgent, "code_writer"),
            "critic": (CriticAgent, "critic")
        }

        for name, (agent_class, checkpoint_name) in agent_configs.items():
            print(f"   Chargement de {name}...")
            # créer instance de base (toujours)
            agent = agent_class()
            magrpo_path = f"checkpoints/magrpo_rl/epoch{self.epoch}_{checkpoint_name}_rl"
            sft_path = f"checkpoints/{checkpoint_name}_lora"

            # par défaut pointer vers MAGRPO
            agent.lora_path = magrpo_path

            try:
                # MAGRPO présent ?
                if os.path.exists(agent.lora_path):
                    try:
                        agent._load_model()
                        print(f"     {name} chargé (MAGRPO)")
                    except Exception as e:
                        print(f"       Erreur chargement MAGRPO pour {name}: {e}")
                        # tenter fallback SFT si disponible
                        if os.path.exists(sft_path):
                            agent.lora_path = sft_path
                            try:
                                agent._load_model()
                                print(f"     {name} chargé (SFT fallback)")
                            except Exception as e2:
                                print(f"       Erreur chargement SFT pour {name}: {e2}")
                        else:
                            print(f"       Aucun checkpoint SFT local trouvé pour {name}, utilisation instance par défaut")
                else:
                    # MAGRPO absent → essayer SFT fallback
                    if os.path.exists(sft_path):
                        agent.lora_path = sft_path
                        print(f"       Checkpoint MAGRPO non trouvé, utilisation du SFT: {sft_path}")
                        try:
                            agent._load_model()
                            print(f"     {name} chargé (SFT)")
                        except Exception as e:
                            print(f"       Erreur chargement SFT pour {name}: {e}")
                            print(f"   → {name} instancié sans weights")
                    else:
                        print(f"       Checkpoint MAGRPO non trouvé: {magrpo_path}")
                        print(f"       Aucun checkpoint SFT local trouvé: {sft_path}")
                        print(f"   → {name} instancié sans weights (offline fallback)")

                # ajouter l'agent à la map quel que soit l'état du chargement
                self.agents[name] = agent

            except Exception as e:
                print(f"        Erreur inattendue lors du chargement de {name}: {e}")
                agent = agent_class()
                self.agents[name] = agent
                print(f"   → Utilisation de l'instance par défaut pour {name}")

        print("  Agents chargés\n")
    
    def reset(self, initial_query: str):
        """Réinitialise le système avec une nouvelle requête"""
        self.history = [f"User: {initial_query}"]
        self.current_agent = "orchestrator"
        # Affecter la requête originale (fixe: était manquante)
        self.original_query = initial_query
        print(f"\n     Nouvelle session: {initial_query}\n")
    
    def step(self):
        """Exécute une étape du workflow multi-agent"""
        if self.current_agent not in self.agents:
            print(f"     Agent inconnu: {self.current_agent}")
            return None, True
        
        agent = self.agents[self.current_agent]
        
        # Construire l'état selon le type d'agent
        if self.current_agent == "orchestrator":
            # L'orchestrator voit l'historique mais limité pour éviter le dépassement
            # Garder seulement les 3 dernières entrées + la requête originale
            recent_history = self.history[-3:] if len(self.history) > 3 else self.history
            current_state = "\n".join([self.history[0]] + recent_history)  # Requête originale + 3 dernières entrées
        else:
            # Les agents exécuteurs voient l'instruction la plus récente de l'orchestrator
            # Chercher la dernière instruction de l'orchestrator
            instruction = None
            for entry in reversed(self.history):
                if "[ORCHESTRATOR_INSTRUCTION]:" in entry:
                    instruction = entry.split(":", 1)[1].strip()
                    break
            
            # Si pas d'instruction trouvée ou instruction vide/invalide, utiliser la requête originale
            if not instruction or len(instruction) < 5:
                instruction = self.original_query
            
            # Pour CodeWriter, améliorer l'instruction si elle semble incorrecte
            if self.current_agent == "code_writer":
                # Si l'instruction ne contient pas de mots-clés de code, utiliser la requête originale
                code_keywords = ['code', 'script', 'function', 'program', 'python', 'calculer', 'max', 'min', 'liste']
                if not any(keyword in instruction.lower() for keyword in code_keywords):
                    print(f"    Instruction semble incorrecte, utilisation de la requête originale")
                    instruction = self.original_query
            
            current_state = instruction
        
        print(f" Agent actif: {self.current_agent.upper()}")
        print(f" État actuel: {current_state[:100]}...\n")
        
        # Agent génère une action
        try:
            if self.fast_mode:
                print(f" Génération en cours (mode rapide)...")
            else:
                print(f" Génération en cours... (cela peut prendre 10-30 secondes avec TinyLlama sur CPU)")
            
            # Génération avec gestion du timeout
            result = agent.act(current_state, fast_mode=self.fast_mode)
            
            if result is None:
                print(f"     {self.current_agent.upper()} a retourné None")
                return None, True
            
            print(f"  Réponse de {self.current_agent}:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            
            # Ajouter à l'historique
            response_text = json.dumps(result, ensure_ascii=False)
            self.history.append(f"[{self.current_agent.upper()}]: {response_text}")
            
            # Déterminer le prochain agent (logique simplifiée)
            if self.current_agent == "orchestrator":
                # Vérifier si c'est une erreur
                if isinstance(result, dict) and result.get("action_type") == "ERROR":
                    error_msg = result.get("error_message", "")
                    raw_output = result.get("raw_output", "")
                    
                    # Si l'erreur est due à un JSON invalide, essayer d'extraire l'information
                    if "JSON" in error_msg and raw_output:
                        print(f"    Orchestrator a généré du texte libre au lieu de JSON")
                        print(f" Tentative d'extraction d'information...")
                        
                        # Détecter automatiquement l'agent basé sur la requête originale
                        obs_lower = self.original_query.lower()
                        if any(word in obs_lower for word in ['code', 'script', 'calculer', 'programme', 'python', 'fonction', 'max', 'min', 'liste', 'tri', 'array']):
                            delegated_agent = "CodeWriter"
                        elif any(word in obs_lower for word in ['cherche', 'trouve', 'recherche', 'information', 'date', 'prix']):
                            delegated_agent = "Researcher"
                        elif any(word in obs_lower for word in ['évalue', 'critique', 'vérifie', 'valide']):
                            delegated_agent = "Critic"
                        else:
                            delegated_agent = "CodeWriter"  # Par défaut
                        
                        # Utiliser la requête originale comme instruction
                        instruction = self.original_query
                        
                        # Transition directe
                        agent_mapping = {
                            "CodeWriter": "code_writer",
                            "Researcher": "researcher",
                            "Critic": "critic"
                        }
                        next_agent_key = agent_mapping.get(delegated_agent, "code_writer")
                        
                        if next_agent_key in self.agents:
                            self.current_agent = next_agent_key
                            print(f"\n  Transition automatique vers: {next_agent_key.upper()} (détection intelligente)")
                            print(f"   Instruction: {instruction[:100]}...")
                            self.history.append(f"[ORCHESTRATOR_INSTRUCTION]: {instruction}")
                            return result, False
                    
                    # Si trop d'erreurs consécutives, terminer
                    error_count = sum(1 for entry in self.history if "ERROR" in entry or "action_type" in entry)
                    if error_count >= 3:
                        print(f"\n    Trop d'erreurs consécutives ({error_count}), arrêt du workflow")
                        return result, True
                    
                    # Sinon, continuer
                    self.current_agent = "orchestrator"
                    return result, False
                
                # L'orchestrator décide du prochain agent
                if isinstance(result, dict):
                    delegated = result.get("delegated_agent", "").strip()
                    
                    # Nettoyer la valeur (enlever les listes, guillemets, etc.)
                    if isinstance(delegated, list) and delegated:
                        delegated = str(delegated[0])
                    elif not isinstance(delegated, str):
                        delegated = str(delegated)
                    
                    # Nettoyer les caractères indésirables
                    delegated = delegated.strip('[]"\'').strip()
                    
                    # Mapping des noms d'agents (CodeWriter → code_writer, etc.)
                    agent_mapping = {
                        "codewriter": "code_writer",
                        "code_writer": "code_writer",
                        "codeworker": "code_writer",  # Variante
                        "researcher": "researcher",
                        "critic": "critic",
                        "finished": "finished",
                        "end": "finished"
                    }
                    
                    next_agent_key = agent_mapping.get(delegated.lower(), None)
                    
                    if next_agent_key == "finished":
                        # Workflow terminé
                        final_answer = result.get("final_answer", "")
                        if final_answer:
                            print(f"\n     Workflow terminé")
                            print(f"   Réponse finale: {final_answer}")
                        else:
                            print(f"\n     Workflow terminé (pas de réponse finale)")
                        return result, True
                    elif next_agent_key and next_agent_key in self.agents:
                        # Vérifier si on a déjà un résultat satisfaisant de cet agent
                        has_valid_result = any("[CODE_VALID]: True" in entry for entry in self.history)
                        if has_valid_result and next_agent_key == "code_writer":
                            # On a déjà un code valide, terminer le workflow
                            print(f"\n   Code valide déjà obtenu, workflow terminé")
                            # Récupérer le dernier résultat valide
                            last_result = None
                            for entry in reversed(self.history):
                                if "[CODE_WRITER_RESULT]" in entry:
                                    try:
                                        result_text = entry.split(":", 1)[1].strip()
                                        last_result = json.loads(result_text)
                                        break
                                    except:
                                        pass
                            
                            if last_result:
                                print(f"\n     Workflow terminé (code Python valide obtenu)")
                                print(f"     Code final:")
                                code = last_result.get("python_code", "")
                                print(f"```python\n{code[:300]}...\n```")
                                return last_result, True
                        
                        # Transition vers l'agent délégué
                        self.current_agent = next_agent_key
                        instruction = result.get("instruction", "").strip()
                        
                        # Nettoyer l'instruction (enlever "User: " au début si présent)
                        if instruction.startswith("User: "):
                            instruction = instruction[6:].strip()
                        
                        # Si l'instruction est vide ou semble incorrecte, utiliser la requête originale
                        if not instruction or len(instruction) < 5:
                            instruction = self.original_query
                            print(f"\n    Instruction vide, utilisation de la requête originale")
                        
                        # Pour CodeWriter, vérifier que l'instruction contient des mots-clés de code
                        if next_agent_key == "code_writer":
                            code_keywords = ['code', 'script', 'function', 'program', 'python', 'calculer', 'max', 'min', 'liste', 'array', 'tri', 'sort']
                            if not any(keyword in instruction.lower() for keyword in code_keywords):
                                print(f"    Instruction ne semble pas être une tâche de code, utilisation de la requête originale")
                                instruction = self.original_query
                        
                        print(f"\n   Transition vers: {next_agent_key.upper()}")
                        if instruction:
                            print(f"     Instruction: {instruction[:100]}...")
                        # Ajouter l'instruction à l'historique pour l'agent suivant
                        self.history.append(f"[ORCHESTRATOR_INSTRUCTION]: {instruction}")
                    else:
                        # Agent inconnu ou erreur, utiliser la détection intelligente
                        print(f"\n    Agent délégué non reconnu: {delegated}")
                        print(f"     Détection intelligente basée sur la requête...")
                        
                        # Détection automatique
                        obs_lower = self.original_query.lower()
                        if any(word in obs_lower for word in ['code', 'script', 'calculer', 'programme', 'python', 'fonction', 'max', 'min', 'liste', 'tri', 'array']):
                            next_agent_key = "code_writer"
                            instruction = self.original_query
                            self.current_agent = next_agent_key
                            print(f"      Transition automatique vers: {next_agent_key.upper()}")
                            print(f"     Instruction: {instruction[:100]}...")
                            self.history.append(f"[ORCHESTRATOR_INSTRUCTION]: {instruction}")
                        else:
                            print(f"      Retour à l'orchestrator")
                            self.current_agent = "orchestrator"
                else:
                    self.current_agent = "orchestrator"
            else:
                # Les autres agents retournent à l'orchestrator avec leur résultat
                # Construire le contexte pour l'orchestrator
                agent_result = json.dumps(result, ensure_ascii=False)
                self.history.append(f"[{self.current_agent.upper()}_RESULT]: {agent_result}")
                self.current_agent = "orchestrator"
                print(f"\n      Retour à l'orchestrator avec le résultat")
            
            return result, False
            
        except KeyboardInterrupt:
            print(f"\n    Interruption par l'utilisateur")
            return None, True
        except Exception as e:
            print(f"     Erreur lors de l'exécution de {self.current_agent.upper()}: {e}")
            import traceback
            traceback.print_exc()
            # En cas d'erreur, retourner à l'orchestrator pour continuer
            if self.current_agent != "orchestrator":
                self.current_agent = "orchestrator"
                self.history.append(f"[ERROR]: {self.current_agent.upper()} a rencontré une erreur: {str(e)}")
                print(f"      Retour à l'orchestrator après erreur")
                return None, False
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
                print(f"\n  Workflow terminé après {turn + 1} tours")
                return result
            
            # Détecter les boucles infinies (même agent plusieurs fois de suite)
            if turn >= 3:
                recent_agents = []
                for entry in self.history[-8:]:
                    if "[" in entry and "]" in entry:
                        agent_name = entry.split("]")[0].replace("[", "").strip()
                        if agent_name:
                            recent_agents.append(agent_name)
                
                if len(recent_agents) >= 4 and len(set(recent_agents)) == 1:
                    print(f"\n    Boucle détectée (agent {recent_agents[0]} répété), arrêt du workflow")
                    return result
            
            # Détecter trop d'erreurs
            error_count = sum(1 for entry in self.history if "ERROR" in entry or "action_type" in entry)
            if error_count >= 5:
                print(f"\n    Trop d'erreurs ({error_count}), arrêt du workflow")
                return result
                
            if turn >= max_turns - 1:
                print(f"\n    Nombre maximum de tours atteint")
                return result
        
        return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Système Multi-Agent MAGRPO")
    parser.add_argument("--epoch", type=int, default=20, help="Époque du checkpoint (10, 15, 20)")
    parser.add_argument("--query", type=str, help="Requête initiale")
    parser.add_argument("--max-turns", type=int, default=10, help="Nombre maximum de tours")
    parser.add_argument("--interactive", action="store_true", help="Mode interactif")
    parser.add_argument("--fast", action="store_true", help="Mode rapide (génération plus rapide mais moins précise)")
    parser.add_argument("--offline", action="store_true", help="Mode hors-ligne (empêche téléchargements réseau HuggingFace)")
    
    args = parser.parse_args()
    
    # Créer le système
    system = MAGRPOMultiAgentSystem(epoch=args.epoch, fast_mode=args.fast, offline=args.offline)
    
    if args.interactive:
        # Mode interactif
        print("="*60)
        print("   Système Multi-Agent MAGRPO - Mode Interactif")
        print("="*60)
        print("\nTapez 'quit' pour quitter\n")
        
        while True:
            query = input("Vous: ")
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("     Au revoir!")
                break
            
            if not query.strip():
                continue
            
            result = system.run(query, max_turns=args.max_turns)
            print("\n" + "="*60 + "\n")
    elif args.query:
        # Mode avec requête spécifiée
        result = system.run(args.query, max_turns=args.max_turns)
    else:
        # Exemples par défaut
        print("="*60)
        print("     Système Multi-Agent MAGRPO")
        print("="*60)
        
        # Exemple 1
        print("\n     Exemple 1: Analyse comparative")
        result1 = system.run("Compare le Pixel 8 et l'iPhone 15", max_turns=5)
        
        # Exemple 2
        print("\n\n     Exemple 2: Recherche et code")
        result2 = system.run("Trouve la date de sortie du Pixel 8 et crée un script pour calculer son prix avec remise", max_turns=6)

