import json
import torch
import os
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig


MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_BASE = "checkpoints/" 


# CLASSE DE BASE

class BaseAgent:
 
    def __init__(self, name: str, system_prompt: str, lora_folder: str):
        self.name = name
        self.system_prompt = system_prompt
       
        self.lora_path = os.path.join(CHECKPOINT_BASE, lora_folder) 
        self.tokenizer = None
        self.model = None
        
        self._load_model()

    def _load_model(self):
        
        print(f"\n--- Chargement de l'agent {self.name} ---")
        print(f"Modèle de base: {MODEL_NAME}")
        print(f"Device: {DEVICE}")
        
        if not os.path.isdir(self.lora_path):
            print(f"     AVERTISSEMENT: Dossier LoRA non trouvé : {self.lora_path}")
            print("Tentative de chargement du modèle de base uniquement (vérifiez le chemin SFT).")
            # Fallback : Charge le modèle de base si les poids LoRA sont manquants
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
                device_map="auto" if DEVICE == "cuda" else None,
            )
            if DEVICE == "cpu":
                self.model = self.model.to(DEVICE)
        else:
            adapter_config_path = os.path.join(self.lora_path, "adapter_config.json")
            if not os.path.exists(adapter_config_path):
                raise FileNotFoundError(
                    f"Fichier adapter_config.json non trouvé dans {self.lora_path}. "
                    f"Vérifiez que le checkpoint LoRA est correctement sauvegardé."
                )
            
            print(f"Application des poids LoRA depuis : {self.lora_path}")
            # 1. Chargement du modèle de base
            try:
                base_model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME,
                    dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
                    device_map="auto" if DEVICE == "cuda" else None,
                )
                if DEVICE == "cpu":
                    base_model = base_model.to(DEVICE)
                
                
                with open(adapter_config_path, 'r') as f:
                    config_dict = json.load(f)
                
                
                essential_params = {
                    'r': config_dict.get('r', 64),
                    'lora_alpha': config_dict.get('lora_alpha', 16),
                    'target_modules': config_dict.get('target_modules', ['q_proj', 'v_proj']),
                    'lora_dropout': config_dict.get('lora_dropout', 0.1),
                    'bias': config_dict.get('bias', 'none'),
                    'task_type': config_dict.get('task_type', 'CAUSAL_LM'),
                    'inference_mode': config_dict.get('inference_mode', True)
                }
                
                lora_config = LoraConfig(**essential_params)
                
                import tempfile
                import shutil
                with tempfile.TemporaryDirectory() as temp_dir:
                   
                    cleaned_config = {k: v for k, v in essential_params.items()}
                    cleaned_config['peft_type'] = 'LORA'
                    if 'base_model_name_or_path' in config_dict:
                        cleaned_config['base_model_name_or_path'] = config_dict['base_model_name_or_path']
                    
                    temp_config_path = os.path.join(temp_dir, "adapter_config.json")
                    with open(temp_config_path, 'w') as f:
                        json.dump(cleaned_config, f, indent=2)
                    
                    
                    adapter_model_path = os.path.join(self.lora_path, "adapter_model.safetensors")
                    if not os.path.exists(adapter_model_path):
                        adapter_model_path = os.path.join(self.lora_path, "adapter_model.bin")
                    
                    if os.path.exists(adapter_model_path):
                        print(f"   Chargement des poids depuis {os.path.basename(adapter_model_path)}")
                        shutil.copy2(adapter_model_path, os.path.join(temp_dir, os.path.basename(adapter_model_path)))
                    else:
                        raise FileNotFoundError(
                            f"Aucun fichier adapter_model trouvé dans {self.lora_path}. "
                            f"Recherché: adapter_model.safetensors et adapter_model.bin"
                        )
                    
                    try:
                        peft_model = PeftModel.from_pretrained(base_model, temp_dir)
                        print(f"        Poids LoRA chargés avec succès")
                        
                        self.model = peft_model.merge_and_unload()
                    except Exception as load_error:
                        
                        error_str = str(load_error).lower()
                        if "alora_invocation_tokens" in error_str or "unexpected keyword" in error_str:
                            print(f"        Tentative alternative de chargement (incompatibilité de version peft)")
                            
                            from peft import get_peft_model
                            peft_model_alt = get_peft_model(base_model, lora_config)
                            
                            if adapter_model_path.endswith('.safetensors'):
                                from safetensors.torch import load_file as safe_load_file
                                adapter_weights = safe_load_file(adapter_model_path)
                            else:
                                adapter_weights = torch.load(adapter_model_path, map_location=DEVICE)
                            
                            peft_model_alt.load_state_dict(adapter_weights, strict=False)
                            self.model = peft_model_alt.merge_and_unload()
                        else:
                            raise
            except Exception as e:
                error_msg = str(e)
                if "gated" in error_msg.lower() or "403" in error_msg.lower():
                    raise RuntimeError(
                        f"Erreur d'accès au modèle gated. Assurez-vous que:\n"
                        f"1. Vous avez accepté la licence sur Hugging Face\n"
                        f"2. Vous avez exporté HF_TOKEN dans votre environnement\n"
                        f"3. Le token a accès au modèle {MODEL_NAME}\n"
                        f"Erreur originale: {error_msg}"
                    ) from e
                else:
                    raise RuntimeError(
                        f"Erreur lors du chargement du modèle LoRA pour {self.name}: {error_msg}"
                    ) from e

        # 4. Chargement du Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        print(f"     Modèle {self.name} prêt.")


    def generate_response(self, user_prompt: str, max_retries: int = 2, fast_mode: bool = False) -> str:
       
        enhanced_user_prompt = (
            f"{user_prompt}\n\n"
            "IMPORTANT: Your response MUST be a valid JSON object starting with {{ and ending with }}. "
            "Do NOT include any text before or after the JSON. Start directly with {{."
        )
        full_prompt = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": enhanced_user_prompt}
        ]
        
        
        try:
            input_ids = self.tokenizer.apply_chat_template(
                full_prompt, 
                tokenize=True, 
                return_tensors="pt",
                add_generation_prompt=True
            ).to(DEVICE)
        except Exception as e:
            
            if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
                input_ids = self.tokenizer.apply_chat_template(
                    full_prompt, 
                    tokenize=True, 
                    return_tensors="pt"
                ).to(DEVICE)
            else:
                
                messages = ""
                for msg in full_prompt:
                    role = msg["role"]
                    content = msg["content"]
                    if role == "system":
                        messages += f"<|system|>\n{content}\n"
                    elif role == "user":
                        messages += f"<|user|>\n{content}\n"
                messages += "<|assistant|>\n"
                input_ids = self.tokenizer(messages, return_tensors="pt").input_ids.to(DEVICE)

       
        if fast_mode:
            max_new_tokens = 256  
            max_retries = 0  
            do_sample = False  
            temperature = 0.1
            top_p = 0.7
        else:
            max_new_tokens = 512
            do_sample = True
            temperature = 0.3
            top_p = 0.85
        
        
        for attempt in range(max_retries + 1):
            try:
                with torch.no_grad():
                    output = self.model.generate(
                        input_ids,
                        max_new_tokens=max_new_tokens,
                        do_sample=do_sample if attempt == 0 else False, 
                        temperature=temperature if attempt == 0 else 0.1,
                        top_p=top_p if attempt == 0 else 0.7,  
                        top_k=50 if not fast_mode else 20, 
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        repetition_penalty=1.2 if fast_mode else 1.3,  
                        no_repeat_ngram_size=2 if fast_mode else 3,  
                    )
            except Exception as e:
                print(f"      Erreur lors de la génération (tentative {attempt + 1}): {e}")
                if attempt == max_retries:
                  
                    return '{"error": "Génération échouée après plusieurs tentatives"}'
                continue

           
            response_text = self.tokenizer.decode(output[0, input_ids.shape[1]:], skip_special_tokens=True)
            response_text = response_text.strip()
            
            
            if response_text.startswith('{'):
                return response_text
            
            if attempt < max_retries:
                continue
        
        return response_text

    def _clean_and_parse_json(self, raw_response: str) -> dict:
   
        cleaned = raw_response.strip()
        
        if cleaned.startswith('{'):
           
            brace_count = 0
            end_pos = -1
            in_string = False
            escape_next = False
            
            for i, char in enumerate(cleaned):
                if escape_next:
                    escape_next = False
                    continue
                    
                if char == '\\':
                    escape_next = True
                    continue
                    
                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue
                    
                if not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_pos = i + 1
                            break
            
            if end_pos > 0:
                json_str = cleaned[:end_pos]
                try:
                    action = json.loads(json_str)
                    if isinstance(action, dict):
                        return action
                except json.JSONDecodeError as e:
                    
                    json_str = self._try_fix_json(json_str)
                    if json_str:
                        try:
                            action = json.loads(json_str)
                            if isinstance(action, dict):
                                return action
                        except json.JSONDecodeError:
                            pass
        
        
        first_brace = cleaned.find('{')
        if first_brace >= 0:
           
            brace_count = 0
            end_pos = -1
            in_string = False
            escape_next = False
            
            for i in range(first_brace, len(cleaned)):
                char = cleaned[i]
                if escape_next:
                    escape_next = False
                    continue
                    
                if char == '\\':
                    escape_next = True
                    continue
                    
                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue
                    
                if not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_pos = i + 1
                            break
            
            if end_pos > first_brace:
                json_str = cleaned[first_brace:end_pos]
                try:
                    action = json.loads(json_str)
                    if isinstance(action, dict):
                        return action
                except json.JSONDecodeError:
                    
                    json_str = self._try_fix_json(json_str)
                    if json_str:
                        try:
                            action = json.loads(json_str)
                            if isinstance(action, dict):
                                return action
                        except json.JSONDecodeError:
                            pass
        
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = list(re.finditer(json_pattern, cleaned, re.DOTALL))
        
        for match in sorted(matches, key=lambda m: len(m.group(0)), reverse=True):
            json_str = match.group(0)
            try:
                action = json.loads(json_str)
                if isinstance(action, dict):
                    return action
            except json.JSONDecodeError:
                json_str = self._try_fix_json(json_str)
                if json_str:
                    try:
                        action = json.loads(json_str)
                        if isinstance(action, dict):
                            return action
                    except json.JSONDecodeError:
                        continue
        
        
        json_patterns = [
            r'"(\w+)":\s*"([^"\\]*(?:\\.[^"\\]*)*)"', 
            r'"(\w+)":\s*(\d+\.?\d*)',      
            r'"(\w+)":\s*(true|false|null)',
            r'"(\w+)":\s*\{',          
        ]
        
        found_keys = {}
        for pattern in json_patterns:
            matches = re.finditer(pattern, cleaned, re.IGNORECASE)
            for match in matches:
                key = match.group(1)
                if len(match.groups()) > 1 and match.group(2):
                    value = match.group(2)
                    # Essayer de convertir les valeurs
                    if value.lower() in ['true', 'false']:
                        found_keys[key] = value.lower() == 'true'
                    elif value.lower() == 'null':
                        found_keys[key] = None
                    elif value.replace('.', '', 1).isdigit():
                        found_keys[key] = float(value) if '.' in value else int(value)
                    else:
                        # Décoder les échappements
                        found_keys[key] = value.encode().decode('unicode_escape')
        
        if found_keys:
            
            return found_keys
        
       
        raise json.JSONDecodeError(
            "Aucun bloc JSON ({...}) trouvé dans la réponse.", 
            raw_response, 
            0
        )
    
    def _try_fix_json(self, json_str: str) -> str:
        
        fixed = json_str.strip()
        
        
        if "'" in fixed and '"' not in fixed:
           
            fixed = re.sub(r"'(\w+)':", r'"\1":', fixed)  
            fixed = re.sub(r":\s*'([^']*)'", r': "\1"', fixed)  
        
       
        fixed = re.sub(r',\s*}', '}', fixed)
        fixed = re.sub(r',\s*]', ']', fixed)
        
       
        quote_count = len(re.findall(r'(?<!\\)"', fixed))
        if quote_count % 2 != 0:
           
            if not fixed.rstrip().endswith('"'):
                fixed = fixed.rstrip() + '"'
        
       
        open_braces = fixed.count('{')
        close_braces = fixed.count('}')
        if open_braces > close_braces:
            fixed += '}' * (open_braces - close_braces)
        
        return fixed
    
    def _camel_to_snake(self, name: str) -> str:
        
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
       
        s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1)
        return s2.lower()
    
    def _normalize_boolean(self, value):
        
        if isinstance(value, str):
            value_lower = value.lower().strip()
            if value_lower in ['true', '1', 'yes', 'on']:
                return True
            elif value_lower in ['false', '0', 'no', 'off', '']:
                return False
           
            if value_lower == 'true/false':
                return True  # Par défaut
        return value
    
    def _normalize_json_keys(self, action: dict, key_mapping: dict = None) -> dict:
        normalized = {}
        
        # Mapping de fautes de frappe communes (typos)
        common_typos = {
            'researh_query': 'research_query',
            'research_query': 'research_query',
            'criteria_ok': 'critique_ok',
            'critique_ok': 'critique_ok',
            'critic_ok': 'critique_ok',
            'pythoncode': 'python_code',
            'python_code': 'python_code',
            'resultexplanation': 'result_explanation',
            'result_explanation': 'result_explanation',
        }
        
        for key, value in action.items():
            # Ignorer les clés d'erreur
            if key in ["action_type", "error_message", "raw_output"]:
                normalized[key] = value
                continue
            
            # Normaliser la clé
            normalized_key = key
            
            # 1. Vérifier les fautes de frappe communes
            key_lower = key.lower()
            if key_lower in common_typos:
                normalized_key = common_typos[key_lower]
            # 2. Utiliser le mapping si fourni
            elif key_mapping:
                # Chercher dans le mapping
                for variant, expected in key_mapping.items():
                    if key_lower == variant.lower() or key_lower == expected.lower():
                        normalized_key = expected
                        break
                
              
                if normalized_key == key:
                    normalized_key = self._camel_to_snake(key)
                    # Vérifier si la version normalisée est dans le mapping
                    if normalized_key in key_mapping.values():
                        pass  
                    else:
                       
                        for variant, expected in key_mapping.items():
                            
                            if (normalized_key.startswith(expected[:3]) or 
                                expected.startswith(normalized_key[:3]) or
                                normalized_key in expected or expected in normalized_key):
                                normalized_key = expected
                                break
            else:
                
                normalized_key = self._camel_to_snake(key)
              
                if normalized_key.lower() in common_typos:
                    normalized_key = common_typos[normalized_key.lower()]
            
           
            if 'ok' in normalized_key.lower() or 'critique' in normalized_key.lower():
                value = self._normalize_boolean(value)
            
            normalized[normalized_key] = value
        
        return normalized

    def act(self, observation: str, fast_mode: bool = False) -> dict:
      
        raw_response = self.generate_response(observation, max_retries=0 if fast_mode else 2, fast_mode=fast_mode)
        
        try:
            action = self._clean_and_parse_json(raw_response)
           
            return action
        except (json.JSONDecodeError, ValueError) as e:
           
            return {
                "action_type": "ERROR", 
                "error_message": f"Erreur de format JSON/Validation: {e}", 
                "raw_output": raw_response
            }


# CLASSES DES AGENTS


class OrchestratorAgent(BaseAgent):
    def __init__(self):
        # Format JSON attendu: {'delegated_agent': 'AGENT_NAME', 'instruction': 'TASK_INSTRUCTION', 'final_answer': 'OPTIONAL_FINAL_ANSWER'}
        orchestrator_system_prompt = (
            "You are the Orchestrator Agent. Your role is to plan, manage the workflow, and delegate tasks "
            "to Executor Agents (Researcher, CodeWriter, Critic).\n\n"
            "DECISION RULES:\n"
            "- If the task requires CODE, SCRIPT, or PROGRAMMING → delegate to CodeWriter\n"
            "- If the task requires RESEARCH, INFORMATION, or DATA → delegate to Researcher\n"
            "- If the task requires EVALUATION, CRITIQUE, or VALIDATION → delegate to Critic\n"
            "- If the task is COMPLETE → set delegated_agent to 'FINISHED'\n\n"
            "CRITICAL OUTPUT FORMAT: You MUST respond ONLY with a valid JSON object. "
            "Your response MUST start with the character '{' and end with '}'. "
            "Do NOT include any text, explanation, or markdown before or after the JSON.\n\n"
            "Required JSON structure:\n"
            "{\n"
            '  "delegated_agent": "Researcher" | "CodeWriter" | "Critic" | "FINISHED",\n'
            '  "instruction": "task description for the delegated agent",\n'
            '  "final_answer": "final answer if task is complete, otherwise empty string"\n'
            "}\n\n"
            "Examples:\n"
            '{"delegated_agent": "CodeWriter", "instruction": "Write a Python function to calculate the maximum of a list", "final_answer": ""}\n'
            '{"delegated_agent": "Researcher", "instruction": "Find information about Pixel 8", "final_answer": ""}\n'
            '{"delegated_agent": "FINISHED", "instruction": "", "final_answer": "Task completed successfully"}\n\n'
            "Remember: Start your response directly with {, no other text."
        )
        super().__init__(
            name="Orchestrator",
            system_prompt=orchestrator_system_prompt,
            lora_folder="orchestrator_lora"
        )
        
    def act(self, observation: str, fast_mode: bool = False) -> dict:
        action = super().act(observation, fast_mode=fast_mode)
        if action.get("action_type") == "ERROR":
            return action
        
       
        key_mapping = {
            'delegatedagent': 'delegated_agent',
            'delegated_agent': 'delegated_agent',
            'delegation_status': 'delegated_agent',
            'delegation_agency': 'delegated_agent',
            'agent': 'delegated_agent',  # Nouveau
            'target_agent': 'delegated_agent',  # Nouveau
            'next_agent': 'delegated_agent',  # Nouveau
            'instruction': 'instruction',
            'instructions': 'instruction',
            'task': 'instruction',
            'question': 'instruction',
            'command': 'instruction',  # Nouveau
            'finalanswer': 'final_answer',
            'final_answer': 'final_answer',
            'answer': 'final_answer',
            'answers': 'final_answer',
            'response': 'final_answer',  # Nouveau
        }
        
        # Normaliser les clés
        action = self._normalize_json_keys(action, key_mapping)
        
        # Chercher les clés requises (avec variantes)
        delegated_agent = None
        instruction = None
        final_answer = None
        
        # Chercher delegated_agent dans toutes les clés possibles
        for key in ['delegated_agent', 'delegation_status', 'delegation_agency', 'agent', 'target_agent', 'next_agent']:
            if key in action:
                val = action[key]
                if isinstance(val, str):
                    # Normaliser la valeur
                    val_upper = val.upper()
                    if 'RESEARCHER' in val_upper or 'RESEARCH' in val_upper:
                        delegated_agent = 'Researcher'
                    elif 'CODE' in val_upper or 'WRITER' in val_upper or 'CODEWRITER' in val_upper:
                        delegated_agent = 'CodeWriter'
                    elif 'CRITIC' in val_upper:
                        delegated_agent = 'Critic'
                    elif 'FINISH' in val_upper or 'DONE' in val_upper or 'END' in val_upper:
                        delegated_agent = 'FINISHED'
                    else:
                        delegated_agent = val
                else:
                    delegated_agent = str(val)
                break
        
        # Chercher instruction
        for key in ['instruction', 'instructions', 'task', 'question', 'command']:
            if key in action:
                instruction = action[key]
                break
        
        # Chercher final_answer
        for key in ['final_answer', 'answer', 'answers', 'response']:
            if key in action:
                final_answer = action[key]
                if isinstance(final_answer, list) and final_answer:
                    final_answer = str(final_answer[0])
                break
        
        # Si aucune clé trouvée, utiliser une détection intelligente basée sur l'observation
        if not delegated_agent:
            obs_lower = observation.lower()
            # Détection intelligente basée sur le contenu de la requête
            if any(word in obs_lower for word in ['code', 'script', 'calculer', 'programme', 'python', 'fonction', 'max', 'min', 'liste', 'array']):
                delegated_agent = 'CodeWriter'
            elif any(word in obs_lower for word in ['cherche', 'trouve', 'recherche', 'information', 'date', 'prix', 'spécification']):
                delegated_agent = 'Researcher'
            elif any(word in obs_lower for word in ['évalue', 'critique', 'vérifie', 'valide', 'analyse']):
                delegated_agent = 'Critic'
            else:
                # Par défaut pour les requêtes de code, déléguer à CodeWriter
                delegated_agent = 'CodeWriter'
        
        # Si instruction pas trouvée, utiliser l'observation
        if not instruction:
            instruction = observation
        
        # Construire la réponse normalisée
        normalized_action = {
            "delegated_agent": delegated_agent or "",
            "instruction": instruction or observation,
            "final_answer": final_answer or ""
        }
        
        # Validation : si delegated_agent est vide, erreur
        if not normalized_action.get("delegated_agent"):
            normalized_action["action_type"] = "ERROR"
            normalized_action["error_message"] = "Orchestrator action missing required keys (delegated_agent, instruction, final_answer)."
            return normalized_action
        
        return normalized_action

class ResearcherAgent(BaseAgent):
    def __init__(self):
        # Format JSON attendu: {'research_query': 'QUERY_STRING', 'final_answer': 'OPTIONAL_FINAL_ANSWER'}
        researcher_system_prompt = (
            "You are the Researcher Agent. Your role is to find factual information based on the instructions from the Orchestrator.\n\n"
            "CRITICAL OUTPUT FORMAT: You MUST respond ONLY with a valid JSON object. "
            "Your response MUST start with the character '{' and end with '}'. "
            "Do NOT include any text, explanation, or markdown before or after the JSON.\n\n"
            "Required JSON structure:\n"
            "{\n"
            '  "research_query": "specific search query string",\n'
            '  "final_answer": "summary of research findings or empty string"\n'
            "}\n\n"
            "Examples:\n"
            '{"research_query": "Google Pixel 8 release date", "final_answer": ""}\n'
            '{"research_query": "Pixel 8 specifications", "final_answer": "The Pixel 8 was released on October 4, 2023"}\n\n'
            "Remember: Start your response directly with {, no other text."
        )
        super().__init__(
            name="Researcher",
            system_prompt=researcher_system_prompt,
            lora_folder="researcher_lora"
        )
        
    def act(self, observation: str, fast_mode: bool = False) -> dict:
        action = super().act(observation, fast_mode=fast_mode)
        if action.get("action_type") == "ERROR":
            return action

        # Mapping des clés pour normalisation
        key_mapping = {
            'researchquery': 'research_query',
            'research_query': 'research_query',
            'researh_query': 'research_query',  # Gérer faute de frappe commune
            'researchquery': 'research_query',
            'search_term': 'research_query',
            'query': 'research_query',
            'finalanswer': 'final_answer',
            'final_answer': 'final_answer',
            'finalanswer': 'final_answer',
            'answer': 'final_answer',
            'response': 'final_answer',
        }
        
        # Normaliser les clés
        action = self._normalize_json_keys(action, key_mapping)

        # Validation spécifique du Researcher
        if not all(k in action for k in ["research_query", "final_answer"]):
            action["action_type"] = "ERROR"
            action["error_message"] = "Researcher action missing required keys (research_query, final_answer)."
        
        return action

class CodeWriterAgent(BaseAgent):
    def __init__(self):
        # Format JSON attendu: {'python_code': 'CODE_TO_EXECUTE', 'result_explanation': 'OPTIONAL_EXPLANATION'}
        codewriter_system_prompt = (
            "You are the Code Writer Agent. Your role is to write and execute Python code to solve tasks, "
            "but ONLY if the instruction explicitly requires computation or script generation.\n\n"
            "CRITICAL OUTPUT FORMAT: You MUST respond ONLY with a valid JSON object. "
            "Your response MUST start with the character '{' and end with '}'. "
            "Do NOT include any text, explanation, or markdown before or after the JSON.\n\n"
            "Required JSON structure:\n"
            "{\n"
            '  "python_code": "Python code to execute",\n'
            '  "result_explanation": "explanation of the result or empty string"\n'
            "}\n\n"
            "Examples:\n"
            '{"python_code": "result = 899 * 0.15", "result_explanation": "The discount is 134.85"}\n'
            '{"python_code": "def calculate(x, y):\\n    return x + y", "result_explanation": ""}\n\n'
            "Remember: Start your response directly with {, no other text."
        )
        super().__init__(
            name="CodeWriter",
            system_prompt=codewriter_system_prompt,
            lora_folder="code_writer_lora"
        )

    def _extract_python_code_from_text(self, text: str) -> tuple:
        """
        Extrait le code Python d'une réponse texte libre (fallback si pas de JSON).
        Cherche dans les blocs de code markdown, les blocs ```python, ou le texte libre.
        
        Returns:
            (code, explanation) tuple
        """
        import re
        
        # Stratégie 1: Chercher dans les blocs de code markdown ```python ... ```
        python_block_pattern = r'```(?:python|py)?\s*\n(.*?)```'
        matches = re.finditer(python_block_pattern, text, re.DOTALL | re.IGNORECASE)
        code_blocks = [match.group(1).strip() for match in matches]
        
        if code_blocks:
            # Prendre le plus grand bloc de code
            code = max(code_blocks, key=len)
            # L'explication est le texte avant ou après le bloc
            explanation = text.replace(f"```python\n{code}\n```", "").replace(f"```\n{code}\n```", "").strip()
            return code, explanation[:200] if explanation else ""
        
        # Stratégie 2: Chercher des patterns Python (def, import, =, etc.)
        lines = text.split('\n')
        code_lines = []
        in_code_block = False
        
        for line in lines:
            # Détecter le début d'un bloc de code
            if any(keyword in line for keyword in ['def ', 'import ', 'from ', 'class ', 'if __name__', '=']):
                in_code_block = True
                code_lines.append(line)
            elif in_code_block:
                # Continuer jusqu'à une ligne vide ou du texte explicatif
                if line.strip() and not line.strip().startswith(('#', '"', "'")):
                    # Vérifier si c'est du code (contient des opérateurs Python)
                    if any(op in line for op in ['=', '(', ')', '[', ']', '.', ':', 'return', 'print']):
                        code_lines.append(line)
                    else:
                        break
                else:
                    code_lines.append(line)
        
        if code_lines:
            code = '\n'.join(code_lines).strip()
            # Nettoyer les commentaires en début de code
            code = re.sub(r'^#.*?\n', '', code, flags=re.MULTILINE)
            explanation = text.replace(code, "").strip()[:200]
            return code, explanation if explanation else ""
        
        # Stratégie 3: Si on trouve juste du code Python simple (une ligne)
        simple_code_pattern = r'(?:result\s*=|def\s+\w+|print\s*\(|import\s+\w+)[^\n]*(?:\n[^\n]*(?:result|return|print))?'
        match = re.search(simple_code_pattern, text, re.IGNORECASE)
        if match:
            code = match.group(0).strip()
            explanation = text.replace(code, "").strip()[:200]
            return code, explanation if explanation else ""
        
        return None, None

    def act(self, observation: str, fast_mode: bool = False) -> dict:
        # Générer la réponse
        raw_response = self.generate_response(observation, max_retries=0 if fast_mode else 2, fast_mode=fast_mode)
        
        # Essayer de parser en JSON d'abord
        try:
            action = self._clean_and_parse_json(raw_response)
        except (json.JSONDecodeError, ValueError):
            # Si le parsing JSON échoue, essayer d'extraire le code Python du texte
            code, explanation = self._extract_python_code_from_text(raw_response)
            
            if code:
                # Construire un dict avec le code extrait
                action = {
                    "python_code": code,
                    "result_explanation": explanation or ""
                }
            else:
                # Si on ne peut pas extraire de code, retourner une erreur
                return {
                    "action_type": "ERROR",
                    "error_message": "Erreur de format JSON/Validation: Aucun bloc JSON ({...}) trouvé et aucun code Python détecté dans la réponse.",
                    "raw_output": raw_response[:500]  # Limiter la taille
                }
        
        # Si c'est une erreur de parsing, essayer d'extraire le code
        if action.get("action_type") == "ERROR" and "raw_output" in action:
            code, explanation = self._extract_python_code_from_text(action["raw_output"])
            if code:
                action = {
                    "python_code": code,
                    "result_explanation": explanation or ""
                }
        
        # Mapping des clés pour normalisation
        key_mapping = {
            'pythoncode': 'python_code',
            'python_code': 'python_code',
            'code': 'python_code',
            'resultexplain': 'result_explanation',
            'result_explanation': 'result_explanation',
            'explanation': 'result_explanation',
            'result': 'result_explanation',
        }
        
        # Normaliser les clés
        action = self._normalize_json_keys(action, key_mapping)
        
        # Si result_explanation manque, utiliser valeur par défaut
        if "result_explanation" not in action:
            action["result_explanation"] = ""

        # Validation spécifique du CodeWriter
        if not all(k in action for k in ["python_code", "result_explanation"]):
            # Dernière tentative: essayer d'extraire du raw_output si disponible
            if "raw_output" in action:
                code, explanation = self._extract_python_code_from_text(action["raw_output"])
                if code:
                    action["python_code"] = code
                    action["result_explanation"] = explanation or ""
                    # Retirer les clés d'erreur
                    action.pop("action_type", None)
                    action.pop("error_message", None)
                    action.pop("raw_output", None)
                else:
                    action["action_type"] = "ERROR"
                    action["error_message"] = "CodeWriter action missing required keys (python_code, result_explanation)."
            else:
                action["action_type"] = "ERROR"
                action["error_message"] = "CodeWriter action missing required keys (python_code, result_explanation)."
        
        return action

class CriticAgent(BaseAgent):
    def __init__(self):
        # Format JSON attendu: {'critique_ok': true/false, 'suggestions': 'SUGGESTIONS_STRING'}
        critic_system_prompt = (
            "You are the Critic Agent. Your role is to evaluate the quality, accuracy, and completeness of a generated solution.\n\n"
            "CRITICAL OUTPUT FORMAT: You MUST respond ONLY with a valid JSON object. "
            "Your response MUST start with the character '{' and end with '}'. "
            "Do NOT include any text, explanation, or markdown before or after the JSON.\n\n"
            "Required JSON structure:\n"
            "{\n"
            '  "critique_ok": true or false,\n'
            '  "suggestions": "concrete suggestions for improvement or confirmation message"\n'
            "}\n\n"
            "Examples:\n"
            '{"critique_ok": true, "suggestions": "The solution is complete and correct"}\n'
            '{"critique_ok": false, "suggestions": "The code has a syntax error on line 3"}\n\n'
            "Remember: Start your response directly with {, no other text."
        )
        super().__init__(
            name="Critic",
            system_prompt=critic_system_prompt,
            lora_folder="critic_lora"
        )

    def act(self, observation: str, fast_mode: bool = False) -> dict:
        action = super().act(observation, fast_mode=fast_mode)
        if action.get("action_type") == "ERROR":
            return action

        # Mapping des clés pour normalisation
        key_mapping = {
            'critiqueok': 'critique_ok',
            'critique_ok': 'critique_ok',
            'criteria_ok': 'critique_ok',  # Gérer faute de frappe commune
            'critic_ok': 'critique_ok',
            'criteoire_ok': 'critique_ok',  # Gérer faute de frappe
            'ok': 'critique_ok',
            'suggestions': 'suggestions',
            'suggestion': 'suggestions',
            'commentaire': 'suggestions',
            'suiviement': 'suggestions',  # Gérer faute de frappe
        }
        
        # Normaliser les clés
        action = self._normalize_json_keys(action, key_mapping)
        
        # Normaliser critique_ok en boolean si présent
        if "critique_ok" in action:
            action["critique_ok"] = self._normalize_boolean(action["critique_ok"])
        
        # Si suggestions manque, utiliser valeur par défaut
        if "suggestions" not in action:
            action["suggestions"] = ""

        # Validation spécifique du Critic
        if not all(k in action for k in ["critique_ok", "suggestions"]):
            action["action_type"] = "ERROR"
            action["error_message"] = "Critic action missing required keys (critique_ok, suggestions)."
        
        if "critique_ok" in action and not isinstance(action["critique_ok"], bool):
            action["action_type"] = "ERROR"
            action["error_message"] = "Critic 'critique_ok' must be a boolean."

        return action