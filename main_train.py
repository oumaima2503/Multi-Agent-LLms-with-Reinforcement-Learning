# main_train.py  (MAGRPO integrated)
import os
import json
import random
import torch
import logging
from typing import List, Dict, Any

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from marl_core.centralized_critic import CentralizedCritic
from marl_core.magrpo_utils import encode_global_state, move_model_to_device, offload_model_to_cpu
from marl_core.magrpo_trainer import MAGRPOTrainer

# CONFIG
# Basé sur l'article "LLM Collaboration with Multi-Agent Reinforcement Learning"
BASE_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
CHECKPOINTS_DIR = "checkpoints"  # Chemin vers les checkpoints SFT
DATASET_PATH = "data/processed_sft/orchestrator_sft.jsonl"  # Dataset pour l'entraînement RL
AGENTS_LIST = ["orchestrator", "researcher", "code_writer", "critic"]
TOTAL_EPOCHS = 10  # Commencez avec 10 époques (ajustez selon résultats)
SAVE_FREQ = 5  # Sauvegarder toutes les 5 époques
SAVE_FOLDER = "checkpoints/magrpo_rl"  # Dossier pour sauvegarder les checkpoints RL
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
torch.cuda.empty_cache()

# utils
def load_agent_policy(agent_name: str, is_training=True):
    lora_path = os.path.join(CHECKPOINTS_DIR, f"{agent_name}_lora")
    if not os.path.exists(os.path.join(lora_path, "adapter_config.json")):
        raise FileNotFoundError(f"Missing adapter for {agent_name} at {lora_path}")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, quantization_config=bnb, device_map="cpu")
    if is_training:
        prepare_model_for_kbit_training(base)
        base.config.use_cache = False
    else:
        base.config.use_cache = True

    lora_cfg = LoraConfig.from_pretrained(lora_path)
    model = get_peft_model(base, lora_cfg)
    return model, tokenizer

# Simple prompt formatting
SYSTEM_PROMPTS = {
    "orchestrator": "Tu es l'Orchestrateur. REPLISSEZ JSON {'AGENT_CIBLE':..., 'COMMANDE':...}",
    "researcher": "Tu es le Researcher. Réponds factuellement.",
    "code_writer": "Tu es CodeWriter. Génère du code Python dans ```python```.",
    "critic": "Tu es Critic. Fais une critique concise."
}
def format_prompt(system_prompt, instruction):
    return f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{instruction} [/INST] "

class LLMAgent:
    def __init__(self, name):
        self.name = name
        self.system_prompt = SYSTEM_PROMPTS[name]
        self.model = None
        self.tokenizer = None

    def load_policy(self):
        self.model, self.tokenizer = load_agent_policy(self.name)

    def generate_action(self, state_text):
        prompt = format_prompt(self.system_prompt, state_text)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to("cuda")
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=128)
        gen = out[0][inputs.input_ids.shape[1]:].detach().cpu()
        text = self.tokenizer.decode(gen, skip_special_tokens=True)
        return text, gen

class MARL_Env:
    def __init__(self, agents_list):
        self.agents = {}
        for name in agents_list:
            a = LLMAgent(name)
            a.load_policy()
            self.agents[name] = a
        self.current_state = ""
        self.current_agent = "orchestrator"
        self.turn_count = 0
        self.max_turns = 10

        self.last_query = ""
        self.last_response_tokens = None

    def reset(self, instr):
        self.current_state = f"Instruction: {instr}"
        self.current_agent = "orchestrator"
        self.turn_count = 0
        return self.current_state

    def step(self):
        agent_name = self.current_agent
        agent = self.agents[agent_name]

        # offload others
        for n,a in self.agents.items():
            if n != agent_name:
                offload_model_to_cpu(a.model)

        # move actor to gpu
        move_model_to_device(agent.model, "cuda")

        # store prompt used
        self.last_query = format_prompt(agent.system_prompt, self.current_state)
        text, gen = agent.generate_action(self.current_state)
        self.last_response_tokens = gen  # CPU tensor

        # offload actor
        offload_model_to_cpu(agent.model)

        # transition logic as before
        reward = 0.0
        done = False
        info = {"response": text}

        if agent_name == "orchestrator":
            try:
                j = json.loads(text)
                tgt = j.get("AGENT_CIBLE","").lower()
                cmd = j.get("COMMANDE","")
                if tgt == "end":
                    done = True
                    reward = 5.0
                elif tgt in self.agents:
                    self.current_agent = tgt
                    self.current_state += f"\n[ORCH->{tgt}]: {cmd}"
                else:
                    done = True
                    reward = -3.0
            except:
                done = True
                reward = -5.0
        else:
            self.current_state += f"\n[{agent_name.upper()}]: {text}"
            self.current_agent = "orchestrator"

        self.turn_count += 1
        if self.turn_count >= self.max_turns:
            done = True
            reward = -5.0

        return self.current_state, reward, done, info

def collect_trajectories(env: MARL_Env, dataset, max_episodes: int):
    trajs = []
    num = min(max_episodes, len(dataset))
    for _ in range(num):
        idx = random.randint(0, len(dataset)-1)
        instr = dataset[idx]["instruction"]
        env.reset(instr)
        episode_steps = []
        done = False
        final_reward = 0.0
        while not done:
            agent_name = env.current_agent
            # encode state
            state_emb = encode_global_state(env.current_state, env.turn_count, agent_name, device="cpu")
            new_state, r, done, info = env.step()
            if info.get("response") is not None:
                # get query tokens and response tokens
                agent = env.agents[agent_name]
                q_ids = agent.tokenizer(env.last_query, return_tensors="pt", truncation=True).input_ids.squeeze(0).cpu()
                resp_ids = env.last_response_tokens.cpu() if env.last_response_tokens is not None else torch.tensor([], dtype=torch.long)
                episode_steps.append({
                    "agent": agent_name,
                    "query": q_ids,
                    "response": resp_ids,
                    "state_emb": state_emb.detach().cpu()
                })
            if done:
                final_reward = r
                for s in episode_steps:
                    trajs.append({
                        "agent": s["agent"],
                        "query": s["query"],
                        "response": s["response"],
                        "reward": float(final_reward),
                        "state_emb": s["state_emb"]
                    })
                break
    logging.info(f"Collected {len(trajs)} transitions.")
    return trajs

def train_marl_magrpo(agents_list):
    env = MARL_Env(agents_list)
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    # instantiate critic (on CPU)
    state_dim = 384
    critic = CentralizedCritic(input_dim=state_dim, hidden=512)
    critic.to("cpu")
    critic_opt = torch.optim.Adam(critic.parameters(), lr=1e-4)

    # prepare trainers per agent
    trainers = {}
    for name in agents_list:
        actor = env.agents[name].model
        ref_model, _ = load_agent_policy(name, is_training=False)
        # ensure ref on cpu
        offload_model_to_cpu(ref_model)
        trainer = MAGRPOTrainer(actor, ref_model, env.agents[name].tokenizer, critic, lr=1.41e-5, clip_epsilon=0.2, device="cuda")
        trainers[name] = trainer
        logging.info(f"Trainer ready for {name}")

    for epoch in range(TOTAL_EPOCHS):
        logging.info(f"Epoch {epoch}")
        transitions = collect_trajectories(env, dataset, max_episodes=2)  # small for T4
        if not transitions:
            logging.warning("No transitions collected.")
            break
        # group by agent
        batches = {n: {"query":[], "response":[], "reward":[], "state":[]} for n in agents_list}
        for t in transitions:
            batches[t["agent"]]["query"].append(t["query"])
            batches[t["agent"]]["response"].append(t["response"])
            batches[t["agent"]]["reward"].append(t["reward"])
            batches[t["agent"]]["state"].append(t["state_emb"])
        # update each agent with MAGRPOTrainer
        for name in agents_list:
            b = batches[name]
            if not b["query"]:
                continue
            stats = trainers[name].step(b["query"], b["response"], b["reward"], b["state"])
            logging.info(f"{name} update: loss {stats['loss']:.4f} kl {stats['kl']:.6f} val_mean {stats['value_mean']:.3f}")
        # save checkpoints
        if (epoch + 1) % SAVE_FREQ == 0:
            os.makedirs(SAVE_FOLDER, exist_ok=True)
            for name in agents_list:
                save_path = os.path.join(SAVE_FOLDER, f"epoch{epoch+1}_{name}_rl")
                try:
                    env.agents[name].model.save_pretrained(save_path)
                    logging.info(f"Saved RL LoRA for {name} -> {save_path}")
                except Exception as e:
                    logging.warning(f"Failed to save {name}: {e}")

    logging.info("Training finished.")

if __name__ == "__main__":
    train_marl_magrpo(AGENTS_LIST)
