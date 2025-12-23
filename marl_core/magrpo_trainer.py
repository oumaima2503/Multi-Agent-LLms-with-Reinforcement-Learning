# magrpo_trainer.py
import torch
import torch.nn.functional as F
from torch.optim import Adam
from typing import List
from magrpo_utils import compute_logprobs, move_model_to_device, offload_model_to_cpu

class MAGRPOTrainer:
    """
    MAGRPO trainer for one agent.
    Uses shared critic (centralized) passed as 'critic' (on CPU).
    Actor/ref are PEFT models (4-bit on CPU) — moved to GPU during update.
    """
    def __init__(self, actor_model, ref_model, tokenizer, critic, lr: float = 1.41e-5, clip_epsilon: float = 0.2, device: str = "cuda"):
        self.actor = actor_model
        self.ref = ref_model
        self.tokenizer = tokenizer
        self.critic = critic  # this is on CPU
        self.clip_epsilon = clip_epsilon
        self.device = device

        # Only optimize LoRA parameters (PEFT exposes them)
        peft_params = [p for n,p in self.actor.named_parameters() if p.requires_grad]
        self.optimizer = Adam(peft_params, lr=lr)

    def compute_gae(self, rewards: List[float], values: List[float], gamma=0.99, lam=0.95):
        advs = []
        gae = 0.0
        values_ext = values + [0.0]
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values_ext[t+1] - values_ext[t]
            gae = delta + gamma * lam * gae
            advs.insert(0, gae)
        return advs

    def step(self, batch_queries: List[torch.Tensor], batch_responses: List[torch.Tensor], batch_rewards: List[float], batch_state_embs: List[torch.Tensor]):
        """
        batch_queries: list of input_ids 1D tensors (CPU)
        batch_responses: list of gen token 1D tensors (CPU)
        batch_rewards: list of floats
        batch_state_embs: list of state embeddings (CPU)
        """
        device = self.device

        # 1) compute values using critic (critic expected on CPU)
        values = []
        for emb in batch_state_embs:
            v = self.critic(emb.to("cpu")).item()
            values.append(v)

        # 2) compute advantages (GAE) using episodic rewards
        advantages = self.compute_gae(batch_rewards, values)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        policy_losses = []
        kls = []
        # loop samples (safe offload model for each sample)
        for q_ids, gen_ids, adv in zip(batch_queries, batch_responses, advantages):
            # move models to GPU
            move_model_to_device(self.actor, device)
            move_model_to_device(self.ref, device)

            # compute old and new log probs
            old_logp = compute_logprobs(self.ref, q_ids, gen_ids, self.tokenizer).to(device)
            new_logp = compute_logprobs(self.actor, q_ids, gen_ids, self.tokenizer).to(device)

            ratio = torch.exp(new_logp - old_logp)
            unclipped = ratio * adv.to(device)
            clipped = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * adv.to(device)
            policy_loss = -torch.min(unclipped, clipped)

            policy_losses.append(policy_loss)
            kls.append((old_logp - new_logp).detach().cpu())

            # offload actor/ref immediately to free GPU before next sample
            offload_model_to_cpu(self.actor)
            offload_model_to_cpu(self.ref)

        if len(policy_losses) == 0:
            return {"loss": 0.0, "kl": 0.0, "value_mean": float(sum(values)/len(values) if values else 0.0)}

        loss = torch.stack(policy_losses).mean()
        # backward on PEFT params
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "kl": float(torch.stack(kls).mean().item()) if kls else 0.0,
            "value_mean": float(sum(values) / len(values))
        }
