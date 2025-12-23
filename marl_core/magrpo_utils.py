# magrpo_utils.py
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import List

# Lightweight sentence encoder for state embedding
# (install sentence-transformers in Colab beforehand)
STATE_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
_state_tokenizer = AutoTokenizer.from_pretrained(STATE_ENCODER)
_state_encoder = AutoModel.from_pretrained(STATE_ENCODER)

def encode_global_state(history_text: str, turn: int, current_agent: str, device: str = "cpu") -> torch.Tensor:
    """
    Encode S_t -> embedding tensor on CPU by default (small model).
    Returns tensor shape [emb_dim] (1D).
    """
    text = f"[TURN={turn}][AGENT={current_agent}] {history_text}"
    inputs = _state_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = _state_encoder(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1).squeeze(0)  # [emb_dim]
    return emb.to(device)

def compute_logprobs(model, input_ids: torch.Tensor, gen_ids: torch.Tensor, tokenizer) -> torch.Tensor:
    """
    Compute sum log-probability of generated tokens under model.
    input_ids: tensor [L_in] (1D)
    gen_ids: tensor [L_gen] (1D)
    Returns scalar tensor (log prob).
    Notes:
      - model must be on correct device.
      - input_ids/gen_ids are 1D CPU tensors; will be moved to model device.
    """
    device = next(model.parameters()).device
    # Concatenate
    full = torch.cat([input_ids.to(device), gen_ids.to(device)], dim=0).unsqueeze(0)  # [1, L_full]
    with torch.no_grad():
        outputs = model(full)
        logits = outputs.logits  # [1, L_full, V]
    # We want log-probs of each generated token conditional on previous tokens.
    # For gen token i at position pos = L_full - L_gen + i:
    L_in = input_ids.shape[0]
    L_gen = gen_ids.shape[0]
    # calculate logsoftmax over vocab
    lps = F.log_softmax(logits, dim=-1)  # [1, L_full, V]
    total = 0.0
    for i in range(L_gen):
        pos = L_in + i  # index in full sequence
        token_id = gen_ids[i].item()
        total = total + lps[0, pos - 1, token_id]  # model predicts token at pos given tokens up to pos-1
        # note: for causal LM, logits[t] correspond to next-token probabilities for position t
    return total  # scalar tensor

def move_model_to_device(model, device: str):
    try:
        model.to(device)
        torch.cuda.empty_cache()
    except Exception:
        pass

def offload_model_to_cpu(model):
    try:
        model.to("cpu")
        torch.cuda.empty_cache()
    except Exception:
        pass
