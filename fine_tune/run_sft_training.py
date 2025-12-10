import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig
from trl import SFTTrainer
import os
import argparse


BASE_MODEL_ID = "microsoft/phi-2"


CHAT_TEMPLATE = (
    "<s>[INST] <<SYS>>\n"
    "{system_prompt}\n"
    "<</SYS>>\n\n"
    "{instruction} [/INST] {response}</s>"
)


def formatting_prompts_func(example):
    """Convertit un batch JSONL → texte formatté"""
    texts = []
    # sécurité : gérer si instruction est une string ou une liste
    if isinstance(example.get("instruction"), list):
        length = len(example["instruction"])
        for i in range(length):
            formatted = CHAT_TEMPLATE.format(
                system_prompt=example["system_prompt"][i],
                instruction=example["instruction"][i],
                response=example["response"][i],
            )
            texts.append(formatted)
    else:
        # cas improbable mais sûr : éléments scalaires
        formatted = CHAT_TEMPLATE.format(
            system_prompt=example.get("system_prompt", ""),
            instruction=example.get("instruction", ""),
            response=example.get("response", ""),
        )
        texts.append(formatted)
    return {"text": texts}


def make_safe_data_collator(tokenizer):
    """Retourne un collator qui supprime 'text' avant le padding/tokenization."""
    base_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    def collator(features):
        # features : liste de dicts. Supprimer 'text' si présent.
        if isinstance(features, (list, tuple)) and len(features) > 0 and isinstance(features[0], dict):
            for f in features:
                if "text" in f:
                    f.pop("text", None)
        # appeler le collator standard (qui attend des input_ids / attention_mask / ...)
        return base_collator(features)

    return collator


def run_sft_training(dataset_path, output_dir, agent_name):

    print(f"\n===== SFT pour agent : {agent_name} =====")
    print(f"Dataset : {dataset_path}\n")

    # ------------ QLoRA CONFIG ------------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )

    # ------------ MODEL ------------
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model.config.use_cache = False

    # ------------ TOKENIZER ------------
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # important pour TRL 0.25.1
    model.tokenizer = tokenizer

    # ------------ LOAD DATASET ------------
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    # Appliquer le template (retourne 'text')
    dataset = dataset.map(formatting_prompts_func, batched=True)

    # S'assurer qu'il n'y a QUE la colonne 'text' (sécurité supplémentaire)
    dataset = dataset.remove_columns([c for c in dataset.column_names if c != "text"])

    print(f"Nombre d'exemples : {len(dataset)}")
    print("Exemple :", dataset[0])

    # ------------ LoRA CONFIG ------------
    peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    # ------------ TRAINING ARGS ------------
    # Note: on peut laisser remove_unused_columns=False mais on a alors besoin du collator safe.
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        optim="paged_adamw_32bit",
        save_steps=500,
        logging_steps=50,
        fp16=True,
        report_to="none",
        remove_unused_columns=False,  # on garde False mais on nettoie via le collator
    )

    # ------------ SFT TRAINER COMPATIBLE TRL 0.25.1 ------------
    safe_collator = make_safe_data_collator(tokenizer)

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        peft_config=peft_config,

        # IMPORTANT : doit renvoyer liste de strings
        formatting_func=lambda batch: batch["text"],

        # collator custom qui supprime 'text' avant padding
        data_collator=safe_collator,
    )

    # ------------ TRAIN ------------
    trainer.train()

    # ------------ SAVE ------------
    print(f"\nSaving model → {output_dir}")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"\n=== SFT TERMINÉ POUR {agent_name} ===\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_name", required=True,
        choices=["orchestrator", "researcher", "code_writer", "critic"])

    args = parser.parse_args()

    dataset = f"data/processed_sft/{args.agent_name}_sft.jsonl"
    out = f"checkpoints/{args.agent_name}_lora"

    os.makedirs(out, exist_ok=True)
    run_sft_training(dataset, out, args.agent_name)
