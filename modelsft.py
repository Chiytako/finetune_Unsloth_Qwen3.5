from unsloth import FastLanguageModel
import torch
from datasets import load_dataset, interleave_datasets
from trl import SFTTrainer, SFTConfig

max_seq_length = 16384

# ─── Format helpers (module-level so Windows spawn can import them) ───────────
def format_reasoning_to_messages(example):
    """Convert ChiTako columns to chat messages, embedding thinking in <think> tags."""
    messages = []
    if example.get("system"):
        messages.append({"role": "system", "content": example["system"]})
    messages.append({"role": "user", "content": example["instruction"]})
    assistant_content = f"<think>\n{example['thinking']}\n</think>\n{example['output']}"
    messages.append({"role": "assistant", "content": assistant_content})
    return {"messages": messages}


def format_with_chat_template(example):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=True,  # Keep reasoning mode ON for Qwen3.5
    )
    return {"text": text}


# ─── Windows multiprocessing guard ───────────────────────────────────────────
if __name__ == "__main__":

    # ─── Model Loading ────────────────────────────────────────────────────────
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen3.5-0.8B",
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        max_seq_length=max_seq_length,
    )

    # ─── Dataset Loading ──────────────────────────────────────────────────────
    # 1. SFT dataset from HuggingFace (private repo)
    sft_dataset = load_dataset(
        "ChiTako/niconico_sft",
        split="train",
        token=True,  # Use HF_TOKEN for private repo access
    )

    # 2. Reasoning dataset from HuggingFace (private repo)
    reasoning_raw = load_dataset(
        "ChiTako/Qwen3.5-27b-ja",
        split="train",
        token=True,  # Use HF_TOKEN for private repo access
    )
    reasoning_raw = reasoning_raw.filter(lambda x: x["quality"]["passed"])

    reasoning_dataset = reasoning_raw.map(
        format_reasoning_to_messages,
        remove_columns=reasoning_raw.column_names,
        num_proc=None,  # None = single-threaded, avoids Windows spawn issues
    )

    # ─── Blend: ≥75% reasoning, ≤25% direct SFT ──────────────────────────────
    blended_dataset = interleave_datasets(
        [reasoning_dataset, sft_dataset],
        probabilities=[0.75, 0.25],
        seed=3407,
        stopping_strategy="first_exhausted",
    )

    # ─── Apply Qwen3.5 Chat Template (enable_thinking=True) ──────────────────
    formatted_dataset = blended_dataset.map(
        format_with_chat_template,
        remove_columns=["messages"],
        num_proc=None,  # None = single-threaded, avoids Windows spawn issues
    )

    # ─── Training ─────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        train_dataset=formatted_dataset,
        tokenizer=tokenizer,
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=10,
            num_train_epochs=3,
            logging_steps=10,
            output_dir="outputs_qwen35",
            optim="adamw_8bit",
            seed=3407,
            dataset_num_proc=None,  # None = single-threaded
        ),
    )

    trainer.train()
