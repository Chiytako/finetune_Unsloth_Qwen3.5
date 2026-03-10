from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
import threading
import queue
import random
from datasets import load_dataset, IterableDataset as HFIterableDataset
from trl import SFTTrainer, SFTConfig

max_seq_length = 16384

# ─── SFT ストリーミング設定 ────────────────────────────────────────────────────
CHUNK_SIZE      = 100   # バックグラウンドで一度に取得するサンプル数
PREFETCH_CHUNKS = 3     # キューに積む最大チャンク数（メモリ = CHUNK_SIZE×PREFETCH_CHUNKS）
MAX_SFT_STEPS   = 5000  # SFT ストリームから消費する最大サンプル数（None = 無制限）

# ─── Reasoning dataset は小規模なので通常ロード ───────────────────────────────
MAX_REASONING_SAMPLES = 1000

# ─── Epoch / batch 設定 ───────────────────────────────────────────────────────
NUM_EPOCHS       = 3
PER_DEVICE_BATCH = 1
GRAD_ACC         = 4

# ─── 日本語アンカー用システムプロンプト ───────────────────────────────────────
JAPANESE_SYSTEM_PROMPT = (
    "あなたは日本語で話す親切なアシスタントです。"
    "常に日本語で、論理的かつ一貫した回答をしてください。"
)


# ─── Format helpers (module-level: Windows spawn でも import 可能) ─────────────
def format_reasoning_to_messages(example):
    messages = []
    if example.get("system"):
        messages.append({"role": "system", "content": example["system"]})
    else:
        messages.append({"role": "system", "content": JAPANESE_SYSTEM_PROMPT})
    messages.append({"role": "user", "content": example["instruction"]})
    assistant_content = f"<think>\n{example['thinking']}\n</think>\n{example['output']}"
    messages.append({"role": "assistant", "content": assistant_content})
    return {"messages": messages}


def format_sft_to_messages(example):
    messages = []
    i = 0
    while f"role_{i}" in example and example[f"role_{i}"] is not None:
        messages.append({"role": example[f"role_{i}"], "content": example[f"content_{i}"] or ""})
        i += 1
    if not messages:
        return {"messages": None}
    if messages[0]["role"] != "system":
        messages = [{"role": "system", "content": JAPANESE_SYSTEM_PROMPT}] + messages
    return {"messages": messages}


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
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        max_seq_length=max_seq_length,
    )

    # ─── Chat template appliers (tokenizer が必要なためここで定義) ────────────
    def apply_reasoning_template(example):
        """推論データ用: enable_thinking=True（<think>タグあり）"""
        if not example.get("messages"):
            return {"text": ""}
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=True,
        )
        return {"text": text}

    def apply_sft_template(example):
        """SFTデータ用: enable_thinking=False（<think>タグなし）"""
        if not example.get("messages"):
            return {"text": ""}
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        return {"text": text}

    # ─── Reasoning dataset: 小規模なので通常ロード ───────────────────────────
    print(f"Loading reasoning dataset (up to {MAX_REASONING_SAMPLES} samples)...")
    reasoning_raw = load_dataset("ChiTako/Qwen3.5-27b-ja", split="train", token=True)
    reasoning_raw = reasoning_raw.filter(lambda x: x["quality"]["passed"])
    if len(reasoning_raw) > MAX_REASONING_SAMPLES:
        reasoning_raw = reasoning_raw.select(range(MAX_REASONING_SAMPLES))
    reasoning_formatted = (
        reasoning_raw
        .map(format_reasoning_to_messages, remove_columns=reasoning_raw.column_names, num_proc=None)
        .map(apply_reasoning_template, remove_columns=["messages"], num_proc=None)
        .filter(lambda x: bool(x["text"]))
    )
    reasoning_list = list(reasoning_formatted)  # メモリに収まるのでリスト化
    print(f"  Loaded {len(reasoning_list)} reasoning samples")

    # ─── SFT streaming pipeline (フォーマット済み) ────────────────────────────
    print(f"Setting up SFT streaming pipeline (chunk={CHUNK_SIZE}, prefetch={PREFETCH_CHUNKS})...")
    sft_stream_formatted = (
        load_dataset("ChiTako/niconico_sft", split="train", token=True, streaming=True)
        .map(format_sft_to_messages)
        .filter(lambda x: x.get("messages") is not None)
        .map(apply_sft_template)
    )
    print(f"  SFT stream ready (downloads {CHUNK_SIZE} samples at a time, max {MAX_SFT_STEPS})")

    # ─── max_steps の計算 ─────────────────────────────────────────────────────
    # SFT が主軸: MAX_SFT_STEPS サンプル + reasoning 挿入分（65%）で総計を推定
    approx_total = MAX_SFT_STEPS + int(MAX_SFT_STEPS * 0.65)
    max_steps = (approx_total * NUM_EPOCHS) // (PER_DEVICE_BATCH * GRAD_ACC)
    print(f"  Estimated max_steps: {max_steps} (≈{NUM_EPOCHS} epochs, ~{approx_total} samples/epoch)")

    # ─── HFIterableDataset.from_generator() で交互供給 + バックグラウンドプリフェッチ ─
    # SFTTrainer が .map() を呼ぶため HuggingFace IterableDataset が必要
    def make_train_generator():
        """
        SFT データを CHUNK_SIZE 件ずつバックグラウンドスレッドで先読みしながら
        reasoning データと 65/35 で交互に供給するジェネレーター。
        """
        rng = random.Random(3407)
        q = queue.Queue(maxsize=PREFETCH_CHUNKS)

        def sft_producer():
            chunk = []
            count = 0
            for item in sft_stream_formatted:
                if MAX_SFT_STEPS and count >= MAX_SFT_STEPS:
                    break
                text = item.get("text", "")
                if text:
                    chunk.append({"text": text})
                    count += 1
                    if len(chunk) >= CHUNK_SIZE:
                        q.put(chunk)
                        chunk = []
            if chunk:
                q.put(chunk)
            q.put(None)  # 終了シグナル

        t = threading.Thread(target=sft_producer, daemon=True)
        t.start()

        reasoning_idx = 0
        sft_buffer = []

        while True:
            # SFT バッファが空になったら次のチャンクを取得
            if not sft_buffer:
                chunk = q.get()
                if chunk is None:
                    break
                sft_buffer = list(chunk)

            sft_item = sft_buffer.pop(0)

            # 65% の確率で reasoning を先に yield（cycling）
            if reasoning_list and rng.random() < 0.65:
                yield reasoning_list[reasoning_idx % len(reasoning_list)]
                reasoning_idx += 1

            yield sft_item

    train_dataset = HFIterableDataset.from_generator(make_train_generator)
    # unsloth が _ex_iterable.batch_size を期待するため MappedExamplesIterable に変換
    # (batched=True でデフォルト batch_size=1000 が設定される。データは遅延処理のまま)
    train_dataset = train_dataset.map(lambda x: x, batched=True)

    # ─── Training ─────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            per_device_train_batch_size=PER_DEVICE_BATCH,
            gradient_accumulation_steps=GRAD_ACC,
            warmup_steps=10,
            max_steps=max_steps,        # HFIterableDataset には max_steps を使用
            learning_rate=2e-4,
            lr_scheduler_type="cosine",
            logging_steps=10,
            output_dir="outputs_qwen35",
            optim="adamw_8bit",
            seed=3407,
            dataset_num_proc=None,
        ),
    )

    # ─── マルチターン対応: アシスタント応答のみ損失を計算 ─────────────────────
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    trainer.train()
