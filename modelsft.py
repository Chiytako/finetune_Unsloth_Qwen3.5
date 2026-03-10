from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
import threading
import queue
from datasets import load_dataset
from torch.utils.data import IterableDataset as TorchIterableDataset
from trl import SFTTrainer, SFTConfig

max_seq_length = 16384

# ─── SFT ストリーミング設定 ────────────────────────────────────────────────────
# SFT データセットは数千万件あるため、CHUNK_SIZE 件ずつバックグラウンドでダウンロードする
CHUNK_SIZE      = 100   # 一度に取得するサンプル数
PREFETCH_CHUNKS = 3     # キューに積む最大チャンク数（メモリ = CHUNK_SIZE×PREFETCH_CHUNKS）
MAX_SFT_STEPS   = 5000  # SFT ストリームから消費する最大ステップ数（None = 無制限）

# ─── Reasoning dataset は小規模なので通常ロード ───────────────────────────────
MAX_REASONING_SAMPLES = 1000  # 全件取得（実態に合わせて調整）

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


# ─── SFTStreamDataset ─────────────────────────────────────────────────────────
class SFTStreamDataset(TorchIterableDataset):
    """
    数千万件の SFT データを CHUNK_SIZE 件ずつバックグラウンドスレッドでダウンロードする。

    動作:
      - プロデューサースレッドが HuggingFace streaming dataset を CHUNK_SIZE 件まとめてキューに積む
      - メインスレッド(トレーニング)はキュー内のデータを消費しながら次のチャンクが自動的に取得される
      - キューが PREFETCH_CHUNKS 個分いっぱいになるとプロデューサーが自動待機し
        メモリを CHUNK_SIZE × PREFETCH_CHUNKS サンプル分に抑制する
      - max_samples に達したら停止（None = 無制限）
    """

    def __init__(self, hf_iterable_ds, chunk_size=CHUNK_SIZE, prefetch_chunks=PREFETCH_CHUNKS,
                 max_samples=None):
        super().__init__()
        self.hf_ds = hf_iterable_ds
        self.chunk_size = chunk_size
        self.prefetch_chunks = prefetch_chunks
        self.max_samples = max_samples

    def _producer(self, q: queue.Queue):
        chunk = []
        count = 0
        for item in self.hf_ds:
            if self.max_samples and count >= self.max_samples:
                break
            text = item.get("text", "")
            if text:
                chunk.append({"text": text})
                count += 1
                if len(chunk) >= self.chunk_size:
                    q.put(chunk)
                    chunk = []
        if chunk:
            q.put(chunk)
        q.put(None)  # 終了シグナル

    def __iter__(self):
        q = queue.Queue(maxsize=self.prefetch_chunks)
        t = threading.Thread(target=self._producer, args=(q,), daemon=True)
        t.start()
        while True:
            chunk = q.get()
            if chunk is None:
                break
            yield from chunk


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
    reasoning_raw = load_dataset(
        "ChiTako/Qwen3.5-27b-ja",
        split="train",
        token=True,
    )
    reasoning_raw = reasoning_raw.filter(lambda x: x["quality"]["passed"])
    if len(reasoning_raw) > MAX_REASONING_SAMPLES:
        reasoning_raw = reasoning_raw.select(range(MAX_REASONING_SAMPLES))
    reasoning_dataset = reasoning_raw.map(
        format_reasoning_to_messages,
        remove_columns=reasoning_raw.column_names,
        num_proc=None,
    )
    reasoning_formatted = reasoning_dataset.map(
        apply_reasoning_template,
        remove_columns=["messages"],
        num_proc=None,
    )
    reasoning_formatted = reasoning_formatted.filter(lambda x: bool(x["text"]))
    print(f"  Loaded {len(reasoning_formatted)} reasoning samples")

    # ─── SFT dataset: 数千万件のためストリーミング + バックグラウンドプリフェッチ ─
    print(f"Setting up SFT streaming pipeline (chunk={CHUNK_SIZE}, prefetch={PREFETCH_CHUNKS})...")
    sft_stream_raw = (
        load_dataset("ChiTako/niconico_sft", split="train", token=True, streaming=True)
        .map(format_sft_to_messages)
        .filter(lambda x: x.get("messages") is not None)
        .map(apply_sft_template)
    )
    sft_stream_dataset = SFTStreamDataset(
        sft_stream_raw,
        chunk_size=CHUNK_SIZE,
        prefetch_chunks=PREFETCH_CHUNKS,
        max_samples=MAX_SFT_STEPS,  # トレーニングで消費する最大サンプル数
    )
    print(f"  SFT stream ready (downloads {CHUNK_SIZE} samples at a time, max {MAX_SFT_STEPS})")

    # ─── max_steps の計算 ─────────────────────────────────────────────────────
    # reasoning: len 確定、SFT: ストリーミングのため推定
    reasoning_count = len(reasoning_formatted)
    sft_count = MAX_SFT_STEPS if MAX_SFT_STEPS else reasoning_count * 3
    # 65/35 ブレンドで reasoning が先に尽きると仮定
    approx_total = int(reasoning_count / 0.65)
    max_steps = (approx_total * NUM_EPOCHS) // (PER_DEVICE_BATCH * GRAD_ACC)
    print(f"  Estimated max_steps: {max_steps} (≈{NUM_EPOCHS} epochs, ~{approx_total} samples/epoch)")

    # ─── カスタム交互 IterableDataset ─────────────────────────────────────────
    class InterleavedDataset(TorchIterableDataset):
        """
        確定長の reasoning データを繰り返しながら、SFT ストリームと 65/35 で交互に供給する。
        reasoning は小規模(~1000件)なので cycle させ、SFT は max_steps まで流し続ける。
        """
        def __init__(self, reasoning_ds, sft_stream_ds, reasoning_prob=0.65, seed=3407):
            super().__init__()
            self.reasoning_ds = list(reasoning_ds)  # メモリに収まるので list 化
            self.sft_stream_ds = sft_stream_ds
            self.reasoning_prob = reasoning_prob
            self.seed = seed

        def __len__(self):
            # unsloth の初期化チェック用: SFT 件数 + reasoning 挿入分の期待値
            sft_count = self.sft_stream_ds.max_samples or 0
            return sft_count + int(sft_count * self.reasoning_prob)

        def __getitem__(self, idx):
            # unsloth が train_dataset[0] でフォーマット確認するため実装
            # reasoning_ds はメモリ上にあるので循環インデックスで返す
            return self.reasoning_ds[idx % len(self.reasoning_ds)]

        def __iter__(self):
            import random
            rng = random.Random(self.seed)
            reasoning_idx = 0
            sft_iter = iter(self.sft_stream_ds)

            while True:
                # SFT ストリームが尽きたら終了
                try:
                    sft_item = next(sft_iter)
                except StopIteration:
                    break

                # reasoning_prob の確率で reasoning を先に yield
                if rng.random() < self.reasoning_prob and self.reasoning_ds:
                    r_item = self.reasoning_ds[reasoning_idx % len(self.reasoning_ds)]
                    reasoning_idx += 1
                    yield r_item

                yield sft_item

    train_dataset = InterleavedDataset(
        reasoning_formatted,
        sft_stream_dataset,
        reasoning_prob=0.65,
        seed=3407,
    )

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
            max_steps=max_steps,        # IterableDataset には max_steps を使用
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
