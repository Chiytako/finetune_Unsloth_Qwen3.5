"""
Upload sft.jsonl to HuggingFace as Parquet dataset (ChiTako/niconico_sft)
Strategy:
  Phase 1: Convert 29GB JSONL -> Parquet chunks (local, skip if exists)
  Phase 2: upload_large_folder (batches into few commits, skips already-uploaded)
"""

import json
import os
import sys
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, create_repo

# Config
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is required")
REPO_ID = "ChiTako/niconico_sft"
INPUT_FILE = "E:/purograming/nikonikodetasets/highquality_dataset/sft.jsonl"
CHUNK_SIZE = 100_000  # match already-uploaded chunks (00000-00127)
LOCAL_DIR = "E:/purograming/nikonikodetasets/highquality_dataset/parquet_upload"

README = """---
language:
- ja
license: other
task_categories:
- text-generation
task_ids:
- language-modeling
tags:
- niconico
- japanese
- sft
- chat
- instruction-following
pretty_name: niconico SFT Dataset
size_categories:
- 100M<n<1B
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*.parquet
---

# niconico SFT Dataset

ニコニコ動画のコメントデータを使った Supervised Fine-Tuning (SFT) データセットです。

## Dataset Structure

各サンプルは 3 ターンの会話形式です。

| カラム | 説明 |
|--------|------|
| `role_0` / `content_0` | system プロンプト（キャラクター設定） |
| `role_1` / `content_1` | user 入力（動画タイトル等） |
| `role_2` / `content_2` | assistant 応答（ニコニコ風コメント） |

## Example

```
role_0: system
content_0: あなたはニコニコ動画のヘビーユーザーです。
role_1: user
content_1: 「【弾いてみた】ヒトリエ 終着点」という音楽動画を見た感想は？
role_2: assistant
content_2: ロックで動かないベーシストほどライブ映えしないものはない。
```
"""

def phase1_convert():
    """Convert JSONL to Parquet chunks locally. Skips existing files."""
    data_dir = os.path.join(LOCAL_DIR, "data")
    os.makedirs(data_dir, exist_ok=True)

    # Write README
    readme_path = os.path.join(LOCAL_DIR, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(README)

    chunk_idx = 0
    batch = []
    total_rows = 0
    skipped_chunks = 0

    print(f"Phase 1: Converting JSONL to Parquet")
    print(f"  Input: {INPUT_FILE}")
    print(f"  Output: {data_dir}")
    print(f"  Chunk size: {CHUNK_SIZE:,} rows\n")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                msgs = record.get("messages", [])
                if len(msgs) >= 3:
                    batch.append(msgs)
            except json.JSONDecodeError:
                continue

            if len(batch) >= CHUNK_SIZE:
                out_path = os.path.join(data_dir, f"train-{chunk_idx:05d}.parquet")
                if os.path.exists(out_path):
                    skipped_chunks += 1
                else:
                    write_parquet(batch, out_path)
                    print(f"  [chunk {chunk_idx:05d}] {len(batch):,} rows -> {os.path.getsize(out_path)/1e6:.1f}MB")
                total_rows += len(batch)
                chunk_idx += 1
                batch = []

                if chunk_idx % 10 == 0:
                    print(f"  Progress: {total_rows:,} rows, {chunk_idx} chunks ({skipped_chunks} skipped)")

    # Last batch
    if batch:
        out_path = os.path.join(data_dir, f"train-{chunk_idx:05d}.parquet")
        if not os.path.exists(out_path):
            write_parquet(batch, out_path)
            print(f"  [chunk {chunk_idx:05d}] {len(batch):,} rows -> {os.path.getsize(out_path)/1e6:.1f}MB")
        total_rows += len(batch)
        chunk_idx += 1

    print(f"\nPhase 1 done: {total_rows:,} rows, {chunk_idx} chunks ({skipped_chunks} skipped)\n")
    return chunk_idx

def write_parquet(batch, out_path):
    col_role0, col_content0 = [], []
    col_role1, col_content1 = [], []
    col_role2, col_content2 = [], []

    for msgs in batch:
        col_role0.append(msgs[0]["role"] if len(msgs) > 0 else "")
        col_content0.append(msgs[0]["content"] if len(msgs) > 0 else "")
        col_role1.append(msgs[1]["role"] if len(msgs) > 1 else "")
        col_content1.append(msgs[1]["content"] if len(msgs) > 1 else "")
        col_role2.append(msgs[2]["role"] if len(msgs) > 2 else "")
        col_content2.append(msgs[2]["content"] if len(msgs) > 2 else "")

    table = pa.table({
        "role_0": pa.array(col_role0, type=pa.string()),
        "content_0": pa.array(col_content0, type=pa.string()),
        "role_1": pa.array(col_role1, type=pa.string()),
        "content_1": pa.array(col_content1, type=pa.string()),
        "role_2": pa.array(col_role2, type=pa.string()),
        "content_2": pa.array(col_content2, type=pa.string()),
    })
    pq.write_table(table, out_path, compression="snappy")

def phase2_upload():
    """Upload local folder to HuggingFace using upload_folder (fewer commits)."""
    print("Phase 2: Uploading to HuggingFace")
    print(f"  Repo: {REPO_ID}")
    print(f"  Local: {LOCAL_DIR}\n")

    api = HfApi(token=HF_TOKEN)

    # Ensure repo exists
    create_repo(
        repo_id=REPO_ID,
        repo_type="dataset",
        private=False,
        token=HF_TOKEN,
        exist_ok=True,
    )

    # Count files to upload
    data_dir = os.path.join(LOCAL_DIR, "data")
    parquet_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".parquet")])
    print(f"  Files to upload: {len(parquet_files)}")

    # upload_large_folder: designed for large repos, batches into multiple commits,
    # skips already-uploaded files by hash, handles rate limits automatically
    # token is passed via HfApi instance initialization above
    api.upload_large_folder(
        repo_id=REPO_ID,
        repo_type="dataset",
        folder_path=LOCAL_DIR,
    )

    print(f"\nDone! Dataset available at:")
    print(f"  https://huggingface.co/datasets/{REPO_ID}")

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    if mode in ("all", "convert"):
        phase1_convert()

    if mode in ("all", "upload"):
        phase2_upload()
