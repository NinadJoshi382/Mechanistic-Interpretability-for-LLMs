"""
prepare.py — Tokenize input.txt into train.bin and val.bin
for nanoGPT character-level training.

Usage:
    python prepare.py

Outputs (written to the same directory as this script):
    train.bin   — 90% of tokens as uint16 numpy array
    val.bin     — 10% of tokens as uint16 numpy array
    meta.pkl    — vocab mappings (stoi, itos, vocab_size)
"""

import os
import pickle
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_FILE  = os.path.join(os.path.dirname(__file__), 'input.txt')
TRAIN_SPLIT = 0.9   # 90% train, 10% val

# ── Read raw text ─────────────────────────────────────────────────────────────
with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    data = f.read()

print(f"Total characters in dataset : {len(data):,}")

# ── Build character-level vocabulary ─────────────────────────────────────────
chars = sorted(list(set(data)))
vocab_size = len(chars)
print(f"Unique characters (vocab)   : {vocab_size}")
print(f"Vocabulary                  : {chars}")

stoi = {ch: i for i, ch in enumerate(chars)}   # char → index
itos = {i: ch for i, ch in enumerate(chars)}   # index → char

encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: ''.join([itos[i] for i in ids])

# ── Encode entire dataset ─────────────────────────────────────────────────────
all_ids = encode(data)
n       = len(all_ids)
split   = int(n * TRAIN_SPLIT)

train_ids = all_ids[:split]
val_ids   = all_ids[split:]

print(f"\nTotal tokens                : {n:,}")
print(f"Train tokens                : {len(train_ids):,}  ({TRAIN_SPLIT*100:.0f}%)")
print(f"Val tokens                  : {len(val_ids):,}  ({(1-TRAIN_SPLIT)*100:.0f}%)")

# ── Save .bin files ───────────────────────────────────────────────────────────
out_dir = os.path.dirname(__file__)

train_arr = np.array(train_ids, dtype=np.uint16)
val_arr   = np.array(val_ids,   dtype=np.uint16)

train_arr.tofile(os.path.join(out_dir, 'train.bin'))
val_arr.tofile(  os.path.join(out_dir, 'val.bin'))

# ── Save meta.pkl ─────────────────────────────────────────────────────────────
meta = {
    'vocab_size': vocab_size,
    'stoi':       stoi,
    'itos':       itos,
    'chars':      chars,
}
with open(os.path.join(out_dir, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print(f"\nFiles written to: {os.path.abspath(out_dir)}")
print(f"  train.bin  — {train_arr.nbytes:,} bytes")
print(f"  val.bin    — {val_arr.nbytes:,} bytes")
print(f"  meta.pkl   — vocab_size={vocab_size}")
print("\nDone! Ready for nanoGPT training.")
