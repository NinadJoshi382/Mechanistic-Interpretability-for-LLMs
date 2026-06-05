"""
prepare.py — converts input.txt into train.bin, val.bin, and meta.pkl
for use with nanoGPT (BPE / tiktoken / GPT-2 tokenizer).

Place this file in the same folder as your input.txt, then run:
    python prepare.py

Each test case in input.txt should be separated by <|endoftext|> so the
model learns the EOS boundary. If your input.txt does NOT already contain
<|endoftext|> separators, this script will treat the entire file as one
document and append a single EOS at the end — which is fine for a small
custom dataset but less ideal. Better practice: add <|endoftext|> after
every "cal: add(...)" block in your input.txt before running this script.

Outputs written to the same directory as this script:
    train.bin  — ~90% of tokens, stored as uint16 numpy array
    val.bin    — ~10% of tokens, stored as uint16 numpy array
    meta.pkl   — vocab size and tokenizer info (stoi/itos not needed for
                 BPE; we store enc name and vocab size for reference)
"""

import os
import pickle
import numpy as np
import tiktoken

# ── config ────────────────────────────────────────────────────────────────────
val_fraction = 0.1          # fraction of tokens reserved for validation
# ──────────────────────────────────────────────────────────────────────────────

# paths relative to this script
script_dir     = os.path.dirname(os.path.abspath(__file__))
input_path     = os.path.join(script_dir, 'input.txt')
train_bin_path = os.path.join(script_dir, 'train.bin')
val_bin_path   = os.path.join(script_dir, 'val.bin')
meta_pkl_path  = os.path.join(script_dir, 'meta.pkl')

# ── load GPT-2 BPE tokenizer ──────────────────────────────────────────────────
enc = tiktoken.get_encoding("gpt2")
# enc.eot_token == 50256  which is <|endoftext|> in the GPT-2 vocabulary
# this is our EOS marker — a real dedicated token, not raw characters
EOT = enc.eot_token          # 50256
VOCAB_SIZE = enc.max_token_value + 1  # 50257

print(f"Tokenizer  : GPT-2 BPE (tiktoken)")
print(f"EOS token  : <|endoftext|>  id={EOT}")
print(f"Vocab size : {VOCAB_SIZE}")

# ── read input.txt ─────────────────────────────────────────────────────────────
with open(input_path, 'r', encoding='utf-8') as f:
    raw = f.read()

print(f"\nRead {len(raw):,} characters from {input_path}")

# ── split into documents on <|endoftext|> ─────────────────────────────────────
# If the user has already placed <|endoftext|> between test cases, split on it.
# Each non-empty chunk becomes one document and gets the EOS token appended.
SEPARATOR = '<|endoftext|>'
documents = [doc.strip() for doc in raw.split(SEPARATOR) if doc.strip()]

if len(documents) <= 1:
    # No separators found — treat the whole file as one document
    print("No <|endoftext|> separators found — treating entire file as one document.")
    documents = [raw.strip()]

print(f"Documents  : {len(documents):,}")

# ── encode every document ─────────────────────────────────────────────────────
# encode_ordinary ignores any special tokens inside the text itself,
# then we manually append the EOT token at the end of each document.
all_ids = []
for doc in documents:
    ids = enc.encode_ordinary(doc)   # BPE-encode the text
    ids.append(EOT)                  # append <|endoftext|> as the EOS marker
    all_ids.extend(ids)

all_ids = np.array(all_ids, dtype=np.uint16)  # uint16 is fine since max id=50256 < 2**16

total_tokens = len(all_ids)
print(f"Total tokens after encoding : {total_tokens:,}")

# ── train / val split ─────────────────────────────────────────────────────────
split_idx  = int(total_tokens * (1 - val_fraction))
train_ids  = all_ids[:split_idx]
val_ids    = all_ids[split_idx:]

print(f"Train tokens : {len(train_ids):,}  ({100*(1-val_fraction):.0f}%)")
print(f"Val tokens   : {len(val_ids):,}  ({100*val_fraction:.0f}%)")

# ── write .bin files ──────────────────────────────────────────────────────────
train_ids.tofile(train_bin_path)
val_ids.tofile(val_bin_path)
print(f"\nWrote {train_bin_path}")
print(f"Wrote {val_bin_path}")

# ── write meta.pkl ────────────────────────────────────────────────────────────
# nanoGPT checks for meta.pkl to decide vocab size.
# For BPE mode we store the encoding name and vocab size.
# stoi / itos are not needed because tiktoken handles all encode/decode.
meta = {
    'vocab_size': VOCAB_SIZE,
    'tokenizer': 'gpt2',            # signals BPE mode to any downstream scripts
    'eot_token': EOT,
    'encoding': 'gpt2',
}
with open(meta_pkl_path, 'wb') as f:
    pickle.dump(meta, f)
print(f"Wrote {meta_pkl_path}")
print(f"\nAll done. vocab_size={VOCAB_SIZE}, eot_token={EOT}")
print("You can now point your nanoGPT config at this data directory.")
