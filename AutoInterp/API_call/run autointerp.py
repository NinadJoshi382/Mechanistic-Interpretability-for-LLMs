"""
run_autointerp.py
-----------------
Standalone AutoInterp script — run in terminal, NOT in Jupyter.

Usage:
    python run_autointerp.py --mode stream     # Step 1: find top activating examples
    python run_autointerp.py --mode explain    # Step 2: generate LLM descriptions
    python run_autointerp.py --mode both       # Both steps back to back

Fill in the CONFIG section before running.
"""

import argparse
import gc
import os
import pickle
import sys
import traceback
from pathlib import Path

# ── STEP 1: confirm script is alive ──────────────────────────────────────────
print("=" * 60)
print("run_autointerp.py — starting")
print(f"Python executable : {sys.executable}")
print(f"Python version    : {sys.version}")
print(f"Working directory : {os.getcwd()}")
print("=" * 60)
sys.stdout.flush()

# ── STEP 2: import torch ─────────────────────────────────────────────────────
print("Importing torch...", end=" ")
sys.stdout.flush()
try:
    import torch
    print(f"OK  (version {torch.__version__})")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
except Exception as e:
    print(f"FAILED\n{e}")
    sys.exit(1)
sys.stdout.flush()

# ── STEP 3: import transformer_lens ──────────────────────────────────────────
print("Importing transformer_lens...", end=" ")
sys.stdout.flush()
try:
    from transformer_lens import HookedTransformer, HookedTransformerConfig
    print("OK")
except Exception as e:
    print(f"FAILED\n{e}")
    sys.exit(1)
sys.stdout.flush()

# ── STEP 4: import clt_forge ─────────────────────────────────────────────────
print("Importing clt_forge...", end=" ")
sys.stdout.flush()
try:
    from clt_forge.autointerp.pipeline import AutoInterp
    from clt_forge.config import AutoInterpConfig
    print("OK")
except Exception as e:
    print(f"FAILED\n{e}")
    print("Make sure CLT-Forge is installed: pip install -e /path/to/CLT-Forge")
    sys.exit(1)
sys.stdout.flush()

print("All imports OK")
print("=" * 60)
sys.stdout.flush()

# ── CONFIG — fill these in ────────────────────────────────────────────────────
NANOGPT_CKPT_PATH   = "out/ckpt.pt"
CLT_CKPT_PATH       = "./clt_checkpoints/final_5001216"
AUTOINTERP_DIR      = "./autointerp_results"
DATASET_PATH        = "./hf_arrow_dataset"
DISK                = True

CONTEXT_SIZE        = 64
TOTAL_TOKENS        = 50_000       # keep small for first test
TOPK                = 5

VLLM_MODEL          = "gpt-4o"     # label only — set actual LLM in client.py
VLLM_MAX_TOKENS     = 200
# ─────────────────────────────────────────────────────────────────────────────


def convert_nanogpt_weights(sd, cfg):
    tl = {}
    d  = cfg.d_model
    nh = cfg.n_heads
    dh = cfg.d_head

    tl["embed.W_E"]       = sd["transformer.wte.weight"]
    tl["pos_embed.W_pos"] = sd["transformer.wpe.weight"]
    tl["ln_final.w"]      = sd["transformer.ln_f.weight"]
    tl["ln_final.b"]      = sd.get("transformer.ln_f.bias", torch.zeros(d))
    lm_w = sd.get("lm_head.weight", sd["transformer.wte.weight"])
    tl["unembed.W_U"] = lm_w.T
    tl["unembed.b_U"] = torch.zeros(cfg.d_vocab)

    for L in range(cfg.n_layers):
        p = f"transformer.h.{L}"
        tl[f"blocks.{L}.ln1.w"] = sd[f"{p}.ln_1.weight"]
        tl[f"blocks.{L}.ln1.b"] = sd.get(f"{p}.ln_1.bias", torch.zeros(d))

        c_attn_w = sd[f"{p}.attn.c_attn.weight"]
        c_attn_b = sd.get(f"{p}.attn.c_attn.bias", torch.zeros(3 * d))
        W_Q, W_K, W_V = c_attn_w.split(d, dim=0)
        b_Q, b_K, b_V = c_attn_b.split(d, dim=0)

        tl[f"blocks.{L}.attn.W_Q"] = W_Q.T.reshape(d, nh, dh).permute(1, 0, 2)
        tl[f"blocks.{L}.attn.W_K"] = W_K.T.reshape(d, nh, dh).permute(1, 0, 2)
        tl[f"blocks.{L}.attn.W_V"] = W_V.T.reshape(d, nh, dh).permute(1, 0, 2)
        tl[f"blocks.{L}.attn.b_Q"] = b_Q.reshape(nh, dh)
        tl[f"blocks.{L}.attn.b_K"] = b_K.reshape(nh, dh)
        tl[f"blocks.{L}.attn.b_V"] = b_V.reshape(nh, dh)

        c_proj_w = sd[f"{p}.attn.c_proj.weight"]
        c_proj_b = sd.get(f"{p}.attn.c_proj.bias", torch.zeros(d))
        tl[f"blocks.{L}.attn.W_O"] = c_proj_w.T.reshape(nh, dh, d)
        tl[f"blocks.{L}.attn.b_O"] = c_proj_b

        tl[f"blocks.{L}.ln2.w"] = sd[f"{p}.ln_2.weight"]
        tl[f"blocks.{L}.ln2.b"] = sd.get(f"{p}.ln_2.bias", torch.zeros(d))

        tl[f"blocks.{L}.mlp.W_in"]  = sd[f"{p}.mlp.c_fc.weight"].T
        tl[f"blocks.{L}.mlp.b_in"]  = sd.get(f"{p}.mlp.c_fc.bias", torch.zeros(cfg.d_mlp))
        tl[f"blocks.{L}.mlp.W_out"] = sd[f"{p}.mlp.c_proj.weight"].T
        tl[f"blocks.{L}.mlp.b_out"] = sd.get(f"{p}.mlp.c_proj.bias", torch.zeros(d))

    return tl


def load_nanogpt():
    print(f"\nLoading NanoGPT from: {NANOGPT_CKPT_PATH}")
    sys.stdout.flush()

    if not os.path.exists(NANOGPT_CKPT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {NANOGPT_CKPT_PATH}")

    checkpoint = torch.load(NANOGPT_CKPT_PATH, map_location="cpu", weights_only=False)
    model_args = checkpoint["model_args"]
    print(f"model_args: {model_args}")

    n_layer    = model_args["n_layer"]
    n_head     = model_args["n_head"]
    n_embd     = model_args["n_embd"]
    block_size = model_args["block_size"]

    raw_sd = checkpoint["model"]
    raw_sd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
              for k, v in raw_sd.items()}
    raw_sd = {(k[len("module."):] if k.startswith("module.") else k): v
              for k, v in raw_sd.items()}

    dataset   = checkpoint.get("config", {}).get("dataset", None)
    meta_path = os.path.join("data", dataset, "meta.pkl") if dataset else None

    if meta_path and os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        vocab_size        = meta["vocab_size"]
        tl_tokenizer_name = None
        print(f"Custom vocab: {vocab_size} tokens")
    else:
        vocab_size        = model_args.get("vocab_size", 50304)
        tl_tokenizer_name = "gpt2"
        print(f"GPT-2 BPE tokeniser, vocab size: {vocab_size}")

    tl_cfg = HookedTransformerConfig(
        n_layers           = n_layer,
        d_model            = n_embd,
        n_heads            = n_head,
        d_head             = n_embd // n_head,
        d_mlp              = 4 * n_embd,
        n_ctx              = block_size,
        d_vocab            = vocab_size,
        act_fn             = "gelu_new",
        normalization_type = "LN",
        tokenizer_name     = tl_tokenizer_name,
        device             = "cpu",
        attn_only          = False,
    )

    print("Converting weights...", end=" ")
    sys.stdout.flush()
    tl_state_dict = convert_nanogpt_weights(raw_sd, tl_cfg)
    print("OK")

    print(f"Building HookedTransformer...", end=" ")
    sys.stdout.flush()
    tl_model = HookedTransformer(tl_cfg)
    tl_model.load_state_dict(tl_state_dict, strict=False)
    tl_model.eval().to(DEVICE)
    for param in tl_model.parameters():
        param.requires_grad_(False)
    print(f"OK  (on {DEVICE})")

    print(f"NanoGPT ready | n_layer={n_layer}, n_embd={n_embd}, vocab={vocab_size}")
    sys.stdout.flush()
    return tl_model, n_layer, n_embd


def run_streaming(tl_model, n_layer, n_embd):
    print("\n" + "=" * 60)
    print("STEP: STREAMING")
    print(f"  Dataset        : {DATASET_PATH}")
    print(f"  Total tokens   : {TOTAL_TOKENS:,}")
    print(f"  Top-K          : {TOPK}")
    print(f"  Save to        : {AUTOINTERP_DIR}")
    print("=" * 60)
    sys.stdout.flush()

    os.makedirs(AUTOINTERP_DIR, exist_ok=True)

    cfg = AutoInterpConfig(
        device                   = DEVICE,
        dtype                    = "float32",
        model_name               = "gpt2",
        n_layers                 = n_layer,
        d_in                     = n_embd,
        use_pretrained_model     = True,
        clt_path                 = CLT_CKPT_PATH,
        dataset_path             = DATASET_PATH,
        disk                     = DISK,
        context_size             = CONTEXT_SIZE,
        total_autointerp_tokens  = TOTAL_TOKENS,
        train_batch_size_tokens  = 2048,
        n_batches_in_buffer      = 20,
        store_batch_size_prompts = 32,
        latent_cache_path        = AUTOINTERP_DIR,
        topk                     = TOPK,
        vllm_model               = VLLM_MODEL,
        vllm_max_tokens          = VLLM_MAX_TOKENS,
    )
    cfg._pretrained_model = tl_model

    print("Initialising AutoInterp...", end=" ")
    sys.stdout.flush()
    autointerp = AutoInterp(cfg)
    print("OK")
    sys.stdout.flush()

    print("Running streaming...")
    sys.stdout.flush()
    autointerp.run(
        job_id                = 0,
        total_jobs            = 1,
        save_dir              = Path(AUTOINTERP_DIR),
        generate_explanations = False,
    )
    print("\nStreaming complete. Parquet saved to:", AUTOINTERP_DIR)
    sys.stdout.flush()


def run_explanations():
    print("\n" + "=" * 60)
    print("STEP: EXPLANATIONS")
    print(f"  Reading parquet from : {AUTOINTERP_DIR}")
    print(f"  LLM model            : {VLLM_MODEL}")
    print("=" * 60)
    sys.stdout.flush()

    # Check parquet files exist before doing anything
    parquet_dir = Path(AUTOINTERP_DIR) / "parquet"
    if not parquet_dir.exists() or not list(parquet_dir.glob("*.parquet")):
        raise FileNotFoundError(
            f"No parquet files found in {parquet_dir}. "
            "Run --mode stream first."
        )

    cfg = AutoInterpConfig(
        device                   = DEVICE,
        dtype                    = "float32",
        model_name               = "gpt2",
        n_layers                 = 4,
        d_in                     = None,
        use_pretrained_model     = True,
        clt_path                 = CLT_CKPT_PATH,
        dataset_path             = DATASET_PATH,
        disk                     = DISK,
        context_size             = CONTEXT_SIZE,
        total_autointerp_tokens  = TOTAL_TOKENS,
        train_batch_size_tokens  = 2048,
        n_batches_in_buffer      = 20,
        store_batch_size_prompts = 32,
        latent_cache_path        = AUTOINTERP_DIR,
        topk                     = TOPK,
        vllm_model               = VLLM_MODEL,
        vllm_max_tokens          = VLLM_MAX_TOKENS,
    )

    # For explain-only we need a minimal model just for the tokeniser
    # Load on CPU — no forward passes happen during explanation
    print("Loading NanoGPT on CPU for tokeniser...", end=" ")
    sys.stdout.flush()
    tl_model, _, _ = load_nanogpt()
    tl_model.cpu()
    cfg._pretrained_model = tl_model
    print("OK")
    sys.stdout.flush()

    print("Initialising AutoInterp...", end=" ")
    sys.stdout.flush()
    autointerp = AutoInterp(cfg)
    print("OK")

    print("Generating explanations via LLM...")
    sys.stdout.flush()
    autointerp.run_explanations_only(
        job_id   = 0,
        save_dir = Path(AUTOINTERP_DIR),
    )
    print("Explanations complete.")
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description="Run AutoInterp for CLT features")
    parser.add_argument(
        "--mode",
        choices=["stream", "explain", "both"],
        required=True,
        help="stream | explain | both"
    )
    args = parser.parse_args()

    print(f"\nMode: {args.mode}")
    sys.stdout.flush()

    if args.mode in ("stream", "both"):
        tl_model, n_layer, n_embd = load_nanogpt()
        run_streaming(tl_model, n_layer, n_embd)

        if args.mode == "both":
            print("\nFreeing GPU before LLM...")
            tl_model.cpu()
            del tl_model
            gc.collect()
            torch.cuda.empty_cache()
            free_gb = (torch.cuda.get_device_properties(0).total_memory
                       - torch.cuda.memory_reserved(0)) / 1024**3
            print(f"VRAM free: {free_gb:.1f} GB")
            sys.stdout.flush()
            run_explanations()

    elif args.mode == "explain":
        run_explanations()

    print("\n" + "=" * 60)
    print("ALL DONE")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(0)
    except Exception as e:
        print("\n" + "=" * 60)
        print("CRASH — full traceback:")
        print("=" * 60)
        traceback.print_exc()
        # Save crash log
        crash_log = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "autointerp_crash_log.txt")
        with open(crash_log, "w") as f:
            f.write(f"Error: {e}\n\n")
            traceback.print_exc(file=f)
        print(f"\nCrash log saved to: {crash_log}")
        sys.exit(1)
