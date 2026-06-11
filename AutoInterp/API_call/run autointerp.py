"""
run_autointerp.py

C:\path\to\python.exe run_autointerp.py --mode stream 2>&1 | Tee-Object -FilePath autointerp_log.txt

-----------------
Standalone script to run AutoInterp outside Jupyter.
Runs in two modes:
    python run_autointerp.py --mode stream       # Step 1: find top examples
    python run_autointerp.py --mode explain      # Step 2: generate LLM descriptions
    python run_autointerp.py --mode both         # Both steps together (needs more VRAM)

Usage:
    1. Fill in the CONFIG section below
    2. Open a terminal in your project folder
    3. Activate your Python environment
    4. Run: python run_autointerp.py --mode stream
    5. Then: python run_autointerp.py --mode explain
"""

import argparse
import gc
import os
import pickle
import sys
from pathlib import Path

import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

# ── CONFIG — fill these in ────────────────────────────────────────────────────
NANOGPT_CKPT_PATH   = "out/ckpt.pt"                        # your NanoGPT checkpoint
CLT_CKPT_PATH       = "./clt_checkpoints/final_5001216"    # your CLT checkpoint dir
CACHED_ACTIVATIONS  = "./cached_activations"               # cached activations dir
AUTOINTERP_DIR      = "./autointerp_results"               # where to save results
DATASET_PATH        = "./hf_arrow_dataset"                 # your Arrow dataset path
DISK                = True                                 # True = local dataset

# CLT training hyperparameters — must match what you trained with
EXPANSION_FACTOR    = 80
CONTEXT_SIZE        = 64

# Streaming settings
TOTAL_TOKENS        = 200_000    # reduce to 50_000 for a quick test
TOPK                = 5          # top K examples per feature

# LLM settings (only used in --mode explain or --mode both)
VLLM_MODEL          = "gpt-4o"   # label only — actual LLM set in client.py
VLLM_MAX_TOKENS     = 200

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ─────────────────────────────────────────────────────────────────────────────


def convert_nanogpt_weights(sd, cfg):
    """Convert NanoGPT state dict to TransformerLens format."""
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
        tl[f"blocks.{L}.mlp.b_in"]  = sd.get(f"{p}.mlp.c_fc.bias",   torch.zeros(cfg.d_mlp))
        tl[f"blocks.{L}.mlp.W_out"] = sd[f"{p}.mlp.c_proj.weight"].T
        tl[f"blocks.{L}.mlp.b_out"] = sd.get(f"{p}.mlp.c_proj.bias", torch.zeros(d))

    return tl


def load_nanogpt(ckpt_path, device):
    """Load NanoGPT checkpoint and convert to HookedTransformer."""
    print(f"Loading NanoGPT from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model_args = checkpoint["model_args"]
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
    else:
        vocab_size        = model_args.get("vocab_size", 50304)
        tl_tokenizer_name = "gpt2"

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

    tl_state_dict = convert_nanogpt_weights(raw_sd, tl_cfg)
    tl_model      = HookedTransformer(tl_cfg)
    tl_model.load_state_dict(tl_state_dict, strict=False)
    tl_model.eval().to(device)
    for p in tl_model.parameters():
        p.requires_grad_(False)

    print(f"NanoGPT loaded | layers={n_layer}, n_embd={n_embd}, vocab={vocab_size}")
    return tl_model, n_layer, n_embd, vocab_size, tl_tokenizer_name


def run_streaming(tl_model, n_layer, n_embd):
    """Run streaming — find top activating examples for each CLT feature."""
    from clt_forge.autointerp.pipeline import AutoInterp
    from clt_forge.config import AutoInterpConfig

    print("\n--- STREAMING ---")
    print(f"Total tokens    : {TOTAL_TOKENS:,}")
    print(f"Top-K examples  : {TOPK}")
    print(f"Device          : {DEVICE}")

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
    cfg._pretrained_model = tl_model.to(DEVICE)

    autointerp = AutoInterp(cfg)
    autointerp.run(
        job_id                = 0,
        total_jobs            = 1,
        save_dir              = Path(AUTOINTERP_DIR),
        generate_explanations = False,
    )
    print("Streaming complete. Parquet saved.")


def run_explanations():
    """Run LLM explanations on already-saved parquet files."""
    from clt_forge.autointerp.pipeline import AutoInterp
    from clt_forge.config import AutoInterpConfig

    print("\n--- EXPLANATIONS ---")
    print("Reading saved parquet and calling LLM...")

    # Minimal config for explanations — no model needed
    cfg = AutoInterpConfig(
        device                   = DEVICE,
        dtype                    = "float32",
        model_name               = "gpt2",
        n_layers                 = 4,           # not used during explain-only
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

    # Pass a dummy pretrained model flag — not used for explain-only
    cfg._pretrained_model = None

    autointerp = AutoInterp(cfg)
    autointerp.run_explanations_only(
        job_id   = 0,
        save_dir = Path(AUTOINTERP_DIR),
    )
    print("Explanations complete.")


def main():
    parser = argparse.ArgumentParser(description="Run AutoInterp for CLT features")
    parser.add_argument(
        "--mode",
        choices=["stream", "explain", "both"],
        required=True,
        help="stream = find top examples | explain = generate LLM descriptions | both = run together"
    )
    args = parser.parse_args()

    print(f"Mode   : {args.mode}")
    print(f"Device : {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    os.makedirs(AUTOINTERP_DIR, exist_ok=True)

    if args.mode in ("stream", "both"):
        tl_model, n_layer, n_embd, vocab_size, tl_tokenizer_name = load_nanogpt(
            NANOGPT_CKPT_PATH, DEVICE
        )
        run_streaming(tl_model, n_layer, n_embd)

        if args.mode == "both":
            # Free GPU before loading LLM
            print("\nFreeing GPU memory before loading LLM...")
            tl_model.cpu()
            del tl_model
            gc.collect()
            torch.cuda.empty_cache()
            free_gb = (torch.cuda.get_device_properties(0).total_memory
                       - torch.cuda.memory_reserved(0)) / 1024**3
            print(f"VRAM free: {free_gb:.1f} GB")
            run_explanations()

    elif args.mode == "explain":
        run_explanations()

    print("\nDone.")


# BEFORE — replace this
if __name__ == "__main__":
    main()

# AFTER — replace with this
if __name__ == "__main__":
    import traceback
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "autointerp_crash_log.txt")
    try:
        main()
    except Exception as e:
        error_msg = traceback.format_exc()
        print(f"\n{'='*60}")
        print(f"CRASH after {e}")
        print(error_msg)
        print(f"{'='*60}")
        with open(log_path, "w") as f:
            f.write(f"Error: {e}\n\n")
            f.write(error_msg)
        print(f"Full error saved to: {log_path}")
        sys.exit(1)
