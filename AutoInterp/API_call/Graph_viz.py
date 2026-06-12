"""
run_attribution_and_viz.py
--------------------------
Standalone script to compute attribution graphs and launch the visual interface.

Modes:
    python run_attribution_and_viz.py --mode attribute --prompt "Once upon a time"
    python run_attribution_and_viz.py --mode visualize
    python run_attribution_and_viz.py --mode both --prompt "Once upon a time"

All arguments:
    --mode          : attribute | visualize | both  (required)
    --prompt        : input prompt for attribution graph (required for attribute/both)
    --graph_name    : filename for saved graph  (default: attribution_graph.pt)
    --max_n_logits  : max number of output logits to consider (default: 10)
    --logit_prob    : desired logit probability threshold (default: 0.95)
    --max_features  : max feature nodes in graph (default: 8192)
    --batch_size    : batch size for attribution (default: 256)
    --feat_threshold: node pruning threshold (default: 0.8)
    --edge_threshold: edge pruning threshold (default: 0.95)
    --no_intervene  : skip intervention step (default: False)
    --port          : port for visual interface (default: 8106)
    --host          : host for visual interface (default: 0.0.0.0)
    --debug_attr    : enable debug mode for attribution (default: False)

Example:
    python run_attribution_and_viz.py --mode both --prompt "The opposite of large is"
    python run_attribution_and_viz.py --mode attribute --prompt "Once upon a time" --max_n_logits 5
    python run_attribution_and_viz.py --mode visualize --port 8080
"""

import argparse
import os
import sys
import traceback

# ── Confirm script is alive immediately ───────────────────────────────────────
print("=" * 60)
print("run_attribution_and_viz.py — starting")
print(f"Python executable : {sys.executable}")
print(f"Working directory : {os.getcwd()}")
print("=" * 60)
sys.stdout.flush()

# ── Import torch ───────────────────────────────────────────────────────────────
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

# ── CONFIG — fill these in ────────────────────────────────────────────────────
CLT_CKPT_PATH        = "./clt_checkpoints/final_5001216"   # <-- your CLT checkpoint dir
ATTRIBUTION_GRAPH_DIR= "./attribution_graphs"               # <-- where graphs are saved
AUTOINTERP_DIR       = "./autointerp_results"               # <-- autointerp parquet dir
MODEL_NAME           = "gpt2"                               # <-- model name for AttributionRunner
# ─────────────────────────────────────────────────────────────────────────────


def run_attribution(args):
    """Compute and save an attribution graph for the given prompt."""
    print("\n" + "=" * 60)
    print("STEP: ATTRIBUTION GRAPH")
    print(f"  Prompt          : {args.prompt!r}")
    print(f"  CLT checkpoint  : {CLT_CKPT_PATH}")
    print(f"  Save directory  : {ATTRIBUTION_GRAPH_DIR}")
    print(f"  Graph filename  : {args.graph_name}")
    print(f"  Max logits      : {args.max_n_logits}")
    print(f"  Logit prob      : {args.logit_prob}")
    print(f"  Max features    : {args.max_features}")
    print(f"  Batch size      : {args.batch_size}")
    print(f"  Feat threshold  : {args.feat_threshold}")
    print(f"  Edge threshold  : {args.edge_threshold}")
    print(f"  Run interventions: {not args.no_intervene}")
    print("=" * 60)
    sys.stdout.flush()

    # Import AttributionRunner
    print("Importing AttributionRunner...", end=" ")
    sys.stdout.flush()
    try:
        from clt_forge.attribution.attribution import AttributionRunner
        print("OK")
    except Exception as e:
        print(f"FAILED\n{e}")
        traceback.print_exc()
        sys.exit(1)
    sys.stdout.flush()

    # Check CLT checkpoint exists
    if not os.path.exists(CLT_CKPT_PATH):
        print(f"ERROR: CLT checkpoint not found at {CLT_CKPT_PATH}")
        sys.exit(1)

    # Initialise AttributionRunner
    print("Initialising AttributionRunner (loading CLT + model)...")
    sys.stdout.flush()
    try:
        runner = AttributionRunner(
            clt_checkpoint = CLT_CKPT_PATH,
            model_name     = MODEL_NAME,
            device         = DEVICE,
            debug          = args.debug_attr,
        )
        print("AttributionRunner initialised.")
    except Exception as e:
        print(f"ERROR during initialisation: {e}")
        traceback.print_exc()
        sys.exit(1)
    sys.stdout.flush()

    # Run attribution
    print(f"\nComputing attribution graph for prompt: {args.prompt!r}")
    print("This may take a few minutes...")
    sys.stdout.flush()
    try:
        os.makedirs(ATTRIBUTION_GRAPH_DIR, exist_ok=True)
        result = runner.run(
            input_string       = args.prompt,
            folder_name        = ATTRIBUTION_GRAPH_DIR,
            graph_name         = args.graph_name,
            max_n_logits       = args.max_n_logits,
            desired_logit_prob = args.logit_prob,
            max_feature_nodes  = args.max_features,
            batch_size         = args.batch_size,
            offload            = "cpu",
            verbose            = True,
            feature_threshold  = args.feat_threshold,
            edge_threshold     = args.edge_threshold,
            run_interventions  = not args.no_intervene,
        )
    except Exception as e:
        print(f"ERROR during attribution: {e}")
        traceback.print_exc()
        sys.exit(1)

    graph_path = os.path.join(ATTRIBUTION_GRAPH_DIR, args.graph_name)
    print("\n" + "=" * 60)
    print("Attribution graph complete!")
    print(f"  Saved to         : {graph_path}")
    print(f"  Input tokens     : {result['token_string']}")
    print(f"  Top logit tokens : {result['logit_token_strings'][:5]}")
    print(f"  Features in graph: {result['feature_indices'].shape[0]}")
    print("=" * 60)
    sys.stdout.flush()

    return graph_path


def run_visualization(graph_path, args):
    """Launch the Dash visual interface."""
    print("\n" + "=" * 60)
    print("STEP: VISUAL INTERFACE")
    print(f"  Graph path      : {graph_path}")
    print(f"  AutoInterp dir  : {AUTOINTERP_DIR}")
    print(f"  CLT checkpoint  : {CLT_CKPT_PATH}")
    print(f"  Host            : {args.host}")
    print(f"  Port            : {args.port}")
    print("=" * 60)
    sys.stdout.flush()

    # Import frontend
    print("Importing frontend...", end=" ")
    sys.stdout.flush()
    try:
        from clt_forge.frontend.app import main as launch_app
        from clt_forge.frontend.config.settings import AppConfig, GraphConfig
        print("OK")
    except Exception as e:
        print(f"FAILED\n{e}")
        traceback.print_exc()
        sys.exit(1)
    sys.stdout.flush()

    # Check graph file exists
    if not os.path.exists(graph_path):
        print(f"ERROR: Attribution graph not found at {graph_path}")
        print("Run --mode attribute first to generate the graph.")
        sys.exit(1)

    # Build AppConfig
    config = AppConfig(
        attr_graph_path                  = graph_path,
        dict_base_folder                 = AUTOINTERP_DIR,
        clt_checkpoint                   = CLT_CKPT_PATH,
        model_name                       = MODEL_NAME,
        model_class_name                 = "HookedTransformer",
        high_frequency_pruning_threshold = 0.25,
        host                             = args.host,
        port                             = args.port,
        debug                            = False,
        graph                            = GraphConfig(),
    )

    print(f"\nLaunching visual interface...")
    print(f"Open your browser at: http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop the server.")
    sys.stdout.flush()

    try:
        launch_app(config)
    except KeyboardInterrupt:
        print("\nServer stopped by user.")
    except Exception as e:
        print(f"ERROR launching interface: {e}")
        traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Compute attribution graphs and launch the visual interface.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_attribution_and_viz.py --mode attribute --prompt "The opposite of large is"
  python run_attribution_and_viz.py --mode visualize
  python run_attribution_and_viz.py --mode both --prompt "Once upon a time"
  python run_attribution_and_viz.py --mode both --prompt "Hello world" --max_n_logits 5 --port 8080
        """
    )

    # Mode
    parser.add_argument(
        "--mode",
        choices=["attribute", "visualize", "both"],
        required=True,
        help="attribute = compute graph | visualize = launch UI | both = compute then launch"
    )

    # Attribution arguments
    parser.add_argument("--prompt",         type=str,   default=None,
                        help="Input prompt for attribution graph (required for attribute/both)")
    parser.add_argument("--graph_name",     type=str,   default="attribution_graph.pt",
                        help="Filename for saved graph (default: attribution_graph.pt)")
    parser.add_argument("--max_n_logits",   type=int,   default=10,
                        help="Max number of output logits to consider (default: 10)")
    parser.add_argument("--logit_prob",     type=float, default=0.95,
                        help="Desired logit probability threshold (default: 0.95)")
    parser.add_argument("--max_features",   type=int,   default=8192,
                        help="Max feature nodes in graph (default: 8192)")
    parser.add_argument("--batch_size",     type=int,   default=256,
                        help="Batch size for attribution computation (default: 256)")
    parser.add_argument("--feat_threshold", type=float, default=0.8,
                        help="Node pruning threshold (default: 0.8)")
    parser.add_argument("--edge_threshold", type=float, default=0.95,
                        help="Edge pruning threshold (default: 0.95)")
    parser.add_argument("--no_intervene",   action="store_true",
                        help="Skip intervention step (faster but less complete)")
    parser.add_argument("--debug_attr",     action="store_true",
                        help="Enable debug mode for attribution runner")

    # Visualization arguments
    parser.add_argument("--port",   type=int, default=8106,
                        help="Port for visual interface (default: 8106)")
    parser.add_argument("--host",   type=str, default="0.0.0.0",
                        help="Host for visual interface (default: 0.0.0.0)")

    args = parser.parse_args()

    # Validate arguments
    if args.mode in ("attribute", "both") and args.prompt is None:
        print("ERROR: --prompt is required for --mode attribute or --mode both")
        print("Example: python run_attribution_and_viz.py --mode attribute --prompt \"Once upon a time\"")
        sys.exit(1)

    print(f"\nMode   : {args.mode}")
    if args.prompt:
        print(f"Prompt : {args.prompt!r}")
    sys.stdout.flush()

    graph_path = os.path.join(ATTRIBUTION_GRAPH_DIR, args.graph_name)

    if args.mode == "attribute":
        run_attribution(args)

    elif args.mode == "visualize":
        run_visualization(graph_path, args)

    elif args.mode == "both":
        run_attribution(args)
        run_visualization(graph_path, args)

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
        crash_log = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "attribution_crash_log.txt"
        )
        with open(crash_log, "w") as f:
            f.write(f"Error: {e}\n\n")
            traceback.print_exc(file=f)
        print(f"\nCrash log saved to: {crash_log}")
        sys.exit(1)
