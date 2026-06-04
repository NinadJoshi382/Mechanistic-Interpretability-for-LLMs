"""
Poor Man's Configurator. Probably a terrible idea. Example usage:
$ python train.py config/override_file.py --batch_size=32
this will first run config/override_file.py, then override batch_size to 32

The code in this file will be run as follows from e.g. train.py:
>>> exec(open('configurator.py').read())

So it's not a Python module, it's just shuttling this code away from train.py.
The code in this script then overrides the globals() of the calling script.

Keys recognised by this configurator come from train.py's config block:
  I/O          : out_dir, eval_interval, log_interval, eval_iters,
                 eval_only, always_save_checkpoint, init_from
  wandb        : wandb_log, wandb_project, wandb_run_name
  data         : dataset_root   <-- replaces the old 'dataset' key
  training     : gradient_accumulation_steps, batch_size, block_size
  model        : n_layer, n_head, n_embd, dropout, bias
  optimiser    : learning_rate, max_iters, weight_decay, beta1, beta2, grad_clip
  lr schedule  : decay_lr, warmup_iters, lr_decay_iters, min_lr
  DDP          : backend
  system       : device, dtype, compile
  checkpointing: checkpoint_interval  <-- new; set to 0 to disable periodic saves
"""

import sys
import os
from ast import literal_eval

for arg in sys.argv[1:]:
    if '=' not in arg:
        # Tokens without '=' are either a config file path or a bare flag
        # injected by torchrun / the shell (e.g. the script name itself).
        # Skip anything that starts with '--' so we never crash on launcher
        # arguments that don't belong to us.
        if arg.startswith('--'):
            continue
        config_file = arg
        # Resolve relative to cwd so the user can pass either an absolute path
        # or a path relative to where they launched train.py from.
        config_file = os.path.realpath(config_file)
        print(f"Overriding config with {config_file}:")
        with open(config_file) as f:
            print(f.read())
        exec(open(config_file).read())
    else:
        # Tokens with '=' that don't start with '--' (e.g. env vars forwarded
        # by some launchers) are not ours — skip them silently.
        if not arg.startswith('--'):
            continue

        # split only on the first '=' so values like --name=a=b work fine
        key, val = arg.split('=', 1)
        key = key[2:]  # strip the leading '--'

        if key not in globals():
            # Show all valid keys so the user knows what is available
            valid = sorted(
                k for k, v in globals().items()
                if not k.startswith('_') and isinstance(v, (int, float, bool, str))
            )
            raise ValueError(
                f"Unknown config key: '{key}'\n"
                f"Valid keys are: {valid}"
            )

        expected_type = type(globals()[key])

        try:
            attempt = literal_eval(val)
        except (SyntaxError, ValueError):
            # Could not parse as a Python literal — treat as a plain string.
            attempt = val

        # Relax the strict type-identity check from the original configurator.
        # The original type(attempt) == type(globals()[key]) rejected valid cases:
        #   --learning_rate=1   (int literal, but key is float)
        #   --min_lr=0          (int literal 0, but key is float 6e-5)
        # We allow int -> float coercion and keep strict checking for everything else.
        if expected_type == float and isinstance(attempt, int):
            attempt = float(attempt)

        if not isinstance(attempt, expected_type):
            raise TypeError(
                f"Config key '{key}' expects type {expected_type.__name__} "
                f"but received {type(attempt).__name__} (value: {val!r}).\n"
                f"Hint: booleans must be True/False (e.g. --compile=False). "
                f"Strings are inferred automatically without quotes."
            )

        print(f"Overriding: {key} = {attempt}")
        globals()[key] = attempt
