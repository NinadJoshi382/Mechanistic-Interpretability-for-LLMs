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
from ast import literal_eval

for arg in sys.argv[1:]:
    if '=' not in arg:
        # assume it's the name of a config file
        assert not arg.startswith('--'), \
            f"Expected a config file path (no '--') but got: {arg}"
        config_file = arg
        print(f"Overriding config with {config_file}:")
        with open(config_file) as f:
            print(f.read())
        exec(open(config_file).read())
    else:
        # assume it's a --key=value argument
        assert arg.startswith('--'), \
            f"Expected '--key=value' syntax but got: {arg}"

        # split only on the first '=' so values like --name=a=b work fine
        key, val = arg.split('=', 1)
        key = key[2:]  # strip the leading '--'

        if key not in globals():
            # CHANGED: print all valid keys so the user knows what is available
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
            # could not parse as a Python literal — treat as a plain string
            attempt = val

        # CHANGED: relax the strict type identity check.
        # The original check (type(attempt) == type(globals()[key])) rejected
        # valid cases such as:
        #   --learning_rate=1   (int literal, but key is float)
        #   --min_lr=0          (int literal 0, but key is float 6e-5)
        # We now allow int -> float coercion and keep the strict check for
        # everything else (bool, str, remaining int keys).
        if expected_type == float and isinstance(attempt, int):
            # silently coerce: --learning_rate=1  ->  1.0
            attempt = float(attempt)

        if not isinstance(attempt, expected_type):
            raise TypeError(
                f"Config key '{key}' expects type {expected_type.__name__} "
                f"but received {type(attempt).__name__} (value: {val!r}).\n"
                f"Hint: booleans must be True/False, strings need no quotes on "
                f"the command line (they are inferred automatically)."
            )

        print(f"Overriding: {key} = {attempt}")
        globals()[key] = attempt
