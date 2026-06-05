"""
find_block_size.py

Analyzes your input.txt and recommends the optimal block_size for nanoGPT.

Usage:
    python find_block_size.py --input_file data/addition/input.txt

Optional flags:
    --percentile   float  Percentile of sequence lengths to cover (default: 100)
                          e.g. 99 means block_size covers 99% of sequences
    --pack          bool  Set True if prepare.py packs all text into one stream
                          (default: False — treats each line as its own sequence)
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# -----------------------------------------------------------------------------

def next_power_of_2(n):
    """Return the smallest power of 2 >= n."""
    power = 1
    while power < n:
        power *= 2
    return power


def analyze(input_file, percentile=100, packed=False):

    # --- 1. Read file ---------------------------------------------------------
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Could not find: {input_file}")

    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"\n{'='*60}")
    print(f"  File      : {input_file}")
    print(f"  Mode      : {'packed stream' if packed else 'per-line sequences'}")
    print(f"  Percentile: {percentile}%")
    print(f"{'='*60}\n")

    # --- 2. Compute sequence lengths ------------------------------------------
    if packed:
        # The whole file is one token stream; block_size is a sliding window over it.
        # What matters is: how many tokens are in the file total, and what window
        # size captures enough context to learn the pattern.
        lines = [l for l in text.split('\n') if l.strip()]
        seq_lengths = [len(l) for l in lines]
        print(f"[Packed mode] Treating each line as one example within a continuous stream.")
    else:
        # Each line is a separate sequence (typical for addition datasets).
        lines = [l for l in text.split('\n') if l.strip()]
        seq_lengths = [len(l) for l in lines]

    seq_lengths = np.array(seq_lengths)
    total_sequences = len(seq_lengths)

    # --- 3. Statistics --------------------------------------------------------
    min_len    = int(seq_lengths.min())
    max_len    = int(seq_lengths.max())
    mean_len   = float(seq_lengths.mean())
    median_len = float(np.median(seq_lengths))
    p90        = int(np.percentile(seq_lengths, 90))
    p95        = int(np.percentile(seq_lengths, 95))
    p99        = int(np.percentile(seq_lengths, 99))
    p100       = int(seq_lengths.max())

    print(f"  Total sequences : {total_sequences:,}")
    print(f"  Min length      : {min_len}")
    print(f"  Max length      : {max_len}")
    print(f"  Mean length     : {mean_len:.2f}")
    print(f"  Median length   : {median_len:.2f}")
    print(f"\n  Percentile breakdown:")
    print(f"    90th percentile : {p90}  chars")
    print(f"    95th percentile : {p95}  chars")
    print(f"    99th percentile : {p99}  chars")
    print(f"    100th (max)     : {p100} chars")

    # --- 4. Recommend block_size ----------------------------------------------
    target_len    = int(np.percentile(seq_lengths, percentile))
    raw_block     = target_len if not packed else target_len * 4  # pack: give more context
    recommended   = next_power_of_2(raw_block)

    # Also show power-of-2 recommendations for common percentiles
    print(f"\n  Recommended block_size (next power of 2 above percentile):")
    for p in [90, 95, 99, 100]:
        val  = int(np.percentile(seq_lengths, p))
        rec  = next_power_of_2(val if not packed else val * 4)
        flag = "  <-- covers ALL sequences" if p == 100 else ""
        flag = "  <-- recommended (packed)" if packed and p == 99 else flag
        print(f"    p{p:3d}: longest={val:4d} chars  →  block_size = {rec}{flag}")

    print(f"\n  ✅ FINAL RECOMMENDATION:")
    print(f"     block_size = {recommended}")
    print(f"     (covers {percentile}% of your sequences, "
          f"next power of 2 above {target_len} chars)\n")

    # --- 5. Unique character vocab check --------------------------------------
    unique_chars = sorted(set(text))
    vocab_size   = len(unique_chars)
    print(f"  Vocabulary info:")
    print(f"    Unique characters : {vocab_size}")
    print(f"    Characters        : {repr(''.join(unique_chars))}")
    print(f"    (This should match vocab_size in your meta.pkl)\n")

    # --- 6. Distribution of sequence lengths ----------------------------------
    length_counts = Counter(seq_lengths.tolist())
    sorted_lengths = sorted(length_counts.keys())

    print(f"  Sequence length distribution:")
    print(f"    {'Length':>8}  {'Count':>8}  {'% of data':>10}  Bar")
    print(f"    {'-'*55}")
    for length in sorted_lengths:
        count   = length_counts[length]
        pct     = 100.0 * count / total_sequences
        bar     = '█' * int(pct / 2)
        print(f"    {length:>8}  {count:>8}  {pct:>9.1f}%  {bar}")

    # --- 7. Save histogram plot -----------------------------------------------
    out_dir = os.path.dirname(input_file)
    plot_path = os.path.join(out_dir, 'sequence_length_distribution.png')

    plt.figure(figsize=(9, 5))
    plt.hist(seq_lengths, bins=range(min_len, max_len + 2), color='steelblue',
             edgecolor='white', align='left')
    plt.axvline(recommended, color='red', linestyle='--', linewidth=1.5,
                label=f'Recommended block_size = {recommended}')
    plt.axvline(max_len, color='orange', linestyle=':', linewidth=1.5,
                label=f'Max sequence length = {max_len}')
    plt.xlabel('Sequence length (characters)')
    plt.ylabel('Count')
    plt.title('Sequence Length Distribution\n'
              f'Recommended block_size = {recommended}  |  '
              f'Max length = {max_len}  |  '
              f'Vocab size = {vocab_size}')
    plt.legend()
    plt.grid(axis='y', alpha=0.4)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"  📊 Distribution plot saved to: {plot_path}\n")
    print('='*60)

    return recommended


# -----------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyze input.txt and recommend block_size for nanoGPT'
    )
    parser.add_argument(
        '--input_file',
        type=str,
        default='data/addition/input.txt',
        help='Path to your input.txt file (default: data/addition/input.txt)'
    )
    parser.add_argument(
        '--percentile',
        type=float,
        default=100,
        help='Sequence length percentile to target (default: 100 = cover all sequences)'
    )
    parser.add_argument(
        '--pack',
        action='store_true',
        default=False,
        help='Set if prepare.py packs all text into one continuous stream (default: False)'
    )
    args = parser.parse_args()

    analyze(
        input_file=args.input_file,
        percentile=args.percentile,
        packed=args.pack
    )
