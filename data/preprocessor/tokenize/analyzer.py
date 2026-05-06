import json
import logging
import os
from pyspark.sql import functions as F
log = logging.getLogger("LLM-Tokenizer")

_MODEL_SIZES = {
    "7B":  7e9,
    "13B": 13e9,
    "70B": 70e9,
}

_A100_TFLOPS = 312e12

def analyze_tokenization(
    spark,
    tokenized_path: str,
    output_dir:     str = None,
) -> dict:

    log.info("Analyzing tokenization output...")
    df = spark.read.parquet(tokenized_path)
    row = df.agg(
        F.count("input_ids").alias("total_sequences"),
        F.sum("num_tokens").alias("total_tokens"),
        F.avg("num_tokens").alias("avg_tokens"),
        F.min("num_tokens").alias("min_tokens"),
        F.max("num_tokens").alias("max_tokens"),
        F.expr("percentile(num_tokens, 0.25)").alias("p25"),
        F.expr("percentile(num_tokens, 0.50)").alias("p50"),
        F.expr("percentile(num_tokens, 0.75)").alias("p75"),
        F.expr("percentile(num_tokens, 0.95)").alias("p95"),
    ).collect()[0]

    total_tokens = row["total_tokens"] or 0

    _print_report(row, total_tokens)

    analysis = {
        "total_sequences": row["total_sequences"],
        "total_tokens":    total_tokens,
        "total_tokens_B":  round(total_tokens / 1e9, 3),
        "avg_tokens":      round(row["avg_tokens"], 1),
        "min_tokens":      row["min_tokens"],
        "max_tokens":      row["max_tokens"],
        "p25_tokens":      row["p25"],
        "p50_tokens":      row["p50"],
        "p75_tokens":      row["p75"],
        "p95_tokens":      row["p95"],
    }

    if output_dir:
        _save_analysis(analysis, output_dir)

    return analysis


# ─────────────────────────────────────────────────────────────────────────── #
# Helper privat                                                                #
# ─────────────────────────────────────────────────────────────────────────── #

def _print_report(row, total_tokens: int) -> None:
    print("\n" + "═" * 60)
    print("  TOKENIZATION ANALYSIS")
    print("═" * 60)
    print(f"  Total Sequences   : {row['total_sequences']:,}")
    print(f"  Total Tokens      : {total_tokens:,}")
    print(f"  Total Tokens (B)  : {total_tokens / 1e9:.3f} B")
    print(f"  Avg Tokens/Seq    : {row['avg_tokens']:.0f}")
    print(f"  Min / Max         : {row['min_tokens']} / {row['max_tokens']}")
    print(f"  P25/P50/P75/P95   : "
          f"{row['p25']:.0f} / {row['p50']:.0f} / "
          f"{row['p75']:.0f} / {row['p95']:.0f}")

    print()
    print("  Estimated Training Compute (Chinchilla rule: 6 × params × tokens):")
    for name, params in _MODEL_SIZES.items():
        flops      = 6 * params * total_tokens
        pflops     = flops / 1e15
        a100_days  = flops / (_A100_TFLOPS * 86_400)
        print(f"    Model {name:>4}: {pflops:,.0f} PFLOPs "
              f"| ~{a100_days:.0f} A100-days")

    print("═" * 60)


def _save_analysis(analysis: dict, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "analysis.json")
    with open(path, "w") as f:
        json.dump(analysis, f, indent=2)
    log.info(f"Analysis disimpan ke: {path}")
