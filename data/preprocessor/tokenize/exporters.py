import logging
import os
import struct

log = logging.getLogger("LLM-Tokenizer")

def export_parquet(df, output_path: str) -> None:
    out = f"{output_path}/tokenized.parquet"
    df.write.mode("overwrite").option("compression", "snappy").parquet(out)
    log.info(f"Saved Parquet: {out}")


def export_jsonl(df, output_path: str) -> None:
    out = f"{output_path}/tokenized.jsonl"
    df.select("input_ids").write.mode("overwrite").json(out)
    log.info(f"Saved JSONL: {out}")


def export_binary(df, output_path: str, vocab_size: int) -> None:
    log.info("Exporting ke binary format (.bin)...")
    use_uint16 = vocab_size < 65_535
    dtype      = "<H" if use_uint16 else "<I" 

    bin_path = os.path.join(output_path, "tokenized.bin")
    token_count = 0
    with open(bin_path, "wb") as f:
        for row in df.select("input_ids").toLocalIterator():
            for token_id in row["input_ids"]:
                f.write(struct.pack(dtype, token_id))
                token_count += 1

    size_gb  = os.path.getsize(bin_path) / (1024**3)
    dtype_str = "uint16" if use_uint16 else "uint32"
    log.info(f"Binary export: {bin_path} ({size_gb:.2f} GB, {dtype_str}, {token_count:,} tokens)")


def export_arrow(df, output_path: str) -> None:
    log.info("Exporting ke Arrow format (HuggingFace datasets)...")
    arrow_path = os.path.join(output_path, "hf_dataset")
    parquet_path = os.path.join(output_path, "tokenized_for_hf.parquet")
    try:
        from datasets import Dataset

        df.select("input_ids", "attention_mask").write.mode("overwrite").parquet(parquet_path)
        hf_dataset = Dataset.from_parquet(parquet_path)
        hf_dataset.save_to_disk(arrow_path)
        log.info(f"Arrow export: {arrow_path}")
    except ImportError:
        log.warning("Library 'datasets' tidak ditemukan. Fallback ke Parquet.")
        df.write.mode("overwrite").parquet(
            os.path.join(output_path, "tokenized_fallback.parquet")
        )


def save_output(df, output_path: str, output_format: str,
                vocab_size: int = 32_000) -> None:

    os.makedirs(output_path, exist_ok=True)

    dispatch = {
        "parquet": lambda: export_parquet(df, output_path),
        "jsonl":   lambda: export_jsonl(df, output_path),
        "bin":     lambda: export_binary(df, output_path, vocab_size),
        "arrow":   lambda: export_arrow(df, output_path),
    }

    handler = dispatch.get(output_format)
    if handler is None:
        raise ValueError(
            f"Format tidak dikenal: '{output_format}'. "
            f"Pilihan: {list(dispatch.keys())}"
        )
    handler()
