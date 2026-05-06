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

    all_ids = (
        df.select("input_ids")
          .rdd
          .flatMap(lambda row: row["input_ids"])
          .collect()
    )

    bin_path = os.path.join(output_path, "tokenized.bin")
    with open(bin_path, "wb") as f:
        for token_id in all_ids:
            f.write(struct.pack(dtype, token_id))

    size_gb  = os.path.getsize(bin_path) / (1024**3)
    dtype_str = "uint16" if use_uint16 else "uint32"
    log.info(f"Binary export: {bin_path} ({size_gb:.2f} GB, {dtype_str})")


def export_arrow(df, output_path: str) -> None:
    log.info("Exporting ke Arrow format (HuggingFace datasets)...")
    arrow_path = os.path.join(output_path, "hf_dataset")
    try:
        from datasets import Dataset

        pandas_df  = df.select("input_ids", "attention_mask").toPandas()
        hf_dataset = Dataset.from_pandas(pandas_df)
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
