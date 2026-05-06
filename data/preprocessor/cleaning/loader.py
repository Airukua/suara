import logging
import time
log = logging.getLogger("LLM-Cleaner")
_COMMON_TEXT_COLUMNS = ("text", "content", "body", "document", "passage", "context")

def _detect_format(input_path: str) -> str:
    if input_path.endswith(".parquet") or "parquet" in input_path:
        return "parquet"
    if input_path.endswith((".jsonl", ".json")):
        return "jsonl"
    if input_path.endswith((".csv", ".tsv")):
        return "csv"
    return "text"

def _read_dataframe(spark, input_path: str, file_format: str, text_column: str):
    if file_format == "parquet":
        return spark.read.parquet(input_path)

    if file_format == "jsonl":
        return spark.read.json(input_path)

    if file_format == "csv":
        return (
            spark.read
            .option("header",      "true")
            .option("inferSchema", "true")
            .option("multiLine",   "true")
            .option("escape",      '"')
            .csv(input_path)
        )

    if file_format == "text":
        df = spark.read.text(input_path)
        return df.withColumnRenamed("value", text_column)

    raise ValueError(f"Format tidak didukung: {file_format}")


def _resolve_text_column(df, text_column: str) -> str:
    if text_column in df.columns:
        return text_column

    fallback = next(
        (col for col in df.columns if col in _COMMON_TEXT_COLUMNS),
        None,
    )
    if fallback:
        log.warning(f"Kolom teks '{text_column}' tidak ditemukan. Auto-detect: '{fallback}'")
        return fallback

    raise ValueError(
        f"Kolom teks '{text_column}' tidak ditemukan. "
        f"Kolom yang tersedia: {df.columns}"
    )


def load_data(
    spark,
    input_path:      str,
    file_format:     str   = "auto",
    text_column:     str   = "text",
    sample_fraction: float = None,
):
    log.info(f"Loading data dari: {input_path}")
    start = time.time()

    if file_format == "auto":
        file_format = _detect_format(input_path)

    df = _read_dataframe(spark, input_path, file_format, text_column)

    if sample_fraction:
        df = df.sample(fraction=sample_fraction, seed=42)
        log.info(f"Sample fraction: {sample_fraction}")

    text_column = _resolve_text_column(df, text_column)
    elapsed = time.time() - start
    count   = df.count()
    log.info(f"Loaded {count:,} records dalam {elapsed:.1f}s")
    return df, text_column
