import json
import logging
import os
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path
import psutil
from data.configuration.preprocess_config import compute_spark_config
from data.preprocessor.cleaning.loader import load_data
from data.preprocessor.cleaning.udfs import register_cleaning_udfs
from pyspark.sql import functions as F
from tqdm.auto import tqdm

log = logging.getLogger("LLM-Cleaner")

RESUME_STAGE_SEQUENCE = [
    ("after_repetition", 10),
    ("after_toxic", 11),
    ("after_language", 12),
    ("after_dedup", 13),
]

def estimate_optimal_partitions(
    total_records:     int,
    avg_doc_size_kb:   float = 5.0,
    target_partition_mb: int = 128,
) -> int:
    total_size_mb = (total_records * avg_doc_size_kb) / 1024
    partitions    = max(1, int(total_size_mb / target_partition_mb))
    cores         = psutil.cpu_count(logical=True)
    return max(cores, partitions)


def _save_output(df, output_path: str, output_format: str, final_count: int) -> None:
    os.makedirs(output_path, exist_ok=True)

    if output_format == "parquet":
        df.write.mode("overwrite").option("compression", "snappy") \
          .parquet(f"{output_path}/cleaned_data.parquet")

    elif output_format == "jsonl":
        df.coalesce(max(1, final_count // 500_000)) \
          .write.mode("overwrite") \
          .json(f"{output_path}/cleaned_data.jsonl")

    elif output_format == "csv":
        df.write.mode("overwrite").option("header", "true") \
          .csv(f"{output_path}/cleaned_data.csv")

    log.info(f"Output disimpan ke: {output_path}")


def _log_separator(msg: str = "") -> None:
    log.info("=" * 60)
    if msg:
        log.info(f"  {msg}")


def _build_resume_key(
    input_path: str,
    text_column: str,
    file_format: str,
    output_format: str,
    target_languages: list | None,
    min_quality_score: float,
    min_chars: int,
    max_chars: int,
    min_words: int,
    url_mode: str,
    dedup_enabled: bool,
    pii_removal: bool,
    toxic_filter: bool,
    sample_fraction: float | None,
) -> dict:
    return {
        "input_path": str(Path(input_path).expanduser().resolve()),
        "text_column": text_column,
        "file_format": file_format,
        "output_format": output_format,
        "target_languages": sorted(target_languages or []),
        "min_quality_score": min_quality_score,
        "min_chars": min_chars,
        "max_chars": max_chars,
        "min_words": min_words,
        "url_mode": url_mode,
        "dedup_enabled": dedup_enabled,
        "pii_removal": pii_removal,
        "toxic_filter": toxic_filter,
        "sample_fraction": sample_fraction,
    }


def _checkpoint_root(output_path: str) -> Path:
    return Path(output_path) / "_cleaning_checkpoints"


def _stage_checkpoint_path(output_path: str, stage_name: str) -> Path:
    return _checkpoint_root(output_path) / stage_name


def _stage_metadata_path(output_path: str, stage_name: str) -> Path:
    return _stage_checkpoint_path(output_path, stage_name) / "_metadata.json"


def _save_stage_checkpoint(
    df,
    output_path: str,
    stage_name: str,
    resume_key: dict,
    stats: dict,
) -> None:
    stage_dir = _stage_checkpoint_path(output_path, stage_name)
    stage_dir.mkdir(parents=True, exist_ok=True)
    data_dir = stage_dir / "data"
    df.write.mode("overwrite").option("compression", "snappy").parquet(str(data_dir))
    metadata = {
        "stage_name": stage_name,
        "saved_at": datetime.now().isoformat(),
        "resume_key": resume_key,
        "stats": stats,
    }
    with open(_stage_metadata_path(output_path, stage_name), "w") as f:
        json.dump(metadata, f, indent=2)
    log.info(f"  Checkpoint fase '{stage_name}' disimpan ke: {data_dir}")


def _load_resume_checkpoint(spark, output_path: str, resume_key: dict):
    for stage_name, completed_step in reversed(RESUME_STAGE_SEQUENCE):
        metadata_path = _stage_metadata_path(output_path, stage_name)
        data_dir = _stage_checkpoint_path(output_path, stage_name) / "data"
        if not metadata_path.exists() or not data_dir.exists():
            continue
        try:
            with open(metadata_path) as f:
                metadata = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            log.warning(f"Gagal membaca metadata checkpoint {metadata_path}: {exc}")
            continue

        if metadata.get("resume_key") != resume_key:
            log.info(
                "Checkpoint fase '%s' diabaikan karena konfigurasi run berubah.",
                stage_name,
            )
            continue

        log.info(
            "Melanjutkan cleaning dari checkpoint fase '%s': %s",
            stage_name,
            data_dir,
        )
        return spark.read.parquet(str(data_dir)), metadata.get("stats", {}), completed_step

    return None, {}, 0


def run_cleaning_pipeline(
    spark,
    input_path:        str,
    output_path:       str,
    text_column:       str   = "text",
    file_format:       str   = "auto",
    output_format:     str   = "parquet",
    target_languages:  list  = None,
    min_quality_score: float = 0.25,
    min_chars:         int   = 40,
    max_chars:         int   = 1_000_000,
    min_words:         int   = 8,
    url_mode:          str   = "placeholder",
    dedup_enabled:     bool  = True,
    pii_removal:       bool  = True,
    toxic_filter:      bool  = False,
    save_stats:        bool  = True,
    checkpoint_dir:    str   = "/tmp/spark-checkpoints",
    sample_fraction:   float = None,
):
    spark.sparkContext.setCheckpointDir(checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    resume_key = _build_resume_key(
        input_path=input_path,
        text_column=text_column,
        file_format=file_format,
        output_format=output_format,
        target_languages=target_languages,
        min_quality_score=min_quality_score,
        min_chars=min_chars,
        max_chars=max_chars,
        min_words=min_words,
        url_mode=url_mode,
        dedup_enabled=dedup_enabled,
        pii_removal=pii_removal,
        toxic_filter=toxic_filter,
        sample_fraction=sample_fraction,
    )

    _log_separator("LLM DATA CLEANING PIPELINE STARTED")
    log.info(f"  Input  : {input_path}")
    log.info(f"  Output : {output_path}")
    _log_separator()

    total_start = time.time()
    progress = tqdm(total=14, desc="Cleaning", unit="step", leave=False)
    try:
        log.info("[2/14] Registering cleaning UDFs...")
        udfs = register_cleaning_udfs(spark)
        progress.update(1)

        stats_cache = {}
        df, stats_cache, resumed_step = _load_resume_checkpoint(spark, output_path, resume_key)

        initial_count = stats_cache.get("initial_count")
        after_boilerplate = stats_cache.get("after_boilerplate_filter")
        after_length = stats_cache.get("after_length_filter")
        after_char_ratio = stats_cache.get("after_char_ratio_filter")
        after_repetition = stats_cache.get("after_repetition_filter")
        after_toxic = stats_cache.get("after_toxic_filter")
        after_lang = stats_cache.get("after_language_filter")
        after_exact_dedup = stats_cache.get("after_exact_dedup")
        after_near_dedup = stats_cache.get("after_near_dedup")

        if resumed_step:
            progress.update(resumed_step - 1)

        if df is None:
            log.info("[1/14] Loading data...")
            df, text_column = load_data(spark, input_path, file_format, text_column, sample_fraction)
            initial_count = df.count()
            log.info(f"  Loaded: {initial_count:,} dokumen")
            progress.update(1)

            if "doc_id" not in df.columns:
                df = df.withColumn("doc_id", F.monotonically_increasing_id())

            log.info("[3/14] Fixing encoding (Unicode/Mojibake)...")
            df = df.withColumn(text_column, udfs["fix_encoding"](F.col(text_column)))
            progress.update(1)

            log.info("[4/14] Normalizing whitespace...")
            df = df.withColumn(text_column, udfs["normalize_whitespace"](F.col(text_column)))
            progress.update(1)

            log.info(f"[5/14] Handling URLs & emails (mode={url_mode})...")
            df = df.withColumn(
                text_column,
                udfs["handle_urls_emails"](F.col(text_column), F.lit(url_mode)),
            )
            progress.update(1)

            if pii_removal:
                log.info("[6/14] Removing PII...")
                df = df.withColumn(text_column, udfs["remove_pii"](F.col(text_column)))
            else:
                log.info("[6/14] PII removal SKIPPED")
            progress.update(1)

            log.info("[7/14] Filtering boilerplate...")
            df = df.filter(~udfs["is_boilerplate"](F.col(text_column)))
            after_boilerplate = df.count()
            log.info(f"  Setelah boilerplate filter: {after_boilerplate:,} "
                     f"(-{initial_count - after_boilerplate:,})")
            progress.update(1)

            log.info(f"[8/14] Filtering by length (min={min_chars}, max={max_chars})...")
            df = df.filter(
                udfs["passes_length_filter"](
                    F.col(text_column),
                    F.lit(min_chars), F.lit(max_chars), F.lit(min_words), F.lit(1),
                )
            )
            after_length = df.count()
            log.info(f"  Setelah length filter: {after_length:,} "
                     f"(-{after_boilerplate - after_length:,})")
            progress.update(1)

            log.info("[9/14] Filtering by character ratio...")
            df = df.filter(udfs["passes_char_ratio_filter"](F.col(text_column)))
            after_char_ratio = df.count()
            log.info(f"  Setelah char ratio filter: {after_char_ratio:,} "
                     f"(-{after_length - after_char_ratio:,})")
            progress.update(1)

            log.info("[10/14] Filtering excessive repetition...")
            df = df.filter(~udfs["has_excessive_repetition"](F.col(text_column)))
            after_repetition = df.count()
            log.info(f"  Setelah repetition filter: {after_repetition:,} "
                     f"(-{after_char_ratio - after_repetition:,})")
            df = df.checkpoint(eager=True)
            progress.update(1)
            _save_stage_checkpoint(
                df,
                output_path,
                "after_repetition",
                resume_key,
                {
                    "initial_count": initial_count,
                    "after_boilerplate_filter": after_boilerplate,
                    "after_length_filter": after_length,
                    "after_char_ratio_filter": after_char_ratio,
                    "after_repetition_filter": after_repetition,
                },
            )

        if resumed_step < 11:
            if toxic_filter:
                log.info("[11/14] Filtering toxic content...")
                df = df.filter(~udfs["has_toxic_content"](F.col(text_column)))
                after_toxic = df.count()
                log.info(f"  Setelah toxic filter: {after_toxic:,} "
                         f"(-{after_repetition - after_toxic:,})")
                df = df.checkpoint(eager=True)
                _save_stage_checkpoint(
                    df,
                    output_path,
                    "after_toxic",
                    resume_key,
                    {
                        "initial_count": initial_count,
                        "after_boilerplate_filter": after_boilerplate,
                        "after_length_filter": after_length,
                        "after_char_ratio_filter": after_char_ratio,
                        "after_repetition_filter": after_repetition,
                        "after_toxic_filter": after_toxic,
                    },
                )
            else:
                after_toxic = after_repetition
                log.info("[11/14] Toxic filter SKIPPED")
            progress.update(1)

        if resumed_step < 12:
            log.info("[12/14] Detecting languages...")
            df = df.withColumn("language", udfs["detect_language"](F.col(text_column)))

            if target_languages:
                log.info(f"  Memfilter bahasa: {target_languages}")
                df = df.filter(F.col("language").isin(target_languages))
                after_lang = df.count()
                log.info(f"  Setelah language filter: {after_lang:,}")
            else:
                after_lang = after_toxic
                log.info("  Language filter: DISABLED (semua bahasa dipertahankan)")

            df = df.checkpoint(eager=True)
            _save_stage_checkpoint(
                df,
                output_path,
                "after_language",
                resume_key,
                {
                    "initial_count": initial_count,
                    "after_boilerplate_filter": after_boilerplate,
                    "after_length_filter": after_length,
                    "after_char_ratio_filter": after_char_ratio,
                    "after_repetition_filter": after_repetition,
                    "after_toxic_filter": after_toxic,
                    "after_language_filter": after_lang,
                },
            )
            progress.update(1)

        if resumed_step < 13:
            if dedup_enabled:
                log.info("[13/14] Running deduplication (exact + near-duplicate)...")
                df, after_exact_dedup, after_near_dedup = _run_deduplication(df, text_column)
                _save_stage_checkpoint(
                    df,
                    output_path,
                    "after_dedup",
                    resume_key,
                    {
                        "initial_count": initial_count,
                        "after_boilerplate_filter": after_boilerplate,
                        "after_length_filter": after_length,
                        "after_char_ratio_filter": after_char_ratio,
                        "after_repetition_filter": after_repetition,
                        "after_toxic_filter": after_toxic,
                        "after_language_filter": after_lang,
                        "after_exact_dedup": after_exact_dedup,
                        "after_near_dedup": after_near_dedup,
                    },
                )
            else:
                after_exact_dedup = after_near_dedup = after_lang
                log.info("[13/14] Deduplication SKIPPED")
            progress.update(1)

        log.info(f"[14/14] Scoring & filtering quality (min={min_quality_score})...")
        df = df.withColumn("quality_score", udfs["compute_quality_score"](F.col(text_column)))
        df = df.filter(F.col("quality_score") >= min_quality_score)
        final_count = df.count()
        progress.update(1)
    finally:
        progress.close()

    total_elapsed = time.time() - total_start
    retention_rate = (final_count / initial_count * 100) if initial_count > 0 else 0

    _log_separator("PIPELINE COMPLETE")
    log.info(f"  Input  : {initial_count:,} dokumen")
    log.info(f"  Output : {final_count:,} dokumen")
    log.info(f"  Kept   : {retention_rate:.1f}%")
    log.info(f"  Time   : {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    _log_separator()

    _save_output(df, output_path, output_format, final_count)

    stats = {
        "pipeline_run_at":          datetime.now().isoformat(),
        "input_path":               input_path,
        "output_path":              output_path,
        "initial_count":            initial_count,
        "after_boilerplate_filter": after_boilerplate,
        "after_length_filter":      after_length,
        "after_char_ratio_filter":  after_char_ratio,
        "after_repetition_filter":  after_repetition,
        "after_toxic_filter":       after_toxic,
        "after_language_filter":    after_lang,
        "after_exact_dedup":        after_exact_dedup if dedup_enabled else None,
        "after_near_dedup":         after_near_dedup  if dedup_enabled else None,
        "final_count":              final_count,
        "retention_rate_pct":       round(retention_rate, 2),
        "elapsed_seconds":          round(total_elapsed, 1),
    }

    if save_stats:
        stats_path = f"{output_path}/pipeline_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        log.info(f"Stats disimpan ke: {stats_path}")

    return df, stats

def _run_deduplication(df, text_column: str):
    normalized_text = F.trim(F.regexp_replace(F.lower(F.col(text_column)), r"\s+", " "))
    df = df.withColumn("_normalized_text", normalized_text)
    before_dedup = df.count()
    exact_dedup_tmp_dir = tempfile.mkdtemp(prefix="suara_exact_dedup_", dir="/tmp")
    exact_dedup_tmp_path = os.path.join(exact_dedup_tmp_dir, "data")
    try:
        (
            df.withColumn("_doc_hash", F.sha2(F.col("_normalized_text"), 256))
              .repartition(F.col("_doc_hash"))
              .dropDuplicates(["_doc_hash"])
              .write.mode("overwrite")
              .parquet(exact_dedup_tmp_path)
        )
        exact_df = df.sparkSession.read.parquet(exact_dedup_tmp_path)
        after_exact_dedup = exact_df.count()
        log.info(f"  Setelah exact dedup: {after_exact_dedup:,} "
                 f"(-{before_dedup - after_exact_dedup:,} duplikat exact)")

        log.info("  Computing lightweight fingerprints untuk near-dedup...")
        canonical_text = F.trim(
            F.regexp_replace(
                F.regexp_replace(F.col("_normalized_text"), r"[^\p{L}\p{N}\s]+", " "),
                r"\s+",
                " ",
            )
        )
        df = (
            exact_df.withColumn("_canonical_text", canonical_text)
              .withColumn("_tokens", F.expr("filter(split(_canonical_text, ' '), x -> x <> '')"))
              .withColumn("_token_count", F.size(F.col("_tokens")))
              .withColumn("_length_bucket", F.floor(F.length(F.col("_canonical_text")) / F.lit(100)))
              .withColumn("_front_tokens", F.concat_ws(" ", F.slice(F.col("_tokens"), 1, 12)))
              .withColumn(
                  "_back_tokens",
                  F.concat_ws(" ", F.reverse(F.slice(F.reverse(F.col("_tokens")), 1, 12))),
              )
              .withColumn(
                  "_head_vocab",
                  F.concat_ws(" ", F.array_sort(F.array_distinct(F.slice(F.col("_tokens"), 1, 24)))),
              )
              .withColumn(
                  "_near_key",
                  F.sha2(
                      F.concat_ws(
                          "||",
                          F.col("_length_bucket").cast("string"),
                          F.col("_token_count").cast("string"),
                          F.col("_front_tokens"),
                          F.col("_back_tokens"),
                          F.col("_head_vocab"),
                      ),
                      256,
                  ),
              )
              .repartition(F.col("_near_key"))
              .dropDuplicates(["_near_key"])
              .drop(
                  "_tokens",
                  "_token_count",
                  "_length_bucket",
                  "_front_tokens",
                  "_back_tokens",
                  "_head_vocab",
                  "_canonical_text",
                  "_normalized_text",
                  "_doc_hash",
                  "_near_key",
              )
        )
        df = df.checkpoint(eager=True)
        after_near_dedup = df.count()
        log.info(f"  Setelah near-dedup: {after_near_dedup:,} "
                 f"(-{after_exact_dedup - after_near_dedup:,} near-duplikat)")

        return df, after_exact_dedup, after_near_dedup
    finally:
        shutil.rmtree(exact_dedup_tmp_dir, ignore_errors=True)

def process_in_batches(
    spark,
    input_paths:      list,
    output_base:      str,
    batch_size_hint:  int  = 50_000_000,
    **pipeline_kwargs,
):
    log.info(f"Batch processing {len(input_paths)} input paths...")
    all_stats = []

    for i, path in enumerate(tqdm(input_paths, desc="Cleaning batches", unit="batch", leave=False)):
        batch_output = f"{output_base}/batch_{i:04d}"
        log.info(f"\n{'='*50}")
        log.info(f"Batch {i+1}/{len(input_paths)}: {path}")
        log.info(f"{'='*50}")
        try:
            _, stats = run_cleaning_pipeline(
                spark=spark,
                input_path=path,
                output_path=batch_output,
                **pipeline_kwargs,
            )
            all_stats.append(stats)
        except Exception as e:
            log.error(f"Batch {i} gagal: {e}")
            continue

    log.info("\nMerging semua batch output...")
    merged_df    = spark.read.parquet(f"{output_base}/batch_*/cleaned_data.parquet")
    final_output = f"{output_base}/final_merged"
    merged_df.write.mode("overwrite").option("compression", "snappy").parquet(final_output)
    total_final = merged_df.count()
    log.info(f"Merged output: {total_final:,} records → {final_output}")
    return merged_df, all_stats
