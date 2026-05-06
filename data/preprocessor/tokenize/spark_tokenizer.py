import json
import logging
import os
import time
import psutil
import sentencepiece as spm
from pyspark.sql import functions as F
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, IntegerType, StructField, StructType
from tokenizers import Tokenizer as HFTokenizer
from transformers import AutoTokenizer
from tqdm.auto import tqdm
from data.preprocessor.tokenize.exporters import save_output

log = logging.getLogger("LLM-Tokenizer")


class SparkTokenizer:
    def __init__(
        self,
        spark,
        tokenizer_type: str,
        model_path: str,
        text_column: str = "text",
        context_length: int = 2048,
        add_bos: bool = True,
        add_eos: bool = True,
    ):
        self.spark = spark
        self.tokenizer_type = tokenizer_type
        self.model_path = model_path
        self.text_column = text_column
        self.context_length = context_length
        self.add_bos = add_bos
        self.add_eos = add_eos

        self._tok = self._load_tokenizer(tokenizer_type, model_path)
        self.vocab_size = self._get_vocab_size()
        log.info(f"Tokenizer loaded: {tokenizer_type} | vocab={self.vocab_size:,}")

    def _load_tokenizer(self, tok_type: str, model_path: str):
        if tok_type == "sentencepiece":
            tokenizer = spm.SentencePieceProcessor()
            tokenizer.load(model_path)
            return tokenizer
        if tok_type == "hf":
            return HFTokenizer.from_file(os.path.join(model_path, "tokenizer.json"))
        if tok_type == "pretrained":
            return AutoTokenizer.from_pretrained(model_path)
        raise ValueError(f"Tokenizer type tidak dikenal: {tok_type}")

    def _get_vocab_size(self) -> int:
        try:
            if self.tokenizer_type == "sentencepiece":
                return self._tok.get_piece_size()
            if self.tokenizer_type == "hf":
                return self._tok.get_vocab_size()
            if self.tokenizer_type == "pretrained":
                return len(self._tok)
        except Exception:
            return -1
        return -1

    def tokenize_dataset(
        self,
        input_path: str,
        output_path: str,
        output_format: str = "parquet",
        packing_strategy: str = "greedy",
        num_partitions: int = None,
        save_token_counts: bool = True,
    ) -> tuple:
        log.info("═" * 62)
        log.info("  DISTRIBUTED TOKENIZATION STARTED")
        log.info(f"  Input    : {input_path}")
        log.info(f"  Output   : {output_path}")
        log.info(f"  Strategy : {packing_strategy}")
        log.info(f"  Context  : {self.context_length} tokens")
        log.info("═" * 62)

        start = time.time()
        progress = tqdm(total=5, desc="Spark tokenize", unit="step", leave=False)
        try:
            df = self._load_input(input_path)
            df = self._repartition(df, num_partitions)
            progress.update(1)

            tok_config = self._make_broadcast_config()

            if packing_strategy == "greedy":
                tokenized_df = self._tokenize_greedy(df, tok_config)
            elif packing_strategy == "truncate":
                tokenized_df = self._tokenize_truncate(df, tok_config)
            else:
                tokenized_df = self._tokenize_none(df, tok_config)
            progress.update(1)

            try:
                stats_row = tokenized_df.agg(
                    F.count(F.lit(1)).alias("total_sequences"),
                    F.sum("num_tokens").alias("total_tokens"),
                ).collect()[0]
                total_sequences = stats_row["total_sequences"] or 0
                total_tokens = stats_row["total_tokens"] or 0
                elapsed = time.time() - start
                progress.update(1)

                stats = self._build_stats(
                    total_sequences,
                    total_tokens,
                    elapsed,
                    packing_strategy,
                )
                self._log_summary(stats)
                save_output(tokenized_df, output_path, output_format, self.vocab_size)
                progress.update(1)

                self._save_stats(stats, output_path)
                progress.update(1)
                return tokenized_df, stats
            finally:
                pass
        finally:
            progress.close()

    def _load_input(self, input_path: str):
        if "parquet" in input_path or input_path.endswith(".parquet"):
            return self.spark.read.parquet(input_path)
        if input_path.endswith((".jsonl", ".json")):
            return self.spark.read.json(input_path)
        return self.spark.read.parquet(input_path)

    def _repartition(self, df, num_partitions: int):
        if num_partitions:
            return df.repartition(num_partitions)
        default_parallelism = self.spark.sparkContext.defaultParallelism
        return df.repartition(max(1, default_parallelism))

    def _make_broadcast_config(self) -> object:
        return self.spark.sparkContext.broadcast(
            {
                "type": self.tokenizer_type,
                "model_path": self.model_path,
                "add_bos": self.add_bos,
                "add_eos": self.add_eos,
                "ctx_len": self.context_length,
            }
        )

    def _tokenize_greedy(self, df, tok_config_bc):
        context_length = self.context_length
        text_col = self.text_column

        schema = StructType(
            [
                StructField("input_ids", ArrayType(IntegerType()), False),
                StructField("attention_mask", ArrayType(IntegerType()), False),
                StructField("num_tokens", IntegerType(), False),
                StructField("num_docs_packed", IntegerType(), False),
            ]
        )

        def tokenize_and_pack_partition(rows):
            cfg = tok_config_bc.value
            encode, pad_id = _build_encoder(cfg)

            current_seq = []
            doc_count = 0

            for row in rows:
                text = getattr(row, text_col, None)
                if not text:
                    continue
                try:
                    ids = encode(str(text))
                except Exception:
                    continue

                if len(ids) > context_length:
                    for chunk_start in range(0, len(ids), context_length):
                        chunk = ids[chunk_start:chunk_start + context_length]
                        if len(chunk) < context_length // 4:
                            continue
                        pad_len = context_length - len(chunk)
                        padded = chunk + [pad_id] * pad_len
                        mask = [1] * len(chunk) + [0] * pad_len
                        yield (padded, mask, len(chunk), 1)
                    continue

                if len(current_seq) + len(ids) <= context_length:
                    current_seq.extend(ids)
                    doc_count += 1
                else:
                    if current_seq:
                        yield _flush(current_seq, context_length, pad_id, doc_count)
                    current_seq = ids
                    doc_count = 1

            if current_seq:
                yield _flush(current_seq, context_length, pad_id, doc_count)

        rdd = df.rdd.mapPartitions(tokenize_and_pack_partition)
        return self.spark.createDataFrame(rdd, schema)

    def _tokenize_truncate(self, df, tok_config_bc):
        @udf(returnType=ArrayType(IntegerType()))
        def tokenize_truncate_udf(text):
            if text is None:
                return None
            cfg = tok_config_bc.value
            try:
                encode, _ = _build_encoder(cfg)
                ids = encode(text)
                return ids[:cfg["ctx_len"]]
            except Exception:
                return None

        text_col = self.text_column
        return (
            df.withColumn("input_ids", tokenize_truncate_udf(F.col(text_col)))
            .filter(F.col("input_ids").isNotNull())
            .withColumn("num_tokens", F.size(F.col("input_ids")))
            .withColumn("attention_mask", F.expr("transform(input_ids, x -> 1)"))
        )

    def _tokenize_none(self, df, tok_config_bc):
        @udf(returnType=ArrayType(IntegerType()))
        def tokenize_full_udf(text):
            if text is None:
                return None
            cfg = tok_config_bc.value
            try:
                encode, _ = _build_encoder(cfg)
                return encode(text)
            except Exception:
                return None

        text_col = self.text_column
        return (
            df.withColumn("input_ids", tokenize_full_udf(F.col(text_col)))
            .filter(F.col("input_ids").isNotNull())
            .withColumn("num_tokens", F.size(F.col("input_ids")))
        )

    def _build_stats(
        self,
        total_sequences: int,
        total_tokens: int,
        elapsed: float,
        packing_strategy: str,
    ) -> dict:
        return {
            "tokenizer_type": self.tokenizer_type,
            "model_path": self.model_path,
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
            "packing_strategy": packing_strategy,
            "total_sequences": total_sequences,
            "total_tokens": total_tokens,
            "total_tokens_B": round(total_tokens / 1e9, 3),
            "elapsed_seconds": round(elapsed, 1),
            "tokens_per_second": round(total_tokens / elapsed) if elapsed > 0 else 0,
        }

    def _log_summary(self, stats: dict) -> None:
        log.info("═" * 62)
        log.info("  Tokenization COMPLETE")
        log.info(f"  Sequences : {stats['total_sequences']:,}")
        log.info(f"  Tokens    : {stats['total_tokens']:,} ({stats['total_tokens_B']:.2f}B)")
        log.info(f"  Speed     : {stats['tokens_per_second']:,} tok/s")
        log.info(
            f"  Time      : {stats['elapsed_seconds']:.0f}s "
            f"({stats['elapsed_seconds']/60:.1f} min)"
        )
        log.info("═" * 62)

    def _save_stats(self, stats: dict, output_path: str) -> None:
        stats_path = f"{output_path}/tokenization_stats.json"
        with open(stats_path, "w") as file:
            json.dump(stats, file, indent=2)
        log.info(f"Stats disimpan: {stats_path}")


def _build_encoder(cfg: dict):
    tok_type = cfg["type"]
    model_path = cfg["model_path"]

    if tok_type == "sentencepiece":
        tokenizer = spm.SentencePieceProcessor()
        tokenizer.load(model_path)
        pad_id = tokenizer.pad_id() if tokenizer.pad_id() >= 0 else 0

        def encode(text):
            ids = tokenizer.encode(text)
            if cfg["add_bos"]:
                ids = [tokenizer.bos_id()] + ids
            if cfg["add_eos"]:
                ids = ids + [tokenizer.eos_id()]
            return ids

    elif tok_type == "hf":
        tokenizer = HFTokenizer.from_file(os.path.join(model_path, "tokenizer.json"))
        bos_id = tokenizer.token_to_id("<s>") or 1
        eos_id = tokenizer.token_to_id("</s>") or 2
        pad_id = tokenizer.token_to_id("<pad>") or 0

        def encode(text):
            ids = tokenizer.encode(text).ids
            if cfg["add_bos"]:
                ids = [bos_id] + ids
            if cfg["add_eos"]:
                ids = ids + [eos_id]
            return ids

    elif tok_type == "pretrained":
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        pad_id = tokenizer.pad_token_id or 0

        def encode(text):
            return tokenizer.encode(text, add_special_tokens=True, truncation=False)

    else:
        raise ValueError(f"Tokenizer type tidak dikenal: {tok_type}")

    return encode, pad_id


def _flush(seq: list, context_length: int, pad_id: int, doc_count: int) -> tuple:
    pad_len = context_length - len(seq)
    padded = seq + [pad_id] * pad_len
    mask = [1] * len(seq) + [0] * pad_len
    return (padded, mask, len(seq), doc_count)
