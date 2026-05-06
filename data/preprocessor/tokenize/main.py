import logging
import os
from contextlib import suppress

from data.configuration.preprocess_config import create_spark_session
from data.preprocessor.system_info import DEFAULT_TOKENIZER_DEPENDENCIES, print_system_info
from data.preprocessor.tokenize.analyzer import analyze_tokenization
from data.preprocessor.tokenize.spark_tokenizer import SparkTokenizer
from data.preprocessor.tokenize.trainer import TokenizerTrainer
from tqdm.auto import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def run(
    input_path:              str,
    output_dir:              str,
    # Tokenizer
    tokenizer_mode:          str   = "train",          # "train" | "load"
    tokenizer_type:          str   = "sentencepiece",  # "sentencepiece" | "hf" | "pretrained"
    tokenizer_subtype:       str   = "bpe",            # "bpe" | "unigram" | "byte_level"
    pretrained_name:         str   = None,             # nama/path jika mode="load"
    vocab_size:              int   = 32_000,
    context_length:          int   = 2_048,
    # Tokenisasi
    packing_strategy:        str   = "greedy",
    output_format:           str   = "parquet",
    # Data
    text_column:             str   = "text",
    corpus_sample_fraction:  float = 0.05,             # 5% untuk training tokenizer
    max_corpus_lines:        int   = 5_000_000,
    add_bos:                 bool  = True,
    add_eos:                 bool  = True,
) -> dict:
    print("\n" + "═" * 62)
    print("  LLM FULL PIPELINE: CLEANING → TOKENIZE → EXPORT")
    print("═" * 62 + "\n")

    print_system_info(
        title="SYSTEM & DEPENDENCY CHECK",
        dependencies=DEFAULT_TOKENIZER_DEPENDENCIES,
        show_gpu=True,
    )
    total_steps = 4 if output_format == "parquet" else 3
    progress = tqdm(total=total_steps, desc="Tokenize pipeline", unit="step", leave=False)
    spark = create_spark_session("LLM-Full-Pipeline", workload="tokenize")
    progress.update(1)

    tok_dir    = os.path.join(output_dir, "tokenizer")
    tokens_dir = os.path.join(output_dir, "tokens")
    corpus_txt = os.path.join(tok_dir, "corpus_sample.txt")

    os.makedirs(tok_dir,    exist_ok=True)
    os.makedirs(tokens_dir, exist_ok=True)

    try:
        model_path = _prepare_tokenizer(
            spark=spark,
            tokenizer_mode=tokenizer_mode,
            tokenizer_type=tokenizer_type,
            tokenizer_subtype=tokenizer_subtype,
            pretrained_name=pretrained_name,
            vocab_size=vocab_size,
            input_path=input_path,
            corpus_txt=corpus_txt,
            tok_dir=tok_dir,
            text_column=text_column,
            corpus_sample_fraction=corpus_sample_fraction,
            max_corpus_lines=max_corpus_lines,
        )
        progress.update(1)

        sp_tok = SparkTokenizer(
            spark=spark,
            tokenizer_type=tokenizer_type,
            model_path=model_path,
            text_column=text_column,
            context_length=context_length,
            add_bos=add_bos,
            add_eos=add_eos,
        )

        _, tok_stats = sp_tok.tokenize_dataset(
            input_path=input_path,
            output_path=tokens_dir,
            output_format=output_format,
            packing_strategy=packing_strategy,
        )
        progress.update(1)

        if output_format == "parquet":
            analyze_tokenization(
                spark,
                tokenized_path=f"{tokens_dir}/tokenized.parquet",
                output_dir=tokens_dir,
            )
            progress.update(1)

        return tok_stats
    finally:
        progress.close()
        # If the JVM has already crashed, stopping Spark raises a secondary
        # connection error and hides the original failure.
        with suppress(Exception):
            spark.stop()

def _prepare_tokenizer(
    spark, tokenizer_mode, tokenizer_type, tokenizer_subtype,
    pretrained_name, vocab_size, input_path, corpus_txt, tok_dir,
    text_column, corpus_sample_fraction, max_corpus_lines,
) -> str:
    if tokenizer_mode == "load":
        model_path = pretrained_name or "gpt2"
        logging.getLogger("LLM-Tokenizer").info(
            f"Memuat pretrained tokenizer: {model_path}"
        )
        return model_path

    if tokenizer_mode != "train":
        raise ValueError("tokenizer_mode harus 'train' atau 'load'")

    trainer = TokenizerTrainer(model_dir=tok_dir)

    TokenizerTrainer.extract_corpus_for_training(
        spark, input_path, corpus_txt,
        text_column=text_column,
        max_lines=max_corpus_lines,
        sample_fraction=corpus_sample_fraction,
    )

    if tokenizer_type == "sentencepiece":
        return trainer.train_sentencepiece(
            input_files=[corpus_txt],
            vocab_size=vocab_size,
            model_type=tokenizer_subtype,
        )
    elif tokenizer_type == "hf":
        return trainer.train_hf_bpe(
            input_files=[corpus_txt],
            vocab_size=vocab_size,
            pretokenizer=tokenizer_subtype,
        )

    raise ValueError(f"Tidak bisa melatih tokenizer type: '{tokenizer_type}'")
