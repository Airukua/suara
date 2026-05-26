import logging
import os
import time
from typing import List
import psutil
import sentencepiece as spm
from pyspark.sql import functions as F
from tokenizers import Tokenizer, trainers
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.decoders import WordPiece as WordPieceDecoder
from tokenizers.models import BPE, WordPiece
from tokenizers.normalizers import BertNormalizer
from tokenizers.pre_tokenizers import BertPreTokenizer, ByteLevel, Whitespace
from transformers import PreTrainedTokenizerFast

log = logging.getLogger("LLM-Tokenizer")

_DEFAULT_SPECIAL_TOKENS = [
    "<pad>", "<unk>", "<s>", "</s>",
    "[INST]", "[/INST]",
    "[SYS]", "[/SYS]",
    "<|user|>", "<|assistant|>", "<|system|>",
    "[URL]", "[EMAIL]", "[PHONE]", "[NIK]",
]

_SENTENCEPIECE_META_TOKENS = {"<pad>", "<unk>", "<s>", "</s>"}


class TokenizerTrainer:
    def __init__(self, model_dir: str = "./tokenizer_model"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

    def train_sentencepiece(
        self,
        input_files: List[str],
        vocab_size: int = 32_000,
        model_type: str = "bpe",
        character_coverage: float = 0.9995,
        model_prefix: str = "sp_tokenizer",
        pad_id: int = 0,
        unk_id: int = 1,
        bos_id: int = 2,
        eos_id: int = 3,
        user_defined_symbols: List[str] = None,
        input_sentence_size: int = 5_000_000,
        shuffle_input_sentence: bool = True,
        num_threads: int = None,
        byte_fallback: bool = True,
        normalization_rule_name: str = "nmt_nfkc_cf",
    ) -> str:
        if num_threads is None:
            num_threads = psutil.cpu_count(logical=True)

        extra = user_defined_symbols or []
        all_symbols = list(dict.fromkeys(_DEFAULT_SPECIAL_TOKENS + extra))
        sp_user_symbols = [sym for sym in all_symbols if sym not in _SENTENCEPIECE_META_TOKENS]
        model_path = os.path.join(self.model_dir, model_prefix)

        log.info(f"Training SentencePiece {model_type.upper()}...")
        log.info(f"  Vocab size    : {vocab_size:,}")
        log.info(f"  Input files   : {len(input_files)}")
        log.info(f"  Max sentences : {input_sentence_size:,}")
        log.info(f"  Threads       : {num_threads}")
        log.info(f"  Byte fallback : {byte_fallback}")
        log.info(f"  User symbols  : {len(sp_user_symbols)}")

        start = time.time()
        spm.SentencePieceTrainer.train(
            input=",".join(input_files),
            model_prefix=model_path,
            vocab_size=vocab_size,
            model_type=model_type,
            character_coverage=character_coverage,
            pad_id=pad_id,
            unk_id=unk_id,
            bos_id=bos_id,
            eos_id=eos_id,
            user_defined_symbols=",".join(sp_user_symbols),
            input_sentence_size=input_sentence_size,
            shuffle_input_sentence=shuffle_input_sentence,
            num_threads=num_threads,
            byte_fallback=byte_fallback,
            normalization_rule_name=normalization_rule_name,
            split_digits=True,
            allow_whitespace_only_pieces=False,
            remove_extra_whitespaces=True,
            seed_sentencepiece_size=1_000_000,
            shrinking_factor=0.75,
            max_sentence_length=8192,
            train_extremely_large_corpus=True,
        )
        elapsed = time.time() - start

        model_file = f"{model_path}.model"
        log.info(f"SentencePiece selesai dalam {elapsed:.1f}s")
        log.info(f"  Model : {model_file}")
        log.info(f"  Vocab : {model_path}.vocab")

        _verify_sentencepiece(model_file)
        return model_file

    def train_hf_bpe(
        self,
        input_files: List[str],
        vocab_size: int = 32_000,
        model_prefix: str = "hf_bpe_tokenizer",
        min_frequency: int = 2,
        special_tokens: List[str] = None,
        pretokenizer: str = "byte_level",
    ) -> str:
        extra = special_tokens or []
        all_specials = list(dict.fromkeys(_DEFAULT_SPECIAL_TOKENS + extra))

        log.info("Training HuggingFace BPE (Rust backend)...")
        log.info(f"  Vocab size    : {vocab_size:,}")
        log.info(f"  Pretokenizer  : {pretokenizer}")
        log.info(f"  Min frequency : {min_frequency}")

        tokenizer = Tokenizer(BPE(unk_token="<unk>"))

        if pretokenizer == "byte_level":
            tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
            tokenizer.decoder = ByteLevelDecoder()
        elif pretokenizer == "whitespace":
            tokenizer.pre_tokenizer = Whitespace()
        elif pretokenizer == "bert":
            tokenizer.pre_tokenizer = BertPreTokenizer()

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=all_specials,
            show_progress=True,
            continuing_subword_prefix="" if pretokenizer == "byte_level" else "##",
        )

        start = time.time()
        tokenizer.train(input_files, trainer)
        elapsed = time.time() - start

        model_path = os.path.join(self.model_dir, model_prefix)
        os.makedirs(model_path, exist_ok=True)
        tokenizer.save(os.path.join(model_path, "tokenizer.json"))

        _save_as_pretrained(tokenizer, model_path)

        log.info(f"HF BPE selesai dalam {elapsed:.1f}s -> {model_path}")
        _verify_hf_tokenizer(tokenizer, "HF BPE")
        return model_path

    def train_hf_wordpiece(
        self,
        input_files: List[str],
        vocab_size: int = 30_522,
        model_prefix: str = "hf_wordpiece_tokenizer",
        min_frequency: int = 2,
        special_tokens: List[str] = None,
    ) -> str:
        default_bert_specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        extra = special_tokens or []
        all_specials = list(dict.fromkeys(default_bert_specials + extra))

        tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
        tokenizer.normalizer = BertNormalizer(lowercase=True)
        tokenizer.pre_tokenizer = BertPreTokenizer()
        tokenizer.decoder = WordPieceDecoder()

        trainer = trainers.WordPieceTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=all_specials,
            show_progress=True,
            continuing_subword_prefix="##",
        )

        log.info("Training HF WordPiece...")
        start = time.time()
        tokenizer.train(input_files, trainer)
        elapsed = time.time() - start

        model_path = os.path.join(self.model_dir, model_prefix)
        os.makedirs(model_path, exist_ok=True)
        tokenizer.save(os.path.join(model_path, "tokenizer.json"))

        log.info(f"WordPiece selesai dalam {elapsed:.1f}s -> {model_path}")
        return model_path

    @staticmethod
    def extract_corpus_for_training(
        spark,
        input_path: str,
        output_txt: str,
        text_column: str = "text",
        max_lines: int | None = 5_000_000,
        sample_fraction: float = None,
    ) -> str:
        log.info("Extracting corpus untuk training tokenizer...")
        log.info(f"  Source    : {input_path}")
        log.info(f"  Target    : {output_txt}")
        log.info(f"  Max lines : {max_lines:,}" if max_lines is not None else "  Max lines : unlimited")

        if "parquet" in input_path or input_path.endswith(".parquet"):
            df = spark.read.parquet(input_path)
        else:
            df = spark.read.json(input_path)

        if sample_fraction:
            df = df.sample(fraction=sample_fraction, seed=42)

        texts_df = df.select(text_column).filter(F.col(text_column).isNotNull())
        if max_lines is not None:
            texts_df = texts_df.limit(max_lines)

        os.makedirs(os.path.dirname(output_txt) or ".", exist_ok=True)
        written = 0
        with open(output_txt, "w", encoding="utf-8") as file:
            for text in texts_df.rdd.map(lambda row: row[0]).toLocalIterator():
                line = text.replace("\n", " ").strip()
                if line:
                    file.write(line + "\n")
                    written += 1

        log.info(f"  Ditulis {written:,} baris ke {output_txt}")
        return output_txt


def _verify_sentencepiece(model_file: str) -> None:
    processor = spm.SentencePieceProcessor()
    processor.load(model_file)

    test = "Ini adalah contoh kalimat untuk uji tokenizer."
    log.info(f"  Verifikasi: '{test}'")
    log.info(f"  Tokens : {processor.encode(test, out_type=str)}")
    log.info(f"  IDs    : {processor.encode(test)}")


def _verify_hf_tokenizer(tokenizer: Tokenizer, label: str) -> None:
    test = "Ini adalah contoh kalimat untuk uji tokenizer BPE."
    encoded = tokenizer.encode(test)
    log.info(f"  Verifikasi {label}: '{test}'")
    log.info(f"  Tokens : {encoded.tokens[:15]}...")
    log.info(f"  IDs    : {encoded.ids[:15]}...")


def _save_as_pretrained(tokenizer: Tokenizer, model_path: str) -> None:
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=os.path.join(model_path, "tokenizer.json"),
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
    )
    fast_tokenizer.save_pretrained(model_path)
    log.info(f"  Disimpan sebagai PreTrainedTokenizerFast: {model_path}")
