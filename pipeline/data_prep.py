import argparse
import json
import logging
import sys
from pathlib import Path

from datasets import DatasetDict, load_dataset
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.preprocessor.cleaning.main import run as run_cleaning
from data.preprocessor.tokenize.main import run as run_tokenize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

log = logging.getLogger("LLM-DataPrep")

_SUPPORTED_LOCAL_SUFFIXES = {
    ".parquet": "parquet",
    ".jsonl": "json",
    ".json": "json",
    ".csv": "csv",
    ".tsv": "csv",
    ".txt": "text",
    ".text": "text",
}

_TOKENIZER_SUBTYPE_COMPATIBILITY = {
    "sentencepiece": {"bpe", "unigram"},
    "hf": {"byte_level", "whitespace", "bert"},
    "pretrained": set(),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Entrypoint utama data preprocessing dengan mode clean, tokenize, atau full."
    )
    parser.add_argument("--stage", choices=["clean", "tokenize", "full"], default="full")
    parser.add_argument("--data-source", choices=["local", "huggingface"], default="local")
    parser.add_argument("--input-path", default="data/raw")
    parser.add_argument("--cleaned-output", default="data/cleaned")
    parser.add_argument("--token-output", default="data/tokens")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--hf-dataset", default=None)
    parser.add_argument("--hf-subset", default=None)
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--hf-cache-dir", default="data/cache/huggingface")
    parser.add_argument("--split-dataset", action="store_true")
    parser.add_argument("--train-ratio", type=float, default=0.98)
    parser.add_argument("--val-ratio", type=float, default=0.01)
    parser.add_argument("--test-ratio", type=float, default=0.01)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--target-languages", default="")
    parser.add_argument("--min-quality-score", type=float, default=0.25)
    parser.add_argument("--sample-fraction", type=float, default=None)
    parser.add_argument("--disable-dedup", action="store_true")
    parser.add_argument("--disable-pii-removal", action="store_true")
    parser.add_argument("--enable-toxic-filter", action="store_true")
    parser.add_argument("--tokenizer-mode", choices=["train", "load"], default="train")
    parser.add_argument(
        "--tokenizer-type",
        choices=["sentencepiece", "hf", "pretrained"],
        default="sentencepiece",
    )
    parser.add_argument(
        "--tokenizer-subtype",
        choices=["bpe", "unigram", "byte_level", "whitespace", "bert"],
        default="bpe",
        help=(
            "Subtype tokenizer saat mode train. "
            "sentencepiece: bpe|unigram, hf: byte_level|whitespace|bert"
        ),
    )
    parser.add_argument("--pretrained-name", default=None)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--packing-strategy", choices=["greedy", "truncate", "none"], default="greedy")
    parser.add_argument("--output-format", choices=["parquet", "jsonl", "bin", "arrow"], default="parquet")
    parser.add_argument("--corpus-sample-fraction", type=float, default=0.05)
    parser.add_argument("--max-corpus-lines", type=int, default=5_000_000)
    parser.add_argument("--no-bos", action="store_true")
    parser.add_argument("--no-eos", action="store_true")
    return parser


def _validate_split_args(args) -> None:
    if not args.split_dataset:
        return

    total = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total - 1.0) > 1e-9:
        raise ValueError("train/val/test ratio harus berjumlah 1.0")

    if min(args.train_ratio, args.val_ratio, args.test_ratio) < 0:
        raise ValueError("train/val/test ratio tidak boleh negatif")


def _validate_tokenizer_args(args) -> None:
    allowed_subtypes = _TOKENIZER_SUBTYPE_COMPATIBILITY[args.tokenizer_type]

    if args.tokenizer_mode == "load":
        if args.tokenizer_type == "pretrained" and not args.pretrained_name:
            raise ValueError("--pretrained-name wajib diisi saat --tokenizer-type pretrained")
        return

    if args.tokenizer_type == "pretrained":
        raise ValueError(
            "--tokenizer-type pretrained hanya didukung saat --tokenizer-mode load"
        )

    if args.tokenizer_subtype not in allowed_subtypes:
        allowed_list = ", ".join(sorted(allowed_subtypes))
        raise ValueError(
            f"--tokenizer-subtype '{args.tokenizer_subtype}' tidak cocok untuk "
            f"--tokenizer-type {args.tokenizer_type}. Pilihan yang valid: {allowed_list}"
        )


def _resolve_input_path(args) -> str:
    if args.data_source == "local":
        return args.input_path
    return _materialize_huggingface_dataset(
        dataset_name=args.hf_dataset,
        subset_name=args.hf_subset,
        split_name=args.hf_split,
        cache_dir=args.hf_cache_dir,
    )


def _materialize_huggingface_dataset(
    dataset_name: str,
    subset_name: str,
    split_name: str,
    cache_dir: str,
) -> str:
    if not dataset_name:
        raise ValueError("--hf-dataset wajib diisi saat --data-source huggingface")

    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    safe_dataset = dataset_name.replace("/", "__")
    safe_subset = (subset_name or "default").replace("/", "__")
    safe_split = split_name.replace("/", "__")
    output_dir = cache_root / f"{safe_dataset}__{safe_subset}__{safe_split}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "dataset.jsonl"

    if output_path.exists():
        log.info(f"Menggunakan cache Hugging Face dataset: {output_path}")
        return str(output_path)

    log.info(
        "Downloading dataset Hugging Face: %s subset=%s split=%s",
        dataset_name,
        subset_name or "default",
        split_name,
    )
    dataset = load_dataset(dataset_name, subset_name, split=split_name)
    if isinstance(dataset, DatasetDict):
        dataset = dataset[split_name]

    dataset.to_json(str(output_path), force_ascii=False)
    log.info(f"Dataset Hugging Face disimpan ke: {output_path}")
    return str(output_path)


def _detect_local_loader(input_path: str) -> tuple[str, list[str]]:
    path = Path(input_path)

    if path.is_file():
        suffix = path.suffix.lower()
        loader = _SUPPORTED_LOCAL_SUFFIXES.get(suffix)
        if not loader:
            raise ValueError(f"Format file lokal tidak didukung untuk split: {path}")
        return loader, [str(path)]

    if not path.exists():
        raise ValueError(f"Input path tidak ditemukan: {input_path}")

    files = []
    for suffix in _SUPPORTED_LOCAL_SUFFIXES:
        files.extend(sorted(path.rglob(f"*{suffix}")))

    if not files:
        raise ValueError(f"Tidak ada file data yang bisa di-split di: {input_path}")

    suffixes = {file.suffix.lower() for file in files}
    if len(suffixes) != 1:
        raise ValueError(f"Mixed file types belum didukung untuk split otomatis: {sorted(suffixes)}")

    suffix = next(iter(suffixes))
    return _SUPPORTED_LOCAL_SUFFIXES[suffix], [str(file) for file in files]


def _load_source_dataset(input_path: str):
    loader, data_files = _detect_local_loader(input_path)

    if loader == "csv":
        delimiter = "\t" if all(file.endswith(".tsv") for file in data_files) else ","
        return load_dataset(loader, data_files=data_files, split="train", delimiter=delimiter)

    return load_dataset(loader, data_files=data_files, split="train")


def _materialize_dataset_splits(args, resolved_input_path: str) -> dict[str, str]:
    split_root = Path(args.hf_cache_dir) / "splits"
    split_root.mkdir(parents=True, exist_ok=True)

    if args.data_source == "huggingface":
        dataset_name = args.hf_dataset
        subset_name = args.hf_subset
        split_name = args.hf_split
        safe_source = dataset_name.replace("/", "__")
        safe_subset = (subset_name or "default").replace("/", "__")
        base_name = f"{safe_source}__{safe_subset}__{split_name}"
    else:
        base_name = Path(resolved_input_path).stem.replace("/", "__")

    output_dir = split_root / base_name
    split_paths = {
        "train": output_dir / "train.jsonl",
        "validation": output_dir / "validation.jsonl",
        "test": output_dir / "test.jsonl",
    }

    if all(path.exists() for path in split_paths.values()):
        log.info(f"Menggunakan cache split dataset: {output_dir}")
        return {name: str(path) for name, path in split_paths.items()}

    if args.data_source == "huggingface":
        dataset = load_dataset(args.hf_dataset, args.hf_subset, split=args.hf_split)
        if isinstance(dataset, DatasetDict):
            dataset = dataset[args.hf_split]
    else:
        dataset = _load_source_dataset(resolved_input_path)

    train_split = dataset.train_test_split(
        test_size=1.0 - args.train_ratio,
        seed=args.split_seed,
    )
    train_dataset = train_split["train"]
    remainder = train_split["test"]

    if args.val_ratio == 0 and args.test_ratio == 0:
        validation_dataset = dataset.select([])
        test_dataset = dataset.select([])
    elif args.val_ratio == 0:
        validation_dataset = dataset.select([])
        test_dataset = remainder
    elif args.test_ratio == 0:
        validation_dataset = remainder
        test_dataset = dataset.select([])
    else:
        validation_fraction = args.val_ratio / (args.val_ratio + args.test_ratio)
        remainder_split = remainder.train_test_split(
            test_size=1.0 - validation_fraction,
            seed=args.split_seed,
        )
        validation_dataset = remainder_split["train"]
        test_dataset = remainder_split["test"]

    output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset.to_json(str(split_paths["train"]), force_ascii=False)
    validation_dataset.to_json(str(split_paths["validation"]), force_ascii=False)
    test_dataset.to_json(str(split_paths["test"]), force_ascii=False)
    log.info(f"Split dataset disimpan ke: {output_dir}")
    return {name: str(path) for name, path in split_paths.items()}


def _run_cleaning_stage(args, input_path: str, output_path: str, target_languages: list[str]) -> dict:
    _, stats = run_cleaning(
        input_path=input_path,
        output_path=output_path,
        target_languages=target_languages or None,
        min_quality_score=args.min_quality_score,
        dedup_enabled=not args.disable_dedup,
        pii_removal=not args.disable_pii_removal,
        toxic_filter=args.enable_toxic_filter,
        sample_fraction=args.sample_fraction,
    )
    return stats


def _run_tokenize_stage(args, input_path: str, output_path: str) -> dict:
    return run_tokenize(
        input_path=input_path,
        output_dir=output_path,
        tokenizer_mode=args.tokenizer_mode,
        tokenizer_type=args.tokenizer_type,
        tokenizer_subtype=args.tokenizer_subtype,
        pretrained_name=args.pretrained_name,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        packing_strategy=args.packing_strategy,
        output_format=args.output_format,
        text_column=args.text_column,
        corpus_sample_fraction=args.corpus_sample_fraction,
        max_corpus_lines=args.max_corpus_lines,
        add_bos=not args.no_bos,
        add_eos=not args.no_eos,
    )


def _resolve_trained_tokenizer_path(args, train_output_path: str) -> str:
    tokenizer_root = Path(train_output_path) / "tokenizer"
    if args.tokenizer_type == "sentencepiece":
        return str(tokenizer_root / "sp_tokenizer.model")
    if args.tokenizer_type == "hf":
        return str(tokenizer_root / "hf_bpe_tokenizer")
    raise ValueError("Auto-reuse tokenizer split hanya didukung untuk sentencepiece atau hf")


def main() -> None:
    args = build_parser().parse_args()
    _validate_split_args(args)
    _validate_tokenizer_args(args)

    target_languages = [lang.strip() for lang in args.target_languages.split(",") if lang.strip()]
    resolved_input_path = _resolve_input_path(args)

    if args.split_dataset:
        split_inputs = _materialize_dataset_splits(args, resolved_input_path)
    else:
        split_inputs = {"all": resolved_input_path}

    results = {}

    if args.stage in {"clean", "full"}:
        cleaning_results = {}
        cleaned_inputs = {}
        split_iterator = tqdm(split_inputs.items(), total=len(split_inputs), desc="Cleaning splits", unit="split", leave=False)
        for split_name, input_path in split_iterator:
            split_output_dir = Path(args.cleaned_output) / split_name if args.split_dataset else Path(args.cleaned_output)
            cleaning_results[split_name] = _run_cleaning_stage(
                args,
                input_path=input_path,
                output_path=str(split_output_dir),
                target_languages=target_languages,
            )
            cleaned_inputs[split_name] = str(split_output_dir / "cleaned_data.parquet")

        results["cleaning"] = cleaning_results if args.split_dataset else cleaning_results["all"]
        tokenize_inputs = cleaned_inputs
    else:
        tokenize_inputs = split_inputs

    if args.stage in {"tokenize", "full"}:
        tokenize_results = {}
        shared_tokenizer_path = args.pretrained_name
        train_split_name = "train" if args.split_dataset and "train" in tokenize_inputs else None

        split_iterator = tqdm(tokenize_inputs.items(), total=len(tokenize_inputs), desc="Tokenize splits", unit="split", leave=False)
        for split_name, input_path in split_iterator:
            split_output_dir = Path(args.token_output) / split_name if args.split_dataset else Path(args.token_output)

            tokenize_args = args
            if args.split_dataset and split_name != train_split_name:
                if args.tokenizer_mode == "train":
                    if not shared_tokenizer_path:
                        shared_tokenizer_path = _resolve_trained_tokenizer_path(
                            args,
                            str(Path(args.token_output) / train_split_name),
                        )
                    tokenize_results[split_name] = run_tokenize(
                        input_path=input_path,
                        output_dir=str(split_output_dir),
                        tokenizer_mode="load",
                        tokenizer_type=args.tokenizer_type,
                        tokenizer_subtype=args.tokenizer_subtype,
                        pretrained_name=shared_tokenizer_path,
                        vocab_size=args.vocab_size,
                        context_length=args.context_length,
                        packing_strategy=args.packing_strategy,
                        output_format=args.output_format,
                        text_column=args.text_column,
                        corpus_sample_fraction=args.corpus_sample_fraction,
                        max_corpus_lines=args.max_corpus_lines,
                        add_bos=not args.no_bos,
                        add_eos=not args.no_eos,
                    )
                    continue

            tokenize_results[split_name] = _run_tokenize_stage(
                tokenize_args,
                input_path=input_path,
                output_path=str(split_output_dir),
            )

            if args.split_dataset and split_name == train_split_name and args.tokenizer_mode == "train":
                shared_tokenizer_path = _resolve_trained_tokenizer_path(args, str(split_output_dir))

        results["tokenize"] = tokenize_results if args.split_dataset else tokenize_results["all"]

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
