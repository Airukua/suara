import argparse
import json
import logging
import hashlib
import sys
from pathlib import Path

import yaml
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.data_prep import _detect_local_loader, _resolve_trained_tokenizer_path
from data.preprocessor.cleaning.main import run as run_cleaning
from data.preprocessor.tokenize.main import run as run_tokenize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

log = logging.getLogger("LLM-ManifestDataPrep")

_TEXT_COLUMN_CANDIDATES = (
    "text",
    "content",
    "document",
    "article",
    "body",
    "sentence",
    "kalimat",
    "caption",
    "utterance",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pipeline preprocessing multi-dataset Hugging Face berbasis manifest."
    )
    parser.add_argument(
        "--manifest",
        default="data/configuration/hf_manifest.yaml",
        help="Path ke manifest YAML multi-dataset Hugging Face.",
    )
    return parser


def _load_manifest(path: str) -> dict:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest tidak ditemukan: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    if not isinstance(data, dict):
        raise ValueError("Manifest harus berupa mapping/dict YAML")

    datasets_cfg = data.get("datasets")
    if not isinstance(datasets_cfg, list) or not datasets_cfg:
        raise ValueError("Manifest harus memiliki daftar `datasets` yang tidak kosong")

    return data


def _slugify_dataset_name(name: str, subset: str | None, split: str) -> str:
    parts = [name.replace("/", "__")]
    if subset:
        parts.append(subset.replace("/", "__"))
    parts.append(split.replace("/", "__"))
    return "__".join(parts)


def _entry_signature(entry: dict) -> str:
    relevant = {
        "name": entry["name"],
        "subset": entry.get("subset"),
        "split": entry.get("split", "train"),
        "text_column": entry.get("text_column", "text"),
        "sample_fraction": entry.get("sample_fraction"),
        "max_samples": entry.get("max_samples"),
    }
    payload = json.dumps(relevant, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_cached_normalized_dataset(output_path: Path) -> Dataset:
    return load_dataset("json", data_files=str(output_path), split="train")


def _infer_text_column(dataset: Dataset, requested_column: str | None) -> str:
    columns = list(dataset.column_names)
    if not columns:
        raise ValueError("Dataset tidak memiliki kolom apa pun")

    normalized_map = {column.casefold(): column for column in columns}
    requested = (requested_column or "").strip()
    if requested:
        direct_match = normalized_map.get(requested.casefold())
        if direct_match:
            return direct_match

    for candidate in _TEXT_COLUMN_CANDIDATES:
        match = normalized_map.get(candidate.casefold())
        if match:
            return match

    string_columns = []
    for column, feature in dataset.features.items():
        dtype = getattr(feature, "dtype", None)
        if dtype in {"string", "large_string"}:
            string_columns.append(column)

    if len(string_columns) == 1:
        return string_columns[0]

    if requested:
        raise ValueError(
            f"Kolom teks '{requested}' tidak ditemukan. "
            f"Kolom tersedia: {columns}. "
            f"Coba isi `text_column` yang benar di manifest atau biarkan autodetect."
        )

    raise ValueError(
        "Gagal auto-detect kolom teks. "
        f"Kolom tersedia: {columns}. "
        "Isi `text_column` secara eksplisit di manifest."
    )


def _ensure_text_column(dataset: Dataset, text_column: str | None, source_name: str) -> tuple[Dataset, str]:
    resolved_text_column = _infer_text_column(dataset, text_column)
    if text_column and text_column.strip() and resolved_text_column != text_column:
        log.warning(
            "Kolom teks '%s' tidak ditemukan untuk %s. Menggunakan kolom hasil autodetect: '%s'",
            text_column,
            source_name,
            resolved_text_column,
        )

    if resolved_text_column != "text":
        dataset = dataset.rename_column(resolved_text_column, "text")

    keep_columns = {"text"}
    removable = [column for column in dataset.column_names if column not in keep_columns]
    if removable:
        dataset = dataset.remove_columns(removable)

    dataset = dataset.filter(lambda row: row["text"] is not None and str(row["text"]).strip() != "")
    return dataset, resolved_text_column


def _load_hf_dataset(entry: dict, cache_dir: str, normalized_cache_dir: Path) -> tuple[str, Dataset, dict]:
    dataset_name = entry["name"]
    subset_name = entry.get("subset")
    split_name = entry.get("split", "train")
    text_column = entry.get("text_column", "text")
    sample_fraction = entry.get("sample_fraction")
    max_samples = entry.get("max_samples")

    source_label = f"{dataset_name}:{subset_name or 'default'}:{split_name}"
    source_slug = _slugify_dataset_name(dataset_name, subset_name, split_name)
    output_path = normalized_cache_dir / f"{source_slug}.jsonl"
    meta_path = normalized_cache_dir / f"{source_slug}.meta.json"
    entry_signature = _entry_signature(entry)

    if output_path.exists() and meta_path.exists():
        try:
            cached_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cached_meta = None

        if cached_meta and cached_meta.get("entry_signature") == entry_signature:
            log.info("Menggunakan cache dataset ternormalisasi: %s", output_path)
            cached_dataset = _load_cached_normalized_dataset(output_path)
            return source_label, cached_dataset, cached_meta

    log.info("Loading HF dataset: %s", source_label)

    dataset = load_dataset(
        dataset_name,
        subset_name,
        split=split_name,
        cache_dir=cache_dir,
    )
    if isinstance(dataset, DatasetDict):
        dataset = dataset[split_name]

    dataset, resolved_text_column = _ensure_text_column(
        dataset,
        text_column=text_column,
        source_name=source_label,
    )

    if sample_fraction is not None:
        if sample_fraction <= 0 or sample_fraction > 1:
            raise ValueError(f"sample_fraction harus di rentang (0, 1], dapat: {sample_fraction}")
        dataset = dataset.shuffle(seed=42).select(range(max(1, int(len(dataset) * sample_fraction))))

    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError(f"max_samples harus > 0, dapat: {max_samples}")
        dataset = dataset.select(range(min(len(dataset), max_samples)))

    dataset.to_json(str(output_path), force_ascii=False)
    cache_meta = {
        "source": source_label,
        "entry_signature": entry_signature,
        "requested_text_column": text_column,
        "resolved_text_column": resolved_text_column,
        "records": len(dataset),
        "output_path": str(output_path),
    }
    meta_path.write_text(json.dumps(cache_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Saved normalized source cache: %s (%s records)", output_path, len(dataset))

    return source_label, dataset, cache_meta


def _materialize_raw_corpus(manifest: dict) -> tuple[str, dict]:
    output_cfg = manifest.get("output", {})
    cache_cfg = manifest.get("cache", {})
    skip_failed_datasets = manifest.get("skip_failed_datasets", True)

    raw_dir = Path(output_cfg.get("raw_merged_dir", "data/raw/merged_hf"))
    raw_dir.mkdir(parents=True, exist_ok=True)

    hf_cache_dir = cache_cfg.get("hf_cache_dir", "data/cache/huggingface")
    normalized_cache_dir = Path(cache_cfg.get("normalized_cache_dir", str(raw_dir)))
    normalized_cache_dir.mkdir(parents=True, exist_ok=True)
    per_source_stats = {}
    skipped_sources = {}
    datasets_to_merge = []
    source_signatures = {}

    for entry in manifest["datasets"]:
        if "name" not in entry:
            raise ValueError("Setiap item datasets wajib memiliki field `name`")

        source_label = (
            f"{entry['name']}:{entry.get('subset') or 'default'}:{entry.get('split', 'train')}"
        )
        try:
            source_label, dataset, cache_meta = _load_hf_dataset(
                entry,
                cache_dir=hf_cache_dir,
                normalized_cache_dir=normalized_cache_dir,
            )
        except Exception as exc:
            if not skip_failed_datasets:
                raise

            source_slug = _slugify_dataset_name(
                entry["name"],
                entry.get("subset"),
                entry.get("split", "train"),
            )
            skipped_sources[source_slug] = {
                "source": source_label,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            log.warning("Skipping dataset %s karena gagal dimuat: %s", source_label, exc)
            continue

        source_slug = _slugify_dataset_name(entry["name"], entry.get("subset"), entry.get("split", "train"))

        per_source_stats[source_slug] = {
            "source": source_label,
            "records": len(dataset),
            "output_path": cache_meta["output_path"],
            "resolved_text_column": cache_meta.get("resolved_text_column"),
        }
        source_signatures[source_slug] = cache_meta["entry_signature"]
        datasets_to_merge.append(dataset)

    if not datasets_to_merge:
        raise ValueError("Tidak ada dataset yang berhasil dimuat dari manifest")

    merged_path = raw_dir / "merged.jsonl"
    merged_meta_path = raw_dir / "merged.meta.json"
    current_merged_signature = hashlib.sha256(
        json.dumps(source_signatures, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()

    if merged_path.exists() and merged_meta_path.exists():
        try:
            merged_meta = json.loads(merged_meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            merged_meta = None

        if merged_meta and merged_meta.get("merged_signature") == current_merged_signature:
            log.info("Menggunakan cache merged corpus: %s", merged_path)
            merged_dataset = _load_cached_normalized_dataset(merged_path)
            return str(merged_path), {
                "raw_merged_path": str(merged_path),
                "raw_merged_records": len(merged_dataset),
                "sources": per_source_stats,
                "skipped_sources": skipped_sources,
            }

    merged = concatenate_datasets(datasets_to_merge) if len(datasets_to_merge) > 1 else datasets_to_merge[0]
    merged.to_json(str(merged_path), force_ascii=False)
    merged_meta_path.write_text(
        json.dumps(
            {
                "merged_signature": current_merged_signature,
                "source_signatures": source_signatures,
                "records": len(merged),
                "output_path": str(merged_path),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    log.info("Merged corpus disimpan ke: %s (%s records)", merged_path, len(merged))

    return str(merged_path), {
        "raw_merged_path": str(merged_path),
        "raw_merged_records": len(merged),
        "sources": per_source_stats,
        "skipped_sources": skipped_sources,
    }


def _build_stage_args(manifest: dict):
    stage_cfg = manifest.get("stage", {})
    output_cfg = manifest.get("output", {})
    split_cfg = manifest.get("split", {})
    cleaning_cfg = manifest.get("cleaning", {})
    token_cfg = manifest.get("tokenize", {})

    class Args:
        pass

    args = Args()
    args.data_source = "local"
    args.input_path = output_cfg.get("raw_merged_dir", "data/raw/merged_hf")
    args.cleaned_output = output_cfg.get("cleaned_dir", "data/cleaned/merged_hf")
    args.token_output = output_cfg.get("token_dir", "data/tokens/merged_hf")
    args.text_column = "text"
    args.hf_cache_dir = manifest.get("cache", {}).get("hf_cache_dir", "data/cache/huggingface")
    args.hf_dataset = None
    args.hf_subset = None
    args.hf_split = "train"
    args.split_dataset = split_cfg.get("enabled", True)
    args.train_ratio = split_cfg.get("train_ratio", 0.98)
    args.val_ratio = split_cfg.get("val_ratio", 0.01)
    args.test_ratio = split_cfg.get("test_ratio", 0.01)
    args.split_seed = split_cfg.get("seed", 42)
    args.target_languages = cleaning_cfg.get("target_languages", [])
    args.min_quality_score = cleaning_cfg.get("min_quality_score", 0.25)
    args.sample_fraction = cleaning_cfg.get("sample_fraction")
    args.disable_dedup = not cleaning_cfg.get("dedup_enabled", True)
    args.disable_pii_removal = not cleaning_cfg.get("pii_removal", True)
    args.enable_toxic_filter = cleaning_cfg.get("toxic_filter", False)
    args.tokenizer_mode = token_cfg.get("tokenizer_mode", "train")
    args.tokenizer_type = token_cfg.get("tokenizer_type", "sentencepiece")
    args.tokenizer_subtype = token_cfg.get("tokenizer_subtype", "bpe")
    args.pretrained_name = token_cfg.get("pretrained_name")
    args.vocab_size = token_cfg.get("vocab_size", 32000)
    args.context_length = token_cfg.get("context_length", 2048)
    args.packing_strategy = token_cfg.get("packing_strategy", "greedy")
    args.output_format = token_cfg.get("output_format", "parquet")
    args.corpus_sample_fraction = token_cfg.get("corpus_sample_fraction", 0.05)
    args.max_corpus_lines = token_cfg.get("max_corpus_lines", 5_000_000)
    args.no_bos = not token_cfg.get("add_bos", True)
    args.no_eos = not token_cfg.get("add_eos", True)
    args.clean_enabled = stage_cfg.get("clean", True)
    args.tokenize_enabled = stage_cfg.get("tokenize", True)
    return args


def _run_cleaning_stage(args, input_path: str) -> dict:
    _, stats = run_cleaning(
        input_path=input_path,
        output_path=args.cleaned_output,
        target_languages=args.target_languages or None,
        min_quality_score=args.min_quality_score,
        dedup_enabled=not args.disable_dedup,
        pii_removal=not args.disable_pii_removal,
        toxic_filter=args.enable_toxic_filter,
        sample_fraction=args.sample_fraction,
    )
    return stats


def _load_existing_cleaning_stats(args, input_path: str) -> dict | None:
    cleaned_dir = Path(args.cleaned_output)
    cleaned_data_path = cleaned_dir / "cleaned_data.parquet"
    stats_path = cleaned_dir / "pipeline_stats.json"

    if not cleaned_data_path.exists() or not stats_path.exists():
        return None

    try:
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None

    if stats.get("input_path") != input_path:
        return None
    if stats.get("output_path") != args.cleaned_output:
        return None

    return stats


def _materialize_local_splits(args, input_path: str) -> dict[str, str]:
    split_root = Path(args.hf_cache_dir) / "splits_manifest"
    run_slug = Path(args.cleaned_output).name or "merged_hf"
    output_dir = split_root / run_slug
    split_paths = {
        "train": output_dir / "train.jsonl",
        "validation": output_dir / "validation.jsonl",
        "test": output_dir / "test.jsonl",
    }

    if all(path.exists() for path in split_paths.values()):
        log.info("Menggunakan cache split manifest: %s", output_dir)
        return {name: str(path) for name, path in split_paths.items()}

    loader, data_files = _detect_local_loader(input_path)
    if loader == "csv":
        delimiter = "\t" if all(file.endswith(".tsv") for file in data_files) else ","
        dataset = load_dataset(loader, data_files=data_files, split="train", delimiter=delimiter)
    else:
        dataset = load_dataset(loader, data_files=data_files, split="train")

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
    log.info("Split manifest disimpan ke: %s", output_dir)
    return {name: str(path) for name, path in split_paths.items()}


def _run_tokenize_stage(args, split_inputs: dict[str, str]) -> dict:
    tokenize_results = {}
    shared_tokenizer_path = args.pretrained_name
    train_split_name = "train" if "train" in split_inputs else next(iter(split_inputs))

    for split_name, input_path in split_inputs.items():
        split_output_dir = Path(args.token_output) / split_name if len(split_inputs) > 1 else Path(args.token_output)

        tokenizer_mode = args.tokenizer_mode
        pretrained_name = shared_tokenizer_path

        if split_name != train_split_name and args.tokenizer_mode == "train":
            if not shared_tokenizer_path:
                shared_tokenizer_path = _resolve_trained_tokenizer_path(
                    args,
                    str(Path(args.token_output) / train_split_name),
                )
            tokenizer_mode = "load"
            pretrained_name = shared_tokenizer_path

        tokenize_results[split_name] = run_tokenize(
            input_path=input_path,
            output_dir=str(split_output_dir),
            tokenizer_mode=tokenizer_mode,
            tokenizer_type=args.tokenizer_type,
            tokenizer_subtype=args.tokenizer_subtype,
            pretrained_name=pretrained_name,
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

        if split_name == train_split_name and args.tokenizer_mode == "train":
            shared_tokenizer_path = _resolve_trained_tokenizer_path(args, str(split_output_dir))

    return tokenize_results


def main() -> None:
    cli_args = build_parser().parse_args()
    manifest = _load_manifest(cli_args.manifest)
    args = _build_stage_args(manifest)

    raw_merged_path, raw_stats = _materialize_raw_corpus(manifest)
    results = {"materialize": raw_stats}

    cleaned_input_path = raw_merged_path
    if args.clean_enabled:
        cleaning_stats = _load_existing_cleaning_stats(args, raw_merged_path)
        if cleaning_stats is not None:
            log.info("Menggunakan hasil cleaning yang sudah ada: %s", Path(args.cleaned_output) / "cleaned_data.parquet")
        else:
            cleaning_stats = _run_cleaning_stage(args, raw_merged_path)
        results["cleaning"] = cleaning_stats
        cleaned_input_path = str(Path(args.cleaned_output) / "cleaned_data.parquet")

    if args.tokenize_enabled:
        if args.split_dataset:
            split_inputs = _materialize_local_splits(args, cleaned_input_path)
        else:
            split_inputs = {"all": cleaned_input_path}
        results["tokenize"] = _run_tokenize_stage(args, split_inputs)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
