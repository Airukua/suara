import logging

from data.configuration.preprocess_config import create_spark_session
from data.preprocessor.system_info import print_system_info
from data.preprocessor.cleaning.pipeline import process_in_batches, run_cleaning_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def run(
    input_path:        str   = "data/raw",
    output_path:       str   = "data/cleaned",
    target_languages:  list  = None,
    min_quality_score: float = 0.35,
    dedup_enabled:     bool  = True,
    pii_removal:       bool  = True,
    toxic_filter:      bool  = True,
    sample_fraction:   float = None,
):
    print_system_info(title="SYSTEM INFO", show_gpu=True)
    spark = create_spark_session("LLM-DataCleaner", workload="cleaning")
    try:
        df, stats = run_cleaning_pipeline(
            spark=spark,
            input_path=input_path,
            output_path=output_path,
            target_languages=target_languages,
            min_quality_score=min_quality_score,
            dedup_enabled=dedup_enabled,
            pii_removal=pii_removal,
            toxic_filter=toxic_filter,
            sample_fraction=sample_fraction,
        )
        return df, stats
    finally:
        spark.stop()


def run_batch(
    input_paths:  list,
    output_base:  str = "data/cleaned_batches",
    **pipeline_kwargs,
):
    print_system_info(title="SYSTEM INFO", show_gpu=True)
    spark = create_spark_session("LLM-DataCleaner-Batch", workload="cleaning")
    try:
        merged_df, all_stats = process_in_batches(
            spark=spark,
            input_paths=input_paths,
            output_base=output_base,
            **pipeline_kwargs,
        )
        return merged_df, all_stats
    finally:
        spark.stop()
