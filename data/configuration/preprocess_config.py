import logging
from pathlib import Path

import psutil
import yaml
from pyspark.sql import SparkSession

log = logging.getLogger("LLM-Preprocess")

_CONFIG_PATH = Path(__file__).with_name("preprocess_config.yaml")


def load_preprocess_config(config_path: str = None) -> dict:
    path = Path(config_path) if config_path else _CONFIG_PATH
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def compute_spark_config(workload: str = "cleaning", config_path: str = None) -> dict:
    settings = load_preprocess_config(config_path)
    spark_settings = settings.get("spark", {})
    workload_settings = spark_settings.get("workloads", {}).get(workload, {})
    defaults = spark_settings.get("defaults", {})

    mem = psutil.virtual_memory()
    cores = psutil.cpu_count(logical=True)
    total_ram_gb = mem.total / (1024**3)
    available_ram_gb = mem.available / (1024**3)
    spark_ram_gb = int(available_ram_gb * defaults.get("available_ram_fraction", 0.75))

    driver_mem = max(
        workload_settings.get("driver_mem_min_gb", 2),
        int(spark_ram_gb * workload_settings.get("driver_mem_fraction", 0.2)),
    )
    executor_mem = max(
        workload_settings.get("executor_mem_min_gb", 4),
        int(spark_ram_gb * workload_settings.get("executor_mem_fraction", 0.8)),
    )
    default_parallelism = max(1, cores * workload_settings.get("parallelism_multiplier", 3))
    shuffle_partitions = max(
        defaults.get("min_shuffle_partitions", 200),
        cores * workload_settings.get("shuffle_multiplier", 4),
    )

    # This project runs Spark in local mode, so overly aggressive driver memory
    # and thread counts can kill the single JVM. Keep some headroom for Python,
    # the OS, and tokenizer libraries.
    local_cores_limit = workload_settings.get("local_cores_max", 8)
    safe_cores = max(1, min(cores, local_cores_limit))
    safe_total_mem_gb = max(2, int(max(available_ram_gb - 1.0, 2)))
    driver_mem = max(2, min(driver_mem, max(2, safe_total_mem_gb // 2)))
    executor_mem = max(2, min(executor_mem, max(2, safe_total_mem_gb - driver_mem)))
    default_parallelism = min(default_parallelism, safe_cores * 3)
    shuffle_partitions = min(shuffle_partitions, max(32, safe_cores * 4))

    driver_overhead = max(
        defaults.get("min_driver_overhead_mb", 384),
        int(driver_mem * 1024 * defaults.get("memory_overhead_fraction", 0.1)),
    )
    executor_overhead = max(
        defaults.get("min_executor_overhead_mb", 384),
        int(executor_mem * 1024 * defaults.get("memory_overhead_fraction", 0.1)),
    )

    config = {
        "spark.driver.memory": f"{driver_mem}g",
        "spark.executor.memory": f"{executor_mem}g",
        "spark.driver.memoryOverhead": f"{driver_overhead}m",
        "spark.executor.memoryOverhead": f"{executor_overhead}m",
        "spark.default.parallelism": str(default_parallelism),
        "spark.sql.shuffle.partitions": str(shuffle_partitions),
        "spark.sql.adaptive.enabled": "true",
        "spark.sql.adaptive.coalescePartitions.enabled": "true",
        "spark.sql.adaptive.skewJoin.enabled": "true",
        "spark.memory.fraction": str(defaults.get("memory_fraction", 0.8)),
        "spark.memory.storageFraction": str(defaults.get("memory_storage_fraction", 0.3)),
        "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
        "spark.kryoserializer.buffer.max": defaults.get("kryo_buffer_max", "512m"),
        "spark.sql.parquet.compression.codec": defaults.get("parquet_codec", "snappy"),
        "spark.sql.parquet.mergeSchema": "false",
        "spark.sql.files.maxPartitionBytes": defaults.get("max_partition_bytes", "256mb"),
        "spark.local.dir": defaults.get("local_dir", "/tmp/spark-temp"),
        "spark.sql.broadcastTimeout": str(defaults.get("broadcast_timeout_seconds", 600)),
        "spark.network.timeout": defaults.get("network_timeout", "600s"),
        "spark.rpc.askTimeout": defaults.get("rpc_timeout", "600s"),
        "spark.python.worker.reuse": "true",
    }

    broadcast_block_size = workload_settings.get("broadcast_block_size")
    if broadcast_block_size:
        config["spark.broadcast.blockSize"] = broadcast_block_size

    print(f"=== AUTO-TUNED SPARK CONFIG ({workload.upper()}) ===")
    print(f"Total RAM       : {total_ram_gb:.1f} GB")
    print(f"Available RAM   : {available_ram_gb:.1f} GB")
    print(f"Spark Driver    : {driver_mem}g")
    print(f"Spark Executor  : {executor_mem}g")
    print(f"CPU Cores       : {cores} (using {safe_cores})")
    print(f"Parallelism     : {default_parallelism}")
    print(f"Shuffle Parts   : {shuffle_partitions}")
    print()

    return config


def create_spark_session(
    app_name: str = "LLM-DataCleaner",
    workload: str = "cleaning",
    config: dict = None,
    config_path: str = None,
):
    if config is None:
        config = compute_spark_config(workload=workload, config_path=config_path)

    cores = psutil.cpu_count(logical=True)
    workload_settings = load_preprocess_config(config_path).get("spark", {}).get("workloads", {}).get(workload, {})
    local_cores = max(1, min(cores, workload_settings.get("local_cores_max", 8)))
    builder = SparkSession.builder.appName(app_name).master(f"local[{local_cores}]")

    for key, value in config.items():
        builder = builder.config(key, value)

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    log.info(f"SparkSession created: {app_name} [{workload}]")
    log.info(f"Spark UI: {spark.sparkContext.uiWebUrl}")
    return spark
