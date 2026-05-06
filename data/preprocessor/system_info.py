import platform
from typing import Dict, Optional
import psutil


DEFAULT_TOKENIZER_DEPENDENCIES = {
    "pyspark": "pyspark",
    "sentencepiece": "sentencepiece",
    "tokenizers": "tokenizers",
    "transformers": "transformers",
    "numpy": "numpy",
    "pyarrow": "pyarrow",
    "datasets": "datasets",
}


def collect_system_info() -> Dict[str, object]:
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    cpu_freq = psutil.cpu_freq()

    return {
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "node_name": platform.node(),
        "processor": platform.processor(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "cpu_freq_mhz": cpu_freq.current if cpu_freq else None,
        "cpu_usage_pct": psutil.cpu_percent(interval=1),
        "ram_total_gb": mem.total / (1024**3),
        "ram_available_gb": mem.available / (1024**3),
        "ram_used_gb": mem.used / (1024**3),
        "ram_usage_pct": mem.percent,
        "disk_total_gb": disk.total / (1024**3),
        "disk_used_gb": disk.used / (1024**3),
        "disk_free_gb": disk.free / (1024**3),
        "disk_usage_pct": disk.percent,
    }


def check_dependencies(dependencies: Optional[Dict[str, str]] = None) -> Dict[str, bool]:
    available = {}
    for name, package_name in (dependencies or {}).items():
        try:
            __import__(package_name)
            available[name] = True
        except ImportError:
            available[name] = False
    return available


def print_system_info(
    title: str = "SYSTEM INFO",
    dependencies: Optional[Dict[str, str]] = None,
    show_gpu: bool = True,
) -> Dict[str, object]:
    info = collect_system_info()

    print("═" * 62)
    print(f"  {title}")
    print("═" * 62)

    print("=== SYSTEM ===")
    print(f"OS              : {info['os']}")
    print(f"Machine         : {info['machine']}")
    print(f"Node Name       : {info['node_name']}")
    print()

    print("=== CPU ===")
    print(f"Processor       : {info['processor']}")
    print(f"Physical cores  : {info['physical_cores']}")
    print(f"Logical cores   : {info['logical_cores']}")
    if info["cpu_freq_mhz"] is not None:
        print(f"CPU freq (MHz)  : {info['cpu_freq_mhz']:.2f}")
    print(f"CPU usage (%)   : {info['cpu_usage_pct']}")
    print()

    print("=== RAM ===")
    print(f"Total RAM       : {info['ram_total_gb']:.2f} GB")
    print(f"Available RAM   : {info['ram_available_gb']:.2f} GB")
    print(f"Used RAM        : {info['ram_used_gb']:.2f} GB")
    print(f"Usage (%)       : {info['ram_usage_pct']}")
    print()

    print("=== DISK ===")
    print(f"Total Disk      : {info['disk_total_gb']:.2f} GB")
    print(f"Used Disk       : {info['disk_used_gb']:.2f} GB")
    print(f"Free Disk       : {info['disk_free_gb']:.2f} GB")
    print(f"Usage (%)       : {info['disk_usage_pct']}")
    print()

    if show_gpu:
        print("=== GPU ===")
        try:
            import torch

            if torch.cuda.is_available():
                print(f"GPU Name        : {torch.cuda.get_device_name(0)}")
                print(f"CUDA Version    : {torch.version.cuda}")
                print(f"GPU Count       : {torch.cuda.device_count()}")
                total_mem = torch.cuda.get_device_properties(0).total_memory
                print(f"GPU Memory      : {total_mem / (1024**3):.2f} GB")
            else:
                print("GPU             : Not available (CPU-only mode)")
        except ImportError:
            print("Torch not installed - GPU check skipped")
        print()

    dependency_status = check_dependencies(dependencies)
    if dependency_status:
        print("=== DEPENDENCIES ===")
        for name, package_name in dependencies.items():
            try:
                module = __import__(package_name)
                version = getattr(module, "__version__", "?")
                print(f"{name:<18} : OK ({version})")
            except ImportError:
                print(f"{name:<18} : NOT INSTALLED")
        print()

        if (
            "sentencepiece" in dependency_status
            and "tokenizers" in dependency_status
            and not dependency_status["sentencepiece"]
            and not dependency_status["tokenizers"]
        ):
            print("Install minimal satu tokenizer library:")
            print("pip install sentencepiece tokenizers transformers pyarrow")
            print()

    return {
        "system": info,
        "dependencies": dependency_status,
    }
