# Suara

## Multi-dataset Hugging Face data prep

Pipeline baru untuk banyak dataset Hugging Face ada di [pipeline/data_prep_manifest.py](/home/a_rkk/suara/pipeline/data_prep_manifest.py:1) dengan contoh manifest di [data/configuration/hf_manifest.yaml](/home/a_rkk/suara/data/configuration/hf_manifest.yaml:1).

Alurnya:
- download banyak dataset Hugging Face
- samakan kolom teks menjadi `text`
- merge corpus mentah
- cleaning gabungan
- split train/validation/test setelah cleaning
- tokenize semua split dengan tokenizer yang sama

Jalankan:

```bash
python3 pipeline/data_prep_manifest.py \
  --manifest data/configuration/hf_manifest.yaml
```

Sebelum menjalankan, sesuaikan daftar dataset, `text_column`, output path, dan konfigurasi cleaning/tokenize di file manifest.
