import hashlib
import re
import struct
import unicodedata
from pyspark.sql.functions import udf
from pyspark.sql.types import BooleanType, FloatType, StringType

_BOILERPLATE_PATTERNS = [
    r"404\s+(not\s+found|error|page|halaman)",
    r"403\s+forbidden",
    r"(please|harap)\s+(enable|aktifkan|nyalakan)\s+javascript",
    r"javascript\s+(is\s+)?(required|disabled|not\s+enabled|dinonaktifkan|tidak\s+aktif)",
    r"captcha|recaptcha",
    r"(click|tap|klik)\s+(here|di\s+sini)\s+to\s+continue",
]

# Dipakai hanya sebagai fallback heuristik pada deteksi bahasa.
# Daftar ini tidak digunakan untuk menghapus kata dari teks.
_ID_STOPWORDS = {
    "yang", "dan", "di", "ke", "dari", "untuk", "dengan", "pada", "adalah",
    "ini", "itu", "dalam", "atau", "juga", "tidak", "karena", "sebagai",
    "oleh", "akan", "lebih", "sudah", "agar", "antara", "bisa", "masih",
}

_EN_STOPWORDS = {
    "the", "and", "of", "to", "in", "for", "with", "on", "is", "this",
    "that", "from", "by", "as", "are", "be", "or", "an", "it", "at",
    "was", "were", "can", "have", "has",
}

_MERSENNE_PRIME = (1 << 61) - 1
_MAX_HASH       = (1 << 32) - 1
_LITERAL_WHITESPACE_PATTERN = re.compile(r"\\+[nrtfv]+")
_REPEATED_PUNCT_PATTERN = re.compile(r"([!?.,;:])\1{1,}")
_DECORATIVE_SYMBOL_RUN_PATTERN = re.compile(r"([^\w\s])\1{2,}")


def register_cleaning_udfs(spark) -> dict:

    @udf(returnType=StringType())
    def fix_encoding(text):
        if text is None:
            return None
        try:
            import ftfy
            text = ftfy.fix_text(text)
        except ImportError:
            try:
                text = text.encode("latin-1").decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError):
                pass
        text = unicodedata.normalize("NFC", text)
        import html
        text = html.unescape(text)
        return text

    @udf(returnType=StringType())
    def normalize_whitespace(text):
        if text is None:
            return None
        # Ubah escape sequence literal dari dataset mentah menjadi whitespace nyata.
        text = _LITERAL_WHITESPACE_PATTERN.sub(
            lambda match: "\n" if "n" in match.group(0) else " ",
            text,
        )
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\x80-\x9f]", "", text)
        text = re.sub(r"[\u200b\u200c\u200d\u200e\u200f\ufeff]", "", text)
        text = _REPEATED_PUNCT_PATTERN.sub(r"\1", text)
        # Pangkas simbol dekoratif panjang seperti blok/ornamen forum yang bikin tokenizer ribut.
        text = _DECORATIVE_SYMBOL_RUN_PATTERN.sub(r"\1", text)
        # Normalisasi spasi dan newline
        text = text.replace("\t", " ").replace("\f", " ")
        cleaned_lines = []
        for line in text.split("\n"):
            tokens = []
            for token in line.split():
                alnum_count = sum(ch.isalnum() for ch in token)
                symbol_count = sum(
                    unicodedata.category(ch).startswith(("S", "P"))
                    for ch in token
                )
                if alnum_count == 0 and symbol_count >= 3:
                    continue
                tokens.append(token)
            cleaned_lines.append(re.sub(r" {2,}", " ", " ".join(tokens)).strip())
        lines = cleaned_lines
        text  = "\n".join(lines)
        text  = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @udf(returnType=StringType())
    def handle_urls_emails(text, mode="placeholder"):
        if text is None:
            return None
        url_pattern   = r'https?://[^\s<>"{}|\\^`\[\]]+|www\.[^\s<>"{}|\\^`\[\]]+'
        email_pattern = r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b"
        if mode == "placeholder":
            text = re.sub(url_pattern,   "[URL]",   text)
            text = re.sub(email_pattern, "[EMAIL]", text)
        else:
            text = re.sub(url_pattern,   "", text)
            text = re.sub(email_pattern, "", text)
        return text

    @udf(returnType=StringType())
    def remove_pii(text):
        if text is None:
            return None
        text = re.sub(
            r"(\+62|62|0)[- ]?8[1-9][0-9][- ]?[0-9]{4,8}[- ]?[0-9]{0,4}",
            "[PHONE]", text,
        )
        text = re.sub(r"\+\d{1,3}[\s\-]?\d{6,14}", "[PHONE]", text)
        text = re.sub(r"\b[1-9][0-9]{15}\b", "[NIK]", text)
        text = re.sub(
            r"\b(?:\d[ -]?){13,16}\b",
            lambda m: "[CARD]"
                      if len(re.sub(r"[^0-9]", "", m.group())) in (13, 15, 16)
                      else m.group(),
            text,
        )
        # Alamat IP
        text = re.sub(
            r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9]{1,2})\.){3}"
            r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9]{1,2})\b",
            "[IP]", text,
        )
        return text

    @udf(returnType=BooleanType())
    def is_boilerplate(text):
        if text is None:
            return True
        text_lower = text.lower()
        for pattern in _BOILERPLATE_PATTERNS:
            if re.search(pattern, text_lower):
                return True
        lines = text.strip().split("\n")
        if len(lines) > 5:
            short_lines = sum(1 for l in lines if 0 < len(l.strip()) < 30)
            if short_lines / len(lines) > 0.7:
                return True
        return False

    @udf(returnType=BooleanType())
    def passes_length_filter(text,
                              min_chars=40,
                              max_chars=1_000_000,
                              min_words=8,
                              min_sentences=1):
        if text is None:
            return False
        if not (min_chars <= len(text) <= max_chars):
            return False
        if len(text.split()) < min_words:
            return False
        sentence_count = sum(1 for s in re.split(r"[.!?]+", text) if s.strip())
        if sentence_count < min_sentences:
            return False
        return True

    @udf(returnType=BooleanType())
    def passes_char_ratio_filter(text):
        if text is None or len(text) == 0:
            return False
        total   = len(text)
        digits  = sum(c.isdigit() for c in text)
        specials = sum(not c.isalnum() and not c.isspace() for c in text)
        alpha   = [c for c in text if c.isalpha()]

        if digits  / total > 0.4:
            return False
        if specials / total > 0.3:
            return False
        if alpha and len(alpha) > 20:
            caps = sum(c.isupper() for c in alpha)
            if caps / len(alpha) > 0.6:
                return False
        return True

    @udf(returnType=BooleanType())
    def has_excessive_repetition(text):
        if text is None or len(text) < 80:
            return False
        words = text.lower().split()
        if len(words) < 16:
            return False

        # Cek rasio trigram unik
        trigrams = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
        if trigrams and len(set(trigrams)) / len(trigrams) < 0.4:
            return True

        # Cek baris duplikat
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if len(lines) > 3 and len(set(lines)) / len(lines) < 0.5:
            return True

        return False

    @udf(returnType=FloatType())
    def compute_quality_score(text):
        if text is None or len(text) < 10:
            return 0.0

        words      = text.split()
        word_count = len(words)
        scores     = []
        if 80 <= word_count <= 50_000:
            len_score = 1.0
        elif word_count < 80:
            len_score = word_count / 80
        else:
            len_score = max(0.0, 1.0 - (word_count - 50_000) / 100_000)
        scores.append(len_score * 0.2)

        unique_words = len(set(w.lower() for w in words))
        ttr = min(1.0, unique_words / max(1, word_count) * 2)
        scores.append(ttr * 0.35)
        sentences     = re.split(r"[.!?]+", text)
        good_sents    = sum(1 for s in sentences if 3 <= len(s.split()) <= 200)
        scores.append((good_sents / max(1, len(sentences))) * 0.25)
        avg_word_len = sum(len(w) for w in words) / max(1, word_count)
        scores.append(min(1.0, max(0.0, (avg_word_len - 3) / 5)) * 0.1)
        punct       = sum(c in ".!?,;:" for c in text)
        punct_ratio = punct / max(1, len(text))
        if 0.005 <= punct_ratio <= 0.08:
            punct_score = 1.0
        elif punct_ratio < 0.005:
            punct_score = punct_ratio / 0.005
        else:
            punct_score = max(0.0, 1.0 - (punct_ratio - 0.08) * 10)
        scores.append(punct_score * 0.1)

        return float(min(1.0, sum(scores)))

    @udf(returnType=StringType())
    def compute_doc_hash(text):
        if text is None:
            return None
        normalized = " ".join(text.lower().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    @udf(returnType=StringType())
    def compute_minhash_signature(text, num_hashes=128, shingle_size=5):
        if text is None or len(text) < 50:
            return None

        words   = text.lower().split()
        shingles = (
            {text[:50]}
            if len(words) < shingle_size
            else {
                " ".join(words[i:i+shingle_size])
                for i in range(len(words) - shingle_size + 1)
            }
        )

        a_vals = [hash(f"a_{i}") & _MAX_HASH for i in range(num_hashes)]
        b_vals = [hash(f"b_{i}") & _MAX_HASH for i in range(num_hashes)]

        signature = [float("inf")] * num_hashes
        for shingle in shingles:
            h = int(hashlib.md5(shingle.encode()).hexdigest(), 16) % _MERSENNE_PRIME
            for i in range(num_hashes):
                sig = (a_vals[i] * h + b_vals[i]) % _MERSENNE_PRIME
                if sig < signature[i]:
                    signature[i] = sig

        sig_bytes = struct.pack(f"{num_hashes}Q", *[int(s) for s in signature])
        return sig_bytes.hex()[:64] 

    @udf(returnType=StringType())
    def detect_language(text):
        if text is None or len(text) < 20:
            return "unknown"
        sample = text[:500]
        try:
            from langdetect import detect
            return detect(sample)
        except Exception:
            pass
        # Fallback berbasis kata kunci
        text_lower = sample.lower()
        id_count = sum(f" {w} " in f" {text_lower} " for w in _ID_STOPWORDS)
        en_count = sum(f" {w} " in f" {text_lower} " for w in _EN_STOPWORDS)
        if id_count > en_count and id_count > 2:
            return "id"
        if en_count > id_count and en_count > 2:
            return "en"
        return "unknown"

    @udf(returnType=BooleanType())
    def has_toxic_content(text):
        return False if text is not None else False

    return {
        "fix_encoding":             fix_encoding,
        "normalize_whitespace":     normalize_whitespace,
        "handle_urls_emails":       handle_urls_emails,
        "remove_pii":               remove_pii,
        "is_boilerplate":           is_boilerplate,
        "passes_length_filter":     passes_length_filter,
        "passes_char_ratio_filter": passes_char_ratio_filter,
        "has_excessive_repetition": has_excessive_repetition,
        "compute_quality_score":    compute_quality_score,
        "compute_doc_hash":         compute_doc_hash,
        "compute_minhash_signature":compute_minhash_signature,
        "detect_language":          detect_language,
        "has_toxic_content":        has_toxic_content,
    }
