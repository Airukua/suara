import hashlib
import re
import struct
import unicodedata
from pyspark.sql.functions import udf
from pyspark.sql.types import BooleanType, FloatType, StringType

_BOILERPLATE_PATTERNS = [
    r"cookie[s]?\s+polic",
    r"privacy\s+polic",
    r"terms?\s+(of\s+)?(use|service|condition)",
    r"accept\s+all\s+cookie",
    r"404\s+(not\s+found|error|page)",
    r"403\s+forbidden",
    r"please\s+(enable|turn\s+on)\s+javascript",
    r"javascript\s+(is\s+)?(required|disabled|not\s+enabled)",
    r"captcha|recaptcha",
    r"(click|tap)\s+here\s+to\s+continue",
    r"subscribe\s+to\s+(our\s+)?newsletter",
    r"follow\s+us\s+on\s+(twitter|facebook|instagram)",
]

_TOXIC_PATTERNS = [
    r"\b(kill|murder|bomb|terrorist|suicide)\b.*\b(instruction|how\s+to|guide|step)\b",
    r"\b(how\s+to\s+make|manufacture|synthesize)\b.{0,50}\b(drug|weapon|explosive)\b",
]

_ID_STOPWORDS = {"yang", "dan", "di", "ke", "dari", "dengan", "untuk", "pada", "adalah"}
_EN_STOPWORDS = {"the", "a", "an", "is", "are", "was", "were", "to", "of", "in",
                 "for", "on", "with", "at", "by", "from", "have", "has"}

_MERSENNE_PRIME = (1 << 61) - 1
_MAX_HASH       = (1 << 32) - 1


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
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\x80-\x9f]", "", text)
        text = re.sub(r"[\u200b\u200c\u200d\u200e\u200f\ufeff]", "", text)
        # Normalisasi spasi dan newline
        text = text.replace("\t", " ").replace("\f", " ")
        lines = [re.sub(r" {2,}", " ", line).strip() for line in text.split("\n")]
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
                              min_chars=100,
                              max_chars=1_000_000,
                              min_words=20,
                              min_sentences=3):
        if text is None:
            return False
        if not (min_chars <= len(text) <= max_chars):
            return False
        if len(text.split()) < min_words:
            return False
        if len(re.split(r"[.!?]+", text)) < min_sentences:
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
        if text is None or len(text) < 50:
            return False
        words = text.lower().split()
        if len(words) < 20:
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
        if 200 <= word_count <= 50_000:
            len_score = 1.0
        elif word_count < 200:
            len_score = word_count / 200
        else:
            len_score = max(0.0, 1.0 - (word_count - 50_000) / 100_000)
        scores.append(len_score * 0.2)

        unique_words = len(set(w.lower() for w in words))
        ttr = min(1.0, unique_words / max(1, word_count) * 2)
        scores.append(ttr * 0.25)
        stopwords     = _ID_STOPWORDS | _EN_STOPWORDS
        content_words = [w for w in words if w.lower() not in stopwords and len(w) > 2]
        content_ratio = len(content_words) / max(1, word_count)
        scores.append(min(1.0, content_ratio * 1.5) * 0.15)
        sentences     = re.split(r"[.!?]+", text)
        good_sents    = sum(1 for s in sentences if 5 <= len(s.split()) <= 200)
        scores.append((good_sents / max(1, len(sentences))) * 0.2)
        avg_word_len = sum(len(w) for w in words) / max(1, word_count)
        scores.append(min(1.0, max(0.0, (avg_word_len - 3) / 5)) * 0.1)
        punct       = sum(c in ".!?,;:" for c in text)
        punct_ratio = punct / max(1, len(text))
        if 0.01 <= punct_ratio <= 0.08:
            punct_score = 1.0
        elif punct_ratio < 0.01:
            punct_score = punct_ratio / 0.01
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
        if text is None:
            return False
        text_lower = text.lower()
        return any(re.search(p, text_lower) for p in _TOXIC_PATTERNS)

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
