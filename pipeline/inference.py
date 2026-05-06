import argparse
import sys
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
import sentencepiece as spm
from tokenizers import Tokenizer as HFTokenizer
from transformers import AutoTokenizer

from data.configuration.config import load_config
from pipeline.model import SuaRA


class InferenceTokenizer:
    def __init__(self, tokenizer_type: str, model_path: str, add_bos: bool = True, add_eos: bool = True):
        self.tokenizer_type = tokenizer_type
        self.model_path = model_path
        self.add_bos = add_bos
        self.add_eos = add_eos

        if tokenizer_type == "sentencepiece":
            tokenizer = spm.SentencePieceProcessor()
            tokenizer.load(model_path)
            self._tokenizer = tokenizer
            self.bos_token_id = tokenizer.bos_id()
            self.eos_token_id = tokenizer.eos_id()
            self.pad_token_id = tokenizer.pad_id() if tokenizer.pad_id() >= 0 else 0
        elif tokenizer_type == "hf":
            tokenizer = HFTokenizer.from_file(str(Path(model_path) / "tokenizer.json"))
            self._tokenizer = tokenizer
            self.bos_token_id = tokenizer.token_to_id("<s>") or 2
            self.eos_token_id = tokenizer.token_to_id("</s>") or 3
            self.pad_token_id = tokenizer.token_to_id("<pad>") or 0
        elif tokenizer_type == "pretrained":
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            self._tokenizer = tokenizer
            self.bos_token_id = tokenizer.bos_token_id or tokenizer.cls_token_id or 0
            self.eos_token_id = tokenizer.eos_token_id
            self.pad_token_id = tokenizer.pad_token_id or 0
        else:
            raise ValueError(f"tokenizer_type tidak didukung: {tokenizer_type}")

    @classmethod
    def from_config(cls, config):
        return cls(
            tokenizer_type=config.tokenizer_type,
            model_path=config.model_path,
            add_bos=config.add_bos,
            add_eos=config.add_eos,
        )

    def __len__(self) -> int:
        if self.tokenizer_type == "sentencepiece":
            return int(self._tokenizer.get_piece_size())
        if self.tokenizer_type == "hf":
            return int(self._tokenizer.get_vocab_size())
        return int(self._tokenizer.vocab_size)

    def encode(
        self,
        text: str,
        add_bos: bool | None = None,
        add_eos: bool | None = None,
    ) -> List[int]:
        if add_bos is None:
            add_bos = self.add_bos
        if add_eos is None:
            add_eos = self.add_eos

        if self.tokenizer_type == "sentencepiece":
            token_ids = list(self._tokenizer.encode(text))
        elif self.tokenizer_type == "hf":
            token_ids = list(self._tokenizer.encode(text).ids)
        else:
            token_ids = list(self._tokenizer.encode(text, add_special_tokens=False))

        if add_bos and self.bos_token_id is not None:
            token_ids = [self.bos_token_id] + token_ids
        if add_eos and self.eos_token_id is not None:
            token_ids = token_ids + [self.eos_token_id]
        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        if self.tokenizer_type == "sentencepiece":
            cleaned_ids = token_ids
            if skip_special_tokens:
                cleaned_ids = [
                    tok_id for tok_id in token_ids
                    if tok_id not in {self.pad_token_id, self.bos_token_id, self.eos_token_id}
                ]
            return self._tokenizer.decode(cleaned_ids)

        if self.tokenizer_type == "hf":
            if skip_special_tokens:
                special_ids = {tok_id for tok_id in (self.pad_token_id, self.bos_token_id, self.eos_token_id) if tok_id is not None}
                token_ids = [tok_id for tok_id in token_ids if tok_id not in special_ids]
            return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

        return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

def build_generation_case(texts, prompt_words=12, target_words=24):
    for text in texts:
        words = text.split()
        if len(words) > prompt_words + 4:
            prompt = ' '.join(words[:prompt_words])
            target = ' '.join(words[prompt_words:prompt_words + target_words])
            return prompt, target
    fallback_prompt = "the history of"
    return fallback_prompt, ""

@torch.inference_mode()
def generate_sample(model, tokenizer, device, prompt, MAX_SEQ=None, max_new_tokens=40,
                    temperature=0.9, top_k=40, ):
    model.eval()
    if MAX_SEQ is None:
        MAX_SEQ = getattr(model, 'max_seq', 128)
    token_ids = tokenizer.encode(prompt, add_eos=False)
    if not token_ids:
        token_ids = [tokenizer.bos_token_id]

    generated = list(token_ids)

    for _ in range(max_new_tokens):
        ctx = generated[-MAX_SEQ:]
        inp = torch.tensor([ctx], dtype=torch.long, device=device)
        logits, _ = model(inp)
        next_logits = logits[0, -1] / max(temperature, 1e-5)

        if top_k is not None and top_k > 0:
            k = min(top_k, next_logits.size(-1))
            top_vals, top_idx = torch.topk(next_logits, k)
            probs = F.softmax(top_vals, dim=-1)
            next_token = top_idx[torch.multinomial(probs, 1)].item()
        else:
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()

        generated.append(next_token)
        if next_token == tokenizer.eos_token_id:
            break
    return tokenizer.decode(generated)


@torch.inference_mode()
def generate(
    model,
    tokenizer,
    device,
    prompt: str,
    MAX_SEQ: Optional[int] = None,
    max_new_tokens: int = 40,
    temperature: float = 0.9,
    top_k: Optional[int] = 40,
    top_p: Optional[float] = 0.9, 
    repetition_penalty: float = 1.1,  
    stop_strings: Optional[List[str]] = None, 
) -> str:
    model.eval()
    if MAX_SEQ is None:
        MAX_SEQ = getattr(model, 'max_seq', 128)

    token_ids: List[int] = tokenizer.encode(prompt, add_eos=False)
    if not token_ids:
        bos = getattr(tokenizer, 'bos_token_id', None)
        token_ids = [bos] if bos is not None else [0]

    generated = list(token_ids)
    prompt_len = len(generated)

    for _ in range(max_new_tokens):
        ctx = generated[-MAX_SEQ:]
        inp = torch.tensor([ctx], dtype=torch.long, device=device)

        logits, _ = model(inp)
        next_logits = logits[0, -1].clone().float()
        next_logits /= max(temperature, 1e-5)

        if repetition_penalty != 1.0:
            ctx_tensor = torch.tensor(ctx, dtype=torch.long, device=next_logits.device)
            unique_tok_ids = torch.unique(ctx_tensor)
            penalized_logits = next_logits.index_select(0, unique_tok_ids)
            penalized_logits = torch.where(
                penalized_logits > 0,
                penalized_logits / repetition_penalty,
                penalized_logits * repetition_penalty,
            )
            next_logits.scatter_(0, unique_tok_ids, penalized_logits)

        if top_k is not None and top_k > 0:
            k = min(top_k, next_logits.size(-1))
            kth_val = torch.topk(next_logits, k).values[-1]
            next_logits = next_logits.masked_fill(next_logits < kth_val, float('-inf'))

        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_logits[cumulative_probs - sorted_probs > top_p] = float('-inf')
            next_logits = torch.zeros_like(next_logits).scatter_(0, sorted_idx, sorted_logits)

        probs = F.softmax(next_logits, dim=-1)
        next_token: int = torch.multinomial(probs, 1).item()
        generated.append(next_token)
        eos_id = getattr(tokenizer, 'eos_token_id', None)
        if eos_id is not None and next_token == eos_id:
            break

        if stop_strings:
            partial = tokenizer.decode(generated[prompt_len:], skip_special_tokens=True)
            if any(s in partial for s in stop_strings):
                break
    return tokenizer.decode(generated[prompt_len:], skip_special_tokens=True).strip()


def _parse_args():
    parser = argparse.ArgumentParser(description="Generate text from a trained SUARA checkpoint.")
    parser.add_argument(
        "prompt",
        nargs="?",
        default=None,
        help="Prompt teks. Jika kosong, akan dibaca dari stdin bila tersedia.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path ke file config YAML. Default pakai data/configuration/config.yaml.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path checkpoint model. Default: best.pt lalu fallback ke last.pt di artifacts/checkpoints.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument(
        "--stop-string",
        action="append",
        default=None,
        help="String penghenti. Bisa dipakai berkali-kali.",
    )
    return parser.parse_args()


def _resolve_prompt(cli_prompt: Optional[str]) -> str:
    if cli_prompt is not None and cli_prompt.strip():
        return cli_prompt
    if not sys.stdin.isatty():
        piped = sys.stdin.read().strip()
        if piped:
            return piped
    raise ValueError("prompt wajib diisi lewat argumen atau stdin")


def _resolve_checkpoint_path(checkpoint_arg: Optional[str], checkpoint_dir: str) -> Path:
    if checkpoint_arg is not None:
        checkpoint_path = Path(checkpoint_arg).expanduser()
        if not checkpoint_path.is_absolute():
            checkpoint_path = (PROJECT_ROOT / checkpoint_path).resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint tidak ditemukan: {checkpoint_path}")
        return checkpoint_path

    base_dir = Path(checkpoint_dir)
    for candidate in (base_dir / "best.pt", base_dir / "last.pt"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"checkpoint default tidak ditemukan di {base_dir}. "
        "Gunakan --checkpoint untuk menentukan file checkpoint."
    )


def _resolve_bundle_path(checkpoint_path: Path) -> Path:
    if checkpoint_path.is_dir():
        candidate = checkpoint_path / "bundle.pt"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(
            f"folder bundle ditemukan tapi file bundle.pt tidak ada di: {checkpoint_path}"
        )
    return checkpoint_path


def _looks_like_inference_bundle(payload: dict) -> bool:
    return (
        isinstance(payload, dict)
        and payload.get("format") == "suara_inference_bundle"
        and "model_state_dict" in payload
        and "model_config" in payload
        and "tokenizer_config" in payload
    )


def _materialize_tokenizer_config(bundle_dir: Path, tokenizer_config_data: dict) -> argparse.Namespace:
    tokenizer_path = tokenizer_config_data["model_path"]
    candidate = Path(tokenizer_path)
    bundle_candidate = (bundle_dir / tokenizer_path).resolve()
    if not candidate.is_absolute() and bundle_candidate.exists():
        candidate = (bundle_dir / tokenizer_path).resolve()

    return argparse.Namespace(
        tokenizer_type=tokenizer_config_data["tokenizer_type"],
        model_path=str(candidate),
        add_bos=tokenizer_config_data.get("add_bos", True),
        add_eos=tokenizer_config_data.get("add_eos", True),
    )


def main():
    args = _parse_args()
    prompt = _resolve_prompt(args.prompt)
    cfg = load_config(args.config)
    checkpoint_path = _resolve_checkpoint_path(args.checkpoint, cfg.checkpoint.output_directory)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_path = _resolve_bundle_path(checkpoint_path)
    checkpoint = torch.load(load_path, map_location=device)

    if _looks_like_inference_bundle(checkpoint):
        bundle_dir = load_path.parent
        tokenizer_cfg = _materialize_tokenizer_config(bundle_dir, checkpoint["tokenizer_config"])
        tokenizer = InferenceTokenizer.from_config(tokenizer_cfg)
        model = SuaRA(
            vocab_size=len(tokenizer),
            **checkpoint["model_config"],
            enable_runtime_checks=False,
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        generation_cfg = checkpoint["generation_config"]
        max_seq = checkpoint["model_config"]["max_seq"]
    else:
        tokenizer = InferenceTokenizer.from_config(cfg.tokenizer)
        model = SuaRA(
            vocab_size=len(tokenizer),
            **cfg.model.to_kwargs(),
            enable_runtime_checks=False,
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        generation_cfg = {
            "max_new_tokens": cfg.generation.max_new_tokens,
            "temperature": cfg.generation.temperature,
            "top_k": cfg.generation.top_k,
            "top_p": cfg.generation.top_p,
            "repetition_penalty": cfg.generation.repetition_penalty,
        }
        max_seq = cfg.model.max_seq

    output = generate(
        model,
        tokenizer,
        device,
        prompt=prompt,
        MAX_SEQ=max_seq,
        max_new_tokens=args.max_new_tokens or generation_cfg["max_new_tokens"],
        temperature=args.temperature if args.temperature is not None else generation_cfg["temperature"],
        top_k=args.top_k if args.top_k is not None else generation_cfg["top_k"],
        top_p=args.top_p if args.top_p is not None else generation_cfg["top_p"],
        repetition_penalty=(
            args.repetition_penalty
            if args.repetition_penalty is not None
            else generation_cfg["repetition_penalty"]
        ),
        stop_strings=args.stop_string,
    )
    print(output)


if __name__ == "__main__":
    main()
