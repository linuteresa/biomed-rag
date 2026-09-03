"""Pluggable text generators for the answer pipeline.

* ``ExtractiveLLM``   — deterministic, dependency-free, offline default. It does
  not *write* an answer: it selects the context sentences most relevant to the
  question and stitches them together with their [PMID:...] tags. Honest about
  being extractive, but enough to exercise and evaluate the whole pipeline with
  no model.
* ``TransformersLLM`` — a local HuggingFace causal LM (chat template, greedy
  decode). Needs the weights on disk (``local_files_only`` by default so it
  never reaches for the network). Point ``GEN_MODEL`` / ``load_llm(name=...)``
  at a cached instruct model, or later a LoRA-tuned biomedical generator.
"""
from __future__ import annotations

import os
import re
from typing import Protocol, Sequence, runtime_checkable

from ..trace import get_logger

log = get_logger(__name__)

_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_WORD_RE = re.compile(r"[a-z0-9][a-z0-9\-]*")
_PMID_IN_HEAD = re.compile(r"PMID:\s*([0-9?]+)")
_STOP = frozenset(
    "the and for was were are with that this from have has had not but which "
    "you your are our its into than then they them their what how why when who "
    "does did done can could should would may might will shall about above after "
    "again against all any because been before being below between both during "
    "each few more most other some such only own same too very".split()
)


@runtime_checkable
class LLM(Protocol):
    name: str

    def generate(self, messages: Sequence[dict], max_new_tokens: int = 256) -> str: ...


def _tokens(text: str) -> set[str]:
    return {
        w
        for w in _WORD_RE.findall((text or "").lower())
        if len(w) > 2 and w not in _STOP
    }


def split_sentences(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    return [s.strip() for s in _SENT_RE.split(text) if s.strip()]


class ExtractiveLLM:
    """Select-and-cite baseline generator."""

    name = "extractive"

    def __init__(self, max_sentences: int = 3, min_overlap: int = 1):
        self.max_sentences = max_sentences
        self.min_overlap = min_overlap

    def generate(self, messages: Sequence[dict], max_new_tokens: int = 256) -> str:
        user = next((m["content"] for m in messages if m["role"] == "user"), "")
        question, sources = self._split_user(user)
        q_tok = _tokens(question)

        scored: list[tuple[float, int, str, str]] = []
        seen_sents: set[str] = set()
        for order, (pmid, sent) in enumerate(self._iter_source_sentences(sources)):
            key = " ".join(_WORD_RE.findall(sent.lower()))
            if key in seen_sents:
                continue
            seen_sents.add(key)
            overlap = len(q_tok & _tokens(sent))
            if overlap >= self.min_overlap:
                denom = (len(_tokens(sent)) ** 0.5) or 1.0
                scored.append((overlap / denom, order, sent, pmid))

        log.debug("ExtractiveLLM: %d candidate sentence(s) overlap the question", len(scored))
        if not scored:
            log.debug("ExtractiveLLM: no overlap -> abstaining")
            return (
                "The retrieved sources do not contain enough information to "
                "answer this question."
            )
        scored.sort(key=lambda t: (-t[0], t[1]))
        picked = scored[: self.max_sentences]
        picked.sort(key=lambda t: t[1])  # restore reading order

        out = []
        for _s, _o, sent, pmid in picked:
            sent = sent.rstrip(".")
            out.append(f"{sent} [PMID:{pmid}].")
        log.debug("ExtractiveLLM: stitched %d sentence(s), citing %s",
                  len(picked), sorted({p for *_x, p in picked}))
        return " ".join(out)

    # -- helpers ---------------------------------------------------------------
    @staticmethod
    def _split_user(user: str) -> tuple[str, str]:
        q = ""
        if "Question:" in user:
            q = user.split("Question:", 1)[1]
            q = q.split("\n", 1)[0].strip()
        src = user.split("Sources:", 1)[1] if "Sources:" in user else user
        src = src.rsplit("Answer", 1)[0]
        return q, src

    @staticmethod
    def _iter_source_sentences(sources: str):
        cur_pmid = "?"
        for line in sources.splitlines():
            line = line.strip()
            if not line:
                continue
            head = _PMID_IN_HEAD.search(line) if line.startswith("[S") else None
            if head:
                cur_pmid = head.group(1)
                # a source header may also carry trailing text on the same line
                body = line.split(")", 1)[1].strip() if ")" in line else ""
                line = body
            for sent in split_sentences(line):
                if len(sent) > 15:
                    yield cur_pmid, sent


class TransformersLLM:
    """Local HuggingFace causal LM, loaded lazily."""

    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
        local_files_only: bool = True,
        device: str | None = None,
    ):
        self.name = model_name
        self._local_only = local_files_only
        self._device = device
        self._model = None
        self._tok = None

    def _ensure(self) -> None:
        if self._model is not None:
            return
        log.info("TransformersLLM: loading %s (local_files_only=%s)",
                 self.name, self._local_only)
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(
            self.name, local_files_only=self._local_only
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.name, local_files_only=self._local_only
        )
        self._model.eval()
        if self._device:
            self._model.to(self._device)

    def generate(self, messages: Sequence[dict], max_new_tokens: int = 256) -> str:
        self._ensure()
        import torch

        prompt = self._tok.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=True
        )
        inputs = self._tok(prompt, return_tensors="pt")
        if self._device:
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                repetition_penalty=1.3,
                no_repeat_ngram_size=3,
                pad_token_id=self._tok.eos_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self._tok.decode(gen, skip_special_tokens=True).strip()


def load_llm(name: str | None = None) -> LLM:
    """`name` None/"extractive" -> ExtractiveLLM; anything else -> TransformersLLM.
    Falls back to ExtractiveLLM if the model cannot be loaded offline."""
    name = name or os.environ.get("GEN_MODEL", "extractive")
    if name in ("extractive", "none", ""):
        log.info("load_llm: using ExtractiveLLM (deterministic, offline)")
        return ExtractiveLLM()
    try:
        log.debug("load_llm: trying TransformersLLM(%r)", name)
        llm = TransformersLLM(name)
        llm._ensure()
        return llm
    except Exception as e:  # no weights cached / no torch
        log.warning("load_llm: could not load %r offline (%s); falling back to ExtractiveLLM", name, e)
        print(f"[generate] could not load {name!r} offline ({e}); using ExtractiveLLM")
        return ExtractiveLLM()
