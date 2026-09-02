"""Component 2 — grounded answer generation over retrieved biomedical nodes.

`AnswerPipeline` = retriever -> prompt builder -> LLM -> citation extraction.
The LLM is pluggable: `ExtractiveLLM` (deterministic, offline, the default) or
`TransformersLLM` (a local HuggingFace causal model; needs the weights cached).
`generate.evaluate` scores answers with reference-free grounding proxies
(faithfulness, answer relevancy, citation support, context precision/recall) —
RAGAS-style in spirit, but heuristic and LLM-judge-free so they run offline.
"""
from .llm import ExtractiveLLM, TransformersLLM, load_llm
from .pipeline import Answer, AnswerPipeline
from .prompt import build_messages, format_contexts

__all__ = [
    "ExtractiveLLM",
    "TransformersLLM",
    "load_llm",
    "Answer",
    "AnswerPipeline",
    "build_messages",
    "format_contexts",
]
