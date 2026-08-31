"""Central configuration, loaded from environment / .env.

Nothing here requires network or API keys at import time, so the module is safe
to import in tests and in --dry-run flows.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

try:
    from dotenv import load_dotenv

    load_dotenv()  # populate os.environ from a local .env if present
except Exception:  # python-dotenv not installed yet — env vars still work
    pass


def _get(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


@dataclass(frozen=True)
class NCBIConfig:
    api_key: str = field(default_factory=lambda: _get("NCBI_API_KEY"))
    tool: str = field(default_factory=lambda: _get("NCBI_TOOL", "biomed-rag"))
    email: str = field(default_factory=lambda: _get("NCBI_EMAIL"))

    @property
    def rate_limit_per_sec(self) -> int:
        # NCBI allows 10 req/s with an API key, 3 without.
        return 10 if self.api_key else 3


@dataclass(frozen=True)
class PineconeConfig:
    api_key: str = field(default_factory=lambda: _get("PINECONE_API_KEY"))
    index_name: str = field(default_factory=lambda: _get("PINECONE_INDEX", "biomed-rag"))
    cloud: str = field(default_factory=lambda: _get("PINECONE_CLOUD", "aws"))
    region: str = field(default_factory=lambda: _get("PINECONE_REGION", "us-east-1"))
    # Hybrid (sparse-dense) search requires the dotproduct metric.
    metric: str = "dotproduct"


@dataclass(frozen=True)
class EmbedConfig:
    model: str = field(default_factory=lambda: _get("EMBED_MODEL", "pritamdeka/S-PubMedBert-MS-MARCO"))
    dim: int = field(default_factory=lambda: int(_get("EMBED_DIM", "768") or "768"))


@dataclass(frozen=True)
class Config:
    ncbi: NCBIConfig = field(default_factory=NCBIConfig)
    pinecone: PineconeConfig = field(default_factory=PineconeConfig)
    embed: EmbedConfig = field(default_factory=EmbedConfig)

    # Namespaces keep abstracts and full-text bodies in separate address spaces
    # inside one Pinecone index, so retrieval can target one or the other.
    DOC_TYPE_ABSTRACT: str = "abstract"
    DOC_TYPE_FULLTEXT: str = "fulltext"


CONFIG = Config()
