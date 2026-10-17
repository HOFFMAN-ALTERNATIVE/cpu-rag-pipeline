"""cpurag — a local, CPU-only retrieval-augmented generation pipeline."""

from cpurag.chunking import Chunk, chunk_text
from cpurag.embeddings import Embedder, HashingEmbedder
from cpurag.pipeline import Pipeline
from cpurag.retrieval import BM25Retriever, DenseRetriever, HybridRetriever
from cpurag.store import FlatIndex, Int8Index

__all__ = [
    "BM25Retriever",
    "Chunk",
    "DenseRetriever",
    "Embedder",
    "FlatIndex",
    "HashingEmbedder",
    "HybridRetriever",
    "Int8Index",
    "Pipeline",
    "chunk_text",
]
__version__ = "1.0.0"
