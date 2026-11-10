import os
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

class NeuralReranker:
    """Neural cross-encoder reranker with pair caching and graceful CPU/GPU handling."""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self._score_cache: Dict[Tuple[str, str], float] = {}
        self._load_model()

    def _load_model(self):
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name, device=self.device)
        except Exception as e:
            print(f"Warning: Could not load CrossEncoder model {self.model_name}: {e}. Running in passthrough mode.")
            self.model = None

    def rerank(self, query: str, candidate_texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Computes neural relevance scores for (query, document) pairs.
        Uses in-memory cache to skip redundant evaluations.
        """
        if self.model is None or len(candidate_texts) == 0:
            return np.zeros(len(candidate_texts), dtype=np.float32)

        pairs_to_compute = []
        indices_to_compute = []
        scores = np.zeros(len(candidate_texts), dtype=np.float32)

        for i, text in enumerate(candidate_texts):
            key = (query, text[:250])  # Key based on query and document snippet
            if key in self._score_cache:
                scores[i] = self._score_cache[key]
            else:
                pairs_to_compute.append((query, text))
                indices_to_compute.append(i)

        if pairs_to_compute:
            computed_scores = self.model.predict(pairs_to_compute, batch_size=batch_size, show_progress_bar=False)
            for idx, raw_score in zip(indices_to_compute, computed_scores):
                sc = float(raw_score)
                scores[idx] = sc
                key = (query, candidate_texts[idx][:250])
                self._score_cache[key] = sc

        return scores
