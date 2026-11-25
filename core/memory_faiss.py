"""
QuantGuard-MAS — Unit III: Persistent FAISS Long-Term Memory & Feedback Adaptation
Model 5: sentence-transformers/all-MiniLM-L6-v2 (with pure-NumPy fallback if FAISS is absent)
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from core.config import Config


SEED_FINANCIAL_BRIEFS = [
    {
        "text": "RELIANCE.NS (Reliance Industries): Diversified conglomerate across O2C, Jio telecom, and Retail; moderate beta with strong domestic cashflows.",
        "memory_type": "company_brief",
        "symbol": "RELIANCE.NS",
    },
    {
        "text": "TCS.NS (Tata Consultancy Services): Defensive Tier-1 Indian IT exporter with strong operating margins, low debt, and USD revenue sensitivity.",
        "memory_type": "company_brief",
        "symbol": "TCS.NS",
    },
    {
        "text": "HDFCBANK.NS (HDFC Bank): Leading private-sector bank in India; sensitive to RBI repo rate cycles and credit-deposit ratio norms.",
        "memory_type": "company_brief",
        "symbol": "HDFCBANK.NS",
    },
    {
        "text": "INFY.NS (Infosys): Large-cap IT services firm with AI/cloud transformation pipeline; higher earnings-day volatility than TCS.",
        "memory_type": "company_brief",
        "symbol": "INFY.NS",
    },
    {
        "text": "ITC.NS (ITC Ltd): Low-volatility defensive FMCG and consumer conglomerate with steady dividend yield; acts as a portfolio stabilizer during market shocks.",
        "memory_type": "company_brief",
        "symbol": "ITC.NS",
    },
    {
        "text": "Default user risk profile: Max drawdown limit is 15%, single-stock cap is 35%, minimum cash reserve is 10%, risk aversion lambda is 1.5.",
        "memory_type": "user_preference",
        "symbol": "PORTFOLIO",
    },
]


class FAISSLongTermMemory:
    """
    Persistent vector store using all-MiniLM-L6-v2 + FAISS IndexFlatIP
    (falling back seamlessly to exact NumPy cosine similarity if faiss-cpu is missing).
    Supports non-destructive feedback adaptation via is_correction=True.
    """

    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, store_dir: Optional[Path] = None, use_local_embeddings: bool = True):
        self.store_dir = Path(store_dir or Config.FAISS_INDEX_PATH)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.store_dir / "metadata.json"
        self.vecs_path = self.store_dir / "vectors.npy"
        self.faiss_path = self.store_dir / "memory.index"

        self._embedder = None
        self._use_local_embeddings = use_local_embeddings
        self._faiss = None
        try:
            import faiss  # type: ignore

            self._faiss = faiss
        except ImportError:
            self._faiss = None

        self.records: List[Dict] = []
        self.vectors: Optional[np.ndarray] = None
        self.index = None

        self._load()
        if not self.records:
            self._seed_defaults()

    def _get_embedder(self):
        if self._embedder is None and self._use_local_embeddings:
            try:
                from sentence_transformers import SentenceTransformer

                self._embedder = SentenceTransformer(self.EMBEDDING_MODEL)
            except Exception:
                self._embedder = None
        return self._embedder

    def _embed(self, text: str) -> np.ndarray:
        embedder = self._get_embedder()
        if embedder is not None:
            vec = embedder.encode(
                [text], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
            )[0]
            return np.asarray(vec, dtype="float32")

        # Deterministic hashed bag-of-words fallback (384-d unit vector)
        vec = np.zeros(384, dtype="float32")
        for tok in text.lower().split():
            idx = abs(hash(tok)) % 384
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def _rebuild_index(self):
        if self.vectors is None or len(self.vectors) == 0:
            self.index = None
            return
        if self._faiss is not None:
            idx = self._faiss.IndexFlatIP(self.vectors.shape[1])
            idx.add(self.vectors)
            self.index = idx

    def _save(self):
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.records, f, indent=2, ensure_ascii=False)
        if self.vectors is not None and len(self.vectors) > 0:
            np.save(self.vecs_path, self.vectors)
            if self._faiss is not None and self.index is not None:
                self._faiss.write_index(self.index, str(self.faiss_path))

    def _load(self):
        if self.meta_path.exists() and self.vecs_path.exists():
            try:
                self.records = json.loads(self.meta_path.read_text(encoding="utf-8"))
                self.vectors = np.load(self.vecs_path).astype("float32")
                self._rebuild_index()
            except Exception:
                self.records = []
                self.vectors = None
                self.index = None

    def _seed_defaults(self):
        for item in SEED_FINANCIAL_BRIEFS:
            self.add_memory(
                text=item["text"],
                memory_type=item["memory_type"],
                symbol=item["symbol"],
                is_correction=False,
            )

    def add_memory(
        self,
        text: str,
        memory_type: str = "user_preference",
        symbol: str = "PORTFOLIO",
        is_correction: bool = False,
    ) -> Dict:
        """
        Store a memory vector. When is_correction=True, marks matching active
        memories of the same (memory_type, symbol) as inactive (superseded).
        """
        now = datetime.now(timezone.utc).isoformat()
        mem_id = str(uuid.uuid4())[:8]
        superseded_ids = []

        if is_correction:
            for rec in self.records:
                if (
                    rec.get("is_active", True)
                    and rec.get("memory_type") == memory_type
                    and rec.get("symbol") == symbol
                ):
                    rec["is_active"] = False
                    rec["superseded_by"] = mem_id
                    rec["updated_at"] = now
                    superseded_ids.append(rec["id"])

        vec = self._embed(text).reshape(1, -1)
        record = {
            "id": mem_id,
            "text": text.strip(),
            "memory_type": memory_type,
            "symbol": symbol,
            "is_active": True,
            "is_correction": is_correction,
            "supersedes": superseded_ids,
            "superseded_by": None,
            "created_at": now,
            "updated_at": now,
        }

        self.records.append(record)
        self.vectors = vec if self.vectors is None else np.vstack([self.vectors, vec])
        self._rebuild_index()
        self._save()
        return record

    def search(self, query: str, top_k: int = 5, active_only: bool = True) -> List[Dict]:
        if not self.records or self.vectors is None:
            return []

        q_vec = self._embed(query).reshape(1, -1)
        k = min(len(self.records), max(top_k * 3, 10))

        if self._faiss is not None and self.index is not None:
            scores, indices = self.index.search(q_vec, k)
            pairs = [(int(i), float(s)) for i, s in zip(indices[0], scores[0]) if i >= 0]
        else:
            sims = (self.vectors @ q_vec.T).ravel()
            order = np.argsort(-sims)[:k]
            pairs = [(int(i), float(sims[i])) for i in order]

        results = []
        for idx, sim in pairs:
            rec = self.records[idx]
            if active_only and not rec.get("is_active", True):
                continue
            item = dict(rec)
            item["similarity"] = round(sim, 4)
            results.append(item)
            if len(results) >= top_k:
                break
        return results

    def list_all(self, active_only: bool = False) -> List[Dict]:
        if active_only:
            return [r for r in self.records if r.get("is_active", True)]
        return list(self.records)
