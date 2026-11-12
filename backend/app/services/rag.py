
import os
import re
import uuid
import numpy as np
import chromadb

from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from app.config import settings


class RAGEngine:

    def __init__(self):
        os.makedirs(settings.chroma_path, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=settings.chroma_path
        )

        self.collection = self.client.get_or_create_collection(
            "youtube_chunks",
            metadata={"hnsw:space": "cosine"}
        )

        self.embedder = SentenceTransformer(
            settings.embedding_model
        )

        # Load reranker only when enabled
        self.reranker = None

        if settings.enable_reranker:
            self.reranker = CrossEncoder(
                settings.reranker_model
            )

    def chunk(self, snippets, max_chars=1000, overlap=2):
        result = []
        current = []
        size = 0

        def save(items):
            if items:
                result.append({
                    "text": " ".join(x["text"] for x in items),
                    "start": items[0]["start"],
                    "end": items[-1]["end"]
                })

        for item in snippets:
            n = len(item["text"])

            if current and size + n > max_chars:
                save(current)
                current = current[-overlap:] if overlap else []
                size = sum(len(x["text"]) for x in current)

            current.append(item)
            size += n

        save(current)
        return result

    def ingest(self, video_id, snippets):
        chunks = self.chunk(snippets)

        if not chunks:
            raise ValueError("No transcript chunks available.")

        self.collection.delete(
            where={"video_id": video_id}
        )

        vectors = self.embedder.encode(
            [c["text"] for c in chunks],
            normalize_embeddings=True
        ).tolist()

        ids = [
            f"{video_id}_{uuid.uuid4().hex}"
            for _ in chunks
        ]

        metas = [
            {
                "video_id": video_id,
                "start": float(c["start"]),
                "end": float(c["end"])
            }
            for c in chunks
        ]

        self.collection.add(
            ids=ids,
            documents=[c["text"] for c in chunks],
            embeddings=vectors,
            metadatas=metas
        )

        return chunks

    @staticmethod
    def tokens(text):
        return re.findall(r"[a-z0-9_]+", text.lower())

    def retrieve(self, video_id, question, k=None):

        k = k or settings.top_k

        count = self.collection.count()

        if count == 0:
            return []

        qvec = self.embedder.encode(
            [question],
            normalize_embeddings=True
        )[0]

        found = self.collection.query(
            query_embeddings=[qvec.tolist()],
            n_results=min(count, max(k * 4, 20)),
            where={"video_id": video_id},
            include=[
                "documents",
                "metadatas",
                "embeddings"
            ]
        )

        raw_docs = found.get("documents")
        raw_metas = found.get("metadatas")
        raw_vectors = found.get("embeddings")

        docs = raw_docs[0] if raw_docs is not None and len(raw_docs) else []
        metas = raw_metas[0] if raw_metas is not None and len(raw_metas) else []

        vectors = (
            np.asarray(raw_vectors[0], dtype=float)
            if raw_vectors is not None and len(raw_vectors)
            else np.empty((0, len(qvec)))
        )

        if not docs:
            return []

        # BM25 keyword retrieval
        all_data = self.collection.get(
            where={"video_id": video_id},
            include=["documents"]
        )

        all_docs = all_data.get("documents") or []

        bm25 = (
            BM25Okapi([self.tokens(d) for d in all_docs])
            if all_docs else None
        )

        bm_scores = (
            bm25.get_scores(self.tokens(question))
            if bm25 else []
        )

        bm_order = sorted(
            range(len(all_docs)),
            key=lambda i: bm_scores[i],
            reverse=True
        )

        bm_rank = {
            all_docs[i]: rank + 1
            for rank, i in enumerate(bm_order)
        }

        sem_rank = {
            d: i + 1
            for i, d in enumerate(docs)
        }

        # Reciprocal Rank Fusion
        fused = []

        for doc, meta, vec in zip(docs, metas, vectors):

            score = (
                1 / (60 + sem_rank.get(doc, 999999))
                + 1 / (60 + bm_rank.get(doc, 999999))
            )

            fused.append((score, doc, meta, vec))

        fused.sort(key=lambda x: x[0], reverse=True)

        pool = fused[:min(len(fused), k * 2)]

        if not pool:
            return []

        # Maximal Marginal Relevance
        matrix = np.asarray(
            [x[3] for x in pool],
            dtype=float
        )

        relevance = matrix @ np.asarray(qvec)

        selected = []
        remaining = list(range(len(pool)))

        while remaining and len(selected) < min(k, len(remaining)):

            best = None
            best_score = -1e9

            for idx in remaining:

                redundancy = max(
                    (
                        float(matrix[idx] @ matrix[j])
                        for j in selected
                    ),
                    default=0.0
                )

                score = (
                    0.65 * float(relevance[idx])
                    - 0.35 * redundancy
                )

                if score > best_score:
                    best = idx
                    best_score = score

            selected.append(best)
            remaining.remove(best)

        # Build retrieved results
        results = []

        for idx in selected:
            results.append({
                "text": pool[idx][1],
                **pool[idx][2],
                "score": float(relevance[idx])
            })

        # Cross-Encoder Reranking
        if self.reranker and results:

            pairs = [
                (question, item["text"])
                for item in results
            ]

            scores = self.reranker.predict(pairs)

            for item, score in zip(results, scores):
                item["rerank_score"] = float(score)

            results.sort(
                key=lambda x: x["rerank_score"],
                reverse=True
            )

        return results


rag = RAGEngine()
