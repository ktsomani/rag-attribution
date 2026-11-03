import os
import glob
from eval_harness.adapter import RAGPipelineAdapter
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from rank_bm25 import BM25Okapi

class LocalPipeline(RAGPipelineAdapter):
    def __init__(self):
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.rerank_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        self.documents = []
        self.ids = []
        for filepath in glob.glob("data/*.txt"):
            doc_id = os.path.basename(filepath).replace(".txt", "")
            with open(filepath, "r") as f:
                text = f.read().strip()
                self.documents.append(text)
                self.ids.append(doc_id)
                
        self.client = chromadb.Client()
        try:
            self.client.delete_collection(name="rag_docs")
        except:
            pass
        self.collection = self.client.create_collection(name="rag_docs")
        
        if self.documents:
            embeddings = self.embed_model.encode(self.documents).tolist()
            self.collection.add(embeddings=embeddings, documents=self.documents, ids=self.ids)
            
        tokenized_corpus = [doc.lower().split(" ") for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str, k: int, mode: str) -> list[str]:
        if not self.documents:
            return []

        if mode == "dense":
            query_embedding = self.embed_model.encode([query]).tolist()
            results = self.collection.query(query_embeddings=query_embedding, n_results=k)
            return results['ids'][0] if results['ids'] else []

        elif mode == "lexical":
            tokenized_query = query.lower().split(" ")
            scores = self.bm25.get_scores(tokenized_query)
            ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
            return [self.ids[i] for i in ranked_indices]

        elif mode == "hybrid_rerank":
            dense_res = self.retrieve(query, k=10, mode="dense")
            lexical_res = self.retrieve(query, k=10, mode="lexical")
            
            rrf_scores = {}
            for rank, doc_id in enumerate(dense_res):
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1.0 / (60 + rank + 1))
            for rank, doc_id in enumerate(lexical_res):
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1.0 / (60 + rank + 1))
                
            fused_docs = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:10]
            doc_text_map = {self.ids[i]: self.documents[i] for i in range(len(self.ids))}
            pairs = [[query, doc_text_map[doc_id]] for doc_id in fused_docs]
            
            if not pairs:
                return []
                
            rerank_scores = self.rerank_model.predict(pairs)
            reranked = sorted(zip(fused_docs, rerank_scores), key=lambda x: x[1], reverse=True)
            return [doc_id for doc_id, score in reranked[:k]]

        return []

    def generate(self, query: str, chunk_ids: list[str]) -> str:
        return f"Based on {chunk_ids}, the answer is found."