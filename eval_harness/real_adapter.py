import os
import glob
from eval_harness.adapter import RAGPipelineAdapter
from sentence_transformers import SentenceTransformer
import chromadb

class LocalPipeline(RAGPipelineAdapter):
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.client = chromadb.Client()
        try:
            self.client.delete_collection(name="rag_docs")
        except:
            pass
        self.collection = self.client.create_collection(name="rag_docs")
        
        documents = []
        ids = []
        for filepath in glob.glob("data/*.txt"):
            doc_id = os.path.basename(filepath).replace(".txt", "")
            with open(filepath, "r") as f:
                text = f.read().strip()
                documents.append(text)
                ids.append(doc_id)
                
        if documents:
            embeddings = self.model.encode(documents).tolist()
            self.collection.add(
                embeddings=embeddings,
                documents=documents,
                ids=ids
            )

    # The signature here MUST match what run_retrieval.py calls
    def retrieve(self, query: str, k: int, mode: str) -> list[str]:
        query_embedding = self.model.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=k
        )
        if results and results['ids'] and len(results['ids']) > 0:
            return results['ids'][0]
        return []

    def generate(self, query: str, chunk_ids: list[str]) -> str:
        return f"Based on {chunk_ids}, the answer is found."