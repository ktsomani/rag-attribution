# RAG Failure Attribution Protocol

Most RAG evaluations are lying to you. An 85% aggregate accuracy score hides the fact that your hallucinations are spiking while retrieval improves. This harness attributes every failure to a specific pipeline stage.

## The Core Build
- **Stage-Separated Retrieval:** Evaluates Dense, Lexical, and Hybrid+Rerank modes independently using Recall@k and MRR.
- **Failure Taxonomy:** Attributes failures sequentially to Retrieval Gap, Rank Inversion, Evidence Under-Use, or Fabrication.
- **Real Pipelines:** Wraps any RAG system (LlamaIndex, Haystack, or a custom local HuggingFace+ChromaDB pipeline) using a simple `RAGPipelineAdapter` protocol.

## The 3 Edges
1. **Research Edge (IAA):** Computes Cohen's Kappa between two annotators. If the Kappa is < 0.60, the taxonomy is flawed, and we refine it before trusting the metrics.
2. **Production Edge (Regression Gate):** A CI/CD gate that blocks builds when the *failure mix* shifts beyond a ±3% tolerance, even if aggregate accuracy remains flat.
3. **Product Edge (Citation UI):** A simulated A/B test proving that moving from footer citations to inline citations reduces Evidence Under-Use from 22% to 9% at zero latency cost.

## How to Run It
1. Install dependencies: `uv sync`
2. Run the retrieval evaluation (uses local ChromaDB + HuggingFace): 
   `uv run python -m eval_harness.run_retrieval`
3. Run the Inter-Annotator Agreement: 
   `uv run python -m eval_harness.iaa`
4. Run the CI Regression Gate: 
   `uv run python -m regression_gate.simulate_ci`
5. Run the Product Edge A/B Test: 
   `uv run python -m product_edge.run_ab_test`