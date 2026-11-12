# RAG Failure Attribution & Regression Gate

A production-grade evaluation harness that diagnoses *why* a RAG pipeline fails, and uses a CI regression gate to prevent silent model degradation.

## 1. The Core Build
Evaluated retrieval on the **SciFact BEIR test split (300 queries, official qrels)** using a failure taxonomy to separate retrieval-stage from generation-stage errors.
- **Dense:** Recall@10 0.783 | MRR 0.579 | nDCG@10 0.628
- **Lexical (BM25):** Recall@10 0.693 | MRR 0.509 | nDCG@10 0.553
- **Hybrid+RRF:** Recall@10 0.803 | MRR 0.634 | nDCG@10 0.675

## 2. The Production Edge: The CI Gate
Aggregate accuracy is a vanity metric. This harness uses **bootstrap 95% confidence intervals** to detect statistically significant shifts in failure-mix proportions.
- **The Catch:** Caught a regression where aggregate accuracy remained flat at **82%**, but the retrieval-attributable share of failures spiked from **10% to 28%** (n=54 failures, bootstrap 95% CI on the delta [0.04, 0.32]). The CI gate successfully blocked the change.

## 3. The Autonomous Edge: Harness Learning
Built an LLM Proposer (Groq API) to auto-generate pipeline patches based on execution reports.
- **Gate Validation:** Evaluated all 9 LLM-generated patches against ground truth. The gate agreed 9/9 times, promoting 3 patches that improved MRR to **0.634**, and quarantining 6 that would have regressed metrics or thrown exceptions.

## 4. The Product Edge: Offline A/B Evaluation
Conducted a paired offline comparison (n=300 queries) to test the impact of inline citations vs. footer citations.
- **Result:** Evidence under-use dropped from 22% to 9% (McNemar p < 0.001), with no measurable change to retrieval latency.

## 5. LLM-as-a-Judge & Cost Tracking
*   **Evaluation on Unlabeled Data:** Implemented an LLM-as-a-Judge pipeline to score answer faithfulness (1-5) on data without exact gold matches. Validated the judge against human annotations using **Cohen's Kappa (κ = 1.0 on a controlled n=10 test set)** to ensure the judge wasn't hallucinating.
*   **Token Economics:** Added token-cost tracking to the evaluation loop. A full 900-query regression suite costs **$0.0081**, proving that automated regression testing is economically viable for CI/CD.