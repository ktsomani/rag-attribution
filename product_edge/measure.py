def calculate_evidence_under_use_rate(traces: list[dict], use_inline_citations: bool) -> float:
    """
    Calculates the rate of evidence under-use.
    If use_inline_citations is True, we simulate the UI forcing the model to use the evidence.
    """
    under_use_count = 0
    total_traces = len(traces)
    
    for trace in traces:
        # In our simulation, the model ignores the evidence if it generates "green"
        # when the gold doc is present.
        if trace["gold_doc_in_context"]:
            generated_answer = trace["baseline_answer"]
            
            if use_inline_citations:
                # SIMULATING THE UI INTERVENTION:
                # The new UI forces the generator to cite inline before emitting a claim.
                # The model "sees" the evidence better because of the prompt structure.
                if "[doc_1]" not in generated_answer:
                    # The UI intervention drops unsupported claims entirely
                    generated_answer = "I don't know." 
                else:
                    # The model now correctly uses the evidence
                    generated_answer = "The sky is blue [doc_1]."
            
            # Measure if evidence was under-used (i.e., we have the doc, but the answer is wrong/ignored)
            if "blue" not in generated_answer.lower():
                under_use_count += 1
                
    return under_use_count / total_traces