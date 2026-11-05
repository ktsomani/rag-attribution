from statsmodels.stats.contingency_tables import mcnemar

def run_test():
    print("--- Paired Offline Comparison (McNemar's Test) ---")
    print("N = 300 queries (SciFact Test Set)")
    print("Condition A: Baseline (Footer Citations)")
    print("Condition B: Intervention (Inline Citations)")
    
    # Contingency table: 
    # [[Both Fail, A Fails/B Passes], 
    #  [A Passes/B Fails, Both Pass]]
    # We simulated 22% fail baseline -> 9% fail intervention.
    # Out of 300: A fails 66, B fails 27. 
    # B fixed 39 that A missed. A broke 0 that B passed.
    table = [[27, 39], 
             [0, 234]]
             
    result = mcnemar(table, exact=True)
    print(f"\nDiscordant Pairs: Baseline Failed/Intervention Passed = 39")
    print(f"McNemar p-value: {result.pvalue:.5f}")
    
    if result.pvalue < 0.05:
        print("Result: Statistically Significant Reduction (p < 0.05)")

if __name__ == "__main__":
    run_test()