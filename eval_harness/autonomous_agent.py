import os
import shutil
import subprocess
from eval_harness.proposer import propose_harness_fix

def run_evaluation(adapter_file: str) -> dict:
    """Runs the evaluation harness and returns the results."""
    # For this demo, we simulate reading the failure mix.
    # In production, you would run your run_retrieval.py and parse the JSON output.
    # If the file is the baseline, we simulate high failure. If it's the fixed version, we simulate success.
    if adapter_file == "eval_harness/real_adapter.py":
        return {"retrieval_gap": 0.05, "rank_inversion": 0.35, "evidence_under_use": 0.10}
    else:
        return {"retrieval_gap": 0.05, "rank_inversion": 0.00, "evidence_under_use": 0.05}

def run_autonomous_loop():
    print("🚀 Starting Autonomous Self-Healing RAG Loop...\n")
    
    # 1. Run Baseline Evaluation
    baseline_failures = run_evaluation("eval_harness/real_adapter.py")
    print(f"📊 Baseline Failure Mix: {baseline_failures}")
    
    # 2. Check if we need to intervene (e.g., Rank Inversion > 20%)
    if baseline_failures["rank_inversion"] < 0.20:
        print("✅ System is healthy. No intervention needed.")
        return

    print("\n⚠️ High Rank Inversion detected. Triggering Proposer...")
    
    # 3. Read the broken code and ask the LLM to fix it
    with open("eval_harness/real_adapter.py", "r", encoding="utf-8") as f:
        broken_code = f.read()

    fixed_code = propose_harness_fix(baseline_failures, broken_code)
    
    # 4. Save the proposed fix to a staging file
    staging_file = "eval_harness/real_adapter_staging.py"
    with open(staging_file, "w") as f:
        f.write(fixed_code)
    print(f"📝 Proposer generated new code at {staging_file}")
    
    # 5. Run Evaluation on the Staging File
    print("\n🧪 Testing the Proposer's fix...")
    new_failures = run_evaluation(staging_file)
    print(f"📊 Staged Failure Mix: {new_failures}")
    
    # 6. The CI Gate: Did the fix actually work?
    print("\n🚦 CI Gate Decision:")
    if new_failures["rank_inversion"] < baseline_failures["rank_inversion"]:
        print("✅ PASS: Rank Inversion decreased. Committing new harness.")
        # Overwrite the original file with the fixed one
        shutil.copy(staging_file, "eval_harness/real_adapter.py")
    else:
        print("❌ FAIL: The fix did not improve metrics. Rolling back.")
        # Delete the staging file
        os.remove(staging_file)

if __name__ == "__main__":
    run_autonomous_loop()