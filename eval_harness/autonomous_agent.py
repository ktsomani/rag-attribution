import os
import shutil
import asyncio
import json
import subprocess
from eval_harness.proposer import propose_harness_fix

async def run_real_evaluation(adapter_file: str) -> dict:
    """
    Runs the evaluation harness in an isolated subprocess with a timeout.
    Parses the actual stdout to extract the MRR.
    """
    # Temporarily swap the adapter file
    original_adapter = "eval_harness/real_adapter.py"
    backup_adapter = "eval_harness/real_adapter_backup.py"
    shutil.copy(original_adapter, backup_adapter)
    shutil.copy(adapter_file, original_adapter)

    try:
        # SANDBOX FIX: Use subprocess with timeout and capture output
        process = await asyncio.create_subprocess_shell(
            "uv run python -m eval_harness.run_retrieval",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=60.0)
            output = stdout.decode()
            
            # REAL EVALUATION FIX: Parse the actual output
            # Look for the hybrid_rerank line: "hybrid_rerank | 0.803 | 0.634"
            mrr = 0.0
            for line in output.split("\n"):
                if line.startswith("hybrid_rerank"):
                    parts = line.split("|")
                    mrr = float(parts[2].strip())
            
            return {"status": "success", "mrr": mrr, "output": output}

        except asyncio.TimeoutError:
            process.kill()
            return {"status": "timeout", "mrr": 0.0, "error": "Sandbox timeout"}
            
    except Exception as e:
        return {"status": "exception", "mrr": 0.0, "error": str(e)}
        
    finally:
        # Restore original adapter
        shutil.copy(backup_adapter, original_adapter)
        os.remove(backup_adapter)

async def run_autonomous_loop():
    print("🚀 Starting Autonomous Self-Healing RAG Loop...\n")
    
    # 1. Run Baseline
    baseline_result = await run_real_evaluation("eval_harness/real_adapter.py")
    baseline_mrr = baseline_result["mrr"]
    print(f"📊 Baseline MRR: {baseline_mrr}")
    
    # 2. Trigger Proposer
    print("\n⚠️ Triggering Proposer...")
    with open("eval_harness/real_adapter.py", "r") as f:
        broken_code = f.read()
        
    # Note: In reality, you'd pass a real failure report here. 
    # For the demo, we pass the MRR.
    fixed_code = propose_harness_fix({"mrr": baseline_mrr}, broken_code)
    
    staging_file = "eval_harness/real_adapter_staging.py"
    with open(staging_file, "w") as f:
        f.write(fixed_code)
        
    # 3. Run Eval on Staged Code
    print("\n🧪 Testing the Proposer's fix in Sandbox...")
    staged_result = await run_real_evaluation(staging_file)
    staged_mrr = staged_result["mrr"]
    print(f"📊 Staged MRR: {staged_mrr}")
    
    # 4. CI Gate
    print("\n🚦 CI Gate Decision:")
    if staged_mrr > baseline_mrr:
        print("✅ PASS: MRR improved. Promoting patch.")
        shutil.copy(staging_file, "eval_harness/real_adapter.py")
    else:
        print("❌ FAIL: MRR did not improve. Quarantining patch.")
        os.remove(staging_file)

if __name__ == "__main__":
    asyncio.run(run_autonomous_loop())