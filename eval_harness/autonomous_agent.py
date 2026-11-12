import os
import shutil
import asyncio
import subprocess
from eval_harness.proposer import propose_harness_fix

def _run_eval_subprocess():
    """Synchronous subprocess call wrapped in a thread."""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    
    try:
        result = subprocess.run(
            ["uv", "run", "python", "-m", "eval_harness.run_retrieval"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            env=env,
            timeout=120 # <--- TIMEOUT ADDED HERE
        )
        return result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return "", "TimeoutExpired: The evaluation took longer than 120 seconds."

async def run_real_evaluation(adapter_file: str) -> float:
    original_adapter = os.path.abspath("eval_harness/real_adapter.py")
    backup_adapter = os.path.abspath("eval_harness/real_adapter_backup.py")
    target_adapter = os.path.abspath(adapter_file)
    
    is_staging = (target_adapter != original_adapter)
    
    if is_staging:
        shutil.copy(original_adapter, backup_adapter)
        shutil.copy(target_adapter, original_adapter)

    try:
        stdout, stderr = await asyncio.to_thread(_run_eval_subprocess)
        
        mrr = 0.0
        for line in stdout.split("\n"):
            if line.startswith("hybrid_rerank"):
                parts = line.split("|")
                mrr = float(parts[2].strip())
        
        if mrr == 0.0 and stderr:
            print(f"⚠️ Subprocess STDERR (first 500 chars):\n{stderr[:500]}...")

        return mrr
            
    except Exception as e:
        print(f"🚨 SANDBOX EXCEPTION: {str(e)}")
        return 0.0
        
    finally:
        if is_staging:
            shutil.copy(backup_adapter, original_adapter)
            os.remove(backup_adapter)

async def run_autonomous_loop():
    print("🚀 Starting Autonomous Self-Healing RAG Loop...\n")
    
    print("📊 Running Baseline Evaluation...")
    baseline_mrr = await run_real_evaluation("eval_harness/real_adapter.py")
    print(f"📊 Baseline MRR: {baseline_mrr}")
    
    print("\n⚠️ Triggering Proposer...")
    with open("eval_harness/real_adapter.py", "r", encoding="utf-8") as f:
        broken_code = f.read()
        
    # AWAIT ADDED HERE
    fixed_code = await propose_harness_fix({"mrr": baseline_mrr}, broken_code)
    
    staging_file = "eval_harness/real_adapter_staging.py"
    with open(staging_file, "w", encoding="utf-8") as f:
        f.write(fixed_code)
        
    print("\n🧪 Testing the Proposer's fix in Sandbox...")
    staged_mrr = await run_real_evaluation(staging_file)
    print(f"📊 Staged MRR: {staged_mrr}")
    
    print("\n🚦 CI Gate Decision:")
    if staged_mrr > baseline_mrr:
        print("✅ PASS: MRR improved. Promoting patch.")
        shutil.copy(staging_file, "eval_harness/real_adapter.py")
    else:
        print("❌ FAIL: MRR did not improve. Quarantining patch.")
        os.remove(staging_file)

if __name__ == "__main__":
    asyncio.run(run_autonomous_loop())