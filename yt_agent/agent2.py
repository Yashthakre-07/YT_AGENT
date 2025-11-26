# agent2.py
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any

AGENT_SCRIPT = str(Path(__file__).with_name("AGENT4_fixed_final.py"))

def run_agent(video_id: str, question: str) -> Dict[str, str]:
    """
    Try in-process import-run first (fast). If import fails, run as subprocess.
    Returns {"answer": str, "summary": str}
    """
    # Fast path: try import & call run_agent directly
    try:
        module = __import__("AGENT4_fixed_final")
        if hasattr(module, "run_agent"):
            try:
                return module.run_agent(video_id, question)
            except Exception as e:
                # If the in-process call raised, continue to subprocess fallback for isolation
                pass
    except Exception:
        # Ignore import errors and fallback to subprocess
        pass

    # Subprocess fallback: set env vars and run script, capture last JSON line
    env = os.environ.copy()
    env["VIDEO_ID"] = video_id
    env["QUESTION"] = question

    try:
        proc = subprocess.run(
            [sys.executable, AGENT_SCRIPT],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            timeout=900  # increase if your agent can take longer
        )
    except Exception as e:
        return {"answer": f"Subprocess failed to start: {repr(e)}", "summary": ""}

    # Prefer stdout; parse last non-empty line as JSON
    out_lines = [l.strip() for l in proc.stdout.splitlines() if l.strip()]
    if not out_lines:
        # if no stdout, include stderr for debugging
        err_text = proc.stderr.strip()
        return {"answer": f"No output from agent. stderr: {err_text}", "summary": ""}

    last = out_lines[-1]
    try:
        parsed = json.loads(last)
        if isinstance(parsed, dict):
            return {"answer": parsed.get("answer", ""), "summary": parsed.get("summary", "")}
        else:
            return {"answer": "Agent returned JSON but not dict.", "summary": str(parsed)}
    except json.JSONDecodeError:
        # helpful debug
        debug_text = "Could not parse JSON from agent. Last stdout line:\n" + last + "\n\nFULL STDOUT:\n" + proc.stdout + "\n\nSTDERR:\n" + proc.stderr
        return {"answer": debug_text, "summary": ""}

# quick manual test when run directly
if __name__ == "__main__":
    print(run_agent("Gfr50f6ZBvo", "What is this video about?"))
