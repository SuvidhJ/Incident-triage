import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_inference_stdout_uses_strict_bracketed_lines():
    env = os.environ.copy()
    env["AGENT_MODE"] = "heuristic"
    env.pop("HF_TOKEN", None)
    env.pop("OPENAI_API_KEY", None)

    result = subprocess.run(
        [sys.executable, "inference.py"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert lines
    assert all(re.match(r"^\[(START|STEP|END)\] ", line) for line in lines)

    start_count = sum(line.startswith("[START]") for line in lines)
    end_count = sum(line.startswith("[END]") for line in lines)

    assert start_count == 3
    assert end_count == 3
