#!/usr/bin/env python3
"""
scripts/run_all.py
==================
Convenience script to start all services in the correct order.
Each component runs in its own subprocess.

Usage:
    python scripts/run_all.py

Stop with Ctrl+C — all child processes are killed automatically.
"""

import signal
import subprocess
import sys
import time

PROCESSES = []


def run(cmd: list[str], name: str) -> subprocess.Popen:
    print(f"▶  Starting {name} …")
    p = subprocess.Popen(cmd)
    PROCESSES.append(p)
    return p


def cleanup(*_):
    print("\n⏹  Shutting down all services …")
    for p in PROCESSES:
        p.terminate()
    sys.exit(0)


signal.signal(signal.SIGINT,  cleanup)
signal.signal(signal.SIGTERM, cleanup)

# 1. Ollama (must already be installed)
run(["ollama", "serve"], "Ollama")
time.sleep(2)

# 2. FastAPI inference server
run(
    [sys.executable, "-m", "uvicorn", "inference_api.main:app",
     "--host", "0.0.0.0", "--port", "8000"],
    "Inference API",
)
time.sleep(1)

# 3. Vite dev server for dashboard (requires `npm install` first)
run(["npm", "run", "dev", "--prefix", "dashboard"], "Dashboard")

print("\n✅  All services started.")
print("   Inference API : http://localhost:8000")
print("   Dashboard     : http://localhost:5173")
print("   API Docs      : http://localhost:8000/docs")
print("\nPress Ctrl+C to stop all services.\n")

# Keep alive
while True:
    time.sleep(1)
