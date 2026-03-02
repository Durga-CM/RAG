import uvicorn
import os
import sys

# Ensure the root directory is in the python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Fix Windows console UTF-8 emoji support
if sys.stdout and getattr(sys.stdout, "encoding", "utf-8").lower() != "utf-8":
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass
if sys.stderr and getattr(sys.stderr, "encoding", "utf-8").lower() != "utf-8":
    try: sys.stderr.reconfigure(encoding="utf-8")
    except Exception: pass

if __name__ == "__main__":
    print("\n--- STARTING FASTAPI PRODUCTION RAG SYSTEM ---")
    uvicorn.run("app.api.main:app", host="127.0.0.1", port=8000, reload=True)
