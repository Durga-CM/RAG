import uvicorn
import os
import sys

# Ensure the root directory is in the python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    print("\n--- 🚀 STARTING FASTAPI PRODUCTION RAG SYSTEM ---")
    uvicorn.run("app.api.main:app", host="127.0.0.1", port=8000, reload=True)
