# startup.py
import sys
import os
sys.path.append(os.path.dirname(__file__))

from api.main import app

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api.main:app", host="0.0.0.0", port=port)