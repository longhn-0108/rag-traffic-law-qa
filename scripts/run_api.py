"""Start the FastAPI server."""

import uvicorn

if __name__ == "__main__":
    uvicorn.run("traffic_law_rag.api:app", host="0.0.0.0", port=8000)
