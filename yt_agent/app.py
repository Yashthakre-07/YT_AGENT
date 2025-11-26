# app.py
import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Try to import run_agent from typical modules
run_agent = None
_import_errors = []
for modname in ("agent", "agent2", "AGENT4_fixed_final"):
    try:
        module = __import__(modname)
        if hasattr(module, "run_agent"):
            run_agent = getattr(module, "run_agent")
            break
    except Exception as e:
        _import_errors.append((modname, repr(e)))

# Path to frontend build folder
FRONTEND_BUILD_DIR = Path(__file__).with_name("frontend_dist")

app = FastAPI(title="AGENT4 Combined API + Frontend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- API --------
class AskRequest(BaseModel):
    video_id: str
    question: Optional[str] = ""

@app.post("/api/ask")
async def api_ask(req: AskRequest):
    if run_agent is None:
        return JSONResponse({
            "ok": False,
            "error": "run_agent bridge not found. Ensure agent2.py or agent.py exports run_agent(video_id, question).",
            "import_attempts": [{"module": m, "error": e} for m, e in _import_errors]
        }, status_code=500)

    try:
        # run_agent expected to return a dict with 'answer' and 'summary'
        result = run_agent(req.video_id, req.question)
        if not isinstance(result, dict):
            return JSONResponse({
                "ok": False,
                "error": "run_agent did not return a dict.",
                "type": str(type(result)),
                "raw": str(result)
            }, status_code=500)

        return {"ok": True, "answer": result.get("answer", ""), "summary": result.get("summary", "")}

    except Exception as exc:
        return JSONResponse({
            "ok": False,
            "error": "Exception while running run_agent",
            "exception": repr(exc)
        }, status_code=500)

@app.get("/health")
async def health():
    return {"ok": True}

# ---- Serve frontend static files ----
# Mount static AFTER API route definitions so /api/* is handled by FastAPI.
if FRONTEND_BUILD_DIR.exists() and FRONTEND_BUILD_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_BUILD_DIR), html=True), name="frontend")
else:
    @app.get("/")
    async def index():
        return JSONResponse({
            "ok": False,
            "msg": f"Frontend not found in: {FRONTEND_BUILD_DIR.resolve()}"
        })

# -------- WEB SERVER START --------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
