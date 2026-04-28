"""
SmartValue Scanner — FastAPI Backend
"""
from __future__ import annotations
import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel
from scanner_core import SmartValueScanner, DEFAULT_UNIVERSE

app = FastAPI(title="SmartValue Scanner API")

API_KEY = os.environ.get("FMP_API_KEY", "")

class ScanRequest(BaseModel):
    sectors: list[str] = list(DEFAULT_UNIVERSE.keys())
    min_score: float = 35
    min_confidence: float = 50
    top_n: int = 15

class SearchRequest(BaseModel):
    ticker: str

@app.get("/api/sectors")
def get_sectors():
    return {"sectors": list(DEFAULT_UNIVERSE.keys())}

@app.post("/api/scan")
def scan(req: ScanRequest):
    universe = {k: DEFAULT_UNIVERSE[k] for k in req.sectors if k in DEFAULT_UNIVERSE}
    if not universe:
        raise HTTPException(status_code=400, detail="Aucun secteur sélectionné.")
    scanner = SmartValueScanner(api_key=API_KEY, universe=universe)
    results = scanner.scan(min_score=req.min_score, min_confidence=req.min_confidence)
    return {"results": results[:req.top_n], "total": len(results)}

@app.post("/api/search")
def search(req: SearchRequest):
    scanner = SmartValueScanner(api_key=API_KEY)
    result = scanner.scan_ticker(req.ticker.upper().strip())
    if not result:
        raise HTTPException(status_code=404, detail=f"Ticker introuvable : {req.ticker}")
    return result

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.get("/api/autocomplete")
def autocomplete(q: str = ""):
    if not q or len(q) < 2:
        return {"results": []}
    try:
        import requests as req
        r = req.get(
            f"https://financialmodelingprep.com/api/v3/search",
            params={"query": q, "limit": 8, "apikey": API_KEY},
            timeout=5
        )
        if r.status_code == 200:
            data = r.json()
            results = [{"symbol": d.get("symbol",""), "name": d.get("name","")} for d in data if d.get("symbol")]
            return {"results": results[:8]}
    except Exception:
        pass
    return {"results": []}

# Static files pour l'app scanner
app.mount("/static", StaticFiles(directory="static"), name="static")

# Redirection / vers /app
@app.get("/")
def root():
    return RedirectResponse(url="/app", status_code=301)

# Scanner sur /app
@app.get("/app")
def scanner_app():
    return FileResponse("static/index.html")

# Preview image pour les réseaux sociaux
@app.get("/preview.png")
def preview_image():
    import os
    path = os.path.join(os.path.dirname(__file__), "preview.png")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Preview not found")
    return FileResponse(path, media_type="image/png")
