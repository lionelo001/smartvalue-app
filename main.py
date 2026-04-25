"""
SmartValue Scanner — FastAPI Backend
"""
from __future__ import annotations
import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
from scanner_core import SmartValueScanner, DEFAULT_UNIVERSE

app = FastAPI(title="SmartValue Scanner API")

API_KEY = os.environ.get("FMP_API_KEY", "")
print(f"[DEBUG] API_KEY present: {bool(API_KEY)}, length: {len(API_KEY)}, first4: {API_KEY[:4] if API_KEY else 'EMPTY'}")

# ── Modèles ──────────────────────────────────────────────────
class ScanRequest(BaseModel):
    sectors: list[str] = list(DEFAULT_UNIVERSE.keys())
    min_score: float = 35
    min_confidence: float = 50
    top_n: int = 15

class SearchRequest(BaseModel):
    ticker: str

# ── Routes API ───────────────────────────────────────────────
@app.get("/api/sectors")
def get_sectors():
    return {"sectors": list(DEFAULT_UNIVERSE.keys())}

@app.post("/api/scan")
def scan(req: ScanRequest):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="Clé API FMP manquante.")
    universe = {k: DEFAULT_UNIVERSE[k] for k in req.sectors if k in DEFAULT_UNIVERSE}
    if not universe:
        raise HTTPException(status_code=400, detail="Aucun secteur sélectionné.")
    
    # Test rapide sur le premier ticker pour debug
    from scanner_core import FMPClient, fetch_metrics
    client = FMPClient(API_KEY)
    first_sector = list(universe.keys())[0]
    first_ticker = universe[first_sector][0]
    
    print(f"[DEBUG] Test ticker: {first_ticker}")
    profile = client.get_profile(first_ticker)
    print(f"[DEBUG] Profile keys: {list(profile.keys()) if profile else 'None'}")
    quote = client.get_quote(first_ticker)
    print(f"[DEBUG] Quote PE: {quote.get('pe') if quote else 'None'}")
    ratios = client.get_ratios(first_ticker)
    print(f"[DEBUG] Ratios keys: {list(ratios.keys())[:5] if ratios else 'None'}")
    m = fetch_metrics(first_ticker, client)
    print(f"[DEBUG] Metrics: {m}")
    
    scanner = SmartValueScanner(api_key=API_KEY, universe=universe)
    results = scanner.scan(min_score=req.min_score, min_confidence=req.min_confidence)
    print(f"[DEBUG] Results count: {len(results)}")
    return {"results": results[:req.top_n], "total": len(results)}

@app.post("/api/search")
def search(req: SearchRequest):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="Clé API FMP manquante.")
    scanner = SmartValueScanner(api_key=API_KEY)
    result = scanner.scan_ticker(req.ticker.upper().strip())
    if not result:
        raise HTTPException(status_code=404, detail=f"Ticker introuvable : {req.ticker}")
    return result

@app.get("/api/health")
def health():
    return {"status": "ok", "fmp_key": bool(API_KEY)}

# ── Static files + SPA ───────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return FileResponse("static/index.html")
