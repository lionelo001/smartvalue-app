"""
SmartValue Scanner — FastAPI Backend V6
"""
from __future__ import annotations
import json
import os
import time
import threading
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from scanner_core import SmartValueScanner, DEFAULT_UNIVERSE

app = FastAPI(title="SmartValue Scanner API")

API_KEY = os.environ.get("FMP_API_KEY", "")
MAINTENANCE = os.environ.get("MAINTENANCE", "false").lower() == "true"
CACHE_INTERVAL = 3600
CACHE_FILE = "cache.json"

_cache = {"results": [], "total": 0, "last_update": None, "updating": False}

def save_cache_to_disk():
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump({"results": _cache["results"], "total": _cache["total"], "last_update": _cache["last_update"]}, f, ensure_ascii=False)
        print(f"[Cache] Sauvegardé ({len(_cache['results'])} résultats)")
    except Exception as e:
        print(f"[Cache] Erreur sauvegarde : {e}")

def load_cache_from_disk():
    if not os.path.exists(CACHE_FILE):
        return False
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("results"):
            _cache["results"] = data["results"]
            _cache["total"] = data["total"]
            _cache["last_update"] = data.get("last_update", "cache disque")
            print(f"[Cache] Chargé depuis disque — {len(_cache['results'])} résultats")
            return True
    except Exception as e:
        print(f"[Cache] Erreur lecture : {e}")
    return False

def refresh_cache():
    if _cache["updating"]:
        return
    _cache["updating"] = True
    try:
        scanner = SmartValueScanner(api_key=API_KEY, universe=DEFAULT_UNIVERSE)
        results = scanner.scan(min_score=20, min_confidence=40)
        if results:
            _cache["results"] = results
            _cache["total"] = len(results)
            _cache["last_update"] = datetime.now().strftime("%H:%M")
            print(f"[Cache] Mis à jour à {_cache['last_update']} — {len(results)} résultats")
            save_cache_to_disk()
        else:
            print("[Cache] Scan vide — retry dans 5 minutes")
            def retry():
                time.sleep(300)
                _cache["updating"] = False
                refresh_cache()
            threading.Thread(target=retry, daemon=True).start()
            return
    except Exception as e:
        print(f"[Cache] Erreur : {e}")
    finally:
        _cache["updating"] = False

def cache_scheduler():
    while True:
        refresh_cache()
        time.sleep(CACHE_INTERVAL)

@app.on_event("startup")
def startup_event():
    load_cache_from_disk()
    threading.Thread(target=cache_scheduler, daemon=True).start()
    print("[Cache] Scheduler démarré")
    try:
        from newsletter import run_scheduler
        threading.Thread(target=run_scheduler, daemon=True).start()
        print("[Newsletter] Scheduler démarré")
    except Exception as e:
        print(f"[Newsletter] Erreur démarrage : {e}")

@app.middleware("http")
async def maintenance_middleware(request: Request, call_next):
    if MAINTENANCE:
        if not request.url.path.startswith("/static"):
            maint_path = os.path.join(os.path.dirname(__file__), "maintenance.html")
            with open(maint_path, "r") as f:
                return HTMLResponse(content=f.read(), status_code=503)
    return await call_next(request)

class ScanRequest(BaseModel):
    sectors: list[str] = list(DEFAULT_UNIVERSE.keys())
    min_score: float = 35
    min_confidence: float = 50
    top_n: int = 25
    profile: str = "universel"

class SearchRequest(BaseModel):
    ticker: str

class WaitlistRequest(BaseModel):
    email: str

@app.get("/api/sectors")
def get_sectors():
    return {"sectors": list(DEFAULT_UNIVERSE.keys())}

@app.post("/api/scan")
def scan(req: ScanRequest):
    if not _cache["results"] and not _cache["updating"]:
        refresh_cache()
    return {"results": _cache["results"], "total": _cache["total"], "last_update": _cache["last_update"], "profile": req.profile}

@app.post("/api/search")
def search(req: SearchRequest):
    scanner = SmartValueScanner(api_key=API_KEY)
    result = scanner.scan_ticker(req.ticker.upper().strip())
    if not result:
        raise HTTPException(status_code=404, detail=f"Ticker introuvable : {req.ticker}")
    return result

@app.get("/api/autocomplete")
def autocomplete(q: str = ""):
    if not q or len(q) < 2:
        return {"results": []}
    try:
        import requests as req_lib
        r = req_lib.get("https://financialmodelingprep.com/api/v3/search", params={"query": q, "limit": 8, "apikey": API_KEY}, timeout=5)
        if r.status_code == 200:
            data = r.json()
            return {"results": [{"symbol": d.get("symbol",""), "name": d.get("name","")} for d in data if d.get("symbol")][:8]}
    except Exception:
        pass
    return {"results": []}

@app.post("/api/waitlist")
async def waitlist(req: WaitlistRequest):
    import requests as req_lib
    BREVO_API_KEY = os.environ.get("BREVO_API_KEY", "")
    if not BREVO_API_KEY:
        raise HTTPException(status_code=500, detail="Clé API manquante")
    try:
        r = req_lib.post("https://api.brevo.com/v3/contacts", headers={"api-key": BREVO_API_KEY, "Content-Type": "application/json"}, json={"email": req.email, "listIds": [2], "updateEnabled": True, "attributes": {"SOURCE": "SmartValue Waitlist"}}, timeout=10)
        if r.status_code in [201, 204, 400]:
            return {"success": True}
        else:
            raise HTTPException(status_code=500, detail=f"Brevo {r.status_code}: {r.text[:200]}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/test-brevo")
def test_brevo():
    key = os.environ.get("BREVO_API_KEY", "")
    return {"key_present": bool(key), "key_length": len(key), "key_start": key[:10] if key else "vide"}

@app.get("/api/newsletter-test-sv2026")
def test_newsletter():
    try:
        from newsletter import send_newsletter
        success = send_newsletter()
        return {"success": success, "message": "Newsletter envoyee !" if success else "Erreur — voir les logs"}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/debug/{ticker}")
def debug_ticker(ticker: str):
    import yfinance as yf
    t = yf.Ticker(ticker.upper())
    info = t.info
    return {"ticker": ticker, "sector": info.get("sector"), "enterpriseToEbitda": info.get("enterpriseToEbitda"), "currency": info.get("currency")}

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    html = """<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8"/>
<meta property="og:title" content="SmartValue — Scanner d'actions fondamental"/>
<meta property="og:description" content="Analysez les actions mondiales — US, Europe, Asie — selon des criteres fondamentaux clairs. Gratuit et en francais."/>
<meta property="og:type" content="website"/>
<meta property="og:url" content="https://smartvaluescanner.com/"/>
<meta property="og:image" content="https://smartvaluescanner.com/preview.png"/>
<meta property="og:image:width" content="1200"/>
<meta property="og:image:height" content="630"/>
<meta property="og:locale" content="fr_FR"/>
<meta name="twitter:card" content="summary_large_image"/>
<meta name="twitter:title" content="SmartValue — Scanner d'actions fondamental"/>
<meta name="twitter:description" content="Analysez les actions mondiales gratuitement. Simple, en francais."/>
<meta name="twitter:image" content="https://smartvaluescanner.com/preview.png"/>
<title>SmartValue — Scanner d'actions fondamental</title>
</head>
<body>
<script>window.location.href='/app'+window.location.search;</script>
</body>
</html>"""
    return HTMLResponse(content=html)

@app.get("/app")
def scanner_app():
    return FileResponse("static/index.html")

@app.get("/preview.png")
def preview_image():
    path = os.path.join(os.path.dirname(__file__), "preview.png")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Preview not found")
    return FileResponse(path, media_type="image/png")

@app.get("/sitemap.xml")
def sitemap():
    from fastapi.responses import Response
    content = '<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"><url><loc>https://smartvaluescanner.com/</loc><changefreq>daily</changefreq><priority>1.0</priority></url><url><loc>https://smartvaluescanner.com/app</loc><changefreq>daily</changefreq><priority>1.0</priority></url></urlset>'
    return Response(content=content, media_type="application/xml")

@app.get("/robots.txt")
def robots():
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse("User-agent: *\nAllow: /\n")
