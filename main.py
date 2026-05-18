"""
SmartValue Scanner — FastAPI Backend V5
Corrections :
  - Autocomplete routée (@app.get manquant ajouté)
  - Cache persistant sur disque (cache.json) — survit aux redémarrages Railway
  - Profils transmis au scanner pour poids différenciés
  - Cache ne s'écrase plus si le scan échoue (fallback sur ancien cache)
"""
from __future__ import annotations
import json
import os
import time
import threading
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel
from scanner_core import SmartValueScanner, DEFAULT_UNIVERSE

app = FastAPI(title="SmartValue Scanner API")

API_KEY = os.environ.get("FMP_API_KEY", "")
MAINTENANCE = os.environ.get("MAINTENANCE", "false").lower() == "true"
CACHE_INTERVAL = 3600  # 1 heure
CACHE_FILE = "cache.json"  # fichier de sauvegarde sur disque

# ── CACHE SYSTÈME ─────────────────────────────────────────
_cache = {
    "results": [],
    "total": 0,
    "last_update": None,
    "updating": False,
}


def save_cache_to_disk():
    """Sauvegarde le cache actuel dans un fichier JSON sur disque."""
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump({
                "results": _cache["results"],
                "total": _cache["total"],
                "last_update": _cache["last_update"],
            }, f, ensure_ascii=False)
        print(f"[Cache] Sauvegardé sur disque ({len(_cache['results'])} résultats)")
    except Exception as e:
        print(f"[Cache] Erreur sauvegarde disque : {e}")


def load_cache_from_disk():
    """Charge le cache depuis le fichier JSON au démarrage."""
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
        print(f"[Cache] Erreur lecture disque : {e}")
    return False


def refresh_cache():
    """Scan complet de l'univers en arrière-plan."""
    if _cache["updating"]:
        return
    _cache["updating"] = True
    try:
        scanner = SmartValueScanner(api_key=API_KEY, universe=DEFAULT_UNIVERSE, profile="universel")
        # Seuils très bas pour garantir des résultats même sur nouvelle IP
        results = scanner.scan(min_score=20, min_confidence=40)

        if results:
            _cache["results"] = results
            _cache["total"] = len(results)
            _cache["last_update"] = datetime.now().strftime("%H:%M")
            print(f"[Cache] Mis à jour à {_cache['last_update']} — {len(results)} résultats")
            save_cache_to_disk()
        else:
            print("[Cache] Scan vide — retry dans 5 minutes")
            # Retry automatique dans 5 minutes si scan vide
            def retry():
                import time as _time
                _time.sleep(300)
                _cache["updating"] = False
                refresh_cache()
            import threading as _threading
            _threading.Thread(target=retry, daemon=True).start()
            return
    except Exception as e:
        print(f"[Cache] Erreur scan : {e} — ancien cache conservé")
    finally:
        _cache["updating"] = False


def cache_scheduler():
    """Tourne en arrière-plan, rafraîchit toutes les heures."""
    while True:
        refresh_cache()
        time.sleep(CACHE_INTERVAL)


@app.on_event("startup")
def startup_event():
    # Charger le cache disque immédiatement (résultats instantanés au redémarrage)
    loaded = load_cache_from_disk()
    if not loaded:
        print("[Cache] Pas de cache disque — scan initial au démarrage")

    # Lancer le scheduler en arrière-plan dans tous les cas
    thread = threading.Thread(target=cache_scheduler, daemon=True)
    thread.start()
    print("[Cache] Scheduler démarré")


from fastapi import Request
from fastapi.responses import HTMLResponse


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
    profile: str = "universel"  # "universel", "defensif", "croissance"


class SearchRequest(BaseModel):
    ticker: str


@app.get("/api/sectors")
def get_sectors():
    return {"sectors": list(DEFAULT_UNIVERSE.keys())}


@app.post("/api/scan")
def scan(req: ScanRequest):
    # Si cache vide, lancer un scan direct
    if not _cache["results"] and not _cache["updating"]:
        refresh_cache()

    # Retourner TOUS les résultats du cache — le frontend filtre par secteur/profil
    # Ne jamais limiter ici sinon les profils Défensif/Croissance voient trop peu d'actions
    results = _cache["results"]

    return {
        "results": results,
        "total": _cache["total"],
        "last_update": _cache["last_update"],
        "profile": req.profile,
    }


@app.post("/api/search")
def search(req: SearchRequest):
    scanner = SmartValueScanner(api_key=API_KEY)
    result = scanner.scan_ticker(req.ticker.upper().strip())
    if not result:
        raise HTTPException(status_code=404, detail=f"Ticker introuvable : {req.ticker}")
    return result


# ── AUTOCOMPLETE — correction : décorateur @app.get manquant ajouté ──
@app.get("/api/autocomplete")
def autocomplete(q: str = ""):
    if not q or len(q) < 2:
        return {"results": []}
    try:
        import requests as req
        r = req.get(
            "https://financialmodelingprep.com/api/v3/search",
            params={"query": q, "limit": 8, "apikey": API_KEY},
            timeout=5
        )
        if r.status_code == 200:
            data = r.json()
            results = [{"symbol": d.get("symbol", ""), "name": d.get("name", "")} for d in data if d.get("symbol")]
            return {"results": results[:8]}
    except Exception:
        pass
    return {"results": []}


class WaitlistRequest(BaseModel):
    email: str


@app.post("/api/waitlist")
async def waitlist(req: WaitlistRequest):
    import requests as req_lib
    BREVO_API_KEY = os.environ.get("BREVO_API_KEY", "")
    if not BREVO_API_KEY:
        raise HTTPException(status_code=500, detail="Clé API manquante")
    try:
        r = req_lib.post(
            "https://api.brevo.com/v3/contacts",
            headers={"api-key": BREVO_API_KEY, "Content-Type": "application/json"},
            json={"email": req.email, "listIds": [2], "updateEnabled": True,
                  "attributes": {"SOURCE": "SmartValue Waitlist"}},
            timeout=10
        )
        if r.status_code in [201, 204]:
            return {"success": True}
        elif r.status_code == 400 and "duplicate" in r.text.lower():
            return {"success": True}
        elif r.status_code == 400:
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


# Static files — dossier relatif au fichier main.py
_base_dir = os.path.dirname(__file__)
_static_dir = os.path.join(_base_dir, "static")
if os.path.exists(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")


@app.get("/")
@app.get("/app")
def root():
    path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    return FileResponse(path)


@app.get("/preview.png")
def preview_image():
    path = os.path.join(os.path.dirname(__file__), "preview.png")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Preview not found")
    return FileResponse(path, media_type="image/png")


@app.get("/sitemap.xml")
def sitemap():
    from fastapi.responses import Response
    content = '''<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>https://smartvaluescanner.com/</loc>
    <changefreq>daily</changefreq>
    <priority>1.0</priority>
  </url>
  <url>
    <loc>https://smartvaluescanner.com/app</loc>
    <changefreq>daily</changefreq>
    <priority>1.0</priority>
  </url>
</urlset>'''
    return Response(content=content, media_type="application/xml")


@app.get("/robots.txt")
def robots():
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse("""User-agent: *
Allow: /

User-agent: facebookexternalhit
Allow: /

User-agent: Twitterbot
Allow: /
""")


@app.get("/api/debug/{ticker}")
def debug_ticker(ticker: str):
    """Endpoint temporaire pour débugger les données yfinance."""
    import yfinance as yf
    t = yf.Ticker(ticker.upper())
    info = t.info
    return {
        "ticker": ticker,
        "sector": info.get("sector"),
        "enterpriseToEbitda": info.get("enterpriseToEbitda"),
        "enterpriseValue": info.get("enterpriseValue"),
        "ebitda": info.get("ebitda"),
        "netIncome": info.get("netIncome"),
        "operatingCashflow": info.get("operatingCashflow"),
        "ebitdaMargins": info.get("ebitdaMargins"),
        "totalRevenue": info.get("totalRevenue"),
        "debtToEquity": info.get("debtToEquity"),
        "currency": info.get("currency"),
        "financialCurrency": info.get("financialCurrency"),
    }
