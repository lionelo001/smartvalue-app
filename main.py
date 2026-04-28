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
            # Contact déjà existant avec updateEnabled devrait passer, retourner succès
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

# Page racine avec meta tags OG pour les réseaux sociaux
@app.get("/")
def root():
    html = """<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8"/>
<meta property="og:title" content="SmartValue — Scanner d'actions fondamental"/>
<meta property="og:description" content="Analysez les actions mondiales — US, Europe, Asie — selon des critères fondamentaux clairs. Gratuit et en français."/>
<meta property="og:type" content="website"/>
<meta property="og:url" content="https://smartvaluescanner.com/"/>
<meta property="og:image" content="https://smartvaluescanner.com/preview.png"/>
<meta property="og:image:width" content="1200"/>
<meta property="og:image:height" content="630"/>
<meta property="og:locale" content="fr_FR"/>
<meta name="twitter:card" content="summary_large_image"/>
<meta name="twitter:title" content="SmartValue — Scanner d'actions fondamental"/>
<meta name="twitter:description" content="Analysez les actions mondiales gratuitement. Simple, en français."/>
<meta name="twitter:image" content="https://smartvaluescanner.com/preview.png"/>
<meta http-equiv="refresh" content="0;url=/app"/>
<title>SmartValue — Scanner d'actions fondamental</title>
</head>
<body>
<script>window.location.href='/app';</script>
</body>
</html>"""
    from fastapi.responses import HTMLResponse
    return HTMLResponse(content=html)

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
    """Endpoint temporaire pour débugger les données yfinance"""
    import yfinance as yf
    t = yf.Ticker(ticker.upper())
    info = t.info
    return {
        "ticker": ticker,
        "enterpriseToEbitda": info.get("enterpriseToEbitda"),
        "enterpriseValue": info.get("enterpriseValue"),
        "ebitda": info.get("ebitda"),
        "netIncome": info.get("netIncome"),
        "taxProvision": info.get("taxProvision"),
        "incomeTaxExpense": info.get("incomeTaxExpense"),
        "interestExpense": info.get("interestExpense"),
        "depreciationAndAmortization": info.get("depreciationAndAmortization"),
        "totalDepreciationAndAmortization": info.get("totalDepreciationAndAmortization"),
        "operatingCashflow": info.get("operatingCashflow"),
        "ebitdaMargins": info.get("ebitdaMargins"),
        "totalRevenue": info.get("totalRevenue"),
        "currency": info.get("currency"),
        "financialCurrency": info.get("financialCurrency"),
    }
