"""
scanner_core.py  —  SmartValue Scanner V4
Source : Financial Modeling Prep (FMP) — fiable, officiel, plan gratuit 250 req/jour
Univers : Mondial (US, Europe, Asie)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests

# =========================
# UNIVERS MONDIAL
# =========================

DEFAULT_UNIVERSE: Dict[str, List[str]] = {
    # ── US ──────────────────────────────────────────────────────
    "Tech US": [
        "AAPL", "MSFT", "GOOGL", "NVDA", "META", "ADBE", "CRM", "ORCL", "INTC", "AMD",
    ],
    "Finance US": [
        "JPM", "BAC", "WFC", "GS", "MS", "AXP", "V", "MA", "BRK-B",
    ],
    "Santé US": [
        "JNJ", "PFE", "UNH", "ABBV", "LLY", "MRK", "ABT", "BMY", "TMO",
    ],
    "Energie US": [
        "XOM", "CVX", "COP", "EOG", "PSX",
    ],
    "Conso US": [
        "PG", "KO", "PEP", "WMT", "COST", "MCD", "NKE", "SBUX", "HD",
    ],
    "Industriels US": [
        "CAT", "HON", "GE", "MMM", "UNP", "UPS", "FDX", "DE", "ETN",
    ],

    # ── EUROPE ──────────────────────────────────────────────────
    "Tech Europe": [
        "ASML", "SAP", "CAP.PA", "STM.PA", "DSY.PA",
        "ERIC-B.ST", "WKL.AS", "PHIA.AS",
    ],
    "Finance Europe": [
        "BNP.PA", "AXA.PA", "GLE.PA", "DBK.DE", "ALV.DE", "MUV2.DE",
        "HSBA.L", "BARC.L", "LLOY.L", "INGA.AS", "NN.AS",
        "SAN.MC", "BBVA.MC", "UCG.MI", "ISP.MI",
        "AGS.BR", "ACKB.BR", "ZURN.SW",
    ],
    "Santé Europe": [
        "ROG.SW", "NOVN.SW", "NESN.SW", "AZN.L", "GSK.L",
        "BAYN.DE", "SAN.PA", "UCB.BR", "ARGX.BR", "EL.PA", "LONN.SW",
    ],
    "Energie Europe": [
        "TTE.PA", "ENGI.PA", "SHEL.L", "BP.L", "ENI.MI",
        "IBE.MC", "ENEL.MI", "NESTE.HE",
    ],
    "Conso Europe": [
        "MC.PA", "OR.PA", "RMS.PA", "CDI.PA", "KER.PA",
        "HEIA.AS", "DGE.L", "ULVR.L", "COLR.BR",
        "ITX.MC", "CFR.SW",
    ],
    "Industriels Europe": [
        "SIE.DE", "ABB.SW", "AIR.PA", "SAF.PA", "ALO.PA",
        "DG.PA", "SGO.PA", "VOLV-B.ST", "RAND.AS",
        "SIKA.SW", "AKZA.AS", "SOLB.BR", "WDP.BR",
    ],

    # ── ASIE / MONDE ────────────────────────────────────────────
    "Tech Asie": [
        "TSM", "SONY", "9988.HK", "700.HK", "005930.KS",
    ],
    "Finance Asie": [
        "MUFG", "8306.T", "939.HK",
    ],
    "Energie Asie": [
        "PTR", "SNP", "STO",
    ],
}

SOFT_DISCLAIMER = (
    "ℹ️ Ces résultats sont fournis à titre indicatif pour vous aider dans votre réflexion. "
    "Ils ne remplacent pas une analyse complète (rapports annuels, contexte sectoriel, risques). "
    "Si une opportunité vous intéresse, prenez le temps d'approfondir avant toute décision."
)

FMP_BASE = "https://financialmodelingprep.com/api/v3"


# =========================
# CONFIG
# =========================

@dataclass
class Thresholds:
    pe_max: float = 35.0
    pb_max: float = 4.0
    ev_ebitda_max: float = 20.0
    roe_min: float = 0.08
    margin_min: float = 0.05
    debt_to_equity_max: float = 1.0
    rev_growth_min: float = 0.03
    dividend_min: float = 0.01


@dataclass
class Weights:
    valuation: float = 0.25
    profitability: float = 0.30
    financial_health: float = 0.20
    growth: float = 0.15
    dividend: float = 0.10


# =========================
# HELPERS
# =========================

def safe_float(x, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, (int, float, np.integer, np.floating)):
            v = float(x)
            return default if (np.isnan(v) or np.isinf(v)) else v
        return float(str(x).replace(",", "."))
    except Exception:
        return default


def clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def normalize_div(dy) -> float:
    """Retourne dividende en % (ex: 3.2).
    yfinance retourne dividendYield comme ratio (0.032 = 3.2%).
    Cap a 15% pour eviter les glitches.
    """
    dy = safe_float(dy, 0.0)
    if dy <= 0:
        return 0.0
    pct = dy * 100 if dy < 1 else dy
    if pct > 15:
        return 0.0
    return round(pct, 2)


def score_badge(s: float) -> str:
    if s >= 70: return "🔥"
    if s >= 55: return "✅"
    if s >= 40: return "⚠️"
    return "🧊"


def confidence_badge(c: float) -> str:
    if c >= 80: return "🟢"
    if c >= 60: return "🟡"
    return "🔴"


def format_div(d: float) -> str:
    return "Non" if d <= 0 else "Oui"


# =========================
# FMP CLIENT
# =========================

class FMPClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()

    def _get(self, endpoint: str, params: dict = None) -> Optional[dict | list]:
        p = params or {}
        p["apikey"] = self.api_key
        try:
            r = self.session.get(f"{FMP_BASE}/{endpoint}", params=p, timeout=8)
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list) and len(data) > 0:
                    return data
                if isinstance(data, dict) and data:
                    return data
            return None
        except Exception:
            return None

    def get_profile(self, ticker: str) -> Optional[dict]:
        data = self._get(f"profile/{ticker}")
        if data and isinstance(data, list):
            return data[0]
        return None

    def get_quote(self, ticker: str) -> Optional[dict]:
        """Endpoint gratuit FMP — prix, PE, volume."""
        data = self._get(f"quote/{ticker}")
        if data and isinstance(data, list):
            return data[0]
        return None

    def get_ratios(self, ticker: str) -> Optional[dict]:
        """Ratios annuels — gratuit sur FMP (pas TTM)."""
        data = self._get(f"ratios/{ticker}", {"limit": 1})
        if data and isinstance(data, list):
            return data[0]
        return None

    def get_income(self, ticker: str) -> Optional[dict]:
        """Income statement annuel — gratuit."""
        data = self._get(f"income-statement/{ticker}", {"limit": 2})
        if data and isinstance(data, list):
            return data
        return None

    def get_balance(self, ticker: str) -> Optional[dict]:
        """Balance sheet annuel — gratuit."""
        data = self._get(f"balance-sheet-statement/{ticker}", {"limit": 1})
        if data and isinstance(data, list):
            return data[0]
        return None

    def search_ticker(self, query: str, limit: int = 10) -> list:
        data = self._get("search", {"query": query, "limit": limit})
        if isinstance(data, list):
            return data
        return []


# =========================
# METRICS FETCHER
# =========================

def fetch_metrics(ticker: str, client: FMPClient = None) -> Optional[dict]:
    """Fetch via yfinance (fonctionne depuis un serveur, gratuit, fiable)."""
    try:
        import yfinance as yf
        # fast_info est plus fiable que .info sur serveur
        t = yf.Ticker(ticker)
        # Essayer fast_info d abord pour le prix
        try:
            fast = t.fast_info
            price_check = float(fast.last_price) if hasattr(fast, "last_price") else 0
        except Exception:
            price_check = 0
        info = t.info
        # Si info semble vide ou stale, retenter
        if not info or len(info) < 5:
            return None
        # Utiliser le prix de fast_info si disponible et different
        if price_check > 0 and abs(price_check - safe_float(info.get("regularMarketPrice") or info.get("currentPrice"), 0)) > 0.01:
            info["regularMarketPrice"] = price_check
    except Exception:
        return None

    price = safe_float(info.get("regularMarketPrice") or info.get("currentPrice"), 0.0)
    mcap = safe_float(info.get("marketCap"), 0.0)
    currency = info.get("currency", "USD")

    if price <= 0:
        return None
    if mcap < 200_000_000:
        return None

    # Conversion de devise — tout en USD ou EUR
    # Actions européennes cotées en EUR/GBp/CHF → EUR
    # Actions Asie/US cotées en KRW/JPY/TWD/HKD → USD
    target_currency = currency
    try:
        if currency not in ("USD", "EUR"):
            import yfinance as _yf
            if currency in ("GBp", "GBX"):
                # Pence → Livres sterling → EUR
                price = price / 100.0
                gbp_eur = _yf.Ticker("GBPEUR=X").fast_info.last_price or 1.17
                price = round(price * gbp_eur, 2)
                target_currency = "EUR"
            elif currency in ("CHF",):
                chf_eur = _yf.Ticker("CHFEUR=X").fast_info.last_price or 1.05
                price = round(price * chf_eur, 2)
                target_currency = "EUR"
            elif currency in ("SEK", "NOK", "DKK"):
                pair = f"{currency}EUR=X"
                rate = _yf.Ticker(pair).fast_info.last_price or 0.09
                price = round(price * rate, 2)
                target_currency = "EUR"
            elif currency in ("KRW", "JPY", "TWD", "HKD", "CNY", "INR"):
                pair = f"{currency}USD=X"
                rate = _yf.Ticker(pair).fast_info.last_price or 0.001
                price = round(price * rate, 2)
                target_currency = "USD"
    except Exception:
        target_currency = currency  # fallback devise originale

    pe = safe_float(info.get("trailingPE") or info.get("forwardPE"), 0.0)
    pb = safe_float(info.get("priceToBook"), 0.0)
    pb = pb if 0 < pb < 100 else 0.0  # filtrer valeurs aberrantes (ex: ASML)

    # EV/EBITDA — fix bug devise yfinance
    enterprise_value = safe_float(info.get("enterpriseValue"), 0.0)
    ebitda_margins = safe_float(info.get("ebitdaMargins"), 0.0)
    total_revenue = safe_float(info.get("totalRevenue"), 0.0)
    currency = info.get("currency", "USD")
    financial_currency = info.get("financialCurrency", currency)

    ev_ebitda = 0.0

    # Méthode 1 : valeur directe yfinance si réaliste (3-100)
    ev_direct = safe_float(info.get("enterpriseToEbitda"), 0.0)
    if 3 <= ev_direct <= 100:
        ev_ebitda = ev_direct

    elif enterprise_value > 0 and ebitda_margins > 0 and total_revenue > 0:
        if currency == financial_currency:
            # Même devise — recalcul via ebitdaMargins × revenue (fiable)
            ebitda_recalc = ebitda_margins * total_revenue
            if ebitda_recalc > 0:
                ev_calc = enterprise_value / ebitda_recalc
                if 3 <= ev_calc <= 100:
                    ev_ebitda = round(ev_calc, 2)
        # Devises différentes (ex: TSM USD/TWD) → impossible sans taux fiable
        # On laisse 0 → affiché — dans l'interface, honnête et transparent
    # PEG Ratio
    peg = safe_float(info.get("trailingPegRatio"), 0.0)
    peg = peg if 0 < peg < 10 else 0.0  # filtrer valeurs aberrantes

    # Capitalisation boursière
    mcap_raw = safe_float(info.get("marketCap"), 0.0)
    if mcap_raw >= 200_000_000_000:
        cap_category = "Large Cap"
    elif mcap_raw >= 10_000_000_000:
        cap_category = "Mid Cap"
    elif mcap_raw >= 2_000_000_000:
        cap_category = "Small Cap"
    else:
        cap_category = "Micro Cap"

    roe = safe_float(info.get("returnOnEquity"), 0.0)
    margin = safe_float(info.get("profitMargins"), 0.0)

    # Dette/Equity — calcul manuel si yfinance ne retourne rien
    dte_raw = safe_float(info.get("debtToEquity"), 0.0)
    if dte_raw > 0:
        dte = (dte_raw / 100.0) if dte_raw > 10 else dte_raw
    else:
        # Recalcul depuis totalDebt / totalStockholderEquity
        total_debt = safe_float(info.get("totalDebt"), 0.0)
        stockholder_equity = safe_float(info.get("totalStockholderEquity") or info.get("stockholdersEquity"), 0.0)
        if total_debt > 0 and stockholder_equity > 0:
            dte = round(total_debt / stockholder_equity, 2)
        else:
            dte = 0.0

    rev_growth = safe_float(info.get("revenueGrowth"), 0.0)
    revenue = safe_float(info.get("totalRevenue"), 0.0)
    ocf = safe_float(info.get("operatingCashflow"), 0.0)

    # Dividende — uniquement trailingAnnualDividendYield, le plus fiable
    div_yield = 0.0
    dy = safe_float(info.get("trailingAnnualDividendYield"), 0.0)
    if 0 < dy < 0.20:  # entre 0% et 20%
        div_yield = dy

    return {
        "ticker": ticker,
        "name": info.get("longName") or info.get("shortName") or ticker,
        "currency": target_currency,
        "exchange": info.get("exchange", ""),
        "country": info.get("country", ""),
        "sector": info.get("sector", ""),
        "price": price,
        "mcap": mcap,
        "pe": pe,
        "pb": pb,
        "ev_ebitda": ev_ebitda,
        "peg": peg,
        "market_cap": mcap_raw,
        "cap_category": cap_category,
        "roe": roe,
        "margin": margin,
        "debt_to_equity": dte,
        "revenue": revenue,
        "ocf": ocf,
        "rev_growth": rev_growth,
        "div_yield": div_yield,
    }


# =========================
# CONFIDENCE MODEL
# =========================

def quality_confidence(m: dict) -> float:
    keys = ["pe", "pb", "ev_ebitda", "roe", "margin", "debt_to_equity", "rev_growth", "div_yield"]

    # 1) Complétude
    present = sum(1 for k in keys if safe_float(m.get(k), 0.0) != 0.0)
    completeness = (present / len(keys)) * 100

    # 2) Sanity checks
    penalties = 0
    pe = safe_float(m.get("pe"), 0.0)
    pb = safe_float(m.get("pb"), 0.0)
    ev = safe_float(m.get("ev_ebitda"), 0.0)
    roe_pct = safe_float(m.get("roe"), 0.0) * 100
    margin_pct = safe_float(m.get("margin"), 0.0) * 100
    dte = safe_float(m.get("debt_to_equity"), 0.0)
    dy = normalize_div(m.get("div_yield"))
    rg_pct = safe_float(m.get("rev_growth"), 0.0) * 100

    if pe and (pe < 1 or pe > 150): penalties += 12
    if pb and (pb < 0.1 or pb > 60): penalties += 8
    if ev and (ev < 1 or ev > 100): penalties += 10
    if roe_pct and (roe_pct < -60 or roe_pct > 200): penalties += 10
    if margin_pct and (margin_pct < -40 or margin_pct > 70): penalties += 10
    if dte and dte > 8: penalties += 10
    if dy and dy > 15: penalties += 12
    if rg_pct and (rg_pct < -60 or rg_pct > 100): penalties += 8

    sanity = clamp(100 - penalties)

    # 3) Fraîcheur
    freshness = 100.0
    if safe_float(m.get("price"), 0.0) <= 0: freshness -= 40
    if safe_float(m.get("mcap"), 0.0) <= 0: freshness -= 25
    if not m.get("currency"): freshness -= 10
    freshness = clamp(freshness)

    conf = 0.40 * completeness + 0.40 * sanity + 0.20 * freshness - 3.0
    return round(clamp(conf, 30.0, 92.0), 1)


# =========================
# SCORER
# =========================

class SmartValueScorer:
    def __init__(self, th: Thresholds = Thresholds(), w: Weights = Weights()):
        self.th = th
        self.w = w

    def score(self, m: dict) -> Tuple[float, dict, List[str], float, List[str], str]:
        details = {}
        why: List[str] = []
        confidence = quality_confidence(m)

        pe = safe_float(m.get("pe"), 0.0)
        pb = safe_float(m.get("pb"), 0.0)
        ev = safe_float(m.get("ev_ebitda"), 0.0)
        roe = safe_float(m.get("roe"), 0.0)
        roe_pct = roe * 100
        margin = safe_float(m.get("margin"), 0.0)
        margin_pct = margin * 100
        dte = safe_float(m.get("debt_to_equity"), 0.0)
        revenue = safe_float(m.get("revenue"), 0.0)
        ocf = safe_float(m.get("ocf"), 0.0)
        rg = safe_float(m.get("rev_growth"), 0.0)
        rg_pct = rg * 100
        dy_pct = normalize_div(m.get("div_yield"))

        # --- VALUATION ---
        val = 0.0
        # PER — score complet si < pe_max, score partiel si PE eleve mais justifie par croissance
        if 1 < pe < self.th.pe_max:
            if pe < 12:
                val += 100 * 0.50; why.append(f"PER très bas ({pe:.1f})")
            elif pe < 18:
                val += 80 * 0.50; why.append(f"PER raisonnable ({pe:.1f})")
            else:
                val += 55 * 0.50
        elif 1 < pe and rg_pct > 20:
            # Entreprise croissance : PER eleve mais tolere si croissance forte
            val += 25 * 0.50
        if 0 < pb < self.th.pb_max:
            val += clamp(100 * (self.th.pb_max - pb) / self.th.pb_max) * 0.30
            if pb < 2: why.append(f"P/B attractif ({pb:.2f})")
        elif 0 < pb and rg_pct > 30:
            # P/B eleve tolere si hyper-croissance
            val += 10 * 0.30
        if 1 < ev < self.th.ev_ebitda_max:
            val += clamp(100 * (self.th.ev_ebitda_max - ev) / self.th.ev_ebitda_max) * 0.20
            if ev < 12: why.append(f"EV/EBITDA sain ({ev:.1f})")
        details["valuation"] = round(val, 1)

        # --- PROFITABILITY ---
        prof = 0.0
        roe_capped = min(roe_pct, 60)  # cap pour eviter distorsion (NVDA ROE 101%)
        if roe_capped > 0:
            if roe_capped > 25:
                prof += 100 * 0.50; why.append(f"ROE exceptionnel ({roe_pct:.1f}%)")
            elif roe_capped > 18:
                prof += 85 * 0.50; why.append(f"ROE très solide ({roe_pct:.1f}%)")
            elif roe_capped > 12:
                prof += 65 * 0.50
            elif roe_capped > 8:
                prof += 45 * 0.50
        if margin > self.th.margin_min:
            prof += clamp((margin - self.th.margin_min) * 350) * 0.50
            if margin > 0.15: why.append(f"Marges solides ({margin_pct:.1f}%)")
        details["profitability"] = round(prof, 1)

        # --- FINANCIAL HEALTH ---
        health = 0.0
        sector = m.get("sector_name", "")
        is_bank = any(s in sector for s in ["Finance", "Bank", "Insurance"])
        if dte > 0:
            if dte < 0.30:
                health += 100 * 0.55; why.append("Dette très faible")
            elif dte < 0.60:
                health += 75 * 0.55
            elif dte < 1.0:
                health += 45 * 0.55
            elif dte < 1.5:
                health += 20 * 0.55
        elif is_bank:
            # Pour les banques sans DTE disponible, score neutre basé sur cashflow seul
            health += 55 * 0.55  # score neutre
        # Cashflow — ignoré pour les banques (OCF naturellement volatile/négatif)
        if revenue > 0 and ocf > 0 and not is_bank:
            cf_m = ocf / revenue
            if cf_m > 0.18:
                health += 100 * 0.45; why.append("Cashflow excellent")
            elif cf_m > 0.12:
                health += 75 * 0.45
            elif cf_m > 0.06:
                health += 45 * 0.45
        elif is_bank:
            # Pour les banques, score cashflow neutre
            health += 65 * 0.45
        details["financial_health"] = round(health, 1)

        # --- GROWTH ---
        growth = 0.0
        if rg > 0.20:
            growth = 100; why.append(f"Croissance forte ({rg_pct:.1f}%)")
        elif rg > 0.12:
            growth = 80
        elif rg > 0.07:
            growth = 60
        elif rg > 0.03:
            growth = 40
        elif rg > 0:
            growth = 20
        details["growth"] = round(growth, 1)

        # --- DIVIDEND ---
        div = 0.0
        if dy_pct > 6:
            div = 100; why.append(f"Dividende élevé ({dy_pct:.1f}%)")
        elif dy_pct > 4:
            div = 80
        elif dy_pct > 3:
            div = 60; why.append(f"Dividende attractif ({dy_pct:.1f}%)")
        elif dy_pct > 2:
            div = 40
        elif dy_pct > 1:
            div = 20
        details["dividend"] = round(div, 1)

        # --- TOTAL ---
        total = (
            details["valuation"] * self.w.valuation
            + details["profitability"] * self.w.profitability
            + details["financial_health"] * self.w.financial_health
            + details["growth"] * self.w.growth
            + details["dividend"] * self.w.dividend
        )

        # --- TAGS ---
        tags: List[str] = []
        if 0 < pe < 15: tags.append("VALUE")
        if 0 < pb < 2: tags.append("ASSET")
        if roe_pct > 20 and margin > 0.12: tags.append("QUALITY")
        if dte < 0.60: tags.append("SAFE")
        if rg_pct > 8: tags.append("GROWTH")
        if dy_pct >= 2: tags.append("DIVIDEND")

        # --- RÉSUMÉ ---
        parts = []
        if "VALUE" in tags: parts.append("valorisation attractive")
        if "QUALITY" in tags: parts.append("business rentable")
        if "SAFE" in tags: parts.append("bilan sain")
        if "GROWTH" in tags: parts.append("croissance solide")
        if "DIVIDEND" in tags: parts.append("dividende intéressant")
        summary = ", ".join(parts[:2]) if parts else "profil équilibré"

        return round(total, 1), details, why[:3], confidence, tags, summary


# =========================
# SCANNER
# =========================

TAG_MAP = {
    "VALUE": "VALUE",
    "QUALITY": "QUALITÉ",
    "SAFE": "SÛR",
    "DIVIDEND": "DIVIDENDE",
    "ASSET": "ACTIFS",
    "GROWTH": "CROISSANCE",
}


def translate_tags(tags: List[str]) -> str:
    return ", ".join(TAG_MAP.get(t, t) for t in tags)


class SmartValueScanner:
    def __init__(
        self,
        api_key: str,
        universe: Dict[str, List[str]] = None,
        th: Thresholds = Thresholds(),
        w: Weights = Weights(),
    ):
        self.client = FMPClient(api_key)
        self.universe = universe or DEFAULT_UNIVERSE
        self.scorer = SmartValueScorer(th, w)

    def scan_ticker(self, ticker: str, sector_label: str = "Recherche") -> Optional[dict]:
        """Scanner un ticker individuel (recherche libre)."""
        m = fetch_metrics(ticker, self.client)
        if not m:
            return None
        score, details, why, confidence, tags, summary = self.scorer.score(m)
        return self._build_result(m, score, details, why, confidence, tags, summary, sector_label)

    def scan(
        self,
        min_score: float = 35,
        min_confidence: float = 50,
        progress_callback=None,
    ) -> List[dict]:
        tickers = [(sector, t) for sector, lst in self.universe.items() for t in lst]
        results: List[dict] = []
        total = len(tickers)

        for i, (sector, ticker) in enumerate(tickers):
            if progress_callback:
                progress_callback(i / total, f"Analyse {ticker}...")

            m = fetch_metrics(ticker, self.client)
            if not m:
                time.sleep(0.1)
                continue

            score, details, why, confidence, tags, summary = self.scorer.score(m)

            if confidence < min_confidence or score < min_score:
                time.sleep(0.05)
                continue

            results.append(
                self._build_result(m, score, details, why, confidence, tags, summary, sector)
            )
            time.sleep(0.12)  # respecter rate limit FMP

        results.sort(key=lambda x: x["Score"], reverse=True)
        return results

    def _build_result(self, m, score, details, why, confidence, tags, summary, sector) -> dict:
        div_pct = normalize_div(m.get("div_yield"))
        pe_val = safe_float(m.get("pe"), 0.0)
        pb_val = safe_float(m.get("pb"), 0.0)
        ev_val = safe_float(m.get("ev_ebitda"), 0.0)
        roe_val = safe_float(m.get("roe"), 0.0) * 100
        margin_val = safe_float(m.get("margin"), 0.0) * 100
        dte_val = safe_float(m.get("debt_to_equity"), 0.0)
        rg_val = safe_float(m.get("rev_growth"), 0.0) * 100

        return {
            "Ticker": m["ticker"],
            "Société": (m["name"][:45] + "…") if len(m["name"]) > 45 else m["name"],
            "Secteur": sector,
            "Pays": m.get("country", ""),
            "Bourse": m.get("exchange", ""),
            "Prix": round(m["price"], 2),
            "Devise": m["currency"],

            "PER": round(pe_val, 2) if pe_val else None,
            "P/B": round(pb_val, 2) if pb_val else None,
            "EV/EBITDA": round(ev_val, 2) if ev_val else None,
            "ROE %": round(roe_val, 1) if roe_val else None,
            "Marge %": round(margin_val, 1) if margin_val else None,
            "Dette/Equity": round(dte_val, 2) if dte_val else None,
            "Croissance CA %": round(rg_val, 1) if rg_val else 0.0,
            "PEG": round(safe_float(m.get("peg"), 0.0), 2) if m.get("peg") else None,
        "Cap boursière": m.get("cap_category", ""),
        "Market Cap": m.get("market_cap", 0),
        "Div %": div_pct,
            "Div affichage": format_div(div_pct),

            "Score": score,
            "Confiance %": confidence,
            "Score badge": score_badge(score),
            "Confiance badge": confidence_badge(confidence),

            "Tags": translate_tags(tags),
            "Résumé": summary,
            "Pourquoi": " | ".join(why),

            "Bloc valuation": details["valuation"],
            "Bloc rentabilité": details["profitability"],
            "Bloc santé": details["financial_health"],
            "Bloc croissance": details["growth"],
            "Bloc dividende": details["dividend"],
        }

    def search(self, query: str) -> list:
        return self.client.search_ticker(query)

    def to_email_markdown(self, results: List[dict], top_n: int = 5) -> str:
        lines = ["# 🔎 SmartValue Scanner | Sélection du moment\n"]
        for i, r in enumerate(results[:top_n], 1):
            lines.append(
                f"## {i}) {r['Score badge']} {r['Ticker']} ({r['Secteur']}) "
                f"| Score {r['Score']}/100 | Confiance {r['Confiance badge']} {r['Confiance %']}%\n"
            )
            lines.append(f"- Prix: {r['Prix']} {r['Devise']} | Pays: {r['Pays']}")
            if r.get("PER"): lines.append(f"- PER: {r['PER']} | ROE: {r['ROE %']}% | Marge: {r['Marge %']}%")
            lines.append(
                f"- Dette/Equity: {r['Dette/Equity']} | Dividende: {r['Div affichage']}% | "
                f"Croissance CA: {r['Croissance CA %']}%"
            )
            lines.append(f"- Tags: {r['Tags']}")
            lines.append(f"- Résumé: {r['Résumé']}")
            lines.append(f"- Pourquoi: {r['Pourquoi']}\n")
        lines.append(f"> {SOFT_DISCLAIMER}\n")
        return "\n".join(lines)
