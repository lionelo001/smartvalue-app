"""
SmartValue Newsletter — Top 5 hebdomadaire
Envoi automatique chaque lundi à 8h via Brevo
Usage : python newsletter.py        → envoi immédiat (test)
        python newsletter.py --schedule → mode automatique lundi 8h
"""
from __future__ import annotations
import os
import json
import requests
from datetime import datetime

BREVO_API_KEY = os.environ.get("BREVO_API_KEY", "")
BREVO_LIST_ID = 2
SCANNER_URL = "https://smartvaluescanner.com"
CACHE_FILE = "cache.json"
SENDER_EMAIL = "newsletter@smartvaluescanner.com"
SENDER_NAME = "SmartValue Scanner"


# =========================
# LECTURE DU CACHE
# =========================

def load_results() -> list:
    """Charge les résultats depuis le cache disque."""
    if not os.path.exists(CACHE_FILE):
        print("[Newsletter] Pas de cache disque trouvé")
        return []
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("results", [])


def get_top5(results: list) -> list:
    """Retourne le Top 5 trié par score."""
    sorted_results = sorted(results, key=lambda x: x.get("Score", 0), reverse=True)
    return sorted_results[:5]


# =========================
# GÉNÉRATION DU CONTEXTE
# =========================

def generate_context(top5: list, all_results: list) -> str:
    """Génère une phrase de contexte automatique selon les résultats."""
    if not top5:
        return "Voici les meilleures opportunités détectées cette semaine."

    # Secteurs dominants dans le top 5
    sectors = [r.get("Secteur", "") for r in top5]
    sector_count = {}
    for s in sectors:
        base = s.split(" ")[0] if s else "Autre"
        sector_count[base] = sector_count.get(base, 0) + 1

    dominant = max(sector_count, key=sector_count.get)
    count = sector_count[dominant]

    # Score moyen du top 5
    avg_score = sum(r.get("Score", 0) for r in top5) / len(top5)

    # Meilleur score
    best = top5[0]
    best_ticker = best.get("Ticker", "")
    best_score = best.get("Score", 0)

    if count >= 3:
        return f"Cette semaine le secteur {dominant} domine avec {count} entrées dans le Top 5. {best_ticker} est en tête avec un score de {best_score}/100."
    elif avg_score >= 70:
        return f"Excellente semaine — le score moyen du Top 5 atteint {avg_score:.1f}/100. {best_ticker} s'impose en tête avec {best_score}/100."
    else:
        return f"Cette semaine {best_ticker} mène le classement avec {best_score}/100. {len(all_results)} actions analysées au total."


# =========================
# GÉNÉRATION DE L'EMAIL HTML
# =========================

def build_email_html(top5: list, context: str, week_date: str) -> str:
    """Génère le HTML de l'email newsletter."""

    medal = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]

    cards_html = ""
    for i, r in enumerate(top5):
        ticker = r.get("Ticker", "")
        name = r.get("Société", ticker)
        score = r.get("Score", 0)
        conf = r.get("Confiance %", 0)
        tags = r.get("Tags", "")
        pourquoi = r.get("Pourquoi", "")
        per = r.get("PER")
        roe = r.get("ROE %")
        div = r.get("Div %", 0)
        secteur = r.get("Secteur", "")
        pays = r.get("Pays", "")

        # Couleur score
        if score >= 70:
            score_color = "#16a34a"
        elif score >= 50:
            score_color = "#d97706"
        else:
            score_color = "#dc2626"

        # Métriques
        per_txt = f"{per:.1f}" if per else "—"
        roe_txt = f"{roe:.1f}%" if roe else "—"
        div_txt = f"{div:.2f}%" if div and div > 0 else "Non"

        # Lien analyse
        link = f"{SCANNER_URL}/?q={ticker}"

        cards_html += f"""
        <tr>
          <td style="padding:0 0 16px 0;">
            <table width="100%" cellpadding="0" cellspacing="0" style="background:#ffffff;border:1px solid #e5e7eb;border-radius:12px;overflow:hidden;">
              <tr>
                <td style="padding:16px 20px;">
                  <!-- Header action -->
                  <table width="100%" cellpadding="0" cellspacing="0">
                    <tr>
                      <td>
                        <span style="font-size:1.1rem;">{medal[i]}</span>
                        <span style="font-size:1rem;font-weight:700;color:#111827;margin-left:6px;">{ticker}</span>
                        <span style="font-size:0.82rem;color:#6b7280;margin-left:4px;">— {name}</span>
                      </td>
                      <td align="right">
                        <span style="font-size:1.4rem;font-weight:800;color:{score_color};">{score}</span>
                        <span style="font-size:0.75rem;color:#6b7280;">/100</span>
                        <br>
                        <span style="font-size:0.7rem;color:#16a34a;background:#f0fdf4;padding:2px 6px;border-radius:10px;">{conf}%</span>
                      </td>
                    </tr>
                  </table>
                  <!-- Secteur & Pays -->
                  <p style="margin:6px 0 8px;font-size:0.75rem;color:#9ca3af;">{pays} · {secteur}</p>
                  <!-- Tags -->
                  <p style="margin:0 0 10px;">{' · '.join(['<span style="font-size:0.7rem;background:#f3f4f6;color:#374151;padding:2px 8px;border-radius:10px;margin-right:4px;">' + t.strip() + '</span>' for t in tags.split(',') if t.strip()])}</p>
                  <!-- Métriques -->
                  <table cellpadding="0" cellspacing="0" style="margin-bottom:10px;">
                    <tr>
                      <td style="padding-right:20px;">
                        <span style="font-size:0.7rem;color:#9ca3af;display:block;">PER</span>
                        <span style="font-size:0.9rem;font-weight:600;color:#111827;">{per_txt}</span>
                      </td>
                      <td style="padding-right:20px;">
                        <span style="font-size:0.7rem;color:#9ca3af;display:block;">ROE</span>
                        <span style="font-size:0.9rem;font-weight:600;color:#111827;">{roe_txt}</span>
                      </td>
                      <td>
                        <span style="font-size:0.7rem;color:#9ca3af;display:block;">DIVIDENDE</span>
                        <span style="font-size:0.9rem;font-weight:600;color:#111827;">{div_txt}</span>
                      </td>
                    </tr>
                  </table>
                  <!-- Pourquoi -->
                  {f'<p style="margin:0 0 12px;font-size:0.78rem;color:#4b5563;font-style:italic;">💡 {pourquoi}</p>' if pourquoi else ''}
                  <!-- CTA -->
                  <a href="{link}" style="display:inline-block;background:#2563eb;color:#ffffff;font-size:0.78rem;font-weight:600;padding:8px 16px;border-radius:8px;text-decoration:none;">
                    Voir l'analyse complète →
                  </a>
                </td>
              </tr>
            </table>
          </td>
        </tr>
        """

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Top 5 SmartValue — {week_date}</title>
</head>
<body style="margin:0;padding:0;background:#f3f4f6;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background:#f3f4f6;padding:32px 16px;">
    <tr>
      <td align="center">
        <table width="600" cellpadding="0" cellspacing="0" style="max-width:600px;width:100%;">

          <!-- HEADER -->
          <tr>
            <td style="background:#111827;border-radius:12px 12px 0 0;padding:28px 32px;text-align:center;">
              <p style="margin:0 0 4px;font-size:0.75rem;color:#9ca3af;letter-spacing:0.1em;text-transform:uppercase;">SmartValue Scanner</p>
              <h1 style="margin:0;font-size:1.4rem;color:#ffffff;font-weight:800;">📊 Top 5 de la semaine</h1>
              <p style="margin:8px 0 0;font-size:0.82rem;color:#9ca3af;">{week_date}</p>
            </td>
          </tr>

          <!-- CONTEXTE -->
          <tr>
            <td style="background:#eff6ff;border:1px solid #bfdbfe;padding:16px 32px;">
              <p style="margin:0;font-size:0.85rem;color:#1e40af;line-height:1.6;">
                🔍 {context}
              </p>
            </td>
          </tr>

          <!-- ACTIONS -->
          <tr>
            <td style="background:#f9fafb;padding:20px 24px 4px;">
              <table width="100%" cellpadding="0" cellspacing="0">
                {cards_html}
              </table>
            </td>
          </tr>

          <!-- CTA SCANNER -->
          <tr>
            <td style="background:#f9fafb;padding:4px 24px 24px;text-align:center;">
              <a href="{SCANNER_URL}/app" style="display:inline-block;background:#111827;color:#ffffff;font-size:0.85rem;font-weight:700;padding:14px 28px;border-radius:10px;text-decoration:none;letter-spacing:0.02em;">
                Accéder au scanner complet →
              </a>
            </td>
          </tr>

          <!-- DISCLAIMER -->
          <tr>
            <td style="background:#f3f4f6;border-radius:0 0 12px 12px;padding:20px 32px;text-align:center;">
              <p style="margin:0 0 8px;font-size:0.7rem;color:#9ca3af;line-height:1.5;">
                Ces résultats sont générés automatiquement par l'algorithme SmartValue et ne constituent pas un conseil en investissement. Toute décision reste sous votre responsabilité.
              </p>
              <p style="margin:0;font-size:0.7rem;color:#d1d5db;">
                © 2026 SmartValue Scanner · <a href="{SCANNER_URL}" style="color:#6b7280;">smartvaluescanner.com</a>
              </p>
            </td>
          </tr>

        </table>
      </td>
    </tr>
  </table>
</body>
</html>"""

    return html


# =========================
# ENVOI VIA BREVO
# =========================

def get_brevo_contacts() -> list:
    """Récupère tous les contacts de la liste Brevo."""
    contacts = []
    offset = 0
    limit = 500

    while True:
        r = requests.get(
            f"https://api.brevo.com/v3/contacts",
            headers={"api-key": BREVO_API_KEY},
            params={"limit": limit, "offset": offset, "listId": BREVO_LIST_ID},
            timeout=10
        )
        if r.status_code != 200:
            print(f"[Newsletter] Erreur Brevo contacts : {r.status_code}")
            break

        data = r.json()
        batch = data.get("contacts", [])
        contacts.extend(batch)

        if len(batch) < limit:
            break
        offset += limit

    return contacts


def send_campaign(html: str, subject: str) -> bool:
    """Envoie la newsletter via Brevo campaign."""
    payload = {
        "name": f"Top 5 SmartValue — {datetime.now().strftime('%d/%m/%Y')}",
        "subject": subject,
        "sender": {"name": SENDER_NAME, "email": SENDER_EMAIL},
        "type": "classic",
        "htmlContent": html,
        "recipients": {"listIds": [BREVO_LIST_ID]},
        "scheduledAt": None,  # Envoi immédiat
    }

    r = requests.post(
        "https://api.brevo.com/v3/emailCampaigns",
        headers={"api-key": BREVO_API_KEY, "Content-Type": "application/json"},
        json=payload,
        timeout=15
    )

    if r.status_code in [200, 201]:
        campaign_id = r.json().get("id")
        print(f"[Newsletter] Campagne créée (ID: {campaign_id})")

        # Envoyer immédiatement
        r2 = requests.post(
            f"https://api.brevo.com/v3/emailCampaigns/{campaign_id}/sendNow",
            headers={"api-key": BREVO_API_KEY},
            timeout=10
        )
        if r2.status_code == 204:
            print("[Newsletter] ✅ Envoyée avec succès !")
            return True
        else:
            print(f"[Newsletter] Erreur envoi : {r2.status_code} — {r2.text}")
            return False
    else:
        print(f"[Newsletter] Erreur création campagne : {r.status_code} — {r.text}")
        return False


# =========================
# FONCTION PRINCIPALE
# =========================

def send_newsletter():
    """Génère et envoie le Top 5 de la semaine."""
    print(f"[Newsletter] Démarrage — {datetime.now().strftime('%d/%m/%Y %H:%M')}")

    # Charger les résultats
    results = load_results()
    if not results:
        print("[Newsletter] Aucun résultat disponible — abandon")
        return False

    # Top 5
    top5 = get_top5(results)
    if not top5:
        print("[Newsletter] Top 5 vide — abandon")
        return False

    print(f"[Newsletter] Top 5 prêt : {[r['Ticker'] for r in top5]}")

    # Contexte automatique
    context = generate_context(top5, results)
    print(f"[Newsletter] Contexte : {context}")

    # Date de la semaine
    week_date = datetime.now().strftime("Semaine du %d %B %Y")

    # Sujet
    top_ticker = top5[0].get("Ticker", "")
    top_score = top5[0].get("Score", 0)
    subject = f"📊 Top 5 SmartValue — {week_date} | {top_ticker} en tête ({top_score}/100)"

    # Générer HTML
    html = build_email_html(top5, context, week_date)

    # Envoyer
    success = send_campaign(html, subject)
    return success


# =========================
# SCHEDULER LUNDI 8H
# =========================

def run_scheduler():
    """Tourne en arrière-plan, envoie chaque lundi à 8h."""
    import time as _time

    print("[Newsletter] Scheduler démarré — envoi chaque lundi à 8h00")

    while True:
        now = datetime.now()
        # Lundi = 0, 8h00
        if now.weekday() == 0 and now.hour == 8 and now.minute == 0:
            print("[Newsletter] C'est lundi 8h — envoi en cours...")
            send_newsletter()
            _time.sleep(61)  # Éviter double envoi dans la même minute
        _time.sleep(30)  # Vérifier toutes les 30 secondes


# =========================
# POINT D'ENTRÉE
# =========================

if __name__ == "__main__":
    import sys

    if "--schedule" in sys.argv:
        run_scheduler()
    else:
        # Envoi immédiat pour test
        print("[Newsletter] Mode test — envoi immédiat")
        send_newsletter()
